Goal of part 4: does a graph network generalise the middle game where the convolution did not?

# Part 4 plan: a graph network for Dot

Status: agreed 2026-11-07. Stage 0 and (i) run first, overnight on G16.
Dot-2 = dot2c gen 16 (runs/dot2c/gen_0016.pt; AZNet2 6x64, 0.51M parameters).

## Why

Part 3 left this:

- The hand generalises late parity (a2: raw 10x7 probe 87.5%), not the middle
  game: 80 rout mistakes held out 2/24 -> 6/24, p 0.14; set mid held out
  (decisive) 86.9 -> 90.5%, p 0.059. Neither passed the rule.
- Search carries the play: MCTS stays at 86-100% where the hand falls to
  53-85% (15-18 safe lines).
- Fine-tuning that moved V damaged play (b1, b2: routs 42-46 of 200, base 11).

The question: is the limit the representation? Part 3 (c) asked for a graph
only if b2 moved the probe and b1 did not. That evidence is weak: b2 82.6
[78-86] vs b1 78.0% overlap, and a3 showed the chain planes add nothing to the
head over a1. So this part starts with cheap checks and stops early.

## 0. Facts measured for this plan (2026-11-06, this machine)

Throwaway benchmark, RTX 4060 Laptop 8 GB, not in the repo.

| item | number |
|---|---:|
| AZNet2 with Evaluator, 10x7, batch 64 (encoding included) | 28k positions/s |
| Prototype graph net, 8 layers, width 110, 0.70M, no feature extraction | 7.0k positions/s |
| The same, 12 layers, width 90, 0.70M | 5.6k positions/s |
| sim/chains.py analyse, Python | 35 us per position |
| chain_solver.graph on random positions with 30-90 safe moves played | 60 of 300 parsed |

The prototype is unoptimised and its width is not the final one (0.5M = width
about 93 at 8 layers). Expect a graph net 2-4x slower than AZNet2.

Correction to the starting assumption: chain_solver.graph does not parse the
board in general. It raises Unsettled on capturable boxes and on any line
between two ends, so it needs the final board F of a settled position, and that
never exists at 20-40 safe lines (docs/chain_solver.md). The general
decomposition has to be new; the logic of Codec.chain_hint (vectorised, same
chains as sim/chains.py) is the closer starting point.

## 1. Board as a graph

Candidates:

- **(a) Graph as it stands.** Box and line nodes, line-box incidence. Works at
  every stage, and the topology is fixed per board size. Two defects: a chain of
  9+ boxes (9% of chains on 10x7) needs 18+ rounds to be crossed, and plain
  message passing (1-WL) cannot tell one 8-loop from two 4-loops, a difference
  the chain count depends on.
- **(b) Chain graph.** Chains and loops are nodes. Parity is nearly local, but
  it is defined only late. Early the board is one blob, so (b) has to fall back
  to (a) anyway.

**Recommended: (c) = (a) plus one segment layer.** Keep (a) as the base and add
segment nodes computed for the current board, not for the unknown final one:

- A segment is a maximal run of link boxes (two undrawn sides), with its length,
  kind (chain / loop), open ends (ground / junction / capturable) and count of
  ground ends. Each junction box (3-4 undrawn sides) and each untouched region
  stays in the base graph.
- Middle game: no "potential chains" are predicted. A partial chain is a short
  segment; a box with one drawn side is a box feature. A line carries: drawn,
  safe, boxes it would give, position in its segment (end / middle). What will
  form is left to the net, because predicting it needs the fight that the net
  has to judge.
- Safe lines need no separate nodes: they are line nodes with a flag, and the
  fact that two safe lines spoil each other travels through their shared box.
- The ground is a flag on lines, not one hub node. A hub would mix the whole
  border and hide the effect of the global readout.
- Not given as input: s, the long chain count, dots + long chains mod 2, any
  solver value. b2 already tested "telling" the count. Here the net must add it
  up. Otherwise arm (c) only repeats b2.

**Move.** Always a score on the line node, with the line mask of today. MCTS
priors, targets and the loss stay as they are, and all lines are covered at all
stages. "Chain + which line" is rejected: it needs chains at every stage and a
factorised target, and the difference inside a chain (the middle of a 2-chain,
the end of a longer one) is given by the position feature.

## 2. Architecture

- GIN-style sum aggregation with residual MLPs. Parity needs counts, and GAT's
  normalised attention loses them. GAT is not tried unless (i) asks for it.
- Layer: boxes <-> lines, lines <-> segments, a global vector into every node.
- 8 layers: segments already shorten long chains. 12 is the alternative tried in
  stage (i) only.
- Global readout: sums per node kind (segments by length class, junctions, safe
  lines), then an MLP, for V and score. Sum, not mean: a mean cannot count.
- Same parameters on every board size. 0.5M +-5%, the budget counted for the
  CNN arm too.
- LayerNorm, no BatchNorm: it removes the trap of part 3 (b) (log 2026-10-31).
- **Own code on torch, no PyG.** Topology is fixed per size, and segments are
  one id per line, padded to n_boxes. So there are no variable batches: fixed
  edge lists, index_add / scatter_add, masks. It fits Evaluator's call
  `boards -> (priors, values)`, so the C tree and az_mcts do not change. PyG adds
  a dependency to install.sh and to torch 2.14 for nothing used here.
- Segments are computed inside forward from the same planes as AZNet2 (torch,
  on the GPU), so Evaluator and the trainers do not change. C (agents/csrc) only
  if it takes over 20% of the time.

## 3. Comparison protocol (the same for both networks)

**Stage 0: capability, ~1 h.** Same data, target = sign of dots + long chains
mod 2 on settled positions, seen from the mover. Positions: random safe-move
playouts to the end of the safe lines, on 5x5 and 10x7. Pass: Gc >= 95% on
positions from other games. If (c) cannot learn this directly, stop and fix the
readout before anything else.

**(i) From scratch on the judge's late positions.** Purpose: parity without
search, no help from self-play.

- Data: data/parity_hand_{5x5,10x7}.jsonl.gz, 5000 per size, loss as part 3
  (mass on correct moves). Probes 1000 (5x5) and 304 (10x7) stay held out. 10%
  of the training positions for validation.
- Arms, 3 seeds each, same steps and batch: **C** AZNet2, **C+h** AZNet2 with
  chain planes, **Ga** graph (a), **Gc** graph (c). C+h is added because Gc
  contains chain features: without it the comparison mixes "graph" with "chain
  view".
- The arm for (ii) is chosen on validation. The probes are read once, after.
- The test against C uses each arm's best-validation seed, McNemar on the same
  probe positions. All seeds are reported.
- Other reference points on the 10x7 probe: greedy 66%, Hard 64%, formula + one
  ply 84% (on the hard judge sets), a2 87.5%.

**(ii) Fine-tuning on mid and the 80 mistakes: the main experiment.**

- A graph net needs a start comparable with Dot-2, and a net trained in (i)
  has never seen a whole game. So: distil from Dot-2 on the dot2c replay buffers
  (scripts/distill.py), policy and V. Entry check: raw Elo 10x7 >= 1400 and V sign
  agreement with Dot-2 >= 90%. If it fails, (ii) is not comparable: stop and
  report.
- Fine-tune: Dot-2 and the distilled G from the same data, 4000 steps,
  buffers mixed as in configs/mid_hand.yaml, the same split (seed, whole games:
  1387 / 538, 199 decisive; mistakes 56 / 24). Two body variants for each net,
  as a1 / a2 of part 3: **frozen** (body and V frozen, bitwise check) and
  **lr0.1** (body and V at 0.1x lr, self-play V loss kept). The trainable heads
  must have parameter counts within 10%: the CNN head may simply lack capacity
  (docs/mid_hand.md, item 2).
- One share fixed in advance as primary: **r30**. r10 is descriptive only.
  Two body variants are two tests, so T1 and T2 use p < 0.025 each (Bonferroni);
  T1 passes if either variant passes, and T2 is read for that variant.
  lr0.1 also reports the V sign agreement with Dot-2 before and after: b1 / b2
  showed that a moved V can damage play. (ii) plays no games, so that risk is
  measured here and only matters for (iii).
- Tests on set mid, decisive held out: **T1** G before -> after, one-sided sign
  test, p < 0.05. **T2** G after vs CNN after, same test on the same positions.
- The 24 mistakes are descriptive only. They are selected from Dot-2's errors,
  so any other net starts higher (regression to the mean) and 24 has no power.
- **Confirmation before (iii):** set mid is already read in part 3, and
  docs/mid_hand.md asks for a new set fixed in advance. 2000 new positions at
  10-15 safe lines, about 700 decisive, up to 16 h at 60 s and 2 workers; same T1.
  Run only if T1 passes.

Power: base is wrong on 26 of 199 held-out positions. p < 0.025 needs about 12 up
/ 2 down or 6 up / 0 down, so a small gain will not pass; r30 on the CNN (11 / 4)
did not reach even p < 0.05.

**(iii) Self-play on 10x7, only if (ii) passes.** 10 rounds from the distilled
and fine-tuned G, gate 55%, as dot2c. Then: Elo with MCTS and raw; routs on 1200
games on the six seed sets of rout_mistakes (base 47, 77 losses); bridge 20 games
vs the real Hard. Dot2c failed all six gates after gen 16, and b1 / b2 failed all
ten: a failed gate here is the expected case, not news. If no candidate passes,
the fine-tuned net is what is evaluated.

## 4. Success and failure

Success:

- **Main:** T1 p < 0.05 on the old held-out, confirmed on the new set.
- **Probe:** raw 10x7 > 80% from scratch, and above C on the same positions
  (p < 0.05). Added to the threshold because 80% alone says little: a CNN
  fine-tune got 87.5%, formula + one ply 84%.
- **Play, full success:** routs <= 31 of 1200 vs base 47 (one-sided p < 0.05), raw
  Elo 10x7 >= 1520, bridge no worse than 19-1, no rout.

Failure and what it means:

| result | reading | action |
|---|---|---|
| Stage 0 fails | readout defect, not a finding | fix, rerun |
| Gc <= C+h in (i) | graph gives nothing beyond chain planes | stop |
| Gc <= C in (i) | representation is not the limit even late | stop |
| Distillation misses the entry check | comparison impossible at this cost; nothing about GNNs | stop, or decide on a 24 h from-scratch curriculum |
| (i) passes, T1 fails | it sees late parity, not the middle game: the limit is the signal (too few, too hard positions), not the architecture | stop; state it as the answer of part 3 and 4 |
| T1 passes, T2 does not | fine-tuning helps but no more than for the CNN | report, no (iii) |
| T1 passes, confirmation fails | the first pass was noise or the used set | stop |
| (ii) passes, routs do not fall | better hand, same play: routs come from V or search | report; next look at V |

No reruns with other shares or steps to pass a threshold.

## 5. Risks

- **Speed.** 2-4x fewer positions per second. At MCTS 200 the app's move time
  grows; the net's share of self-play sets the real cost. Measured first; if over
  4x, move segment ids to C before (iii).
- **Variable batches.** Avoided by fixed topology and padded segments. A batch
  of mixed sizes is already split per size in Evaluator.
- **Symmetries.** Lines and boxes carry no coordinates, so the graph net is
  equivariant by construction: no augmentation, an advantage. Equivalent lines
  get equal scores. CNN arms keep augmentation. Test: a reflected board gives
  the reflected policy to fp32 error.
- **Overfitting on 80.** It hits any net (12.5 -> 100% on train, as seen). The
  mistakes are therefore a small share of a large mix, and they never decide.
- **The graph is still most of the board early.** Part 3 flagged this. The
  routs start at 7-20 safe lines, where segments exist, but before them (c) has
  little over (a). Accepted: the tests are at 10-15 safe lines.
- **Distilled knowledge.** G starts from Dot-2's beliefs, and a frozen body may
  hide what a graph could learn. Hence the lr0.1 variant in (ii). (i) stays the
  only test free of this.
- **Selection.** Arms are chosen on validation, the primary test is fixed here.

## 6. Time and where

Machines: G16 (this one): RTX 4060 Laptop 8 GB, 30 GB RAM, 8 cores / 20 threads.
G18: Ryzen 9 9955HX (16 cores / 32 threads), RTX 5070 8 GB, 32 GB RAM, Windows, no
Linux; WSL2 not installed, CUDA in it not checked. The plan runs on G16; G18 is the
reserve for (iii), if it comes to that.

Overnight limits (user, 2026-11-07): workers <= 2, judge <= 2, nice, GPU 40%,
free memory >= 12 GB. Runs over 10 minutes go in tmux.

| step | machine | time, estimate |
|---|---|---|
| Graph code, Evaluator path, tests | any | 1-2 days of work |
| Stage 0 | G16 | 1 h |
| (i) 12 runs + probes | G16 | one night |
| Distillation + entry check | G16 | 2-4 h |
| (ii) fine-tune and tests | G16 | 1 h |
| Confirmation set | G16, 2 judge workers | up to 16 h |
| (iii) 10 rounds | G16, G18 in reserve | 10-20 h (dot2c 25 min per round x 2-4, plus gating) |
| Elo, 1200 games, bridge | G16 (bridge needs the built game) | 4 h |

Up to the verdict of (ii): about 2 days of machine time at most. With (iii):
about 4. Reading the first part only: stage 0 + (i) fit in one night.

