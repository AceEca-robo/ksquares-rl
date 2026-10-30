Goal of part 3: can the policy ("the hand") see chain parity without search?

# Part 3 plan: parity without search

Status: draft 2026-10-30, experiment (a) started the same day.
Dot-2 here = dot2c gen 16 (runs/best10x7, the app's 10x7 network; AZNet2
6x64 with global pooling in blocks 2 and 4, 0.51M parameters, trained on
5x5 / 7x7 / 10x7).

## Why

Part 2 (docs/transfer.md, docs/results.md) left two facts:

- The raw policy of every Dot-2 sits at Elo ~1400-1480 on 10x7 (Hard
  1420) no matter how long it trained; all the gain of dot2c came from a
  calibrated value head plus search (MCTS 1755).
- On parity positions the raw policy scores about what "play a safe line"
  scores (10x7: 69% vs greedy 66%, Hard 64%). With search, the 10x7 game
  is still lost in ~9% of games vs Hard, and every such loss is a rout
  (1:69 ... 17:53): the net loses control of the chains in the middle game,
  and V only notices ~10 lines later (docs/overnight.md, 2026-10-29).

So the question is whether the hand can carry the parity count itself,
and what it takes: (a) only the right training signal, (b) an architecture
that sees the whole board more easily, or (c) a different representation.

## Metrics (the same for every experiment)

1. **Parity probe without search** (main): policy argmax correct on the
   held-out probe sets: 5x5 docs/parity_probe_positions.json (1000, 1-6
   safe lines) and 10x7 docs/transfer_parity_10x7.json (304, 1-2 safe
   lines); 95% Wilson intervals. Baselines: greedy 66%, Hard 64%, Dot-2
   policy before training. Also reported: MCTS 200 on the same positions
   and V after the move (does a policy change disturb the rest).
2. **Elo without search** (raw policy argmax) on 5x5 and 10x7: 200 games
   per opponent vs greedy, medium, hard, strong, Bradley-Terry against the
   fixed ladders (5x5: az_eval LADDER; 10x7: docs/ladder_10x7*.json). Guards
   against "learned the probe, forgot to play".
3. **Routs vs Hard on 10x7**: (i) 20 games over the bridge against the real
   KSquares Hard (scripts/az_vs_hard.py protocol, MCTS 200, seed 11), (ii)
   because 20 games cannot see a 9% rate move, also 200 games vs the
   simulator's Hard at MCTS 50 (the app's lowest setting, where the routs
   were found). Rout = a loss by 20+ boxes (of 70).

Time per evaluation of one net: probe ~5 min GPU, Elo ~20 min (CPU-bound
opponents), 200 sim games ~15 min, bridge 20 games ~15 min.

## (a) The hand learns directly from the judge's answers

Hypothesis: the body already contains what is needed (V on 10x7 is
calibrated after dot2c); the policy head never got a clean parity signal,
because self-play targets (MCTS visits) are noisy exactly there. Train the
head on exact answers.

- Data: parity positions with the judge's per-move values, 5000 per size
  (5x5, 10x7), held-out probe sets excluded (§ Data). Target: the set of
  correct moves; loss = -log(sum of the policy over correct moves), so the
  head may split its mass among them.
- Against forgetting: every batch also has self-play positions from
  dot2c's replay buffers (policy targets = MCTS visits, as in training).
- Arms: **a1** policy head only (body and value head frozen, BatchNorm
  statistics frozen); **a2** the whole net, body and value head at 0.1x lr,
  with the self-play value loss kept so V stays calibrated.
- GPU: minutes per arm (5000+5000 positions, a few thousand steps).
- **Worked if**: on the held-out 10x7 set the raw policy rises clearly
  above Hard/greedy (>= 80%, interval not overlapping 66%), raw Elo on
  10x7 and 5x5 does not drop by more than its error (~±40), and routs at
  MCTS 50 do not become more frequent. Partial: the probe rises but Elo
  or routs do not move (the hand learned positions, not a rule).
- Limit: the judge only solves late positions on 10x7 (1-2 safe lines,
  15 s); routs start earlier. A positive result here is "the head can
  express parity when it is nearly decided", not yet "in the middle game".

## (b) Architecture: stronger global pooling; chain count as an input hint

Hypothesis: parity is a global count (dots + long chains, mod 2); six
3x3 convolutions with two pooled bias layers see a 21x15 grid only
partially. Two arms, both from dot2c gen 16 by the same curriculum as
dot2c phase 2 (10x7 0.6 / 7x7 0.3 / 5x5 0.1):

- **b1 global pooling everywhere**: GPoolBlock in all six blocks, pooled
  channels 32 -> 64, plus a pooled term in the policy head per block.
  New blocks zero-initialised (as Dot -> Dot-2), so gen 0 plays as dot2c.
- **b2 chain hint on the input**: extra planes from sim/chains.py at every
  position: per box the length of the chain / loop it belongs to (scaled),
  a constant plane with (dots + long chains formed) mod 2 relative to the
  mover, and the number of safe lines left (scaled). Stem widened, new
  input weights zero, so gen 0 = dot2c. This is "telling" the net the
  count; if even this does not move the raw probe, the head is the limit.
- GPU: ~25 min per generation (dot2c phase 2); 12 generations per arm
  = ~5 h each, ~10 h both (or 5 h in parallel at half the self-play rate).
  Plus the experiment (a) fine-tuning on top of each if (a) helped.
- **Worked if**: raw 10x7 probe >= 80% and raw Elo on 10x7 >= Hard + 100
  (1520) at equal generations, and routs at MCTS 50 below 3%. b2 working
  and b1 not = the count is learnable but not found by pooling.
- Ready to start after (a): the self-play / gating code is the dot2c one;
  needed: `gpool_blocks: all` and a width option in AZNet2, the hint
  planes in Codec (N_PLANES_3) + Evaluator, a config per arm.

## (c) Chain graph network (GNN): scope estimate only

Representation in which parity is local: nodes = junction boxes (3-4
undrawn sides) and the ground; edges = chains with their length, loops as
self-contained nodes; a capturable box marked. Message passing over this
graph (5-10 rounds) + a global readout for V and per-edge policy (a line
belongs to one chain or to a junction).

- Work: graph extraction in C or vectorised Python at MCTS speed (the
  graph changes with every move; sim/chains.py is ~50 us per position,
  too slow at 10k evaluations/s), batching variable graphs (plain scatter
  ops, no new dependency), a policy head that maps edges back to lines,
  an Evaluator path next to the CNN one, distillation from Dot-2 for a
  warm start. Estimate: 2-3 days of code, then ~24 h GPU (like dot2s).
- Risk: early in the game there are few chains and many junctions; the
  graph is then most of the board and the gain over a CNN is unclear.
- Do it only if (b2) shows that an explicit chain view moves the probe
  and (b1) does not (the representation, not capacity, is the limit).

## Data

docs/parity_hand_data.md (scripts/parity_hand.py stats): sizes,
distribution of safe lines, long chains, loops, chain lengths, sources.
Generated by scripts/parity_hand.py data (configs/parity_hand.yaml):
games of ladder agents, judge with a time limit (5x5 2 s, 1-6 safe lines;
10x7 15 s, 1-2 safe lines), only solved positions where the move decides.
