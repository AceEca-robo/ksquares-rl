Goal of part 2: does the understanding of parity transfer from 5x5 to 10x7 without further training?

# Part 2 plan: Dot for any board size

Status: agreed 2026-10-28 (decisions below), implementation started.
Dot = az5b gen 54 (ResNet 6x64, 0.51M parameters, 5x5 only).
Dot-2 = the size-independent network proposed here.

## 1. What is tied to 5x5 today

Already size-generic (no change needed): `sim/board.py`, `sim/chains.py`,
`sim/ksquares_ai.py` (verified identical to the game on 10x7 too,
docs/results.md), `agents/strong.py` (`Engine(w, h)`, line bitmask of any
length), `agents/az_mcts.py`, the RL bridge (`env/bridge.py`
`reset(w, h, ...)`, the game allows 2..50 x 1..30), `launcher` `BoardView`,
`Codec(w, h)` itself (grid, symmetries).

Tied to one size:

| where | what | change |
|---|---|---|
| `agents/az_net.py` `AZNet` | `p_fc = Linear(4*cells, n_lines)`, `v_fc`/`s_fc = Linear(2*cells, 128)`: the heads have a fixed input size; `w, h` stored in the net | fully convolutional heads (§2); the net no longer knows a size |
| `agents/az_net.py` `make_net`, `Evaluator` | one `cfg["board"]`, one `Codec` | `Codec` cache per (w, h); `Evaluator` groups boards by size |
| `agents/az_play.py` (self-play, match) | `w, h = cfg["board"]` for every game | per-game size from `cfg["boards"]` (list + weights) |
| `scripts/az_train.py` `Buffer`, `FIELDS` | one `n_lines`/`n_boxes` per buffer (`FIELDS` comments say 60/25); augmentation with one codec | one buffer per size, each batch from one size (no padding), size mix by weights |
| `scripts/az_eval.py` | `LADDER` Elo anchors measured on 5x5; titles "5x5" | ladder per size (`eval_ladder.py` already takes `board`); anchors for 10x7 are a new measurement |
| `agents/strong.py` params | `node_budget: 23000` calibrated on 5x5 (`calibrate_strong.py` hard-codes `Board(5, 5)`); on 10x7 strong needs ~270 ms per move and searches much shallower | calibrate per size; report strong-on-10x7 as a separate, weaker anchor |
| judge (`launcher/brain.py` `Solver`, `configs/app.yaml` `solver`) | exact verdicts only from `max_undrawn: 34` with `node_limit: 150000`: on 5x5 that is the last 57% of the game, on 10x7 the last 22%, and positions with junctions may exceed the node limit | per-size `max_undrawn`/`node_limit`; measure the verdict coverage on 10x7 |
| `launcher/core.py` `check`, `launcher/brain.py` `fits` | refuse any size but the one in the net's config (`err_size`) | a net declares `boards: any` (or a list); the error stays for Dot |
| `configs/app.yaml` | `board_sizes: [[5,5],[10,7]]`, one `best` | add 7x7; `best` per size or one Dot-2 for all |
| `scripts/selfcheck.py`, `calibrate_strong.py`, `configs/eval_strong.yaml`, probe position sets (`docs/*_positions.json`) | 5x5 literals / 5x5 positions | parameter from config; new 10x7 probe sets later |

## 2. Dot-2 architecture

Input: unchanged. The `(2h+1) x (2w+1)` grid with the 11 planes of `Codec`
is already size-independent: dots, lines and boxes are cells, plane 10
(ones) marks the board inside the zero padding, so edges and corners are
visible to convolutions. Plane 8 (score difference / boxes) stays
normalised; add plane 11 = undrawn lines / all lines (game phase), constant.
10x7 -> 15 x 21 grid, 7x7 -> 15 x 15.

Body: the ResNet as in Dot (6x64 to start, same stem), plus global pooling
inside the body (KataGo style): in 2 of the blocks, the mean and max over
the board (masked by plane 10) of 32 channels, through a linear layer,
added as a bias to the other channels. Reason: the outcome of Dots and
Boxes is decided by chain parity, a global quantity; 13 3x3 convolutions
see 27 cells, which covers 5x5 (11x11) but not 10x7 (15x21) in one view.

Policy head (fully convolutional): 1x1 conv -> 32, global-pool bias as
above, 1x1 conv -> 1 logit per cell; gather the logits at `line_cell`
(legal mask as now). No size-dependent parameters.

Value and score heads ("intuition"): 1x1 conv -> 32, masked global mean,
max and mean x (sqrt(boxes) - 5) / 10 (the size term of KataGo, so the
net can tell board sizes apart), then MLP 96 -> 128 -> 1 (tanh) and a second MLP for
score / boxes.

Mixed sizes: a batch holds one size (no padding needed; the masks are for
the future option of padding). Transfer from Dot: stem and body weights
load as they are (same planes), heads are new.

Symmetries: `Codec` already keeps only the transforms that preserve the
grid shape, so the code needs no change: 8 on square boards (5x5, 7x7),
4 on 10x7 (identity, rotation by 180°, the two mirror flips). Transposing
10x7 gives 7x10, a different board for KSquares; it could serve as an
extra training size, but is not a symmetry of the game record.
Test to add: for every size, encode(transform(state)) == transform(encode)
and the policy permutation round-trips (exists for 5x5).

## 3. Speed of self-play and MCTS on 10x7

10x7: 157 lines (5x5: 60), 70 boxes, a game has 157 moves, 2.6x as many
as 5x5 (not 2x). Grid 315 cells vs 121.

Measured today (`scripts/az_bench.py`, untrained az5 net, one process,
RTX 4060 Laptop, 200 sims):

| board | games (batch) | sims/s | net share | ms per net call |
|---|---:|---:|---:|---:|
| 5x5 | 64 | 19,900 | 37% | 1.20 |
| 5x5 | 192 | 21,900 | 27% | 2.33 |
| 10x7 | 64 | 2,460 | 9% | 2.27 |
| 10x7 | 192 | 2,240 | 6% | 5.48 |

Profile on 10x7: 94% in the Python tree, of it `_select` 70 times per
simulation (5x5: 2.4). The cost per node is the same (~4 µs, numpy
argmax over the children); the trees are deep. With an untrained net and
FPU reduction the search follows one line, so this depth is partly an
artefact of the untrained net. It has to be measured again with a trained
Dot-2 (step (a) gives one) before deciding anything.

Estimate per game: 157 moves x 200 sims ~ 31k evaluations (5x5 ~10.4k),
3x the work. The network part scales ~2.4x per call (cells); at batch 192
the GPU alone would give ~35k evaluations/s on 10x7.

Bottleneck: the Python tree, not the GPU. Decision rule: if Dot-2 after
(a) does < 8k sims/s per process on 10x7, port the tree walk
(`_select`, `_descend`, `_backup_path`, child arrays) to C via cffi, the
board stays in Python only at expansion; ~1 day of work, expected 5-10x on
the tree. Above 8k: stay in Python, 3 workers already saturate the
GPU-share at ~15k/s.

## 4. Experiments

(a) Dot-2 on 5x5 up to Dot's level.
   Init the body from Dot, new heads; first distil (policy + value targets
   from Dot on the az5b buffer, ~1 h), then AlphaZero as az5b. Goal: MCTS
   200 ladder Elo >= 2164 (Dot gen 54) +- the ladder's error, and >= 90%
   vs Hard over the bridge. Also check Dot-2 raw policy on the strategy and
   parity probes, so the global pooling does not lose what Dot knew.
   Fallback if it stalls below: 8x64 or more pooled blocks.

(b) Transfer to 10x7 without training.
   First the yardstick: ladder on 10x7 (random, greedy, medium, hard,
   strong calibrated for 10x7; `eval_ladder.py`, 200 games per pair),
   new Elo anchors. Then Dot-2 (from (a)) raw and MCTS 200, 100 games per
   opponent, and over the bridge vs KSquares Hard (20 games, as for Dot).
   Judge: share of moves on 10x7 positions with <= max_undrawn where the
   solver gives a verdict, and share of "sees" verdicts; same for 7x7.
   Expectation: local tactics (captures, 2-chain handouts) carry over, chain
   parity on a bigger board probably not; the numbers say how much.

(c) Curriculum vs from scratch on 10x7.
   Arm C: Dot-2 from (a) -> mixed 7x7/5x5 (weights 0.7/0.3) -> mixed
   10x7/7x7/5x5 (0.6/0.3/0.1). Arm S: Dot-2 from random weights, 10x7 only.
   Same GPU budget per arm (wall clock incl. the 5x5/7x7 phases of C,
   but not (a) itself, which is reported separately), same hyperparameters
   apart from the size mix. Measured every generation on the 10x7 ladder
   (MCTS 200 Elo) and vs Hard; the curve over GPU hours is the result, not
   only the end point. 1000 games per generation on 10x7 (instead of 2500)
   to get enough generations.

## 5. GPU time and disk

GPU (RTX 4060 Laptop; generations at today's Python speed / with a C tree):

| step | estimate |
|---|---|
| code: Dot-2 net, multi-size buffer/self-play/eval, tests | no GPU, 1-2 days |
| (a) distillation + ~40 generations on 5x5 (~11-15 min each) | 8-10 h |
| 10x7 / 7x7 ladders (CPU, 16 processes) | ~1 h |
| (b) evaluation | ~1 h |
| C port of the tree (only if needed) | no GPU, ~1 day |
| (c) 10x7 generation, 1000 games: 31M evaluations | ~80 min (Python, ~6.6k/s for 3 workers) / ~35 min (C, ~15k/s) |
| (c) per arm | budget 24 h each |
| total | ~60 h GPU, i.e. 4-5 nights |

Disk: a checkpoint of a 0.5M-parameter net ~2.2 MB; az5b took 182 MB for
87 rounds. A 10x7 position in the buffer ~650 bytes (lines, pi in fp16,
sides, owner) -> 500k positions ~0.33 GB in RAM and in the buffer
snapshot. Expect ~0.5 GB per run, ~1.5-2 GB for all of part 2. Free: 31 GB.

## Decisions (2026-10-28)

- 7x10 (transposed) is not added as a training size.
- Dot-2 keeps 6x64 as Dot, for comparability.
- "Dot level" in (a) means both: ladder Elo (MCTS 200) >= 2100 and
  >= 90% vs Hard over the bridge.
- The C port of the tree walk is done first, not conditionally.
- Judge on 10x7: a time limit instead of `max_undrawn`; the share of the
  game it covers is measured and recorded.
- Stopping rule for (a): 6 rejected generations in a row, or the level
  above reached.
