# Experiment log

## 2026-10-23 — project setup, reconnaissance, simulator, baselines

Chronology:
1. `apt-get source ksquares` (Ubuntu 22.04, 4:21.12.3-0ubuntu1). The first
   commit is the upstream `orig.tar.xz` unpacked into `game/` unchanged;
   the Debian packaging (`debian/`, `.pc/`) is left out. `LICENSE` (GPL-2+)
   and `AUTHORS.md` (from `main.cpp` KAboutData and `debian/copyright`).
2. Build **not done**: the KF5/Qt development packages are not installed and
   there is no passwordless sudo. Dependency list in `docs/game_spec.md` §6.
3. Reconnaissance of `game/src` → `docs/game_spec.md`.
4. `sim/board.py`: two-player board with KSquares line numbering, `step`,
   `clone`, `legal_moves`, ASCII `__str__`. Tests check the geometry against
   a literal port of `aiController::squaresFromLine`.
5. Baselines `agents/baselines.py` (random, greedy), `scripts/eval_baselines.py`,
   `scripts/bench_sim.py` → `docs/results.md`.

Decisions:
- The sim uses KSquares' own line indices so that a bridge needs no
  remapping and recorded games from the app replay directly in the sim.
- Sim supports only two players (KSquares allows 2–4); RL targets 1v1.
- Pure Python first: ~1.9k random games/s on 10x7 (~300k moves/s). Enough
  for evaluation; for self-play training at scale a vectorised (numpy over
  a batch of boards) or C version may be needed.

Observations:
- greedy beats random 100% on 5x5 and 10x7, 99.7% on 3x3.
- greedy vs greedy on 3x3: the first mover wins only 41.8% (1000 games);
  on 5x5 and 10x7 it is ~50%. Chain parity decides greedy endgames, so the
  board size matters — worth checking with exact solving on small boards.

Unclear points (need a built binary):
- Whether any linked library calls `srand()` (otherwise KSquares' AI is
  deterministic per fresh process, glibc seed 1).
- Whether the binary runs from the build tree without installation.

Ideas for later:
- Port `aiController::chooseLine` (all three levels) to Python, including
  the `(n-1)` random-index quirk and glibc `rand()`, and verify move-by-move
  against the real game via the bridge with a fixed `srand(seed)`. Then the
  RL agent can train against KSquares Easy/Medium/Hard in the sim at full
  speed and the app is only needed for final evaluation.
- Exact solver for small boards (2x2, 3x3; retrograde / alpha-beta with
  transposition table) as a ground-truth opponent and for value targets.
- Chain-aware baseline (long chain rule, double-dealing) stronger than
  KSquares Hard, as a sanity bar for the RL agent.

## 2026-10-23 (session 2) — build, bridge, AI port, strong reference, ladder

Chronology:
1. Dependencies installed by the user; `cmake -S game -B build && cmake
   --build build -j` builds cleanly (Qt 5.15.3). The binary is
   `build/src/ksquares` (not `build/bin/`).
2. Open questions from session 1, both resolved on the binary:
   - `srand`: an `LD_PRELOAD` shim logging `srand`/`srandom`/`rand` over
     whole games saw no seeding call from any library; the first `rand()`
     is 1804289383 (glibc seed 1).
   - Runs from the build tree without installation, offscreen without X.
3. Bridge `game/src/rlbridge.{h,cpp}` (+ `// RL bridge` lines in
   `ksquareswindow.*` and `CMakeLists.txt`): `KSQUARES_RL_SOCKET`, JSON
   lines, `reset` (size, AI seats, difficulty, `srand` seed) / `step`;
   0 ms AI timer, score dialogs skipped. Client `env/bridge.py` replays and
   checks every move in the simulator (mover, completed boxes, full state).
4. AI port `sim/ksquares_ai.py`: glibc `rand()`, all three levels of
   `chooseLine` literally. Primitives checked against the originals (libc
   via ctypes; a small Qt 5 program for `QSet` order, `QMultiMap` order and
   the index formula, 20,000 cases). `scripts/verify_port.py`: 1200/1200
   games identical to the real game on 5x5 and 10x7, all levels, external
   random player and AI-vs-AI; plus 400/400 Hard games with a random QHash
   seed. Matched on the first full run.
5. Strong reference `agents/strong.py` (alpha-beta + TT + chain analysis,
   see its docstring). The first version won only 6/10 vs Hard: the
   all-but-four line of an opened 4-loop was off by one (it drew the
   capture instead of the middle line), so "exact" endgame values were
   wrong. Found by checking per-move values for consistency (an exact value
   cannot drop against any opponent) and brute-forcing the position. After
   the fix: exact values equal brute-force negamax on 850 random 2x2–3x3
   positions and 300 late 5x5 game positions (≤ 18 undrawn lines).
6. `scripts/eval_ladder.py` (process pool), `configs/eval_strong.yaml`,
   `configs/eval_ladder.yaml`; `scripts/bench_sim.py` now covers all
   agents via `agents/registry.py`. Results in `docs/results.md`.

Results (5x5, 200 games per pair):
- strong (0.5 s/move) vs Hard 94.5% (+8.1 boxes), vs greedy 98.5%; in the
  ladder strong beats every agent with 93.5–100%.
- Ladder order random < greedy ≈ easy < medium < hard < strong;
  hard vs medium 96.5%, medium vs easy 96%.
- Speed per move: port Hard 0.17 ms on 5x5, 1.9 ms on 10x7 (Python);
  the real game over the bridge plays ~50 games/s on 5x5, ~20 on 10x7.

Decisions:
- The game is the socket client and Python the server, as in pacman-rl.
  External players are KSquares "human" seats, so their lines go through
  the same `addLineToIndex` path as mouse clicks.
- The bridge sets `QT_HASH_SEED=0`: Qt 5 otherwise seeds QHash randomly per
  process, and the Hard AI iterates a `QSet` in the hard-hearted handout.
  If several lines qualify, the choice would differ between processes. Never
  happened in 1600 games, but the port emulates the seed-0 order anyway.
- Strong values = net future boxes for the side to move, TT keyed only by
  the line bitmask. Captures that never lose are applied without
  branching; only all-but-two / all-but-four decision points branch.
- Each KSquares AI instance in the simulator has its own `GlibcRand`; in the
  real game all computer seats share one, which only matters for
  AI-vs-AI reproduction (`verify_port.py` shares one there).

Unclear points:
- The strong player still loses ~5% to Hard and ~6% to Medium. Not yet
  analysed; the likely cause is the safe-move phase: until about 15 safe
  lines are left, the search only reaches depth 1–2 and the leaf
  heuristic (one pseudo-random safe-move playout + exact endgame) is noisy.
- Strong's results depend on wall-clock time and machine load (time
  limit), so ladder rows with strong are not bit-reproducible.

Ideas for later:
- Strong: stop iterative deepening once all root values are exact (on
  3x3 it currently uses the whole time limit); average several playouts or
  use a proper safe-move parity count at leaves; component-level endgame
  solver for positions with junctions; a fixed node budget instead of
  wall time for reproducible evaluations.
- Speed up the Hard/Medium port for training in the simulator (it is
  O(candidates × boxes²) per move like the C++; incremental side counts
  would help), or train against the real game over the bridge, which is
  already faster for Hard on 10x7.
- gymnasium wrapper `env/ksquares_env.py` over the simulator (opponent =
  port) with the same observation as over the bridge; the bridge for final
  evaluation against the real KSquares.

## 2026-10-23/24 (night) — AlphaZero on 5x5

Chronology:
1. The task started with "commit the uncommitted analysis of strong's
   losses": there was none — clean tree, no stash, nothing in `runs/`.
   The strong losses (~5–6% vs Medium/Hard) stay unanalysed (see the
   previous section's unclear points).
2. `torch==2.14.1` (CUDA 13), `matplotlib==3.10.9` installed into `.venv`
   (optional extra `az` in `pyproject.toml`).
3. `agents/az_net.py` (encoding, symmetries, ResNet 6x64, 0.51M
   parameters), `agents/az_mcts.py` (batched PUCT), `agents/az_play.py`
   (self-play, net-vs-net matches), `scripts/az_train.py` (loop),
   `scripts/az_eval.py` (ladder, table, plot, Elo), `scripts/az_bench.py`,
   `configs/az5.yaml`, `tests/test_az.py`.
4. MCTS test on 2x2: with uniform priors and value 0 PUCT spreads visits
   breadth-first and never reaches a terminal in 20k simulations on an
   8-line position (all Q = 0) — not a bug, the test now uses random
   playouts as the leaf value (pure MCTS), and the root value then has the
   sign of the exact minimax result in 25/25 positions, with 1 and 4
   leaves per tree.
5. Speed (`scripts/az_bench.py`, one process, untrained net, fp16 on the
   RTX 4060 Laptop, 200 simulations from random positions):

   | games (= batch) | sims/s | net share of the time | ms per net call |
   |---:|---:|---:|---:|
   | 1 | 970 | 93% | 0.96 |
   | 8 | 6,201 | 81% | 1.04 |
   | 32 | 15,457 | 53% | 1.10 |
   | 64 | 22,143 | 41% | 1.18 |
   | 128 | 23,101 | 32% | 1.75 |
   | 1 game, 8 leaves (virtual loss) | 6,354 | 82% | 0.99 |

   The pure-Python tree is ~10x above the 2k sims/s threshold, so nothing
   was moved to C (cffi). Without the network the tree alone does ~30k
   sims/s; at batch 64 it is ~60% of the time, i.e. the bottleneck from
   batch 64 up is Python tree code, below that the per-call GPU latency.
6. Smoke test, one generation (480 games, 300 train steps, gating, eval
   with 20 games per cell): self-play 113 s (6 workers x 48 games,
   42.5k evaluations/s), training 12 s, gating 48 s (96-4 vs the random
   initial net, accepted), evaluation 60 s; whole generation 4 min.
   Self-play layouts, one generation each:

   | workers x games | evaluations/s | GPU util | GPU memory |
   |---|---:|---:|---:|
   | 6 x 48 | 42.5k | — | — |
   | 4 x 128 | 58.2k | 65–80% | 0.8 GB |
   | 8 x 64 | 55.6k | ~90% | 1.4 GB |
   | 3 x 192 | 52.5k | ~50% | 0.65 GB |

   Several processes share the GPU by time slicing, so per process it is
   ~14k/s instead of 22k. Chosen 4 x 128. A generation is then sized to
   ~30 min: 6000 games (~18 min, ~320k positions), 2500 steps of 512.
7. Overnight run in tmux `az5`: `scripts/az_train.py --config
   configs/az5.yaml`, run dir `runs/az5`.

Decisions:
- Representation: one (2h+1)x(2w+1) = 11x11 grid holding dots, lines and
  boxes (lines at (even, odd)/(odd, even), boxes at (odd, odd)) rather
  than a 6x6 dot grid or separate H (6x5) / V (5x6) planes. On this grid
  all 8 symmetries are plain rotations/flips and the transpose maps
  horizontal lines to vertical ones by itself; with separate H/V planes
  a transpose swaps and re-pads channels, and on a 6x6 dot grid two lines
  share a point so they need two channels with off-by-one offsets anyway.
  Planes: drawn, undrawn, my box, opponent's box, free box with 0/1/2/3
  sides, score difference, mover-moved-first, ones (11). Everything is
  relative to the mover, so the value needs no colour flip.
- Heads AlphaZero-style (1x1 conv + FC) so the policy can use global
  counts (chain parity); policy over the 60 lines in KSquares order, masked.
- Extra turn after a capture: every node stores its mover; a backed-up
  value flips sign only on edges where the mover changes.
- Terminal = winner fixed (more than half of the boxes), not only a full
  board: exact values earlier in the tree and shorter self-play games.
  Value target is the result (+1/-1) for the mover; score margins are
  not learned.
- One leaf per game per round in self-play (no virtual loss needed, the
  batch comes from 128 parallel games); virtual loss only for single-game
  search. No tree reuse between moves (200 fresh simulations per move).
- Positions with one legal move are played without search and not stored.
- Rejected candidates: the trainer keeps its weights (AlphaGo Zero); only
  self-play stays on the best net. Gating 100 games, 200 sims, temperature
  1 for the first 4 lines, no noise.
- Evaluation games stop when the winner is fixed (win % only). Opponents
  run in 8 CPU processes; strong has 0.5 s per move by wall clock, so its
  strength drops somewhat while self-play loads the CPU. random was added
  to the opponents: without it, an all-lose net only gets an Elo bound.
- Elo: anchors from the 200-game ladder (random 0, easy 888, greedy 905,
  medium 1235, hard 1488, strong 1815), each net fitted alone against them,
  one virtual half/half game per pair.
- Safe stop: `touch runs/az5/STOP`; the loop finishes the current phase
  (self-play, train or gating), checkpoints, stops the evaluator and exits,
  which closes the tmux session. Resume = start the same command.
