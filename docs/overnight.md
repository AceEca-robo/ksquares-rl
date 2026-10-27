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

Overnight notes (az5):
- 23:53: the pacman distillation (`distill-collect`, 6 processes, ~4.7 GB)
  started on the same GPU; GPU at 100% from then on. Our self-play fell
  from 58k (smoke test, GPU free) to 40k evaluations/s in gen 1 and 16k
  in gen 2 (6000 games took 65 min instead of ~18). Training 2500 steps
  took 420–440 s instead of ~100 s. Memory was never the problem (6.2 of
  8 GB used), so the self-play batch was not reduced; the neighbour was
  left alone. The worker processes sit at 100% CPU while waiting for
  their GPU time slice.
- 01:29: safe stop after gen 2 training (stop → checkpoint → tmux closed,
  works as intended) and resume with 3 x 192 games per worker (fewer,
  larger GPU calls), 2500 games and 1100 steps per generation to get back
  to ~30 min, gating in 2 tasks of 50 games instead of 4 x 25, progress
  lines per quarter of each worker's games.
- Gen 1: accepted 100-0 vs the random initial net. Eval: raw net beats
  only random; with MCTS 200 random 100%, greedy 21%, medium 10%, hard 3%,
  strong 0%.
- Value check after gen 2 training (6000 buffer positions): MSE 0.99 for
  the first 30 lines (self-play outcome is barely predictable there with
  temperature-1 openings), 0.79 at 40–50 lines, 0.40 at 50+ (corr 0.67).
  So the overall value loss of ~0.90 is not a bug.
- 02:59–04:59 the neighbour was in a lighter phase (`dist0`): self-play
  25–35k evaluations/s, generations ~20 min; from 04:59 (`distill-collect`
  again) 13–21k/s, ~40 min. Training 1100 steps: 36 s with the lighter
  neighbour, ~3 min with the heavier one.

Results by 07:25 (15 generations, 11 accepted: 1, 2, 3, 5, 7, 8, 9, 10,
11, 14, 15; rejected 4, 6, 12, 13). Full table `docs/az5_ladder.md`,
curves `docs/az5_ladder.png`. Net win % (100 games per cell):

| gen | raw: greedy | medium | hard | strong | Elo | mcts 200: greedy | medium | hard | strong | Elo |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 454 | 21 | 10 | 3 | 0 | 750 |
| 2 | 46 | 14 | 3 | 0 | 894 | 92 | 76 | 27 | 2 | 1351 |
| 5 | 84 | 46 | 18 | 2 | 1206 | 94 | 80 | 80 | 4 | 1539 |
| 11 | 88 | 65 | 19 | 15 | 1320 | 99 | 87 | 91 | 40 | 1743 |
| 15 | 85 | 73 | 46 | 10 | 1402 | 100 | 94 | 91 | 33 | 1747 |

Anchors: greedy 905, medium 1235, hard 1488, strong 1815.
- With MCTS 200 the net passed greedy and Medium in gen 2, Hard in gen 5
  (80%); since gen 11 it beats Hard ~90% and wins 33–40% against strong
  (Elo ~1745 vs 1815). The raw policy alone passed greedy in gen 3,
  Medium in gen 9 (53%), and is below Hard (46% in gen 15).
- MCTS adds ~350 Elo over the raw policy at every generation.
- Progress per generation is slowing: gating scores fell from 100% to
  55–61% with two rejections in a row (12, 13) before gen 14.

Bottleneck: GPU time shared with the pacman distillation. The training
itself is cheap (36 s per generation with a free-ish GPU); self-play is
85–90% of a generation. Within one process the Python tree is ~60% of
the time at batch 64, so with a free GPU the next limit would be the
tree code (then cffi/C would pay off).

Ideas for later:
- Learning-rate drop (0.02 → 0.002) once gating stalls; currently
  constant.
- KataGo-style playout cap randomisation and an auxiliary score-margin
  head (the value only knows win/loss; endgame margins are free signal).
- Tree reuse between moves; C tree (cffi) once the GPU is not shared.
- Evaluate against strong with a fixed node budget instead of 0.5 s wall
  clock, which depends on the machine load during self-play.
- `az` agent in `agents/registry.py` for the ladder script and the
  bridge (final games against the real KSquares).
- az5 kept running after the morning report until the az5b request: 31
  generations, 21 accepted, best gen 29 (MCTS 200: Hard 93%, strong
  49–60% at 0.5 s wall clock). Safe stop at 11:04 after gen 31 training.

## 2026-10-24 — az5b, strategy probe, video

Chronology:
1. az5 stopped safely (STOP → checkpoint → tmux empty); best accepted
   generation 29.
2. Strong with a node budget (`node_budget` in `agents/strong.py`; the
   search stops after that many nodes instead of at a wall-clock
   deadline). Calibration `scripts/calibrate_strong.py`: strong (0.5 s)
   vs Hard, 12 games, 170 searches that hit the limit; median 46k nodes
   per CPU second → **23,000 nodes** = 0.5 s of one core. Measured in CPU
   time because the machine was fully loaded by the pacman `distill-eval`
   (1700% CPU); the wall-clock node count per search came out the same
   (23k), so the single process got a full core. Caveat: a laptop under
   full load runs at a lower turbo clock, so on a really idle machine
   0.5 s may be somewhat more than 23k nodes.
3. az5b (`configs/az5b.yaml`, tmux `az5b`, `runs/az5b`): initialised from
   az5 gen 29 (`init_from`, the new head random), lr 0.01 (half), second
   value head for the score difference (MSE on margin / 25, weight 0.25;
   the target is the margin when the winner became fixed, because games
   stop there), self-play simulations drawn uniformly from 100–300 per game
   and move, fresh replay buffer, same gating and ladder, strong by node
   budget in the ladder. Gen 1 accepted 72-28 vs az5 gen 29; MCTS 200 vs
   Hard 98%, vs strong (23k nodes) 51%; then 2, 3, 4 rejected (47, 41, 54%).
4. Strategy probe `scripts/strategy_probe.py`, `docs/strategy_probe.md`,
   `docs/strategy_probe.png`, positions `docs/strategy_probe_positions.json`.
5. Game: `KSQUARES_RL_AI_DELAY_MS` (bridge only; delay before computer
   moves, default 0) so a bridge game can be watched. `scripts/az_video.py`
   runs the game with a window on :0, places it with xdotool, records with
   ffmpeg x11grab, scales to 400 px: `docs/az5_vs_hard.mp4` (az5b gen 1 +
   MCTS 200, moving first, vs the game's own Hard: 18–7, 61 s; first take,
   seed 11; re-recorded with az5b gen 5 once it was accepted: 15–10, also
   the first take). Check over the bridge without a window: az5 gen 29 vs real
   Hard 18/20 (all 10 as second player, 8/10 as first).

Decisions and findings, strategy probe:
- "Correct by full brute force" is only feasible late: parity decisions
  on 5x5 fall at ~30–36 undrawn lines (safe lines run out at ~30, median
  over 60 Hard-Medium games), 2^30+ states. So dd positions (≤ 19 undrawn
  lines) are judged by brute force, parity positions by the strong engine
  solving every child to the end (no depth limit, no heuristic). The
  engine agreed with brute force on every move of 8144/8144 positions
  met while generating.
- Correct = reaches the best result (win/tie/loss) from the current score,
  not the most boxes; positions where all moves give the same result are
  dropped. 1610 critical positions from 600 games, 150 + 150 sampled.
- Result: double-dealing is learnt by the policy itself (net without
  search: 0% in gens 2–3, ~50% gens 5–10, 82% gen 14, 92–98% from gen 15;
  with MCTS 92% at gen 2 and 100% from gen 3 — the search finds it
  tactically long before the policy does). The long chain rule is not
  in the policy: on parity positions the raw net stays at 41–49% in every
  generation, the level of greedy (42%, i.e. a random safe line) and Hard
  (44%); a uniform random legal move gets 14%. Only the search knows
  parity, and it learnt it gradually: 33% (gen 2) → 59% (gen 14) → 83%
  (gen 19) → 89% (az5b gen 1); strong 96%.
- Self-play: the first crude count of "hands over two boxes" was
  useless — the random net (gen 0) did it 238 times in 100 games, by
  accident (declining captures at random). Replaced by exact verdicts:
  every loony-phase position with a capture available (40 games per
  generation) is solved by the engine. Positions where only declining
  gets the most boxes: the net declined right 1/7 in gen 1, 12/19 in
  gen 2 (first appearance), 69–87% in gens 3–11, 83–100% from gen 14;
  wrong hand-overs (DD that cost boxes) fell from 14–33% (gens 2–3) to
  0–8% from gen 15. It happens ~46–48 lines into the game (of 60) with
  ~11–14 boxes left, i.e. in the loony endgame right after the safe lines
  run out.

Ideas for later:
- The policy never learns parity: add chain-structure input planes
  (chain/loop lengths, safe-line count parity) or a parity auxiliary
  target, and check whether the raw-net parity score moves.
- Score target from games played to the end instead of the margin at
  decision.
- 12:25 az5b gen 5 accepted (58-42 vs gen 1): MCTS 200 vs Hard 95%, vs
  strong 54%, Elo 1875 (gen 1: 1870, same within noise). Probe: net only
  dd 99 / parity 49, MCTS dd 100 / parity 85; self-play declined right
  24/27, no DD that cost boxes.
- 12:36 az5b gen 6 accepted (56-44 vs gen 5): MCTS 200 vs Hard 96%, vs
  strong 60%, Elo 1885. Probe: net only dd 97 / parity 43, MCTS dd 100 /
  parity 84; self-play declined right 21/21. Video re-recorded with gen 6
  (14–11, first take, seed 11). Gen 7 rejected (52%).
- Probe snapshots for a later animation ("how the net discovers chain
  parity"): scripts/az_probe_maps.py. A fixed set of 20 probe positions
  (16 parity, 4 dd; only those where at most a third of the legal moves
  are correct, spread evenly over the probe file) is frozen in
  docs/probes/positions.json. For every accepted generation it writes
  docs/probes/{run}/gen_NN.json (policy over the legal lines, value and
  score head, MCTS 200 visits / Q / chosen move / principal variation)
  and gen_NN.png (boards with the policy as a fixed-scale heat map,
  correct moves ringed, MCTS move starred, PV numbered). Per run, since
  az5b numbers its generations from 1 again. az_eval.py now calls it after
  each evaluated generation; az5 (21 gens) and az5b (gens 1-14) were
  backfilled (~3 s per generation). On the 16 parity positions the raw
  policy stays near flat: argmax right 5-7/16 from az5 gen 1 to az5b gen
  14; MCTS 4/16 (az5 gen 1) -> 16/16 (az5 gen 17), 15-16 since.
- Value-sign check on the 16 parity snapshot positions
  (docs/probes/summary.md, `az_probe_maps.py table`; snapshots redone
  with the value of every child position). All 16 are "mover wins" by
  construction, so V > 0 at the root is not a test (a constant scores
  16/16; the nets get 2-12/16, V there is near 0 or negative). The test
  that discriminates is the sign of V after each move: 46-59% (chance 50)
  up to az5 gen 7, 72-74% at gens 8-11, 76-87% from gen 17, 84-87% in
  az5b; one-ply search with V finds a winning move 7-15/16. The policy
  argmax stays at 3-9/16 (random 2.2), MCTS 15-16/16 from az5 gen 17.
  Parity is known by the value head, not by the policy; search turns it
  into moves.

## 2026-10-24 (afternoon) — the app

- A desktop app on the pacman-rl pattern: scripts/launcher.py (one 1920x1080
  pygame window, menu, EN/RU/UK, Esc to the menu), launcher/ (core:
  settings, networks, checks, private Xvfb; session: the games in a worker
  thread; brain: policy / V / MCTS / PV and the exact verdict; i18n;
  desktop), install.sh / uninstall.sh, menu entry ksquares-rl. Settings
  in configs/app.yaml. Training untouched.
- Modes: network vs KSquares (the real game over the bridge on a private
  Xvfb, window grabbed with Pillow; Easy / Medium / Hard; who moves first),
  play the network (clicks; "show what the network thinks of my move":
  its prior for your line, its rank, the search value after it, and the
  exact verdict), network vs itself (two generations from az5 / az5b,
  pause, one move, speed). Search none / 50 / 200 / 800.
- Thought panel: policy heat map, MCTS visits with the PV (10 lines)
  numbered on the board, V and a who-is-winning bar, the score head, the
  long chains and loops on the board (sim/chains.py) with the side the
  long chain rule favours (dots + long chains even: the first player), and
  "the network sees / does not see": the strong engine solves every move
  to the end once at most 34 lines are left (node limit 150k, in its own
  thread so the game never waits); V timeline below.
- runs/best/best.pt: a link to the last accepted az5b generation, renewed
  by scripts/update_best.py (called by az_eval.py after each evaluated
  generation and by the app at start).
- 10x7: the net does not play other sizes at all. The body is
  convolutional, but the policy head (4 x 121 -> 60) and the value head
  (2 x 121 -> 128) are fully connected over the 11 x 11 grid, so a 10x7
  input (15 x 21) fails on the shapes (tests/test_app.py). The menu offers
  10x7 and refuses with this explanation. A fully convolutional policy
  head and global pooling in the value head would lift this; not done.
- Colours follow KSquares: first player red, second blue.
- --record (mp4 or gif via ffmpeg, 1280x720), --screenshot,
  --human-agent for unattended "play" (screenshots, tests). Screenshots:
  docs/app_menu.png, app_ksquares.png, app_play.png, app_self.png.
- Tests (tests/test_app.py): strings in three languages, desktop entry,
  chain analysis, the size check, the KSquares session over the bridge
  with a stop mid-game (game and Xvfb gone), clicks with illegal lines,
  pause / one move, and the app quitting by itself with no children left.
- The exact solver moved into agents/strong.py (exact_move_values,
  line_mask); strategy_probe.py uses it from there.
- App fixes after the first look: the chains block is hidden at game over
  and once the winner is fixed (it said "0 long chains ()" on a full
  board; empty brackets are gone too); the no-search caption sits under
  the maps; at a decided position the panel says the search is not needed
  instead of "no search"; the selected mode card is highlighted (←/→ in
  the menu, remembered); --board for scripted runs (they read the saved
  settings); the game-over frame is published before `over` is set.
  Two games against KSquares Hard with MCTS 200 (az5b gen 18): 13:11 as
  first player, 14:11 as second (docs/app_ksquares_end.png).
- Menu split in two: "Mode" with the three cards, then "Settings" under a
  thin rule with only the selected mode's rows (vs KSquares: level, first,
  search, board; play: first, search, board, show my move; vs itself: the
  two generations, search, board), a Start button, and the common
  "Network: ..." line at the bottom. A click on a card selects it, a
  second click, Start or Enter starts it.

## 2026-10-24/25 (night) — az5b continued, stop rule

Plan: continue az5b from gen 38 in tmux `az5b` (window `train`); stop
rule: once 6 generations in a row are rejected, stop it safely (STOP) and
call az5b finished. The streak counts the generations before the restart
too (34–37 were rejected, so at the start it stood at 4). A watcher in
tmux window `stopwatch` reads `state.json` every minute and creates STOP
when the streak reaches 6 (log `runs/az5b/stopwatch.out`). Disk before
start: 29 GB free of 144 GB, runs/ 441 MB.

- Strategy probe and the usual cycle backfilled for every accepted az5b
  generation without them (8, 13, 14, 17, 18, 20 as well as 23–33; the
  ladder and probe snapshots of those already existed). docs/results.md
  now lists every accepted az5b generation.
- Video of the probe snapshots over the generations:
  `scripts/az_probe_video.py` → `docs/probes/dynamics.mp4` (every
  docs/probes/{az5,az5b}/gen_NN.png, 0.8 s each). Redone after each
  accepted generation.
- 00:27 gen 38 accepted (61-39 vs gen 33). MCTS 200: Hard 100%, strong
  84%, Elo 2116 (gen 33: 98 / 81 / 2051); raw: Hard 78%, strong 33%,
  Elo 1684 (best raw so far; gen 33: 74 / 30 / 1591). Probe: raw dd 99 /
  parity 52 (first time above 50; the raw net was at 42–50 in every
  earlier generation, ±4 points of noise on 150 positions), MCTS dd 100 /
  parity 90. Self-play declined right 17/19, 1 of 19 DD moves cost boxes.
  Snapshots: policy argmax 6/16, MCTS 16/16, V sign after a move 88%.
  Video re-recorded with gen 38: 16–9 vs Hard (first take, seed 11).
- Gen 39 rejected (37%).
- Gens 40–43 rejected (48, 48, 46, 49%; streak 5).
- 01:35 gen 44 accepted (60-40 vs gen 38). MCTS 200: Hard 97%, strong
  86%, Elo 2106; raw: Hard 80%, strong 32%, Elo 1656 — level with gen 38
  within noise (100 games per cell). Probe: raw dd 100 / parity 48, MCTS
  dd 100 / parity 88; self-play declined right 8/9. Snapshots: argmax
  6/16, MCTS 16/16, V sign after a move 85%. Video re-recorded with gen
  44: 18–7 vs Hard (first take, seed 11).
- Gen 45 rejected (49%).
- 01:57 gen 46 accepted (58-42 vs gen 44). MCTS 200: Hard 98%, strong
  86%, Elo 2086; raw: Hard 79%, strong 35%, Elo 1677. Probe: raw dd 100 /
  parity 52, MCTS dd 100 / parity 89. Self-play declined right only 13/19
  (68%, lowest since az5 gen 2; 40 games, so few positions), no DD move
  cost boxes. Snapshots: argmax 8/16, MCTS 16/16, V sign after a move
  84%. Video with gen 46: 15–10 vs Hard (first take, seed 11).
- Gen 47 rejected (53%).
- 02:19 gen 48 accepted (55-45 vs gen 46). MCTS 200: Hard 97%, strong
  88% (best so far), Elo 2106; raw: Hard 75%, strong 34%, Elo 1645.
  Probe: raw dd 99 / parity 49, MCTS dd 100 / parity 91 (best so far).
  Self-play declined right 20/21, no DD move cost boxes. Snapshots:
  argmax 5/16, MCTS 15/16, V sign after a move 85%. Video with gen 48:
  15–10 vs Hard (first take, seed 11).
- Gens 49–51 rejected (51, 47, 38%).
- 03:03 gen 52 accepted (63-37 vs gen 48). MCTS 200: Hard 100%, strong
  86%, Elo 2127 (best so far); raw: Hard 76%, medium 94%, strong 45%
  (az5b raw had been at 20–38% vs strong), Elo 1724 (best raw so far).
  Probe: raw dd 99 / parity 51, MCTS dd 100 / parity 90. Self-play
  declined right 19/24, no DD move cost boxes. Snapshots: argmax 7/16,
  MCTS 15/16, V sign after a move 89% (highest so far). Video with gen
  52: 15–10 vs Hard (first take, seed 11; the same score as gens 46 and 48).
- Gen 53 rejected (33%).
- 03:25 gen 54 accepted (62-38 vs gen 52). MCTS 200: Hard 99%, strong
  89%, Elo 2164 (best so far); raw: Hard 83%, strong 34%, Elo 1680.
  Probe: raw dd 99 / parity 49, MCTS dd 100 / parity 90. Self-play
  declined right 12/14, 1 of 15 DD moves cost boxes. Snapshots: argmax
  8/16, MCTS 15/16, V sign after a move 84%. Video with gen 54: 18–7 vs
  Hard (first take, seed 11).
- Gens 55–57 rejected (41, 47, 48%).
- Clock note: summer time ended at 04:00, so log times after gen 57 go
  back an hour (gen 58 gated at "03:10", eleven minutes after gen 57).
- gen 58 accepted (59-41 vs gen 54). MCTS 200: Hard 100%, strong 83%,
  Elo 2106; raw: Hard 75%, strong 41%, Elo 1691. Probe: raw dd 99 /
  parity 53 (highest raw so far, within noise of 48–52), MCTS dd 100 /
  parity 90. Self-play declined right 14/17, no DD move cost boxes.
  Snapshots: argmax 8/16, MCTS 16/16, V sign after a move 84%. Video
  with gen 58: 15–10 vs Hard (first take, seed 11).
- Gen 59 rejected (45%).
- gen 60 accepted (59-41 vs gen 58). MCTS 200: Hard 100%, strong 87%,
  Elo 2151; raw: Hard 83%, strong 39%, Elo 1713. Probe: raw dd 97 /
  parity 50, MCTS dd 100 / parity 89. Self-play declined right 17/21, 1
  of 21 DD moves cost boxes. Snapshots: argmax 5/16, MCTS 15/16, V sign
  after a move 86%. Video with gen 60: 16–9 vs Hard (first take, seed 11).
- Gens 61–63 rejected (51, 50, 49%).
- 04:18 gen 64 accepted (59-41 vs gen 60). MCTS 200: Hard 96%, strong
  83%, Elo 2068; raw: Hard 73%, strong 42%, Elo 1680. Probe: raw dd 99 /
  parity 53, MCTS dd 100 / parity 89. Self-play declined right 27/32, no
  DD move cost boxes. Snapshots: argmax 7/16, MCTS 14/16, V sign after a
  move 85%. Video with gen 64: 14–11 vs Hard (first take, seed 11).
- Observation: gating keeps passing (every 2–4 generations), but the
  ladder has flattened since gen 38: MCTS Elo 2068–2164, vs strong
  83–89%, vs Hard 96–100% (100 games per cell, ±4–7 points). Each new
  net beats the previous best 55–63% yet does not score better against
  the fixed opponents.
- Gens 65–68 rejected (48, 46, 39, 35%; streak 4).
- 05:14 gen 69 accepted (63-37 vs gen 64). MCTS 200: Hard 99%, strong
  82%, Elo 2077; raw: Hard 79%, strong 34%, Elo 1691. Probe: raw dd 99 /
  parity 57 — the highest the raw net has reached (41–53 in every
  earlier generation of az5/az5b; ~4 points standard error on 150
  positions, so a hint, not proof, that the policy starts to pick up
  parity), MCTS dd 100 / parity 91. Self-play declined right 22/23, 1 of
  23 DD moves cost boxes. Snapshots: argmax 6/16, MCTS 15/16, V sign
  after a move 88%. Video with gen 69: 14–11 vs Hard (first take, seed 11).
- Gens 70–72 rejected (32, 39, 49%).
- 05:59 gen 73 accepted (58-42 vs gen 69, though with a negative mean
  margin, -0.68). MCTS 200: Hard 100%, strong 89%, Elo 2151; raw: Hard
  68%, strong 35%, Elo 1645. Probe: raw dd 98 / parity 59 (gen 69: 57;
  two accepted generations in a row above the old 41–53 band), MCTS dd
  100 / parity 91. Self-play declined right 16/21, no DD move cost
  boxes. Snapshots: argmax 7/16, MCTS 16/16, V sign after a move 88%.
  Video with gen 73: 16–9 vs Hard (first take, seed 11).
- Gens 74–77 rejected (43, 52, 51, 51%; streak 4).
- 06:56 gen 78 accepted (55-45 vs gen 73, right at the threshold). MCTS
  200: Hard 98%, strong 83%, Elo 2086; raw: Hard 80%, strong 40%, Elo
  1688. Probe: raw dd 100 / parity 54, MCTS dd 100 / parity 93 (best so
  far; strong 96). Self-play declined right 17/23, no DD move cost
  boxes. Snapshots: argmax 6/16, MCTS 16/16, V sign after a move 86%.
  Video with gen 78: 16–9 vs Hard (first take, seed 11).
- Gens 79–83 rejected (54, 50, 50, 49, 52%; streak 5, one short of the
  stop rule).
- 08:04 gen 84 accepted (65-35 vs gen 78). MCTS 200: Hard 97%, strong
  75% (lowest since gen 13), Elo 2005; raw: Hard 73%, strong 38%, Elo
  1684. Probe: raw dd 100 / parity 55, MCTS dd 100 / parity 89.
  Self-play declined right 17/21, no DD move cost boxes. Snapshots:
  argmax 5/16, MCTS 16/16, V sign after a move 88%. Video with gen 84:
  15–10 vs Hard (first take, seed 11).
- Morning status (08:15): the stop rule has not fired; az5b keeps
  running (gen 85 self-play). Night: gens 38–84 (47 generations), 13
  accepted (38, 44, 46, 48, 52, 54, 58, 60, 64, 69, 73, 78, 84), longest
  rejection streak 5 (gens 39–43 and 79–83). By the ladder the best is
  gen 54 (MCTS Elo 2164, Hard 99%, strong 89%); gens 60 and 73 are level
  within noise (2151). Best raw: gen 52 (Elo 1724, strong 45%).
  runs/best/best.pt follows the last accepted generation (now 84), which
  is not the ladder best — the gate is against the previous best only.

Ideas for later:
- Gate on a fixed panel as well (e.g. 100 games vs strong) or keep the
  ladder-best net for the app, since gating against the previous best
  accepts nets that score worse against fixed opponents (gen 84).
- Raw parity rose from 41–53 to 54–59 over gens 69–84: re-run the probe
  with more parity positions to tell signal from noise.
- Gens 85–87 rejected (35, 48, 54%).
- 08:43 stopped by hand at the user's request (STOP; the stop rule had
  not fired, streak 3). The trainer finished gen 88 self-play and exited
  cleanly at 08:48 ("stopped cleanly", state saved at gen 88 phase
  train, so a restart continues with gen 88 training); tmux session
  closed. Best accepted generation 84; ladder best gen 54.

## 2026-10-25 (evening) — best net pinned, parity probe on 1000 positions

- az5b stays stopped (stopped by hand at 08:43, exited cleanly after
  gen 88 self-play; see above). Final: best az5b net = gen 54, the
  ladder best (MCTS Elo 2164, Hard 99%, strong 89%).
- runs/best/best.pt is pinned to gen 54: `best.gen: 54` in
  configs/app.yaml; scripts/update_best.py points the link at that
  generation instead of the last accepted one (the app calls it at every
  start, so the pin has to live in the config, not in the link).
- Parity question from the night ("raw parity rose from 41-53 to 54-59
  over gens 69-84"): scripts/parity_probe.py, 1000 fresh parity
  positions, gens 33, 54, 73, 84 (docs/results.md, "Parity probe, 1000
  positions"). Generation took 35 s (no brute force: parity positions
  are all engine-solved).
- Answer: the rise is real for the raw policy. Net only: 50.0 / 52.5 /
  57.3 / 59.1% (gen 33 / 54 / 73 / 84); 33 -> 73 p 7e-6, 33 -> 84
  p 7e-8, 54 -> 73 p 0.0015 (paired McNemar); 73 -> 84 not
  significant (p 0.21). Gen 33 is at the level of KSquares Hard and
  greedy (49-50%), gen 84 is 9 points above them; strong is at 95%.
- What it is not: a jump at gen 70. On the old 150 set gens 1-60 hover
  at 42-53% and 69-84 at 54-59%; on the 1000 set 33 -> 54 is +2.5
  (p 0.09) and 54 -> 73 +4.8, a gradual climb that is clearest after
  gen 54. And it is modest: 59% against 95% for the solver; with MCTS 200
  the nets are at 87-91% and gens 54-84 are level, so the search, not
  the policy, still does most of the parity work.
- For the video, the honest caption: "without search the policy now
  gets the long chain rule right in 59% of decisive positions, up from
  50% (KSquares Hard's level) at gen 33" — not "it began to see parity
  after 70 generations".

## 2026-10-26 — video script, parity curves, gen 54 recordings

- Video script `docs/video_script_ru.md` / `_en.md` (nine scenes, theme
  "the value head knows who wins, the policy does not know how, search
  closes the gap"); pipeline and materials in `video/` (not in git, as in
  pacman-rl). Thumbnail C picked: `docs/thumb_c_{ru,en}.png`
  (`video/video_thumbnail.py`; probe #172, gen 54: policy heat map vs the
  value's sign after each move).
- The three curves for the video were on different sets (policy and MCTS
  on the 150 strategy-probe parity positions, the value's sign on the 16
  snapshot positions). `scripts/parity_probe.py curves` / `plot` now
  measure all of them on the 1000 parity positions for every accepted
  generation (49: az5 21, az5b 28), ~10 s each: `docs/parity_curves.md`,
  `docs/parity_curves.png`, summary in docs/results.md. The earlier
  `evaluate` numbers for az5b 33/54/73/84 came out identical.
- What changed against the 16-position picture: the value's sign is at
  chance up to az5 gen 7 (50–57%), 66% at gen 8 (not 74%), 76% at gen 11,
  79–88% in az5b. A one-ply search with the value alone gets 80–84% from
  az5b gen 46 on, close to MCTS 200 (87–92%).
- Game videos re-recorded with gen 54 (ladder best), seed 11, net first:
  18–7 against KSquares Hard in all three. `docs/az5_vs_hard.mp4` via
  `scripts/az_video.py` on a private Xvfb (`windowactivate` is now
  optional: a bare Xvfb has no window manager); app recordings
  `video/rec/hard_gen54_{ru,en}.mp4` at 1920x1080 (new
  `scripts/launcher.py --record-size`; the en take with LANG=en_US.UTF-8
  so KSquares' own menus are English too).
- Russian voice and first cut (video/, not in git): edge-tts
  ru-RU-DmitryNeural, 460 s of voice; the service refused longer texts for
  a while (NoAudioReceived on scenes 3 and 7–9 for many tries), so
  video_voice.py now falls back to one sentence at a time joined with
  0.35 s pauses (used for scene 8). Scene 2 is animated (video_anim.py:
  the chain position, then the endgame played by the exact solver for both
  sides: red keeps control with all-but-two, 19–6). Stills from
  video_cards.py (terminal shots with real command output, the curves on
  the 1000 positions, the Elo ladder, the night table). Cut:
  video/ksquares_rl_ru.mp4, 8:21, soft Russian subtitles; copied to
  ~/YouTube/ksquares-rl/ with thumbnail_ru.png for review.

## 2026-10-26 (evening) — publication on GitHub

- Public repository github.com/AceEca-robo/ksquares-rl; release v0.1 with
  az5b gen 54 as the asset `best.pt` (2,207,519 bytes, sha256
  fbf37b1d…f44e304b). runs/ and video/ stay out of git.
- The app fetches the network when runs/best/best.pt is missing (a fresh
  clone), as in pacman-rl: a dialog in the menu (download / not now,
  progress, cancel, sha256 check, retry on errors), written straight into
  runs/best/best.pt with a best.json saying az5b gen 54. URL, size and
  checksum in configs/app.yaml (`best.download`). Unlike pacman-rl it
  goes into the checkout, not ~/.local/share: the app already writes
  runs/best/ through update_best.py. "Network vs itself" lists the
  downloaded network when the runs are not on disk (gen 54 vs gen 54).
- README rewritten for the public: what it is, the ladder and the parity
  curves, the app, build and run, license. The video link is a
  placeholder until the videos are out.

## 2026-10-27 (night) — video: the new Russian voice, Dot

- New voice text for scenes 1–9 (the network is the hero "Dot", «Дот»
  in the voice); the storyboard kept, cuts moved to the new sentences,
  on-screen words per the dictionary in docs/video_script_ru.md
  (intuition = value, judge = strong, built-in opponent = Hard, round =
  generation, training hall = simulator). Thumbnail C: «Dot знает, кто
  выиграет. / Но не знает, как.» Cut: video/ksquares_rl_ru.mp4, 7:27.
- edge-tts was refusing or hanging for long stretches (one call hung for
  28 min): timeout per call, by-sentence voicing resumes from its parts.
  Voicing all scenes took ~2 h.
- EN waits for the author's OK on RU.

## 2026-10-27 (morning) — "Train your own Dot", the English cut

- The app's fourth mode, "Train your own Dot" (launcher/train.py,
  scripts/launcher.py): checks the CUDA GPU (name, memory), free disk and
  a training already running (runs/.training.json, one at a time); run
  name and rounds (10 / no limit); starts scripts/az_train.py detached
  (own session) with runs/<name>/train.yaml = configs/az5b.yaml from
  scratch (init_from null, publish: false so az_eval.py writes nothing to
  docs/ or runs/best); progress from state.json / log.jsonl / eval/;
  Stop = the STOP file. Accepted rounds join "play" (new network picker)
  and "self" as "<name> gen N". Terminal: python -m launcher.train.
- Real run my_dot, 4 rounds from scratch on the RTX 4060 Laptop: all
  accepted (100-0, 98-2, 85-15, 69-31), 11:04–11:27 per round (az5b
  median 11.3 min: the README table), vs Hard with search 2, 2, 46, 46%.
- Video: RU scene 9 new last sentence (re-voiced), scene 8 closes with the
  training screen (video/rec/train_{ru,en}.png). RU 7:36. The RU voice of
  scene 8 still says "три режима" (not re-voiced, as asked): the menu
  shot there is the old three-card one.
- EN cut: docs/video_script_en.md rewritten sentence for sentence from
  the Russian (Dot, intuition, the judge, the built-in opponent, round);
  English cards, scene 2 animation and thumbnail C ("Dot knows who wins. /
  Just not how."). video/storyboard_en.py moves every Russian cut to the
  same place of the same sentence in the English voice (two sentence
  starts in scene 6 set by hand). EN 6:32.

## 2026-10-27 (afternoon) — the training screen, install from scratch

- Training screen: progress inside the round from az_train.py's
  `progress.jsonl` (self-play games, training steps, the match and its
  score), a pulse with the time since the trainer last wrote, the self-play
  game followed live (`live.json`), a timeline of the rounds; after the
  first accepted round, buttons into play / vs Dot / vs KSquares with it.
  "vs KSquares" and "play" choose between the released Dot and the own
  runs. Thought panel and ladder legend in the video's words.
- A test run from the app, test_New_dot, 5 rounds from scratch: all
  accepted (100-0, 98-2, 89-11, 71-29, 74-26), ~12 min each; vs Hard with
  MCTS 200: 0, 3, 47, 70, 70%.
- Install from scratch in clean podman containers (ubuntu:22.04 and
  ubuntu:24.04, as root, only git installed before the clone). The first
  run failed in the self-check: KSquares quits at once without a D-Bus
  session bus (KDBusService: "DBus session bus not found"), as on any
  machine without a desktop session (ssh, a container). Fix: env/bridge.py
  starts a private dbus-daemon for the game when DBUS_SESSION_BUS_ADDRESS
  is unset. After it both pass: ~2.5 min, 250 MB of packages, 1.1-1.2 GB
  in build/ + .venv with torch's CPU build. install.sh now installs the
  apt packages itself (only the missing ones), torch by the driver's CUDA
  version (cu130 / cu126 / CPU with a warning), and ends with
  scripts/selfcheck.py.

## 2026-10-28 (night) — part 2 starts: C tree, Dot-2, judge on 10x7

Plan agreed (docs/plan_part2.md, "Decisions"): goal "does the
understanding of parity transfer from 5x5 to 10x7 without further
training"; no 7x10; Dot-2 is 6x64; Dot level = MCTS ladder Elo >= 2100 and
>= 90% vs Hard.

- C tree walk (agents/csrc/mcts_tree.c, cffi, built on first use into
  agents/_mcts_tree*.so; the Python walk stays as fallback / reference).
  Nodes stay Python objects, the C side shares their N/W/P arrays; the
  test checks identical N, W, n and evaluation counts on 3x3/5x5/10x7 with
  1 and 8 leaves per tree. Untrained Dot, batch 192: 10x7 3.0k -> 15.5k
  sims/s, 5x5 19.1k -> 30.7k (docs/results.md).
- Order changed slightly: the training-side parametrisation of step 2
  (board mix, per-size buffers) went in before Dot-2 so that step 4 could
  start at 01:05; ladder anchors, judge, calibration followed while the
  distillation ran.
- Dot-2 (AZNet2): Dot's body + global pooling in blocks 2 and 4 (zero-init
  bias, so Dot's body gives identical features — tested), convolutional
  heads, phase plane. 0.51M parameters. 12.4k sims/s on 10x7 (batch 192).
- Distillation (scripts/distill.py, 30k steps, 27 min): held-out argmax
  agreement with Dot 81%, value MSE to the game result 0.605 (Dot 0.610).
- Judge with a time limit: on 10x7 1 s per position covers 50% of the
  positions of a hard-vs-hard game (the last 78 of 157 lines); 0.25 s 48%,
  4 s 51%. 5x5: 57% (as max_undrawn 34). Unclear: net games may have more
  junctions late than Hard's; re-measure on Dot-2 games.
- strong on 10x7: 28.9k nodes per CPU second -> node budget 14.4k for the
  ladder (load-independent, unlike the 5x5 ladder's 0.5 s wall clock).
- Reference over the bridge with az_vs_hard.py: Dot gen 54, 20 games,
  seed 11: 20-0.
- Stop rule note: on az5b's history it would have fired at gen 38 (Elo
  2116, 100% vs hard); az5b's longest reject streak was 5.
- Step (a) ended by the stop rule after 2 generations (01:36): gen 1
  accepted 57-43 vs the distilled student, MCTS 200 ladder Elo 2192
  (Dot gen 54: 2164), vs hard 100%, vs strong 90%; raw policy vs strong
  43%. Gen 2 rejected (44%). Over the bridge (az_vs_hard.py, 20 games,
  seed 11): 19-1 (Dot: 20-0). Caveat: one evaluation of 100 games per
  opponent, Elo error ~±60; "Dot level" is reached, not "better than Dot".
  Dot-2 for (b): runs/dot2a/gen_0001.pt.
- 10x7 ladder done (30 min, 12 processes, during training): Elo random 0,
  greedy 901, easy 905, medium 1261, hard 1493, strong 1297. strong at
  14.4k nodes is not above Hard on 10x7 (49%) and is non-transitive
  (greedy takes 19.5% from it, 0.8% from Hard). For (b) a stronger anchor
  would help: strong with a larger budget (e.g. 4x) as a separate entry.
- Not done tonight: launcher / app.yaml `best` per size and "boards: any"
  in core.check/brain.fits (the app still refuses Dot-2 on 10x7);
  selfcheck.py and the probe position sets are still 5x5.
