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
