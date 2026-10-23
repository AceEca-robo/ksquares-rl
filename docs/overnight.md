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
