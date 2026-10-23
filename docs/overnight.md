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
