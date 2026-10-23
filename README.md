# ksquares-rl

Reinforcement learning for Dots and Boxes as implemented in
[KSquares](https://apps.kde.org/ksquares/) 21.12.3 (KDE, GPL-2.0-or-later).

Status: game sources imported (`game/`) and built; source reconnaissance
(`docs/game_spec.md`); Python simulator (`sim/`); a socket bridge to the
real game (`game/src/rlbridge.cpp`, `env/bridge.py`); a move-exact port of
the KSquares Easy/Medium/Hard AI (`sim/ksquares_ai.py`, verified against the
game over the bridge); a strong alpha-beta reference player
(`agents/strong.py`); baselines and the agent ladder (`docs/results.md`).
No RL training yet.

```
python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest
.venv/bin/python scripts/bench_sim.py
.venv/bin/python scripts/eval_baselines.py
.venv/bin/python scripts/eval_ladder.py              # round robin, 5x5
cmake -S game -B build && cmake --build build -j     # the game (+ bridge)
.venv/bin/python scripts/verify_port.py              # AI port vs real game
```

Building KSquares itself: see `docs/game_spec.md` §6.

License: GPL-2.0-or-later (`LICENSE`); authors of the game: `AUTHORS.md`.
