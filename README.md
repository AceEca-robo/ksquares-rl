# ksquares-rl

Reinforcement learning for Dots and Boxes as implemented in
[KSquares](https://apps.kde.org/ksquares/) 21.12.3 (KDE, GPL-2.0-or-later).

Status: game sources imported (`game/`), source reconnaissance
(`docs/game_spec.md`), Python simulator (`sim/`) and baselines
(`docs/results.md`). No bridge to the game and no training yet.

```
python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest
.venv/bin/python scripts/bench_sim.py
.venv/bin/python scripts/eval_baselines.py
```

Building KSquares itself: see `docs/game_spec.md` §6.

License: GPL-2.0-or-later (`LICENSE`); authors of the game: `AUTHORS.md`.
