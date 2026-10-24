# ksquares-rl

Reinforcement learning for Dots and Boxes as implemented in
[KSquares](https://apps.kde.org/ksquares/) 21.12.3 (KDE, GPL-2.0-or-later).

Status: game sources imported (`game/`) and built; source reconnaissance
(`docs/game_spec.md`); Python simulator (`sim/`); a socket bridge to the
real game (`game/src/rlbridge.cpp`, `env/bridge.py`); a move-exact port of
the KSquares Easy/Medium/Hard AI (`sim/ksquares_ai.py`, verified against the
game over the bridge); a strong alpha-beta reference player
(`agents/strong.py`); baselines and the agent ladder (`docs/results.md`);
AlphaZero on 5x5 (`scripts/az_train.py`, runs az5 and az5b, `docs/overnight.md`);
the strategy probe (`docs/strategy_probe.md`, `docs/probes/`); the app.

## The app

![network vs KSquares Hard](docs/app_ksquares.png)

One pygame window with a menu (English, Russian, Ukrainian; Esc goes back
to the menu, Esc in the menu quits):

- **Network vs KSquares**: the real KSquares on a private Xvfb, its window
  grabbed into the centre; level Easy / Medium / Hard, who moves first.
- **Play the network**: click a line; optionally the network's view of
  your own moves (`docs/app_play.png`).
- **Network vs itself**: any two accepted generations, pause / one move /
  speed (`docs/app_self.png`).

The thought panel shows the policy as a heat map over the lines, V with a
"who is winning" bar, the MCTS visits with the principal variation numbered
on the board, the long chains on the board with the side the long chain
rule favours, and whether the move keeps the win (exact solving once at
most 34 lines are left); below, V move by move. Search: none or MCTS
50 / 200 / 800. The network is `runs/best/best.pt`, a link to the last
accepted az5b generation (`scripts/update_best.py`). It plays 5x5 only:
its policy and value heads are fully connected over the 11 x 11 grid, so
10x7 is offered in the menu but refused with that explanation.

```
./install.sh                       # builds the game, .venv, the menu entry (ksquares-rl)
ksquares-rl                        # or: .venv/bin/python scripts/launcher.py
.venv/bin/python scripts/launcher.py --mode self --gen-a "az5 gen 1" --record self.mp4   # or .gif
./uninstall.sh [--purge]
```

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
