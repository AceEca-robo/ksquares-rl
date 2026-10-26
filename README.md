# ksquares-rl

An AlphaZero network that taught itself Dots and Boxes, playing the real
[KSquares](https://apps.kde.org/ksquares/) 21.12.3 (KDE) through a small
socket bridge built into the game. Trained on 5x5 by self-play only: no
human games and no hand-written chain rules. With 200 MCTS simulations it
beats KSquares Hard 99 games out of 100 and an alpha-beta reference player
("strong") 89 out of 100.

![network vs KSquares Hard](docs/app_ksquares.png)

What is in here:

- `game/`: the KSquares sources (upstream tarball, unchanged in the first
  commit) plus the RL bridge (`game/src/rlbridge.cpp`, every change marked
  `// RL bridge`); `env/bridge.py` talks to it over a Unix socket.
- `sim/`: a pure-Python simulator with KSquares' own line numbering, and a
  move-exact port of the KSquares Easy / Medium / Hard AI
  (`sim/ksquares_ai.py`), checked against the real game on 1200 games
  (130,200 moves, all identical).
- `agents/`: baselines, the strong alpha-beta player (`agents/strong.py`),
  the AlphaZero network and MCTS.
- `scripts/`: training (`az_train.py`, `az_eval.py`), evaluation, the
  probes, the app (`launcher.py`). Every hyperparameter is in `configs/*.yaml`.
- `docs/`: `game_spec.md` (rules, storage, main loop and the built-in AI as
  KSquares implements them), `results.md` (all tables), `overnight.md` (the
  experiment log, decisions and dead ends).

## Results

The ladder on 5x5 (`docs/results.md`): win % of the network, 100 games per
cell, first mover alternating; raw = the policy's argmax, no search;
MCTS = 200 simulations. Elo is Bradley-Terry against fixed anchors
(random 0, greedy 905, Medium 1235, Hard 1488, strong 1815).

| network | raw vs Hard | raw vs strong | raw Elo | MCTS vs Hard | MCTS vs strong | MCTS Elo |
|---|---:|---:|---:|---:|---:|---:|
| az5 gen 29 (first run) | 63 | 24 | 1517 | 93 | 49 | 1829 |
| az5b gen 1 | 72 | 28 | 1614 | 98 | 51 | 1870 |
| **az5b gen 54 (the released one)** | 83 | 34 | 1680 | 99 | **89** | **2164** |

![ladder per generation](docs/az5b_ladder.png)

Full tables: `docs/az5_ladder.md`, `docs/az5b_ladder.md`.

**Parity curves.** Dots and Boxes on 5x5 is decided by the long chain rule:
who has to open the first long chain. `docs/parity_curves.md` takes 1000
positions where nothing can be captured, safe lines are left and the move
decides exactly that, and asks every accepted generation how often it picks
a winning move. The policy alone stays near 50% (it does not "see" parity),
while the value head learns it (88% balanced accuracy at gen 54) and MCTS
turns that into 90% winning moves; KSquares Hard gets 49.6%, strong 94.7%.

![parity curves](docs/parity_curves.png)

## The app

One pygame window with a menu (English, Russian, Ukrainian):

- **Network vs KSquares**: the real KSquares on a private Xvfb, its window
  grabbed into the centre; level Easy / Medium / Hard, who moves first.
- **Play the network**: click a line; optionally see what the network
  thinks of your own moves.
- **Network vs itself**: two generations against each other; pause, one
  move, speed.

The thought panel shows the policy as a heat map over the lines, the value
with a "who is winning" bar, the MCTS visits with the principal variation
numbered on the board, the long chains with the side the long chain rule
favours, and whether the move keeps the win (exact solving once at most 34
lines are left). Search: none or MCTS 50 / 200 / 800. The network plays 5x5
only: its heads are tied to the board size.

| menu | play the network | network vs itself |
|---|---|---|
| ![menu](docs/app_menu.png) | ![play](docs/app_play.png) | ![self](docs/app_self.png) |

The network is `runs/best/best.pt` (az5b gen 54, pinned in
`configs/app.yaml`). `runs/` is not in git: in a fresh clone the app offers
to download it (2.2 MB, checked by sha256) from the
[v0.1 release](https://github.com/AceEca-robo/ksquares-rl/releases/tag/v0.1).

## Build and run

Linux (tested on Ubuntu 22.04), Python 3.10+. `install.sh` checks the
system packages and prints the `sudo apt install ...` line for the missing
ones (Qt 5 / KDE Frameworks 5 dev packages, cmake, Xvfb, xdotool,
x11-utils, ffmpeg; see `docs/game_spec.md` §6); nothing else needs root.

```
git clone https://github.com/AceEca-robo/ksquares-rl && cd ksquares-rl
./install.sh            # builds KSquares with the bridge, .venv, the menu entry
./install.sh --cpu      # the same with torch's CPU-only build (a much smaller download)
ksquares-rl             # or "KSquares RL" in the applications menu
./uninstall.sh [--purge]
```

The app runs fine on a CPU. Recording a game:

```
.venv/bin/python scripts/launcher.py --mode self --record self.mp4   # or .gif
```

Development:

```
python3 -m venv .venv && .venv/bin/pip install -e '.[test,az,app]'
.venv/bin/python -m pytest
cmake -S game -B build && cmake --build build -j     # the game (+ bridge)
.venv/bin/python scripts/verify_port.py              # AI port vs the real game
.venv/bin/python scripts/eval_ladder.py              # round robin, 5x5
.venv/bin/python scripts/az_train.py --config configs/az5.yaml   # then az5b.yaml
```

## Videos

Coming soon.
<!-- TODO: links to the videos once they are published -->

## License

GPL-2.0-or-later, as KSquares itself (`LICENSE`). KSquares is by Matt
Williams and the KDE contributors (`AUTHORS.md`); this repository adds the
RL bridge, the simulator, the agents and the app.
