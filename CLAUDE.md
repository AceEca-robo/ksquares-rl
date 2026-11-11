# ksquares-rl
Goal: train an RL agent to play Dots and Boxes in KSquares 21.12.3 (KDE, Matt Williams et al., GPL-2+).
- game/ — the game's sources (upstream tarball, unchanged in the first commit). Only minimal changes, each marked "// RL bridge".
- Building the game: cmake -S game -B build && cmake --build build -j. Dependencies: see docs/game_spec.md §6 (or `sudo apt-get build-dep ksquares`). Build artifacts are in .gitignore.
- sim/ — pure-Python Dots and Boxes simulator, line numbering identical to KSquares (lines/boxes as in KSquaresGame).
- env/ — the gymnasium wrapper; the bridge to the game talks over a Unix socket (as in pacman-rl).
- agents/, scripts/, configs/ — agents and training. All hyperparameters only in configs/*.yaml.
- docs/game_spec.md — rules, storage, main loop and built-in AI as implemented in KSquares.
- docs/overnight.md — the experiment log: chronology, decisions, unclear points and "Ideas for later"; each session adds a dated section. docs/results.md — the results tables.
- runs/ — runs and checkpoints, not in git (.gitignore).
- Python 3.10+, environment .venv. Run pytest before committing. Training longer than 10 minutes only in tmux.
- Mid-run checkpoints (user, 2026-11-11): every 500 steps a run saves net, optimizer, lr schedule, step, torch/numpy/python RNG states and loop state to runs/<series>/<run>.ckpt (temporary file + rename; scripts/graph_hand.py save_checkpoint / load_checkpoint). A started run resumes from its last checkpoint; the .ckpt is deleted when the run finishes. Resume was checked bit-exact against an uninterrupted run (tests/test_checkpoint.py and a real 1500-step series run).
- Command "stop" for a series: touch runs/<series>/STOP; every run exits at its next checkpoint and no new run starts; wait until all have exited, commit, then answer "можно выключать". Delete STOP before continuing the series.
- GPU limits: memory at most 40%; GPU load is not limited.
- Commits use AceEca-robo@users.noreply.github.com (set in the project's git config).
- Language: answer the user in Russian; code, documentation and commit messages in English.
