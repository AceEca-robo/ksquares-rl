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
- Commits use AceEca-robo@users.noreply.github.com (set in the project's git config).
- Language: answer the user in Russian; code, documentation and commit messages in English.
