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
- Choosing the best generation (user, 2026-11-19, after series 10): pick the best generation by matches between
  candidates, not by the ladder. Candidates = the top 3-4 generations by ladder Elo plus the last accepted one; a round
  robin between them (MCTS 200, first mover alternating, opening temperature, games per pair fixed in the protocol)
  picks the best. The ladder (Bradley-Terry Elo against scripted agents) is only for the early rounds, while the net
  does not yet beat Strong; once nets beat every scripted agent, the ladder cannot rank them (part 1: gen 54 picked by
  the ladder, gen 84 beats it 68 %; series 9: g5 gen 49 by the ladder, gen 114 beats it 93 %).

## Before publishing (user, 2026-11-19; a rule for every future project)

A video, a release or a conclusion in the README goes out only after every claim in it has passed these checks:
1. Every number in the script and the README points to a data file and was recomputed from the saved weights or
   logs, not copied from a summary.
2. "The best network" was chosen by matches between candidates, not by the ladder.
3. A conclusion like "stopped growing", "cannot", "stronger" is backed by a direct comparison whose interval
   excludes a draw, on a number of games written down beforehand.
4. For every main conclusion, a check that could refute it is named and was run. Without such a check the
   conclusion is marked as an observation, not a result.
5. Training, selection and evaluation data do not overlap, and this was checked.
6. Before a script, list the claims with a status "checked / observation / not checked"; only the first two go into
   the video, observations with a spoken caveat.
The lesson behind this rule: part 1 picked gen 54 by the ladder, and "Dot did not grow after round 54" went into
the video and the README; series 10 showed gen 84 beats gen 54 68 % [61, 74].
