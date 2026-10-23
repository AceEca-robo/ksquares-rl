# KSquares 21.12.3 — game specification (source reconnaissance)

All references are to `game/src/` (upstream sources; the only changes are
the RL bridge, §7). Everything below comes from reading the code and was
confirmed on the built binary where noted (2026-10-23).

## 1. Rules as implemented

Implemented in `ksquaresgame.cpp` (`KSquaresGame`).

- Players take turns drawing one line between two adjacent dots.
- `addLineToIndex(index)`: if the line already exists it only prints a
  `qWarning` and returns — the turn does **not** pass and nothing else
  happens. Otherwise the line is set and, if the game is running,
  `checkForNewSquares()` scans **every** box; each unowned box whose four
  lines are drawn goes to the current player (`playerSquareComplete`).
  One line can complete two boxes.
- `tryEndGo()`: if at least one box was completed the same player moves
  again (`takeTurnSig` re-emitted), otherwise `nextPlayer()` (round robin).
- The game ends when the sum of scores reaches `width*height`
  (`gameOver` signal); at that point every line is drawn as well.
- Winner: most boxes. Ties are possible (no tie-break). The high-score entry
  for a winning human is `score - boxes/players`.
- Players: 2–4 (`spinNumOfPlayers` 2..4), each seat human or computer.
  Player ids 0..n-1; **player 0 always moves first** (`i_currentPlayerId`
  starts at -1 and `createGame` calls `nextPlayer()`).
- "Quick start" option: before `start()`, `aiController::autoFill(8)` draws
  random safe lines (no box gets a third side) until no safe line is left,
  then removes 8 of them at random, so about 8 safe moves remain. Uses
  `rand()` and `QRandomGenerator::global()` (the latter is randomly seeded).

## 2. Board size and storage

- Size in boxes: width 2..50, height 1..30 (`newgamewidget.ui`); default
  **10 × 7** (`ksquares.kcfg`). The `--demo` mode uses 15 × 10 with 4 AI
  players and restarts forever.
- `QList<bool> lineList` — `2*w*h + w + h` lines; `QList<int>
  squareOwnerTable` — `w*h` boxes, owner id or -1.
- Line numbering: row by row with stride `2w+1`; in row `r` first the `w`
  horizontal lines (top edges of row `r`), then `w+1` vertical lines; the
  last row has only the `w` bottom lines. Box `i` has lines
  `top=(i/w)*(2w+1)+i%w`, `left=top+w`, `right=left+1`, `bottom=right+w`.
  `sim/board.py` uses exactly this numbering (tested against a literal port
  of `aiController::squaresFromLine`), so line indices pass through a bridge
  unchanged.
- Scores live in `KSquaresPlayer` (`ksquaresplayer.h`).

## 3. Main loop and move processing

Plain Qt event loop (`main.cpp` → `app.exec()`), no game thread.

```
KSquaresGame::takeTurnSig(player) ──► KSquaresWindow::playerTakeTurn
   human: m_scene->enableEvents(); wait for a click
          GameBoardScene::mouseReleaseEvent → getTwoNearestPoints →
          indexFromPointPair → lineDrawn(index) ──► KSquaresGame::addLineToIndex
   AI:    m_scene->disableEvents(); QTimer::singleShot(200 ms, aiChooseLine)
          aiChooseLine: new aiController(lines, squares, w, h) (stateless,
          rebuilt every move) → chooseLine() → sGame->addLineToIndex(index)
KSquaresGame::addLineToIndex → drawLine → checkForNewSquares →
   playerSquareComplete (drawSquare, gameOver?) → tryEndGo →
   takeTurnSig(same player) | nextPlayer() → takeTurnSig(next)
```

- New game: `gameNew()` always shows a **modal** `NewGameDialog`
  (also at startup via `initObject`); `gameReset()` ("Restart") builds a game
  from `Settings` without a dialog.
- Game over: `KSquaresWindow::gameOver` shows a **modal** `ScoresDialog`, and
  for a winning human also a modal `KScoreDialog`. Both block an automated
  driver and must be bypassed by a bridge.
- Settings: KConfig `ksquaresrc` (`ksquares.kcfg`), generated `Settings`
  singleton. AI difficulty is one global value `Settings::difficulty()`
  (0 Easy, 1 Medium, 2 Hard) shared by all computer seats; the kcfg
  default is **0 (Easy)**. Default seats: all human (`HumanList` 2,2,2,2).

## 4. Built-in AI (`aicontroller.cpp`, `aiController::chooseLine`)

No game-tree search at any level: a one-ply rule-based player, with a greedy
"follow the chain" rollout to estimate how many boxes a move gives away.
The header itself says the fourth stage (fighting for control by counting
chains, the "Nim-like" part) is **not implemented**.

Phases of `chooseLine()`:

1. **Capture available** (some box has 3 sides).
   - Easy, Medium: random line among those that complete a box.
   - Hard: `chooseLeastDamaging(all open lines)` with chain analysis (below).
2. **Safe moves exist** (lines that give no box a third side): random safe
   line — at **all** levels. No parity / long-chain-rule reasoning.
3. **Forced phase** (every line hands something over): candidates are lines
   next to a box with 2 sides.
   - Easy: random candidate.
   - Medium: `chooseLeastDamaging` — for each candidate, draw it and let the
     opponent greedily take every box that reaches 3 sides (repeatedly
     scanning the whole board); pick the line with the smallest count, i.e.
     sacrifice the shortest chain.
   - Hard: same plus the hard-specific logic below.

Hard-only logic in `chooseLeastDamaging` (`difficulty > 1`):

- First simulates completing all chains the opponent has offered
  (`myLines`), collecting `ownChains`; `ownMoves` = the shortest one
  (comment: avoids alternating between chains).
- Counts the chains the candidate moves would give away (deduplicated line
  sets): "short" ≤ 3 lines, "long" otherwise. Loops are not detected (TODO
  in the code).
- **Double-dealing ("evil move")**: if exactly 2 own lines remain (or 3
  lines yielding 4 boxes — a loop), there is at least one long chain, and no
  safe moves are left, it draws the line that leaves the last 2 (or 4)
  boxes to the opponent so that the opponent must open the next chain.
  `magicLine` finds the right line because the chain walk can visit lines
  out of spatial order.
- Otherwise takes `ownMoves[0]` (keeps capturing).
- **Hard-hearted handout**: when the least damaging sacrifice gives 2 boxes,
  it draws the middle line of the 2-chain, so the opponent cannot decline it
  with a double-dealing move.
- Fallback: Medium behaviour.

So Hard knows the endgame tactics (all-but-two, hard-hearted handout,
sacrificing the shortest chain) but plays the opening/middle game at random
among safe moves and does not fight for control. Its weaknesses: the
double-dealing condition counts own lines over *all* offered chains, the
"long chain" threshold is in lines, not boxes, and loops are handled only by
the 3-lines/4-boxes special case.

Implementation quirks that matter for reproduction:

- Random choice everywhere is
  `float f = ((float)rand() / (RAND_MAX + 1.0)) * (n - 1); idx = (short)f`,
  so with n ≥ 2 the **last candidate is practically never chosen** (with
  n = 2 it is almost always the first). Exactly: `(float)rand()` rounds to
  2^31 for rand() ≥ 2^31−64, and the product is rounded to float again, so
  the last index comes out with probability ≈ 3·10⁻⁸ per call. Candidate
  lists are in increasing line order, so the AI is biased away from the
  highest-indexed line. Candidate lists may contain duplicates: phase 1 and
  phase 3 append a line once per adjacent box with 3 (resp. 2) sides, and
  duplicates are kept through `chooseLeastDamaging`.
- `rand()` is called even when there is one candidate (Hard phase 3 always
  draws from `goodChoiceList`), which matters for reproducing the sequence;
  Hard's phase 1 (`choices.at(0)`) does not call it.
- QMultiMap `values(key)` and `begin()` return the most recently inserted
  value first, i.e. Hard's `choices.at(0)` prefers the highest-indexed
  equally-good line.
- `squaresFromLine` returns (box below, box above) for horizontal lines and
  (box right, box left) for vertical ones; the Hard chain walk keeps the
  *last* adjacent box with two sides, so this order decides where it goes.
- In the damage count, a line that would complete a second box is checked
  against `myLines`, not the current copy `linesCopy` (counts can be off).
- The hard-hearted handout iterates `QSet<int>::values()`, whose order is
  the QHash bucket order. Qt 5 seeds QHash randomly per process unless
  `QT_HASH_SEED` is set, so if several lines qualify the choice is not
  reproducible across processes. In 1200 verification games (§7) no such
  tie occurred; the bridge runs the game with `QT_HASH_SEED=0` anyway and the
  port emulates the Qt 5.15 bucket order for that seed.

## 5. Determinism

- `chooseLine()` uses only `rand()`; **`srand()` is never called** anywhere
  in KSquares, and **no linked library calls it either** (verified with an
  `LD_PRELOAD` shim logging `srand`/`srandom`/`rand` over whole games: no
  seeding call, the first `rand()` returns 1804289383 = glibc `srand(1)`).
  So a fresh process playing the same human moves gets the same AI replies.
  Across games in one process the sequence continues, so the AI is
  reproducible only per process, not per game.
- Quick start uses `QRandomGenerator::global()` (non-deterministic).
- The AI is a pure function of (lines, owners, w, h, difficulty, rand state);
  the 200 ms delay is only a `QTimer`. A bridge can make it fully
  deterministic per game by calling `srand(seed)` at reset.

## 6. Build

Ubuntu 22.04 packages (from `debian/control` of the source package):

```
sudo apt-get build-dep ksquares     # or explicitly:
sudo apt-get install build-essential cmake extra-cmake-modules gettext \
  qtbase5-dev libkf5config-dev libkf5configwidgets-dev libkf5coreaddons-dev \
  libkf5crash-dev libkf5dbusaddons-dev libkf5doctools-dev libkf5i18n-dev \
  libkf5kdegames-dev libkf5widgetsaddons-dev libkf5xmlgui-dev \
  libkf5completion-dev
cmake -S game -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
./build/src/ksquares          # headless: QT_QPA_PLATFORM=offscreen
```

CMake ≥ 3.16, Qt ≥ 5.15, KF5 ≥ 5.85, KF5KDEGames ≥ 7.3 (all satisfied by
22.04; built here with Qt 5.15.3). The binary lands in `build/src/` and
runs from the build tree without `make install` (the XMLGUI file is
compiled in via `ksquares.qrc`); confirmed, also with
`QT_QPA_PLATFORM=offscreen` and no X display.

## 7. RL bridge

`game/src/rlbridge.{h,cpp}` plus a few lines in `ksquareswindow.{h,cpp}`
and `CMakeLists.txt`, all marked `// RL bridge`. Active only when
`KSQUARES_RL_SOCKET` names a Unix socket; then the game connects to it as
a client instead of opening the new-game dialog, and speaks one JSON object
per line (protocol in `rlbridge.h`): `reset` (board size, which players are
computer players, difficulty, optional `srand` seed) and `step` (line for
an external player). Replies are `state` messages sent whenever an
external player is to move or the game is over, listing every line drawn
since the last message with its player and completed boxes. Computer
players move on a 0 ms timer instead of 200 ms; the score dialogs are
skipped. External players are KSquares "human" seats, and their lines go
through `KSquaresGame::addLineToIndex` exactly like a mouse click.

Python side: `env/bridge.py` (`KSquaresBridge`) starts the game offscreen
with a private `XDG_CONFIG_HOME` (the user's `ksquaresrc` is never
touched) and `QT_HASH_SEED=0`, and replays every reported move in a
`sim.board.Board`, raising `BridgeMismatch` on the first difference in
mover, completed boxes, lines, owners, scores or side to move.

The AI port `sim/ksquares_ai.py` was checked against the game with
`scripts/verify_port.py`: every game is played in the simulator and through
the bridge with the same seeds and the move lists must be identical
(results in `docs/results.md`).
