# Results

## Simulator speed

`scripts/bench_sim.py` (`configs/bench_sim.yaml`), self-play, pure Python
3.10, one core, 2026-10-23 (re-measured with all agents; the random/greedy
rows came out ~30% slower than in the first measurement on the same
machine, so treat ±30% as run-to-run noise). A move is one drawn line (a
game has `2wh+w+h` moves). `easy`/`medium`/`hard` = the KSquares AI port
(`sim/ksquares_ai.py`); `strong` = `agents/strong.py` with a 0.5 s limit
per move, so its speed is set by the limit.

| board | agents | games/s | moves/s | ms/move |
|---|---|---:|---:|---:|
| 3x3 | random vs random | 22,283 | 534,795 | 0.002 |
| 3x3 | greedy vs greedy | 8,097 | 194,330 | 0.005 |
| 3x3 | easy vs easy | 4,168 | 100,032 | 0.010 |
| 3x3 | medium vs medium | 2,072 | 49,738 | 0.020 |
| 3x3 | hard vs hard | 1,345 | 32,275 | 0.031 |
| 3x3 | strong vs strong | 0.25 | 6 | 164 |
| 5x5 | random vs random | 5,985 | 359,120 | 0.003 |
| 5x5 | greedy vs greedy | 1,964 | 117,868 | 0.008 |
| 5x5 | easy vs easy | 958 | 57,468 | 0.017 |
| 5x5 | medium vs medium | 216 | 12,944 | 0.077 |
| 5x5 | hard vs hard | 96 | 5,756 | 0.174 |
| 5x5 | strong vs strong | 0.07 | 4 | 239 |
| 10x7 | random vs random | 1,225 | 192,315 | 0.005 |
| 10x7 | greedy vs greedy | 304 | 47,791 | 0.021 |
| 10x7 | easy vs easy | 152 | 23,914 | 0.042 |
| 10x7 | medium vs medium | 10.6 | 1,657 | 0.603 |
| 10x7 | hard vs hard | 3.3 | 521 | 1.921 |
| 10x7 | strong vs strong | 0.02 | 4 | 269 |

The real game over the bridge (`scripts/verify_port.py`, one process,
computer moves in C++, external moves from Python, every move replayed and
checked in the simulator): 0.015–0.02 s per game on 5x5 and 0.03–0.08 s on
10x7 at any level, i.e. ~50 games/s on 5x5 and ~20 on 10x7. For Medium and
Hard on 10x7 that is several times faster than the Python port.

## Baselines in the simulator

`scripts/eval_baselines.py` (`configs/eval_baselines.yaml`, seed 12345),
1000 games per row, seats alternate (each agent moves first in 500 games).
`random` = uniform legal line; `greedy` = complete a box if possible, else a
line that gives no box a third side, else uniform random. "±" is the 95%
normal-approximation interval (degenerate at 0/100%). Margin is in boxes.

| board | A vs B | games | A wins | B wins | ties | A win % (±95%) | mean margin A−B | first-mover wins % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3x3 | random vs random | 1000 | 482 | 518 | 0 | 48.2 ± 3.1 | -0.16 | 50.6 |
| 3x3 | greedy vs random | 1000 | 997 | 3 | 0 | 99.7 ± 0.3 | +7.45 | 49.7 |
| 3x3 | greedy vs greedy | 1000 | 492 | 508 | 0 | 49.2 ± 3.1 | -0.05 | 41.8 |
| 5x5 | random vs random | 1000 | 530 | 470 | 0 | 53.0 ± 3.1 | +0.74 | 50.2 |
| 5x5 | greedy vs random | 1000 | 1000 | 0 | 0 | 100.0 ± 0.0 | +23.30 | 50.0 |
| 5x5 | greedy vs greedy | 1000 | 516 | 484 | 0 | 51.6 ± 3.1 | +0.24 | 50.6 |
| 10x7 | random vs random | 1000 | 493 | 489 | 18 | 49.3 ± 3.1 | +1.05 | 51.1 |
| 10x7 | greedy vs random | 1000 | 1000 | 0 | 0 | 100.0 ± 0.0 | +67.91 | 50.0 |
| 10x7 | greedy vs greedy | 1000 | 448 | 498 | 54 | 44.8 ± 3.1 | -0.58 | 50.0 |

Odd box counts (3x3, 5x5) cannot tie; 10x7 (70 boxes) can.

## KSquares AI port vs the real game

`scripts/verify_port.py` (`configs/verify_port.yaml`, seed 2026,
`QT_HASH_SEED=0`), 2026-10-23. Each game is played twice with the same
seeds — entirely in the simulator (port + `GlibcRand(seed)`) and in
KSquares 21.12.3 over the bridge (`srand(seed)` at reset) — and the move
lists (line, player, completed boxes) are compared. `ext` is a seeded
random agent (first mover alternates), `ai-ai` is KSquares against itself.
"Ambiguous handouts" counts hard-hearted handouts where the QHash order
would matter (see `docs/game_spec.md` §4).

| board | level | seats | games | identical | moves compared | AI moves | ambiguous handouts | bridge s/game | sim s/game |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 5x5 | easy | ext-ai | 100 | 100 | 6000 | 4112 | 0 | 0.018 | 0.001 |
| 5x5 | easy | ai-ai | 100 | 100 | 6000 | 6000 | 0 | 0.015 | 0.002 |
| 5x5 | medium | ext-ai | 100 | 100 | 6000 | 4117 | 0 | 0.016 | 0.002 |
| 5x5 | medium | ai-ai | 100 | 100 | 6000 | 6000 | 0 | 0.017 | 0.009 |
| 5x5 | hard | ext-ai | 100 | 100 | 6000 | 4079 | 0 | 0.019 | 0.008 |
| 5x5 | hard | ai-ai | 100 | 100 | 6000 | 6000 | 0 | 0.018 | 0.020 |
| 10x7 | easy | ext-ai | 100 | 100 | 15700 | 11125 | 0 | 0.046 | 0.008 |
| 10x7 | easy | ai-ai | 100 | 100 | 15700 | 15700 | 0 | 0.042 | 0.012 |
| 10x7 | medium | ext-ai | 100 | 100 | 15700 | 11166 | 0 | 0.038 | 0.013 |
| 10x7 | medium | ai-ai | 100 | 100 | 15700 | 15700 | 0 | 0.044 | 0.153 |
| 10x7 | hard | ext-ai | 100 | 100 | 15700 | 10973 | 0 | 0.056 | 0.126 |
| 10x7 | hard | ai-ai | 100 | 100 | 15700 | 15700 | 0 | 0.031 | 0.205 |

1200 of 1200 games identical (130,200 moves, 98,773 of them by the AI).
Same check for Hard with `QT_HASH_SEED` unset (the game picks a random
QHash seed, as for a normal user; seed 31337): 400 of 400 games identical
(5x5 and 10x7, ext-ai and ai-ai, 100 each).

## Strong reference player

`agents/strong.py`, 0.5 s per move (wall clock, 16 games in parallel on
20 cores), `scripts/eval_ladder.py --config configs/eval_strong.yaml`
(seed 4242), 5x5, 200 games per pair, first mover alternating,
2026-10-23. Hard = the verified port of KSquares Hard.

| A vs B | games | A wins | B wins | ties | A win % (±95%) | mean margin A−B | first-mover wins % |
|---|---:|---:|---:|---:|---:|---:|---:|
| strong vs hard | 200 | 189 | 11 | 0 | 94.5 ± 3.2 | +8.13 | 49.5 |
| strong vs greedy | 200 | 197 | 3 | 0 | 98.5 ± 1.7 | +10.08 | 49.5 |

Mean 210 ms per strong move (max 516 ms); Hard 0.32 ms, greedy 0.02 ms.

## Ladder on 5x5

`scripts/eval_ladder.py` (`configs/eval_ladder.yaml`, seed 777), every
pair 200 games, first mover alternating, strong at 0.5 s per move,
2026-10-23. Wall time 7.8 min on 16 processes.

| A vs B | games | A wins | B wins | ties | A win % (±95%) | mean margin A−B | first-mover wins % |
|---|---:|---:|---:|---:|---:|---:|---:|
| random vs greedy | 200 | 0 | 200 | 0 | 0.0 ± 0.0 | -23.02 | 50.0 |
| random vs easy | 200 | 0 | 200 | 0 | 0.0 ± 0.0 | -23.04 | 50.0 |
| random vs medium | 200 | 0 | 200 | 0 | 0.0 ± 0.0 | -23.39 | 50.0 |
| random vs hard | 200 | 0 | 200 | 0 | 0.0 ± 0.0 | -23.26 | 50.0 |
| random vs strong | 200 | 0 | 200 | 0 | 0.0 ± 0.0 | -23.71 | 50.0 |
| greedy vs easy | 200 | 105 | 95 | 0 | 52.5 ± 6.9 | +0.16 | 42.5 |
| greedy vs medium | 200 | 16 | 184 | 0 | 8.0 ± 3.8 | -9.60 | 46.0 |
| greedy vs hard | 200 | 14 | 186 | 0 | 7.0 ± 3.5 | -8.49 | 51.0 |
| greedy vs strong | 200 | 2 | 198 | 0 | 1.0 ± 1.4 | -10.78 | 49.0 |
| easy vs medium | 200 | 8 | 192 | 0 | 4.0 ± 2.7 | -10.86 | 49.0 |
| easy vs hard | 200 | 15 | 185 | 0 | 7.5 ± 3.7 | -8.76 | 45.5 |
| easy vs strong | 200 | 6 | 194 | 0 | 3.0 ± 2.4 | -10.43 | 51.0 |
| medium vs hard | 200 | 7 | 193 | 0 | 3.5 ± 2.5 | -8.87 | 51.5 |
| medium vs strong | 200 | 12 | 188 | 0 | 6.0 ± 3.3 | -7.59 | 50.0 |
| hard vs strong | 200 | 13 | 187 | 0 | 6.5 ± 3.4 | -8.15 | 52.5 |

Win % of the row agent against the column agent:

| | random | greedy | easy | medium | hard | strong |
|---|---:|---:|---:|---:|---:|---:|
| **random** | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **greedy** | 100.0 | — | 52.5 | 8.0 | 7.0 | 1.0 |
| **easy** | 100.0 | 47.5 | — | 4.0 | 7.5 | 3.0 |
| **medium** | 100.0 | 92.0 | 96.0 | — | 3.5 | 6.0 |
| **hard** | 100.0 | 93.0 | 92.5 | 96.5 | — | 6.5 |
| **strong** | 100.0 | 99.0 | 97.0 | 94.0 | 93.5 | — |

Mean time per move: random 0.00 ms, greedy 0.01, easy 0.03, medium 0.10,
hard 0.22, strong 204 ms (max 529 ms).

Easy plays like greedy (random capture, random safe line, random
sacrifice); the step to Medium is sacrificing the shortest chain, the step
to Hard (96.5% vs Medium) is all-but-two and hard-hearted handouts. Strong
vs Hard came out 93.5% here and 94.5% in the separate run above.

## AlphaZero on 5x5 (overnight 2026-10-23/24)

`scripts/az_train.py` / `scripts/az_eval.py` (`configs/az5.yaml`):
ResNet 6x64 (0.51M parameters), 200 simulations per move in self-play,
gating at 55% over 100 games. Every accepted generation plays 100 games
against each opponent (first mover alternating), as the raw policy
(argmax, no search) and with MCTS 200 (no noise). Elo: Bradley-Terry
against the fixed ladder anchors above (random 0, greedy 905, easy 888,
medium 1235, hard 1488, strong 1815). strong has 0.5 s per move by wall
clock on a loaded machine. Full table per generation: `docs/az5_ladder.md`;
curves: `docs/az5_ladder.png`.

| gen | raw vs greedy | raw vs medium | raw vs hard | raw vs strong | raw Elo | mcts vs greedy | mcts vs medium | mcts vs hard | mcts vs strong | mcts Elo |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 454 | 21 | 10 | 3 | 0 | 750 |
| 2 | 46 | 14 | 3 | 0 | 894 | 92 | 76 | 27 | 2 | 1351 |
| 5 | 84 | 46 | 18 | 2 | 1206 | 94 | 80 | 80 | 4 | 1539 |
| 11 | 88 | 65 | 19 | 15 | 1320 | 99 | 87 | 91 | 40 | 1743 |
| 15 | 85 | 73 | 46 | 10 | 1402 | 100 | 94 | 91 | 33 | 1747 |

### az5 continued and az5b

az5 ran to gen 31 (21 accepted); best gen 29: MCTS 200 vs Hard 93%, vs
strong (0.5 s wall clock) 49%, Elo 1829 (table `docs/az5_ladder.md`).
az5b continues from it with lr 0.01, a score-difference head and random
self-play simulations (100–300); strong there plays with a fixed budget of
23,000 nodes per move (= 0.5 s of one core, `scripts/calibrate_strong.py`).
Table `docs/az5b_ladder.md`, curves `docs/az5b_ladder.png`.

| run gen | raw vs greedy | raw vs medium | raw vs hard | raw vs strong | raw Elo | mcts vs greedy | mcts vs medium | mcts vs hard | mcts vs strong | mcts Elo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| az5 29 | 90 | 74 | 63 | 24 | 1517 | 99 | 97 | 93 | 49 | 1829 |
| az5b 1 | 95 | 86 | 72 | 28 | 1614 | 99 | 99 | 98 | 51 | 1870 |
| az5b 5 | 96 | 80 | 59 | 26 | 1549 | 100 | 99 | 95 | 54 | 1875 |
| az5b 6 | 94 | 73 | 63 | 27 | 1536 | 99 | 95 | 96 | 60 | 1885 |
| az5b 8 | 98 | 75 | 64 | 29 | 1565 | 99 | 97 | 97 | 70 | 1954 |
| az5b 13 | 90 | 80 | 61 | 26 | 1536 | 100 | 96 | 99 | 71 | 1972 |
| az5b 14 | 95 | 80 | 73 | 30 | 1604 | 100 | 96 | 99 | 63 | 1926 |
| az5b 17 | 100 | 83 | 68 | 38 | 1642 | 99 | 97 | 98 | 59 | 1900 |
| az5b 18 | 94 | 75 | 66 | 26 | 1549 | 97 | 97 | 98 | 74 | 1972 |
| az5b 20 | 99 | 74 | 67 | 26 | 1565 | 99 | 97 | 97 | 69 | 1948 |
| az5b 23 | 96 | 74 | 70 | 21 | 1549 | 99 | 96 | 97 | 74 | 1972 |
| az5b 25 | 94 | 71 | 75 | 27 | 1568 | 100 | 95 | 96 | 76 | 1979 |
| az5b 29 | 92 | 80 | 66 | 32 | 1578 | 100 | 97 | 97 | 74 | 1985 |
| az5b 31 | 94 | 75 | 77 | 20 | 1565 | 99 | 98 | 99 | 77 | 2020 |
| az5b 32 | 91 | 83 | 69 | 22 | 1562 | 100 | 98 | 99 | 84 | 2086 |
| az5b 33 | 95 | 75 | 74 | 30 | 1591 | 100 | 98 | 98 | 81 | 2051 |
| az5b 38 | 99 | 91 | 78 | 33 | 1684 | 100 | 100 | 100 | 84 | 2116 |
| az5b 44 | 97 | 84 | 80 | 32 | 1656 | 100 | 100 | 97 | 86 | 2106 |

Real game over the bridge: az5 gen 29 + MCTS 200 vs KSquares Hard 18/20.
Video `docs/az5_vs_hard.mp4` (az5b gen 44 vs the game's Hard, 18–7).

## Strategy probe (5x5)

`docs/strategy_probe.md` (method and full table per generation),
`docs/strategy_probe.png`. 300 critical late positions, % of moves that
reach the best result (exact: brute force for dd, solver for parity):

| player | double-dealing (150) | parity / long chain rule (150) |
|---|---:|---:|
| strong (23k nodes) | 100 | 96 |
| KSquares Hard | 100 | 44 |
| greedy | 0 | 42 |
| random legal move (expected) | 7 | 14 |
| az5b gen 1, net only | 99 | 44 |
| az5b gen 1, net + MCTS 200 | 100 | 89 |
| az5b gen 5, net only | 99 | 49 |
| az5b gen 5, net + MCTS 200 | 100 | 85 |
| az5b gen 6, net only | 97 | 43 |
| az5b gen 6, net + MCTS 200 | 100 | 84 |
| az5b gen 33, net only | 99 | 45 |
| az5b gen 33, net + MCTS 200 | 100 | 85 |
| az5b gen 38, net only | 99 | 52 |
| az5b gen 38, net + MCTS 200 | 100 | 90 |
| az5b gen 44, net only | 100 | 48 |
| az5b gen 44, net + MCTS 200 | 100 | 88 |
