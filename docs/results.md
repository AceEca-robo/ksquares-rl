# Results

## Simulator speed

`scripts/bench_sim.py` (`configs/bench_sim.yaml`), pure Python 3.10, one
core, 2026-10-23. A move is one drawn line (a game has `2wh+w+h` moves).

| board | agents | games/s | moves/s |
|---|---|---:|---:|
| 3x3 | random vs random | 33,796 | 811,116 |
| 3x3 | greedy vs greedy | 12,697 | 304,729 |
| 5x5 | random vs random | 9,340 | 560,381 |
| 5x5 | greedy vs greedy | 2,736 | 164,149 |
| 10x7 | random vs random | 1,893 | 297,218 |
| 10x7 | greedy vs greedy | 463 | 72,756 |

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
