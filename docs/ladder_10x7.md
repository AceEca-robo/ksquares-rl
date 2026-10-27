Board 10x7, 200 games per pair, seed 1077, 12 processes, wall 30.4 min.

| A vs B | games | A wins | B wins | ties | A win % (±95%) | mean margin A−B | first-mover wins % |
|---|---:|---:|---:|---:|---:|---:|---:|
| random vs greedy | 200 | 0 | 200 | 0 | 0.0 ± 0.0 | -68.11 | 50.0 |
| random vs easy | 200 | 0 | 200 | 0 | 0.0 ± 0.0 | -67.65 | 50.0 |
| random vs medium | 200 | 0 | 200 | 0 | 0.0 ± 0.0 | -68.04 | 50.0 |
| random vs hard | 200 | 0 | 200 | 0 | 0.0 ± 0.0 | -67.47 | 50.0 |
| random vs strong | 200 | 0 | 200 | 0 | 0.0 ± 0.0 | -68.27 | 50.0 |
| greedy vs easy | 200 | 99 | 91 | 10 | 49.5 ± 6.9 | +1.10 | 49.5 |
| greedy vs medium | 200 | 0 | 200 | 0 | 0.0 ± 0.0 | -34.46 | 50.0 |
| greedy vs hard | 200 | 0 | 197 | 3 | 0.0 ± 0.0 | -29.72 | 50.3 |
| greedy vs strong | 200 | 31 | 153 | 16 | 15.5 ± 5.0 | -9.12 | 48.4 |
| easy vs medium | 200 | 1 | 199 | 0 | 0.5 ± 1.0 | -38.24 | 49.5 |
| easy vs hard | 200 | 1 | 198 | 1 | 0.5 ± 1.0 | -26.65 | 49.7 |
| easy vs strong | 200 | 45 | 147 | 8 | 22.5 ± 5.8 | -8.41 | 52.1 |
| medium vs hard | 200 | 0 | 199 | 1 | 0.0 ± 0.0 | -27.58 | 49.7 |
| medium vs strong | 200 | 84 | 109 | 7 | 42.0 ± 6.8 | -2.29 | 55.4 |
| hard vs strong | 200 | 100 | 96 | 4 | 50.0 ± 6.9 | +2.66 | 50.0 |

Win % of the row agent against the column agent (ties count half):

| | random | greedy | easy | medium | hard | strong |
|---|---:|---:|---:|---:|---:|---:|
| **random** | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **greedy** | 100.0 | — | 52.0 | 0.0 | 0.8 | 19.5 |
| **easy** | 100.0 | 48.0 | — | 0.5 | 0.8 | 24.5 |
| **medium** | 100.0 | 100.0 | 99.5 | — | 0.2 | 43.8 |
| **hard** | 100.0 | 99.2 | 99.2 | 99.8 | — | 51.0 |
| **strong** | 100.0 | 80.5 | 75.5 | 56.2 | 49.0 | — |

| agent | moves timed | mean ms/move | max ms/move |
|---|---:|---:|---:|
| random | 46200 | 0.01 | 0.2 |
| greedy | 76720 | 0.02 | 3.1 |
| easy | 76440 | 0.06 | 6.1 |
| medium | 88158 | 0.42 | 44.8 |
| hard | 95621 | 1.29 | 38.0 |
| strong | 87861 | 246.41 | 1799.9 |
