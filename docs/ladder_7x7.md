Board 7x7, 200 games per pair, seed 777, 6 processes, wall 310.5 min.

| A vs B | games | A wins | B wins | ties | A win % (±95%) | mean margin A−B | first-mover wins % |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy vs random | 200 | 200 | 0 | 0 | 100.0 ± 0.0 | +46.85 | 50.0 |
| easy vs random | 200 | 200 | 0 | 0 | 100.0 ± 0.0 | +46.86 | 50.0 |
| medium vs random | 200 | 200 | 0 | 0 | 100.0 ± 0.0 | +47.18 | 50.0 |
| hard vs random | 200 | 200 | 0 | 0 | 100.0 ± 0.0 | +46.91 | 50.0 |
| strong vs random | 200 | 200 | 0 | 0 | 100.0 ± 0.0 | +47.38 | 50.0 |
| greedy vs easy | 200 | 107 | 93 | 0 | 53.5 ± 6.9 | +0.60 | 46.5 |
| medium vs greedy | 200 | 197 | 3 | 0 | 98.5 ± 1.7 | +23.20 | 49.5 |
| hard vs greedy | 200 | 195 | 5 | 0 | 97.5 ± 2.2 | +17.90 | 51.5 |
| strong vs greedy | 200 | 186 | 14 | 0 | 93.0 ± 3.5 | +12.94 | 49.0 |
| medium vs easy | 200 | 195 | 5 | 0 | 97.5 ± 2.2 | +23.65 | 48.5 |
| hard vs easy | 200 | 200 | 0 | 0 | 100.0 ± 0.0 | +19.61 | 50.0 |
| strong vs easy | 200 | 185 | 15 | 0 | 92.5 ± 3.7 | +12.98 | 45.5 |
| hard vs medium | 200 | 194 | 6 | 0 | 97.0 ± 2.4 | +19.02 | 49.0 |
| strong vs medium | 200 | 172 | 28 | 0 | 86.0 ± 4.8 | +9.68 | 47.0 |
| strong vs hard | 200 | 164 | 36 | 0 | 82.0 ± 5.3 | +9.10 | 52.0 |
| strong4 vs medium | 200 | 179 | 21 | 0 | 89.5 ± 4.2 | +12.06 | 49.5 |
| strong4 vs hard | 200 | 176 | 24 | 0 | 88.0 ± 4.5 | +10.36 | 51.0 |
| strong4 vs strong | 200 | 134 | 66 | 0 | 67.0 ± 6.5 | +3.87 | 47.0 |

Win % of the row agent against the column agent (ties count half):

| | random | greedy | easy | medium | hard | strong | strong4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **random** | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | — |
| **greedy** | 100.0 | — | 53.5 | 1.5 | 2.5 | 7.0 | — |
| **easy** | 100.0 | 46.5 | — | 2.5 | 0.0 | 7.5 | — |
| **medium** | 100.0 | 98.5 | 97.5 | — | 3.0 | 14.0 | 10.5 |
| **hard** | 100.0 | 97.5 | 100.0 | 97.0 | — | 18.0 | 12.0 |
| **strong** | 100.0 | 93.0 | 92.5 | 86.0 | 82.0 | — | 33.0 |
| **strong4** | — | — | — | 89.5 | 88.0 | 67.0 | — |

| agent | moves timed | mean ms/move | max ms/move |
|---|---:|---:|---:|
| random | 33657 | 0.01 | 0.1 |
| greedy | 54399 | 0.03 | 6.0 |
| easy | 54049 | 0.05 | 4.1 |
| medium | 71196 | 0.38 | 35.6 |
| hard | 76269 | 0.90 | 24.6 |
| strong | 76599 | 532.26 | 2984.8 |
| strong4 | 37031 | 1907.42 | 11193.0 |
