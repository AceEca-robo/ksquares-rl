# Part 4, series 3': stage 0 kept as an auxiliary share of stage 2

Protocol: docs/plan_part4.md (series 3'), configs/graph_s3p.yaml. Stage 2 has 40000 steps, every 4th a stage 0 step (10x7, V-sign loss); the switch and the best point count policy steps; 3 runs at once. Steps below are total stage 2 steps. Stage 1: school (a) of series 2' on the stage 0 target (20000 steps). Stage 2: the judge's positions from the stage 1 weights, 30000 steps, 5x5 to 5500, then 10x7; best point by 10x7 validation after the switch. Stage 0 columns: V-sign accuracy on the stage 0 10x7 validation (diagnostic only).

## Table 1. Runs

| arm | seed | s1 5x5 click | s1 10x7 click | s1 last 10x7 | best step | val 5x5 | val 10x7 | probe 5x5 | probe 10x7 [95%] | stage0 10x7 at start / best / end |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C | 1 | - | - | 51.1 | 24000 | 49.2 | 60.2 | 45.5 | 69.4 [64-74] | 51.1 / 49.5 / 50.8 |
| C | 2 | - | - | 49.2 | 9000 | 46.2 | 61.4 | 45.0 | 64.8 [59-70] | 49.2 / 51.9 / 49.5 |
| C | 3 | - | - | 50.6 | 8000 | 48.8 | 63.0 | 49.8 | 72.0 [67-77] | 50.6 / 50.6 / 49.6 |
| C | 4 | - | - | 48.5 | 25000 | 48.6 | 60.0 | 48.7 | 71.4 [66-76] | 48.5 / 49.1 / 51.3 |
| C | 5 | - | - | 50.0 | 15500 | 47.8 | 62.6 | 51.8 | 70.7 [65-76] | 50.0 / 48.3 / 49.5 |
| Ch | 1 | 250 | 250 | 100.0 | 16000 | 41.0 | 60.2 | 45.8 | 70.4 [65-75] | 100.0 / 100.0 / 100.0 |
| Ch | 2 | 250 | 250 | 100.0 | 23500 | 42.8 | 61.4 | 45.7 | 69.1 [64-74] | 100.0 / 100.0 / 100.0 |
| Ch | 3 | 250 | 250 | 100.0 | 16000 | 41.2 | 61.4 | 44.4 | 68.1 [63-73] | 100.0 / 100.0 / 100.0 |
| Ch | 4 | 250 | 250 | 100.0 | 7500 | 47.4 | 60.6 | 48.0 | 73.7 [68-78] | 100.0 / 99.9 / 100.0 |
| Ch | 5 | 250 | 250 | 100.0 | 7500 | 44.8 | 61.6 | 45.5 | 72.0 [67-77] | 100.0 / 100.0 / 100.0 |
| Ch14 | 1 | 500 | 1750 | 100.0 | 21000 | 46.2 | 62.2 | 47.5 | 69.4 [64-74] | 100.0 / 97.5 / 99.0 |
| Ch14 | 2 | 250 | 1500 | 100.0 | 18000 | 35.2 | 61.4 | 36.1 | 67.1 [62-72] | 100.0 / 96.2 / 99.1 |
| Ch14 | 3 | 250 | 1500 | 100.0 | 12000 | 43.2 | 62.2 | 45.9 | 67.8 [62-73] | 100.0 / 97.1 / 100.0 |
| Ch14 | 4 | 250 | 1500 | 100.0 | 7500 | 43.2 | 62.8 | 47.1 | 70.7 [65-76] | 100.0 / 99.5 / 100.0 |
| Ch14 | 5 | 250 | 1500 | 100.0 | 8500 | 47.6 | 64.2 | 48.5 | 67.4 [62-72] | 100.0 / 99.8 / 100.0 |
| Ga | 1 | - | - | 50.6 | 8500 | 44.4 | 61.4 | 51.1 | 70.4 [65-75] | 50.6 / 48.8 / 52.1 |
| Ga | 2 | - | - | 50.6 | 15500 | 45.4 | 63.2 | 45.1 | 70.4 [65-75] | 50.6 / 49.1 / 51.1 |
| Ga | 3 | - | - | 50.6 | 35500 | 42.2 | 62.6 | 45.2 | 68.8 [63-74] | 50.6 / 50.5 / 51.1 |
| Ga | 4 | - | - | 50.6 | 8500 | 47.0 | 62.8 | 46.4 | 69.4 [64-74] | 50.6 / 50.1 / 50.5 |
| Ga | 5 | - | - | 50.6 | 8000 | 48.0 | 62.0 | 49.0 | 70.4 [65-75] | 50.6 / 48.7 / 50.8 |
| Gc | 1 | 1000 | 1500 | 100.0 | 22500 | 54.0 | 90.0 | 59.2 | 94.1 [91-96] | 100.0 / 100.0 / 100.0 |
| Gc | 2 | 750 | 1000 | 100.0 | 15000 | 44.6 | 66.0 | 44.2 | 72.4 [67-77] | 100.0 / 100.0 / 100.0 |
| Gc | 3 | 1500 | 1750 | 100.0 | 30500 | 47.8 | 65.2 | 46.6 | 72.4 [67-77] | 100.0 / 50.3 / 50.6 |
| Gc | 4 | 1750 | 2250 | 100.0 | 35500 | 47.8 | 66.2 | 47.0 | 70.1 [65-75] | 100.0 / 53.6 / 53.9 |
| Gc | 5 | 500 | 2750 | 100.0 | 8500 | 46.8 | 63.8 | 47.8 | 72.4 [67-77] | 100.0 / 99.6 / 100.0 |

## Gate (stage 0 V-sign on 10x7 >= 90.0% at the best point; a pair needs >= 3 of 5 passing runs in both arms)

| arm | passing | seeds passing | probe 10x7 mean, passing runs | probe 10x7 mean, all runs |
|---|---:|---|---:|---:|
| C | 0/5 | - | - | 69.7 |
| C+h | 5/5 | 1 2 3 4 5 | 70.7 | 70.7 |
| C+h14 | 5/5 | 1 2 3 4 5 | 68.5 | 68.5 |
| Ga | 0/5 | - | - | 69.9 |
| Gc | 3/5 | 1 2 5 | 79.6 | 76.2 |

## Tests on the 10x7 probe (bootstrap 10000 resamples over passing seeds and positions; one-sided 95% lower bound)

| pair | role | runs | difference, points | lower bound | McNemar seed pairs: first better / second better / n.s. |
|---|---|---|---:|---:|---|
| Gc - C+h | primary | 3 / 5 | +8.9 | -0.3 | 5 / 0 / 10 |
| Gc - C+h14 | second (pre-registered) | 3 / 5 | +11.1 | +1.9 | 5 / 0 / 10 |
| Gc - C | descriptive | 3 / 0 | not comparable | - | - |
| Ga - C | descriptive | 0 / 0 | not comparable | - | - |
| Gc - Ga | descriptive | 3 / 0 | not comparable | - | - |
| C+h - C+h14 | descriptive | 5 / 5 | +2.2 | -0.9 | 3 / 0 / 22 |

## Verdict by the decision table (docs/plan_part4.md, series 3')

Gc passing runs 3/5, C+h 5/5, C+h14 5/5. Primary Gc - C+h: +8.9 points, lower bound -0.3; Gc probe 10x7 mean (passing runs) 79.6%. Second Gc - C+h14: +11.1, lower bound +1.9. 

Row: Gc <= C+h, Gc - C+h14 bound > 0: graph beats the chain count alone, not with the target plane. Next: stop part 4; reported.

## Descriptive (outside the verdict; logs and saved weights only, no new runs)

### (1) Gc s1 (the jump) next to Gc s2 (sign kept, no jump)

Per validation (every 500 steps of 40000; policy step = step x 3/4): judge validation 10x7, stage 0 V-sign on 10x7, mean policy loss and mean stage 0 loss over the 500 steps before.

| step | policy step | s1 val 10x7 | s1 V-sign | s1 policy loss | s1 stage 0 loss | s2 val 10x7 | s2 V-sign | s2 policy loss | s2 stage 0 loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 27.6 | 100.0 | - | - | 3.0 | 100.0 | - | - |
| 500 | 375 | 57.2 | 99.8 | 1.106 | 0.866 | 56.6 | 98.5 | 1.120 | 1.002 |
| 1000 | 750 | 57.8 | 100.0 | 1.019 | 0.467 | 57.2 | 99.5 | 1.042 | 0.316 |
| 2000 | 1500 | 56.4 | 97.8 | 0.853 | 0.311 | 59.8 | 99.8 | 0.959 | 0.176 |
| 3000 | 2250 | 57.0 | 99.5 | 0.479 | 0.382 | 58.2 | 100.0 | 0.724 | 0.057 |
| 4000 | 3000 | 56.0 | 87.2 | 0.218 | 0.106 | 57.8 | 99.5 | 0.367 | 0.025 |
| 5000 | 3750 | 55.8 | 99.9 | 0.130 | 0.046 | 57.4 | 99.8 | 0.177 | 0.055 |
| 6000 | 4500 | 56.2 | 99.9 | 0.095 | 0.040 | 57.6 | 100.0 | 0.107 | 0.087 |
| 7000 | 5250 | 56.4 | 99.9 | 0.074 | 0.079 | 57.4 | 99.9 | 0.087 | 0.108 |
| 7500 | 5625 | 61.6 | 92.5 | 0.567 | 0.181 | 60.8 | 84.7 | 0.517 | 0.274 |
| 8000 | 6000 | 81.8 | 99.9 | 0.549 | 0.223 | 62.2 | 99.6 | 0.668 | 0.271 |
| 9000 | 6750 | 84.0 | 100.0 | 0.197 | 0.013 | 63.0 | 99.7 | 0.370 | 0.032 |
| 10000 | 7500 | 86.2 | 99.9 | 0.132 | 0.030 | 62.4 | 99.7 | 0.189 | 0.012 |
| 11000 | 8250 | 87.2 | 99.6 | 0.092 | 0.024 | 63.4 | 99.9 | 0.117 | 0.012 |
| 12000 | 9000 | 85.8 | 99.9 | 0.067 | 0.016 | 61.8 | 99.6 | 0.085 | 0.023 |
| 13000 | 9750 | 86.8 | 99.8 | 0.052 | 0.020 | 62.8 | 99.9 | 0.067 | 0.026 |
| 14000 | 10500 | 86.8 | 100.0 | 0.050 | 0.008 | 62.4 | 99.9 | 0.054 | 0.013 |
| 15000 | 11250 | 85.4 | 99.9 | 0.037 | 0.026 | 66.0 | 100.0 | 0.044 | 0.012 |
| 16000 | 12000 | 88.2 | 97.2 | 0.030 | 0.010 | 63.4 | 99.9 | 0.055 | 0.015 |
| 17000 | 12750 | 87.8 | 99.9 | 0.028 | 0.009 | 65.0 | 99.7 | 0.029 | 0.015 |
| 18000 | 13500 | 89.6 | 99.8 | 0.018 | 0.017 | 62.8 | 100.0 | 0.021 | 0.003 |
| 19000 | 14250 | 88.6 | 99.8 | 0.016 | 0.004 | 65.8 | 99.6 | 0.022 | 0.004 |
| 20000 | 15000 | 88.6 | 99.9 | 0.007 | 0.006 | 63.4 | 99.9 | 0.018 | 0.003 |
| 21000 | 15750 | 87.2 | 99.9 | 0.015 | 0.003 | 65.0 | 100.0 | 0.017 | 0.002 |
| 22000 | 16500 | 88.4 | 99.8 | 0.010 | 0.003 | 63.2 | 100.0 | 0.013 | 0.002 |
| 23000 | 17250 | 89.0 | 100.0 | 0.000 | 0.000 | 64.4 | 100.0 | 0.008 | 0.001 |
| 24000 | 18000 | 89.2 | 100.0 | 0.000 | 0.000 | 63.6 | 99.9 | 0.007 | 0.002 |
| 25000 | 18750 | 89.2 | 100.0 | 0.000 | 0.000 | 64.6 | 100.0 | 0.005 | 0.000 |
| 26000 | 19500 | 89.2 | 100.0 | 0.000 | 0.000 | 63.8 | 100.0 | 0.003 | 0.000 |
| 27000 | 20250 | 89.4 | 100.0 | 0.000 | 0.000 | 64.4 | 100.0 | 0.002 | 0.001 |
| 28000 | 21000 | 89.8 | 100.0 | 0.000 | 0.000 | 64.8 | 100.0 | 0.000 | 0.000 |
| 29000 | 21750 | 89.8 | 100.0 | 0.000 | 0.000 | 65.0 | 100.0 | 0.000 | 0.000 |
| 30000 | 22500 | 89.8 | 100.0 | 0.000 | 0.000 | 65.0 | 100.0 | 0.000 | 0.000 |
| 31000 | 23250 | 89.8 | 100.0 | 0.000 | 0.000 | 64.8 | 100.0 | 0.000 | 0.000 |
| 32000 | 24000 | 89.2 | 100.0 | 0.000 | 0.000 | 65.0 | 100.0 | 0.000 | 0.000 |
| 33000 | 24750 | 89.2 | 100.0 | 0.000 | 0.000 | 65.2 | 100.0 | 0.000 | 0.000 |
| 34000 | 25500 | 89.2 | 100.0 | 0.000 | 0.000 | 65.4 | 100.0 | 0.000 | 0.000 |
| 35000 | 26250 | 89.0 | 100.0 | 0.000 | 0.000 | 65.2 | 100.0 | 0.000 | 0.000 |
| 36000 | 27000 | 89.2 | 100.0 | 0.000 | 0.000 | 65.2 | 100.0 | 0.000 | 0.000 |
| 37000 | 27750 | 89.2 | 100.0 | 0.000 | 0.000 | 65.2 | 100.0 | 0.000 | 0.000 |
| 38000 | 28500 | 89.2 | 100.0 | 0.000 | 0.000 | 65.2 | 100.0 | 0.000 | 0.000 |
| 39000 | 29250 | 89.2 | 100.0 | 0.000 | 0.000 | 65.2 | 100.0 | 0.000 | 0.000 |
| 40000 | 30000 | 89.2 | 100.0 | 0.000 | 0.000 | 65.2 | 100.0 | 0.000 | 0.000 |

Largest rise between two validations of s1: +20.2 points at step 8000 (policy step 6000), 61.6 -> 81.8; policy loss 0.567 -> 0.549, stage 0 loss 0.181 -> 0.223, V-sign 92.5 -> 99.9. Rows above: every 1000 steps plus every validation with a change >= 5 points.

### (2) Losing the V sign in the first 500 steps of stage 2: Gc s3, s4 against the others

Stage 2 logs have no V histogram (only V-sign accuracy and the mean stage 0 loss per 500 steps), and the step 500 weights are not kept (checkpoints are deleted when a run ends). Histograms below are recomputed by inference on the 2000 stage 0 10x7 validation positions from the weights that exist: end of stage 1 (= stage 2 step 0), the best point and the end of stage 2. Bins of the V pre-activation: <-10, -10..-3, -3..-1, -1..0, 0..1, 1..3, 3..10, >10.

| run | V-sign 0 / 500 / 1000 / 2000 | stage 0 loss 0-500 / 500-1000 / 1000-2000 | V hist step 0 | V hist best | V hist end |
|---|---|---|---|---|---|
| Gc s1 | 100.0 / 99.8 / 100.0 / 97.8 | 0.866 / 0.467 / 0.267 | 1012 0 0 0 0 0 1 987 | 995 17 0 0 0 0 14 974 | 1009 3 0 0 0 0 1 987 |
| Gc s2 | 100.0 / 98.5 / 99.5 / 99.8 | 1.002 / 0.316 / 0.204 | 1011 2 0 0 0 0 0 987 | 906 101 4 2 0 0 120 867 | 1006 7 0 0 0 0 3 984 |
| Gc s3 | 100.0 / 50.3 / 50.2 / 49.4 | 1.053 / 0.694 / 0.695 | 1012 1 0 0 0 0 0 987 | 0 11 63 409 1436 65 15 1 | 4 44 134 339 1299 120 54 6 |
| Gc s4 | 100.0 / 49.4 / 48.1 / 50.2 | 1.188 / 0.699 / 0.695 | 1011 2 0 0 0 0 0 987 | 31 50 17 12 1789 52 46 3 | 43 49 10 8 1768 51 68 3 |
| Gc s5 | 100.0 / 99.6 / 99.2 / 99.3 | 0.487 / 0.405 / 0.140 | 1012 0 0 1 0 0 0 987 | 19 892 79 16 14 39 941 0 | 1010 2 0 0 0 0 1 987 |
| C+h s1 | 100.0 / 100.0 / 93.0 / 99.5 | 0.002 / 0.000 / 0.000 | 1013 0 0 0 0 0 0 987 | 1002 11 0 0 0 0 9 978 | 1009 4 0 0 0 0 0 987 |
| C+h s2 | 100.0 / 100.0 / 99.5 / 85.7 | 0.002 / 0.001 / 0.000 | 1013 0 0 0 0 0 0 987 | 1013 0 0 0 0 0 1 986 | 1013 0 0 0 0 0 2 985 |
| C+h s3 | 100.0 / 97.2 / 98.4 / 100.0 | 0.003 / 0.001 / 0.001 | 1013 0 0 0 0 0 0 987 | 968 45 0 0 0 0 0 987 | 1005 8 0 0 0 0 0 987 |
| C+h s4 | 100.0 / 98.3 / 99.0 / 99.8 | 0.002 / 0.000 / 0.000 | 1013 0 0 0 0 0 0 987 | 977 36 2 0 3 15 271 696 | 1013 0 0 0 0 0 1 986 |
| C+h s5 | 100.0 / 51.7 / 100.0 / 99.8 | 0.002 / 0.001 / 0.001 | 1013 0 0 0 0 0 0 987 | 988 23 2 0 0 0 62 925 | 1013 0 0 0 0 0 0 987 |
| C+h14 s1 | 100.0 / 32.6 / 10.8 / 9.4 | 0.095 / 0.018 / 0.069 | 1009 3 0 0 1 0 1 986 | 511 367 73 40 29 86 587 307 | 754 212 32 13 15 35 374 565 |
| C+h14 s2 | 100.0 / 58.5 / 51.0 / 61.5 | 0.041 / 0.042 / 0.015 | 1006 7 0 0 0 0 0 987 | 602 216 85 34 21 36 107 899 | 685 263 43 4 5 11 26 963 |
| C+h14 s3 | 100.0 / 25.8 / 33.2 / 42.5 | 0.048 / 0.084 / 0.055 | 1008 5 0 0 0 0 6 981 | 419 436 67 39 29 39 257 714 | 996 16 1 0 0 0 28 959 |
| C+h14 s4 | 100.0 / 32.6 / 34.8 / 24.3 | 0.055 / 0.079 / 0.072 | 1006 7 0 0 0 0 0 987 | 647 342 17 3 6 29 647 309 | 954 59 0 0 0 0 40 947 |
| C+h14 s5 | 100.0 / 25.4 / 30.6 / 93.1 | 0.032 / 0.033 / 0.051 | 1008 4 0 1 0 0 2 985 | 589 404 17 4 2 18 563 403 | 986 27 0 0 0 0 19 968 |

Reading aid: ln 2 = 0.693 is the stage 0 loss of a V that says nothing; a loss near 1 with V-sign ~99% means few, very confident wrong answers; positions are 1013 negative / 987 positive.

## Data checks

Written 2026-11-13 at the user's request while the series runs; no training, read-only on the runs.

### 1. Overlap of stage 0 positions with the judge's 10x7 positions

Key: drawn lines only, minimum over the 4 grid symmetries. Cells: shared distinct positions.

| | judge train (4500 distinct) | judge validation (500 distinct) | judge probe (304 distinct) |
|---|---:|---:|---:|
| stage 0 train (aux) (20000 distinct) | 0 | 0 | 0 |
| stage 0 validation (2000 distinct) | 0 | 0 | 0 |

Key: full encoding (17 planes), minimum over the 4 grid symmetries. Cells: shared distinct positions.

| | judge train (4500 distinct) | judge validation (500 distinct) | judge probe (304 distinct) |
|---|---:|---:|---:|
| stage 0 train (aux) (20000 distinct) | 0 | 0 | 0 |
| stage 0 validation (2000 distinct) | 0 | 0 | 0 |

Undrawn lines per position (min / median / max): stage 0 train (aux) 73 / 80 / 88; stage 0 validation 73 / 80 / 86; judge train 75 / 80 / 83; judge validation 76 / 80 / 83; judge probe 75 / 79 / 82.

### 2. Is the 10x7 validation of series 3 and series 3' the same

- Config: both series read parity_config = configs/parity_hand.yaml / configs/parity_hand.yaml; split seed 3031 / 3031, val_share 0.1 / 0.1.
- Positions are read from runs/parity_hand/data_10x7.jsonl (sha256 59b9125b84246db5, last modified 2026-10-30 16:12, before both series); byte-identical to the git-tracked data/parity_hand_10x7.jsonl.gz (commit 2167ff1): True.
- 10x7 validation positions: series 3 500, hash c25d79bea6f35898; series 3' 500, hash c25d79bea6f35898: identical.
- Metric: the same validate() of scripts/graph_s3.py in both series (share of positions whose argmax legal move is one of the judge's correct moves). Recomputed now from saved weights:

| net | weights | logged 10x7 val | recomputed |
|---|---|---:|---:|
| series 3 Gc s1 (paused) | checkpoint, step 19500 (best so far 65.2) | 64.8 | 64.8 |
| series 3 Gc s2 (paused) | checkpoint, step 8500 (best so far 64.8) | 59.4 | 59.4 |
| series 3' Gc_s1 | best, step 22500 | 90.0 | 90.0 |

