# Part 4, series 3'

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

