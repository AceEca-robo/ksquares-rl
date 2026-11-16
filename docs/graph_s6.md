# Part 4, series 6: what decides the jump, the stage 1 weights or the stage 2 seed

Protocol: docs/plan_part4.md (series 6), configs/graph_s6.yaml. Stage 2 only, exactly the stage 2 of series 5 (40000 steps, every 4th a stage 0 step, switch at policy step 5500 = step 7333, best point by 10x7 judge validation after the switch). Stage 1 weights: the finished series 5 weights of seeds 8, 9, 6, 11 (runs/graph_s5/a_Gc_s<seed>.pt, copied). Stage 2 seeds 101-105: the seed of torch, numpy and python in stage 2 (batch order and everything else random there). Slots: 2 runs at once, GPU memory fraction 0.4 per run, nice -n 10 from the start (2026-11-16 10:17) to the stop at checkpoints at 21:19 (14 runs finished, w9 / 104 and w6 / 104 at step 13500); from the restart 6 runs at once, no GPU memory limit, no nice (user, 2026-11-16 21:15; protocol and seeds unchanged). The table marks the runs that ran wholly or partly at 6 slots. Steps below are total stage 2 steps.

## Table 1. Runs

| stage 1 weights | series 5 run of these weights | stage 2 seed | slots | best step | val 5x5 | best val 10x7 | >= 80% | V-sign 10x7 at start / best / end | sign kept (>= 90% at best) | largest rise after the switch | probe 5x5 | probe 10x7 [95%] |
|---:|---|---:|---|---:|---:|---:|---|---|---|---|---:|---:|
| s8 | jump (92.0) | 101 | 2 | 33000 | 47.0 | 91.0 | yes | 100.0 / 100.0 / 100.0 | yes | +17.4 at 8000 (62.0 -> 79.4) | - | - |
| s8 | jump (92.0) | 102 | 2 | 20000 | 43.2 | 90.8 | yes | 100.0 / 97.0 / 99.6 | yes | +16.6 at 7500 (53.6 -> 70.2) | - | - |
| s8 | jump (92.0) | 103 | 2 | 27500 | 57.4 | 95.4 | yes | 100.0 / 49.4 / 49.4 | no | +25.2 at 8000 (61.2 -> 86.4) | - | - |
| s8 | jump (92.0) | 104 | 2 | 20500 | 50.0 | 91.6 | yes | 100.0 / 99.9 / 100.0 | yes | +13.6 at 7500 (56.0 -> 69.6) | - | - |
| s8 | jump (92.0) | 105 | 6 | at checkpoint 13000 |||||||||
| s9 | jump (89.4) | 101 | 2 | 16500 | 44.0 | 66.0 | no | 100.0 / 100.0 / 99.9 | yes | +15.6 at 7500 (44.0 -> 59.6) | - | - |
| s9 | jump (89.4) | 102 | 2 | 24000 | 46.2 | 69.8 | no | 100.0 / 99.9 / 99.9 | yes | +4.0 at 7500 (55.4 -> 59.4) | - | - |
| s9 | jump (89.4) | 103 | 2 | 16500 | 43.2 | 68.8 | no | 100.0 / 99.8 / 100.0 | yes | +4.8 at 8000 (56.6 -> 61.4) | - | - |
| s9 | jump (89.4) | 104 | 2, then 6 from step 13500 | at checkpoint 23500 |||||||||
| s9 | jump (89.4) | 105 | 6 | at checkpoint 13000 |||||||||
| s6 | no jump (63.4) | 101 | 2 | 9000 | 52.4 | 67.0 | no | 100.0 / 49.2 / 100.0 | no | +5.2 at 7500 (54.8 -> 60.0) | - | - |
| s6 | no jump (63.4) | 102 | 2 | 14500 | 49.2 | 67.0 | no | 100.0 / 50.1 / 99.8 | no | +5.0 at 7500 (55.2 -> 60.2) | - | - |
| s6 | no jump (63.4) | 103 | 2 | 37000 | 44.0 | 83.4 | yes | 100.0 / 99.8 / 99.8 | yes | +7.0 at 8000 (57.6 -> 64.6) | - | - |
| s6 | no jump (63.4) | 104 | 2, then 6 from step 13500 | at checkpoint 23500 |||||||||
| s6 | no jump (63.4) | 105 | 6 | at checkpoint 13000 |||||||||
| s11 | no jump (64.6) | 101 | 2 | 17000 | 45.4 | 61.6 | no | 100.0 / 99.7 / 100.0 | yes | +2.8 at 22000 (57.4 -> 60.2) | - | - |
| s11 | no jump (64.6) | 102 | 2 | 10000 | 44.8 | 61.6 | no | 100.0 / 99.1 / 100.0 | yes | +2.6 at 8000 (58.6 -> 61.2) | - | - |
| s11 | no jump (64.6) | 103 | 2 | 8500 | 50.8 | 63.8 | no | 100.0 / 99.1 / 100.0 | yes | +4.8 at 23000 (57.6 -> 62.4) | - | - |
| s11 | no jump (64.6) | 104 | 6 | at checkpoint 13000 |||||||||
| s11 | no jump (64.6) | 105 | 6 | not started |||||||||

## Table 2. Jumps by stage 1 weights

A jump: best 10x7 judge validation after the switch >= 80%. The series 5 runs (stage 2 seed = stage 1 seed) are shown for reference and are not counted.

| stage 1 weights | series 5 run (not counted) | runs finished | jumps | stage 2 seeds with a jump | best val 10x7 by stage 2 seed 101-105 | V sign kept at best |
|---:|---|---:|---:|---|---|---:|
| s8 | jump, 92.0 at 25000 | 4 | 4 | 101 102 103 104 | 91.0 / 90.8 / 95.4 / 91.6 / - | 3 |
| s9 | jump, 89.4 at 36000 | 3 | 0 | - | 66.0 / 69.8 / 68.8 / - / - | 3 |
| s6 | no jump, 63.4 at 11000 | 3 | 1 | 103 | 67.0 / 67.0 / 83.4 / - / - | 1 |
| s11 | no jump, 64.6 at 10000 | 3 | 0 | - | 61.6 / 61.6 / 63.8 / - / - | 3 |

## Primary metric and reading

- weights s8 and s9 (jump in series 5): **4 of 7** runs jump;
- weights s6 and s11 (no jump in series 5): **1 of 6** runs jump.

Row of the protocol: series not finished (7 and 6 of 10 runs per group): no reading.

## Reproducibility control (not counted)

Stage 2 from the weights s8 with the stage 2 seed 8 of series 5, rerun here (runs/graph_s6/control) and compared with runs/graph_s5/s2_Gc_s8.

- log (81 entries: losses, judge validation of both sizes, stage 0 V-sign; wall time aside): equal entry by entry;
- best point: series 5 92.0 at step 25000, here 92.0 at step 25000;
- weights (best point, last step and the 18 kept steps): equal bit for bit;
- V pre-activations from the kept weights of steps 250, 500, 1000: equal.

Control passed: the rerun equals the series 5 run bit for bit.

## Table 3. V on the stage 0 10x7 validation at steps 250, 500, 1000 of stage 2

From the weights kept at those steps, as Table 2 of docs/graph_s5.md (same bins: <-10, -10..-3, -3..-1, -1..0, 0..1, 1..3, 3..10, >10).

| stage 1 weights | stage 2 seed | V-sign 250 / 500 / 1000 | stage 0 loss 0-500 / 500-1000 | V hist 250 | V hist 500 | V hist 1000 |
|---:|---:|---|---|---|---|---|
| s8 | 101 | 77.7 / 59.3 / 60.6 | 1.088 / 0.616 | 0 229 311 911 0 17 532 0 | 7 131 268 1417 0 11 166 0 | 0 290 148 439 795 156 172 0 |
| s8 | 102 | 95.7 / 99.7 / 99.5 | 0.782 / 0.391 | 4 349 647 97 140 382 381 0 | 419 578 9 4 3 22 949 16 | 473 523 13 0 0 8 809 174 |
| s8 | 103 | 60.1 / 99.2 / 99.4 | 0.840 / 0.307 | 0 162 231 248 901 211 230 17 | 113 696 176 25 16 353 610 11 | 549 435 15 4 5 14 506 472 |
| s8 | 104 | 71.8 / 77.8 / 99.5 | 0.986 / 0.388 | 1 204 508 840 117 128 202 0 | 1 537 25 891 1 1 506 38 | 14 963 24 1 1 12 760 225 |
| s9 | 101 | 89.8 / 99.7 / 65.3 | 0.767 / 0.325 | 2 905 57 170 38 36 787 5 | 152 771 77 7 2 212 779 0 | 325 346 167 218 77 474 110 283 |
| s9 | 102 | 100.0 / 99.2 / 100.0 | 0.593 / 0.451 | 322 682 6 2 2 4 618 364 | 44 936 12 7 4 26 782 189 | 266 739 6 1 0 3 902 83 |
| s9 | 103 | 92.1 / 97.7 / 13.7 | 0.938 / 0.510 | 15 603 297 162 364 221 306 32 | 20 918 23 47 32 35 833 92 | 5 289 440 172 165 355 573 1 |
| s6 | 101 | 63.3 / 50.6 / 46.8 | 1.152 / 0.704 | 0 0 326 84 1438 152 0 0 | 0 0 0 639 1361 0 0 0 | 0 0 0 517 1481 2 0 0 |
| s6 | 102 | 50.1 / 50.5 / 48.4 | 1.127 / 0.701 | 0 0 15 365 1620 0 0 0 | 0 0 0 1998 2 0 0 0 | 0 0 0 661 1339 0 0 0 |
| s6 | 103 | 47.9 / 50.2 / 50.6 | 1.090 / 0.698 | 0 0 0 164 1836 0 0 0 | 0 0 0 96 1904 0 0 0 | 0 0 1 1999 0 0 0 0 |
| s11 | 101 | 95.1 / 98.9 / 77.5 | 0.366 / 0.151 | 422 512 61 30 33 95 427 420 | 215 760 5 10 21 20 638 331 | 4 593 264 151 151 264 500 73 |
| s11 | 102 | 76.1 / 97.5 / 97.8 | 0.768 / 0.376 | 0 420 371 284 308 197 420 0 | 0 950 14 3 73 81 849 30 | 0 948 21 3 65 15 948 0 |
| s11 | 103 | 99.4 / 97.4 / 99.5 | 0.533 / 0.205 | 407 567 27 12 6 32 718 231 | 374 560 43 16 9 26 367 605 | 399 587 17 3 3 30 619 342 |

## Kept (runs/graph_s6, not in git)

- w<stage 1 seed>/s2_Gc_s<stage 2 seed>.json (log), .pt (best point), _last.pt, _vhist.json, snap/ (weights at the steps of series 5: 250, 500, 1000 and every validation 7000-14000), probe.json; control/ the same under the series 5 names.
- stage 1 weights used, sha256 of the series 5 files: s6 dc31da322acd6be2; s8 e4456967cc5f27a7; s9 899f54d90bec3b1f; s11 c826d1dcd44e241b.
