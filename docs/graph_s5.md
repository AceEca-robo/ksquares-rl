# Part 4, series 5: how often does Gc make the move-choice jump

Protocol: docs/plan_part4.md (series 5), configs/graph_s5.yaml. Gc only, seeds 6-25; stage 1 and stage 2 are series 3' unchanged (40000 stage 2 steps, every 4th a stage 0 step, switch at policy step 5500 = step 7333, best point by 10x7 judge validation after the switch). Slots: 3 runs at once, GPU memory fraction 0.4, nice -n 10, no work below 12 GB free RAM (user, 2026-11-14). Steps below are total stage 2 steps.

## Table 1. Runs

| seed | s1 5x5 click | s1 10x7 click | s1 last 10x7 | best step | val 5x5 | best val 10x7 | >= 80% | V-sign 10x7 at start / best / end | sign kept (>= 90% at best) | largest rise after the switch | probe 5x5 | probe 10x7 [95%] |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|
| 6 | 1000 | 1750 | 100.0 | 11000 | 46.6 | 63.4 | no | 100.0 / 51.2 / 99.9 | no | +2.6 at 11000 (60.8 -> 63.4) | - | - |
| 7 | 1000 | 1750 | 100.0 | 23000 | 52.8 | 88.8 | yes | 100.0 / 50.1 / 99.6 | no | +16.0 at 7500 (47.6 -> 63.6) | - | - |
| 8 | 1250 | 1500 | 100.0 | 25000 | 48.4 | 92.0 | yes | 100.0 / 49.4 / 49.3 | no | +26.6 at 7500 (53.6 -> 80.2) | - | - |
| 9 | 1000 | 1750 | 100.0 | 36000 | 52.6 | 89.4 | yes | 100.0 / 100.0 / 100.0 | yes | +7.0 at 7500 (53.0 -> 60.0) | - | - |
| 10 | 1250 | 1500 | 100.0 | 27500 | 46.6 | 88.8 | yes | 100.0 / 100.0 / 100.0 | yes | +10.2 at 9000 (65.6 -> 75.8) | - | - |
| 11 | 2000 | 2250 | 100.0 | 10000 | 48.8 | 64.6 | no | 100.0 / 99.8 / 100.0 | yes | +5.4 at 9500 (58.2 -> 63.6) | - | - |
| 12 | 750 | 1250 | 100.0 | 29500 | 49.6 | 86.2 | yes | 100.0 / 100.0 / 100.0 | yes | +7.8 at 8000 (64.4 -> 72.2) | - | - |
| 13 | 750 | - | 49.4 | 10500 | 45.2 | 64.6 | no | 49.4 / 49.6 / 50.0 | no | +4.0 at 7500 (57.8 -> 61.8) | - | - |
| 14 | 500 | 750 | 100.0 | 23500 | 47.4 | 64.6 | no | 100.0 / 59.8 / 99.9 | no | +7.2 at 7500 (53.4 -> 60.6) | - | - |
| 15 | 1000 | 1250 | 100.0 | 12500 | 44.8 | 63.6 | no | 100.0 / 99.6 / 100.0 | yes | +3.8 at 22500 (59.6 -> 63.4) | - | - |
| 16 | 750 | 1000 | 100.0 | 19000 | 51.4 | 68.4 | no | 100.0 / 49.3 / 50.3 | no | +9.0 at 7500 (53.6 -> 62.6) | - | - |
| 17 | 1750 | 2000 | 100.0 | 20500 | 46.0 | 65.0 | no | 100.0 / 100.0 / 100.0 | yes | +3.2 at 8000 (57.6 -> 60.8) | - | - |
| 18 | stage 2 running ||||||||||||
| 19 | stage 2 running ||||||||||||
| 20 | stage 1 running ||||||||||||
| 21 | not started ||||||||||||
| 22 | not started ||||||||||||
| 23 | not started ||||||||||||
| 24 | not started ||||||||||||
| 25 | not started ||||||||||||

## Primary metric and verdict

Runs with best 10x7 validation >= 80% after the switch: **5 of 12** finished (20 planned); seeds 7 8 9 10 12.

| | V sign kept | V sign lost |
|---|---:|---:|
| best val 10x7 >= 80% | 3 | 2 |
| best val 10x7 < 80% | 3 | 4 |

Row of the protocol: series not finished (12 of 20 runs): no verdict.

## Table 2. V on the stage 0 10x7 validation at steps 250, 500, 1000 of stage 2

From the weights kept at those steps (inference after the run; 2000 positions, 1013 negative / 987 positive). Bins of the V pre-activation: <-10, -10..-3, -3..-1, -1..0, 0..1, 1..3, 3..10, >10. Stage 0 loss: mean over the stage 0 steps of the 500 steps before (ln 2 = 0.693 is a V that says nothing).

| seed | V-sign 250 / 500 / 1000 | stage 0 loss 0-500 / 500-1000 | V hist 250 | V hist 500 | V hist 1000 |
|---:|---|---|---|---|---|
| 6 | 49.4 / 49.4 / 50.5 | 1.146 / 0.704 | 0 0 1 1 1995 3 0 0 | 0 0 0 13 1970 15 2 0 | 2 1 8 1981 8 0 0 0 |
| 7 | 50.5 / 49.8 / 48.9 | 1.252 / 0.698 | 0 0 1638 360 0 0 1 1 | 0 0 0 1760 239 1 0 0 | 0 0 0 1723 277 0 0 0 |
| 8 | 77.9 / 62.9 / 99.7 | 0.959 / 0.421 | 0 211 218 1026 1 76 468 0 | 0 407 328 851 115 102 197 0 | 175 791 39 2 9 196 594 194 |
| 9 | 99.0 / 99.4 / 60.8 | 0.694 / 0.269 | 215 772 21 5 1 17 573 396 | 293 716 2 6 4 2 469 508 | 31 586 434 185 179 262 294 29 |
| 10 | 99.4 / 99.9 / 99.8 | 0.825 / 0.456 | 464 459 74 9 5 9 403 577 | 328 685 0 0 1 0 653 333 | 1 1002 5 1 3 7 605 376 |
| 11 | 98.6 / 98.6 / 97.3 | 0.524 / 0.195 | 333 636 13 3 66 2 740 207 | 46 924 12 2 14 57 508 437 | 1 914 42 12 57 46 448 480 |
| 12 | 74.6 / 58.7 / 98.3 | 0.697 / 0.360 | 4 156 430 238 192 510 470 0 | 0 254 671 539 282 193 61 0 | 0 918 40 31 29 54 928 0 |
| 13 | 49.4 / 49.5 / 48.4 | 0.695 / 0.695 | 0 0 0 0 2000 0 0 0 | 0 0 0 200 1800 0 0 0 | 0 0 0 576 1419 5 0 0 |
| 14 | 50.5 / 50.0 / 50.0 | 1.178 / 0.697 | 1 2 7 284 1706 0 0 0 | 0 0 0 82 1918 0 0 0 | 0 2 6 1282 710 0 0 0 |
| 15 | 99.7 / 88.1 / 99.8 | 0.740 / 0.400 | 790 215 3 4 3 63 358 564 | 3 688 258 142 120 147 510 132 | 320 681 6 3 5 8 432 545 |
| 16 | 49.4 / 50.7 / 49.6 | 1.150 / 0.696 | 1 0 0 1890 106 3 0 0 | 0 0 0 47 1953 0 0 0 | 0 0 0 707 1293 0 0 0 |
| 17 | 95.2 / 99.1 / 98.7 | 1.066 / 0.241 | 310 576 115 40 27 87 649 196 | 368 558 81 15 33 178 747 20 | 406 547 42 15 9 51 476 454 |

## Kept for later analysis (runs/graph_s5, not in git)

- snap/s2_Gc_s<seed>_step<step>.pt: weights at steps 250, 500, 1000 and at every validation step 7000-14000 (18 per run; 222 files now).
- s2_Gc_s<seed>_vhist.json: V pre-activations of all stage 0 validation positions (5x5 and 10x7) at steps 250, 500, 1000.
- s2_Gc_s<seed>.pt (best point), s2_Gc_s<seed>_last.pt, a_Gc_s<seed>.pt (end of stage 1), logs in the .json files.

## For reference: Gc seeds 1-5 of series 3' (not counted here)

| seed | best step | best val 10x7 | V-sign 10x7 at best |
|---:|---:|---:|---:|
| 1 | 22500 | 90.0 | 100.0 |
| 2 | 15000 | 66.0 | 100.0 |
| 3 | 30500 | 65.2 | 50.3 |
| 4 | 35500 | 66.2 | 53.6 |
| 5 | 8500 | 63.8 | 99.6 |
