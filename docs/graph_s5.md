# Part 4, series 5: how often does Gc make the move-choice jump

Protocol: docs/plan_part4.md (series 5), configs/graph_s5.yaml. Gc only, seeds 6-25; stage 1 and stage 2 are series 3' unchanged (40000 stage 2 steps, every 4th a stage 0 step, switch at policy step 5500 = step 7333, best point by 10x7 judge validation after the switch). Slots: 3 runs at once, GPU memory fraction 0.4, nice -n 10, no work below 12 GB free RAM (user, 2026-11-14). Steps below are total stage 2 steps.

## Table 1. Runs

| seed | stage 1 | s1 5x5 click | s1 10x7 click | s1 last 10x7 | best step | val 5x5 | best val 10x7 | >= 80% | V-sign 10x7 at start / best / end | sign kept (>= 90% at best) | largest rise after the switch | probe 5x5 | probe 10x7 [95%] |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|
| 6 | copy | 1000 | 1750 | 100.0 | 11000 | 46.6 | 63.4 | no | 100.0 / 51.2 / 99.9 | no | +2.6 at 11000 (60.8 -> 63.4) | 44.6 | 71.7 [66-76] |
| 7 | copy | 1000 | 1750 | 100.0 | 23000 | 52.8 | 88.8 | yes | 100.0 / 50.1 / 99.6 | no | +16.0 at 7500 (47.6 -> 63.6) | 55.6 | 95.4 [92-97] |
| 8 | copy | 1250 | 1500 | 100.0 | 25000 | 48.4 | 92.0 | yes | 100.0 / 49.4 / 49.3 | no | +26.6 at 7500 (53.6 -> 80.2) | 49.7 | 95.1 [92-97] |
| 9 | copy | 1000 | 1750 | 100.0 | 36000 | 52.6 | 89.4 | yes | 100.0 / 100.0 / 100.0 | yes | +7.0 at 7500 (53.0 -> 60.0) | 52.3 | 93.1 [90-95] |
| 10 | copy | 1250 | 1500 | 100.0 | 27500 | 46.6 | 88.8 | yes | 100.0 / 100.0 / 100.0 | yes | +10.2 at 9000 (65.6 -> 75.8) | 48.3 | 92.1 [89-95] |
| 11 | fresh | 2000 | 2250 | 100.0 | 10000 | 48.8 | 64.6 | no | 100.0 / 99.8 / 100.0 | yes | +5.4 at 9500 (58.2 -> 63.6) | 47.2 | 71.4 [66-76] |
| 12 | fresh | 750 | 1250 | 100.0 | 29500 | 49.6 | 86.2 | yes | 100.0 / 100.0 / 100.0 | yes | +7.8 at 8000 (64.4 -> 72.2) | 50.8 | 90.8 [87-94] |
| 13 | fresh | 750 | - | 49.4 | 10500 | 45.2 | 64.6 | no | 49.4 / 49.6 / 50.0 | no | +4.0 at 7500 (57.8 -> 61.8) | 47.9 | 76.3 [71-81] |
| 14 | fresh | 500 | 750 | 100.0 | 23500 | 47.4 | 64.6 | no | 100.0 / 59.8 / 99.9 | no | +7.2 at 7500 (53.4 -> 60.6) | 48.3 | 72.0 [67-77] |
| 15 | fresh | 1000 | 1250 | 100.0 | 12500 | 44.8 | 63.6 | no | 100.0 / 99.6 / 100.0 | yes | +3.8 at 22500 (59.6 -> 63.4) | 48.2 | 72.4 [67-77] |
| 16 | fresh | 750 | 1000 | 100.0 | 19000 | 51.4 | 68.4 | no | 100.0 / 49.3 / 50.3 | no | +9.0 at 7500 (53.6 -> 62.6) | 48.9 | 73.0 [68-78] |
| 17 | fresh | 1750 | 2000 | 100.0 | 20500 | 46.0 | 65.0 | no | 100.0 / 100.0 / 100.0 | yes | +3.2 at 8000 (57.6 -> 60.8) | 47.5 | 76.0 [71-80] |
| 18 | fresh | 1500 | 1750 | 100.0 | 22000 | 45.6 | 63.8 | no | 100.0 / 99.8 / 100.0 | yes | +9.2 at 7500 (53.2 -> 62.4) | 48.4 | 69.7 [64-75] |
| 19 | fresh | 1500 | 2000 | 100.0 | 9000 | 49.0 | 64.2 | no | 100.0 / 98.1 / 100.0 | yes | +5.8 at 7500 (55.2 -> 61.0) | 48.1 | 74.3 [69-79] |
| 20 | fresh | 1500 | 2000 | 100.0 | 20500 | 46.2 | 64.2 | no | 100.0 / 100.0 / 100.0 | yes | +3.8 at 7500 (57.0 -> 60.8) | 46.1 | 70.7 [65-76] |
| 21 | fresh | 1000 | 1500 | 100.0 | 19500 | 47.6 | 66.8 | no | 100.0 / 100.0 / 100.0 | yes | +4.4 at 15000 (60.2 -> 64.6) | 52.5 | 71.7 [66-76] |
| 22 | fresh | 1000 | 1500 | 100.0 | 21500 | 45.6 | 70.2 | no | 100.0 / 100.0 / 100.0 | yes | +5.0 at 16000 (64.2 -> 69.2) | 49.4 | 74.7 [69-79] |
| 23 | fresh | 1000 | 1500 | 100.0 | 9000 | 52.6 | 63.0 | no | 100.0 / 99.9 / 100.0 | yes | +8.0 at 7500 (52.4 -> 60.4) | 50.3 | 74.7 [69-79] |
| 24 | fresh | 500 | 1500 | 100.0 | 9500 | 47.4 | 62.4 | no | 100.0 / 99.2 / 100.0 | yes | +4.0 at 8000 (57.6 -> 61.6) | 49.4 | 73.7 [68-78] |
| 25 | fresh | 1500 | 2000 | 100.0 | 22500 | 45.6 | 64.2 | no | 100.0 / 100.0 / 100.0 | yes | +6.0 at 7500 (55.2 -> 61.2) | 47.0 | 71.4 [66-76] |

## Primary metric and verdict

Runs with best 10x7 validation >= 80% after the switch: **5 of 20** finished (20 planned); seeds 7 8 9 10 12.

| | V sign kept | V sign lost |
|---|---:|---:|
| best val 10x7 >= 80% | 3 | 2 |
| best val 10x7 < 80% | 11 | 4 |

Row of the protocol: >= 5 of 20: the jump is reproducible.

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
| 18 | 88.2 / 97.0 / 99.0 | 1.088 / 0.396 | 0 1 235 657 926 181 0 0 | 0 670 274 105 68 390 493 0 | 8 941 40 15 12 72 912 0 |
| 19 | 93.3 / 99.2 / 99.2 | 1.516 / 0.203 | 0 623 264 63 45 100 836 69 | 8 932 50 16 5 40 840 109 | 130 852 17 4 11 11 889 86 |
| 20 | 99.9 / 95.1 / 99.6 | 0.504 / 0.237 | 636 366 6 4 0 7 170 811 | 0 662 188 149 53 196 752 0 | 156 835 12 4 8 20 820 145 |
| 21 | 92.5 / 99.5 / 94.8 | 0.719 / 0.307 | 0 558 66 243 188 587 357 1 | 187 787 34 1 2 140 848 1 | 2 682 226 80 101 237 646 26 |
| 22 | 52.6 / 47.6 / 56.9 | 1.262 / 0.694 | 0 1 8 1251 740 0 0 0 | 0 0 0 370 1630 0 0 0 | 0 0 0 1290 710 0 0 0 |
| 23 | 98.9 / 98.9 / 99.5 | 0.523 / 0.240 | 0 898 88 26 12 102 612 262 | 1 912 63 39 9 3 973 0 | 361 636 0 16 1 8 571 407 |
| 24 | 98.8 / 99.9 / 95.0 | 1.141 / 0.579 | 2 789 180 37 21 48 923 0 | 17 989 0 4 4 12 709 265 | 149 781 68 54 65 104 474 305 |
| 25 | 99.7 / 99.5 / 99.9 | 0.475 / 0.528 | 989 17 0 1 2 3 96 892 | 518 474 13 4 4 7 619 361 | 149 842 19 3 3 50 589 345 |

## Kept for later analysis (runs/graph_s5, not in git)

- snap/s2_Gc_s<seed>_step<step>.pt: weights at steps 250, 500, 1000 and at every validation step 7000-14000 (18 per run; 360 files now).
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

## Check: stage 1 copied from series 2' (a) or trained here (user, 2026-11-15; not part of the verdict)

Seeds 6-10 use the stage 1 weights of series 2' (a) (same stage 1 config; copied, as series 3' did for seeds 1-5); seeds 11-25 trained stage 1 in this series.

| group | stage 1 | runs finished | best val 10x7 >= 80% | seeds |
|---|---|---:|---:|---|
| series 3' seeds 1-5 (not counted in series 5) | copy | 5 | 1 | 1 |
| series 5 seeds 6-10 | copy | 5 | 4 | 7 8 9 10 |
| series 5 seeds 11-25 | fresh | 15 | 1 | 12 |

Copy 5 of 10, fresh 1 of 15; Fisher exact, two-sided: p = 0.023 (descriptive: the split was not planned before the data).

Retrain of stage 1, seed 8 (runs/graph_s5_s1check; scripts/graph_s5.py stage1; current code, configs/graph_s5.yaml; torch 2.14.1+cu130, CUDA 13.0, NVIDIA GeForce RTX 4060 Laptop GPU), against the copy in /home/ace/projects/ksquares-rl/runs/graph_s2p:

- weights: equal bit for bit (388 tensors, 0 differ, largest absolute difference 0); .pt file bytes equal;
- log (81 entries: loss, gradient norms, lr, V-sign accuracy, saturation and V histograms of both sizes at every 250 steps; wall time aside): equal entry by entry;
- switch step 1250 / 1250 (copy / retrain); click and hold summary equal.

### Notes on the stage 1 check
- Result of the retrain: the copied stage 1 of seed 8 is exactly what the current code and the series 5 config produce. Series 2' (a) ran with 6 slots, GPU memory fraction 1.0 and scripts/graph_s2.py before a8cfb4b (series 2'' option keep_5x5, off here); torch, the net code and configs/graph_gc.yaml are unchanged since. None of these changes the result. No stage 2 rerun of seed 8 from the retrained weights: they are the same weights. Only seed 8 was retrained (seeds 1-7, 9, 10 were not).
- One more difference between the groups, checked the same evening (not asked for): for a copied seed stage 2 starts in a new process, for a fresh seed it runs in the process that has just trained stage 1. Stage 2 of the fresh seeds 11 (no jump) and 12 (jump) was rerun in a new process from their stored stage 1 weights to step 1500: all 4 log entries and the kept weights at steps 250, 500, 1000 equal the series runs bit for bit. So stage 2 does not depend on what the process did before. Checked to step 1500 only, not to the end.
- What this leaves: no difference in procedure was found between the groups; the runs differ by the seed alone (net init and batch order of both stages). The group difference is then chance in the seeds, as far as these checks go. It is still a large one: inside series 5, copy 4 of 5 against fresh 1 of 15 (Fisher exact, two-sided, p = 0.005; with series 3' seeds 1-5, 5 of 10 against 1 of 15, p = 0.023). Both p-values are after the fact: the split was noticed in the data and other splits could have been looked at.
- For the reading of the verdict: the protocol count is 5 of 20 and is not changed by this section. 4 of the 5 are seeds 7-10.
