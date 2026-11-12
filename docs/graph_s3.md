# Part 4, series 3: parity probe on real positions

Status: **paused** (user, 2026-11-12, 10:52) with the series stop command (runs/graph_s3/STOP). Reason:
the stage 2 diagnostic shows every net that came out of stage 1 with the chain count (C+h, C+h14, Gc)
losing it in stage 2 (stage 0 V-sign accuracy on 10x7 100% -> ~50% within the first 500 steps).
Finished runs and checkpoints are kept untouched; probes were not run. No conclusions are drawn from
the partial data. Follow-up: series 3' (stage 0 kept as an auxiliary task in stage 2), draft in docs/plan_part4.md.

State at the pause: stage 1 finished for 14 of 25 runs (C s1-s3, C+h s1-s2, C+h14 s1-s2, Ga s1-s2, Gc s1-s5
copied from series 2' (a)); stage 2 finished for 5 (C s1, C s2, C+h s1, C+h14 s1, Ga s1); paused at
checkpoints: C s3 16500, C+h s2 17000, C+h14 s2 16500, Ga s2 11000, Gc s1 19500, Gc s2 8500 (stage 2).
Not started: C s4-s5, C+h s3-s5, C+h14 s3-s5, Ga s3-s5, stage 2 of Gc s3-s5.
