# Part 4, series 2: transfer to 10x7

Protocol: docs/plan_part4.md, configs/graph_s2.yaml. Recipe B + V-sign, 20000 steps, deterministic, validation every 250 steps. Click: first validation >= 95.0%; hold: every later 10x7 validation >= 90.0%.

## 2a Gc: 10x7 click 0/10, click and hold 0/10

| seed | 5x5 click | switch | 10x7 click | click - switch | hold | best 10x7 | last 10x7 | last 5x5 |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 1000 | - | - | - | - | 68.5 | 60.3 | 100.0 |
| 2 | 750 | - | - | - | - | 76.0 | 75.1 | 100.0 |
| 3 | 1500 | - | - | - | - | 50.6 | 22.1 | 100.0 |
| 4 | 1750 | - | - | - | - | 57.2 | 42.2 | 100.0 |
| 5 | 500 | - | - | - | - | 51.5 | 51.4 | 100.0 |
| 6 | 1000 | - | - | - | - | 50.6 | 39.2 | 100.0 |
| 7 | 1000 | - | - | - | - | 75.0 | 68.4 | 100.0 |
| 8 | 1250 | - | - | - | - | 60.7 | 49.1 | 100.0 |
| 9 | 1000 | - | - | - | - | 86.1 | 86.1 | 100.0 |
| 10 | 1250 | - | - | - | - | 50.6 | 32.2 | 100.0 |

## 2b Gc: 10x7 click 7/10, click and hold 7/10

| seed | 5x5 click | switch | 10x7 click | click - switch | hold | best 10x7 | last 10x7 | last 5x5 |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 1000 | 1000 | 1500 | 500 | True | 100.0 | 100.0 | 27.4 |
| 2 | 750 | 750 | 1000 | 250 | True | 100.0 | 100.0 | 28.8 |
| 3 | 1500 | 1500 | 1750 | 250 | True | 100.0 | 100.0 | 0.4 |
| 4 | 1750 | 1750 | 4500 | 2750 | True | 100.0 | 100.0 | 37.1 |
| 5 | 500 | 500 | - | - | - | 51.0 | 49.1 | 50.4 |
| 6 | 1000 | 1000 | - | - | - | 60.8 | 49.4 | 44.9 |
| 7 | 1000 | 1000 | 1500 | 500 | True | 100.0 | 100.0 | 91.0 |
| 8 | 1250 | 1250 | 1500 | 250 | True | 100.0 | 100.0 | 27.2 |
| 9 | 1000 | 1000 | 1250 | 250 | True | 100.0 | 100.0 | 10.3 |
| 10 | 1250 | 1250 | - | - | - | 50.6 | 49.4 | 51.0 |

## 2c C: 10x7 click 0/5, click and hold 0/5

| seed | 5x5 click | switch | 10x7 click | click - switch | hold | best 10x7 | last 10x7 | last 5x5 |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | - | - | - | - | - | 51.5 | 51.1 | 50.7 |
| 2 | - | - | - | - | - | 51.8 | 49.2 | 49.9 |
| 3 | - | - | - | - | - | 51.1 | 50.6 | 50.6 |
| 4 | - | - | - | - | - | 51.4 | 48.5 | 52.3 |
| 5 | - | - | - | - | - | 51.2 | 50.0 | 51.1 |

## 2c Ch: 10x7 click 5/5, click and hold 5/5

| seed | 5x5 click | switch | 10x7 click | click - switch | hold | best 10x7 | last 10x7 | last 5x5 |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 250 | 250 | 250 | 0 | True | 100.0 | 100.0 | 100.0 |
| 2 | 250 | 250 | 250 | 0 | True | 100.0 | 100.0 | 100.0 |
| 3 | 250 | 250 | 250 | 0 | True | 100.0 | 100.0 | 100.0 |
| 4 | 250 | 250 | 250 | 0 | True | 100.0 | 100.0 | 100.0 |
| 5 | 250 | 250 | 250 | 0 | True | 100.0 | 100.0 | 100.0 |

## Fisher and criterion

2b 7/10 vs 2a 0/10: one-sided p = 0.00155, two-sided p = 0.0031.

C 0/5 vs 2b, two-sided p = 0.0256: "the convolutions click the same" = False. C+h 5/5 is not a test (its plane 15 is the target).

By clicks: 2b row = series 2'. By click + hold: series 2'.
Next: series 2'.

## 2d: 2b runs without a 10x7 click

Medians over the last quarter of the logs. Histogram: raw V on 10x7 validation, bins of 1 from -10 to 10.

| seed | class | train loss | readout grad | sat | V at the switch | V at the end |
|---:|---|---:|---:|---:|---|---|
| 5 | does not find the rule | 0.692 | 9.51e-02 | 0.00 | 954 13 9 12 10 7 12 18 13 7 10 11 11 12 8 9 7 5 8 864 | 0 0 0 0 0 1 0 0 2 640 1357 0 0 0 0 0 0 0 0 0 |
| 6 | does not find the rule | 0.693 | 3.52e-02 | 0.00 | 979 7 3 8 5 8 8 8 2 10 15 18 22 17 18 25 25 27 24 771 | 0 0 0 0 0 0 0 0 0 0 2000 0 0 0 0 0 0 0 0 0 |
| 10 | does not find the rule | 0.693 | 3.43e-02 | 0.00 | 891 14 43 28 31 23 21 26 19 24 25 30 19 22 21 16 25 21 26 675 | 0 0 0 0 0 0 0 0 0 0 2000 0 0 0 0 0 0 0 0 0 |

## Reading (written by hand after the report; rerunning the report drops this section)

- By the protocol: 2b 7/10, click and hold 7/10: series 2'. 2a 0/10, so no transfer without 10x7 examples; 2b above 2a, Fisher one-sided p = 0.0016.
- C says nothing about 10x7. It never reached 95% on 5x5 (best 51-53%, final loss 0.69), so by the event rule it never switched: 5x5 only for all 20000 steps. "C 0/5" is a failure on 5x5. The question "do the convolutions click on 10x7 at the same school" stays open.
- C+h 5/5 at step 250 is the leak of plane 15, as stated in the protocol.
- 2d: all three runs classify as "does not find the rule" by the fixed rule (gradient alive, loss 0.69, not frozen). The histograms add detail: V is spread to the bin edges at the switch and ends in one or two bins around 0, a constant output.

Post hoc, read after the runs, not part of the protocol. Train loss over the first 250 steps after the switch:

| seed | 10x7 val at switch | sat 10x7 at switch | first loss | then | 10x7 click |
|---:|---:|---:|---:|---|---:|
| 1 | 61.0 | 0.90 | 0.50 | learns | 1500 |
| 2 | 62.2 | 0.69 | 0.29 | learns | 1000 |
| 3 | 48.4 | 0.75 | 0.45 | learns | 1750 |
| 7 | 43.1 | 0.92 | 0.53 | learns | 1500 |
| 8 | 60.7 | 0.86 | 0.30 | learns | 1500 |
| 9 | 33.4 | 0.66 | 0.40 | learns | 1250 |
| 4 | 54.7 | 0.93 | 0.97 | 0.69 plateau, recovers | 4500 |
| 5 | 44.8 | 0.94 | 1.07 | 0.69 to the end | - |
| 6 | 37.0 | 0.93 | 1.03 | 0.69 to the end | - |
| 10 | 33.1 | 0.88 | 1.01 | 0.69 to the end | - |

- The split is clean: first loss <= 0.53 learns in 250-500 steps; first loss >= 0.97 collapses to V = constant at 0.69, and only 1 of 4 recovers. Hypothesis for series 2': overconfident wrong outputs at the switch cause a loss spike that pushes the net to the constant solution. 4 vs 6 runs, selected after seeing the data: to be tested, not a result.
- 10x7 accuracy at the switch does not separate the groups (33-62% in both).
- After the switch 5x5 is forgotten in most runs (last 5x5: 0.4-37% in 6 of 7 clicked runs, 91% in s7, down from 100%). Values far below 50% mean the sign of the rule is size-dependent. A net clicked on 10x7 is not a net for both sizes.
- Seeds 1-5 were used to choose V-sign in series 1; 2b clicks there 4/5, on 6-10 3/5.
