# Part 4, series 2': what gets in the way of transfer

Protocol: docs/plan_part4.md (series 2'), configs/graph_s2p.yaml. Recipe of series 2, 20000 steps, validation every 250. Click: first validation >= 95.0%; hold 10x7: every validation after the click >= 90.0%; hold 5x5: every validation from the end of the mix (from the switch when there is no mix) >= 90.0%; both: the last validation >= 90.0% on both sizes.

Restart (2026-11-11): the series was stopped on 2026-11-10 with g, b and a s1 finished and a s2-s7 in progress (no mid-run checkpoints then). Finished runs were kept; a s2-s7 were restarted from scratch with the same seeds, a s8-s10, cC and cCh started then. Runs are deterministic, so the restart does not change results; from the restart on, runs checkpoint every 500 steps (resume checked bit-exact). The b predictions for s16-s20 were checked against the saved logs: switch and first loss equal the committed values.

## Table 1. (g) What the 2b nets learned instead of parity on 5x5

Seeds with last 5x5 < 40.0% in 2b: [1, 2, 3, 4, 8, 9]; reference s7. Error % on the 2000 5x5 validation positions, by class (n = positions in the class).

| run | reproduced bit-exact | 5x5 acc |
|---|---|---:|
| s1 | True | 27.4 |
| s2 | True | 28.8 |
| s3 | True | 0.4 |
| s4 | True | 37.1 |
| s8 | True | 27.2 |
| s9 | True | 10.3 |
| s7 | True | 91.0 |

Error % by long:

| run | 0 (n=0) | 1 (n=50) | 2 (n=308) | 3+ (n=1642) |
|---|---:|---:|---:|---:|
| s1 | - | 0.0 | 48.7 | 79.4 |
| s2 | - | 58.0 | 100.0 | 66.2 |
| s3 | - | 88.0 | 99.7 | 100.0 |
| s4 | - | 46.0 | 100.0 | 56.5 |
| s8 | - | 98.0 | 100.0 | 66.9 |
| s9 | - | 2.0 | 48.7 | 100.0 |
| s7 | - | 60.0 | 49.0 | 0.0 |

Error % by long_len:

| run | 0 (n=0) | 3-5 (n=13) | 6-8 (n=56) | 9-11 (n=149) | 12+ (n=1782) |
|---|---:|---:|---:|---:|---:|
| s1 | - | 0.0 | 35.7 | 67.1 | 74.8 |
| s2 | - | 76.9 | 92.9 | 98.0 | 68.2 |
| s3 | - | 76.9 | 96.4 | 99.3 | 99.9 |
| s4 | - | 92.3 | 91.1 | 85.2 | 59.9 |
| s8 | - | 100.0 | 100.0 | 100.0 | 69.5 |
| s9 | - | 7.7 | 42.9 | 74.5 | 93.0 |
| s7 | - | 69.2 | 50.0 | 27.5 | 5.8 |

Error % by loops:

| run | 0 (n=1615) | 1 (n=339) | 2+ (n=46) |
|---|---:|---:|---:|
| s1 | 76.7 | 56.9 | 47.8 |
| s2 | 67.5 | 86.1 | 91.3 |
| s3 | 99.7 | 99.7 | 97.8 |
| s4 | 62.0 | 65.8 | 71.7 |
| s8 | 68.2 | 91.2 | 100.0 |
| s9 | 92.3 | 81.4 | 56.5 |
| s7 | 6.4 | 18.9 | 30.4 |

Agreement % of the net's sign with candidate rules: +1 iff (count % 2) == player; 'long' is the true rule. 'chains2' counts chains of 2+, 'short' chains of 1-2. Also the player alone (+1 iff player 0). Below 50 = the inverted rule.

| run | long | chains2 | chains | short | loops | long+loops | chains+loops | player |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s1 | 27.4 | 49.9 | 49.2 | 49.9 | 45.9 | 29.7 | 49.8 | 43.8 |
| s2 | 28.8 | 49.3 | 47.9 | 48.9 | 55.4 | 41.1 | 48.2 | 58.6 |
| s3 | 0.4 | 47.8 | 49.2 | 50.0 | 50.2 | 17.3 | 49.4 | 50.8 |
| s4 | 37.1 | 49.4 | 51.6 | 52.5 | 51.4 | 42.5 | 52.8 | 47.4 |
| s8 | 27.2 | 49.9 | 47.9 | 48.8 | 56.4 | 41.2 | 48.5 | 58.9 |
| s9 | 10.3 | 47.5 | 48.9 | 49.8 | 53.6 | 21.0 | 49.0 | 56.5 |
| s7 | 91.0 | 51.4 | 50.8 | 50.0 | 47.9 | 80.4 | 50.3 | 42.9 |

## Table 2. (a) Soft switch: mix of sizes for 1000 steps after the event, then 10x7 only

Gc, 10/10 runs. 10x7 click 10/10, hold 10x7 6/10, hold 5x5 2/10, both at the end 2/10.

| seed | 5x5 click | switch | 10x7 click | click - switch | hold 10x7 | hold 5x5 | both | best 10x7 | last 10x7 | last 5x5 |
|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| 1 | 1000 | 1000 | 1500 | 500 | False | False | False | 100.0 | 100.0 | 53.8 |
| 2 | 750 | 750 | 1000 | 250 | True | False | False | 100.0 | 100.0 | 36.4 |
| 3 | 1500 | 1500 | 1750 | 250 | True | False | False | 100.0 | 100.0 | 45.1 |
| 4 | 1750 | 1750 | 2250 | 500 | False | False | False | 100.0 | 100.0 | 62.9 |
| 5 | 500 | 500 | 2750 | 2250 | False | False | False | 100.0 | 100.0 | 73.2 |
| 6 | 1000 | 1000 | 1750 | 750 | False | False | False | 100.0 | 100.0 | 56.5 |
| 7 | 1000 | 1000 | 1750 | 750 | True | True | True | 100.0 | 100.0 | 100.0 |
| 8 | 1250 | 1250 | 1500 | 250 | True | True | True | 100.0 | 100.0 | 100.0 |
| 9 | 1000 | 1000 | 1750 | 750 | True | False | False | 100.0 | 100.0 | 80.3 |
| 10 | 1250 | 1250 | 1500 | 250 | True | False | False | 100.0 | 100.0 | 87.1 |

## Table 3. (b) Plain switch (as 2b), seeds 11-20, first-loss prediction

Gc, 10/10 runs. 10x7 click 8/10, hold 10x7 5/10, hold 5x5 0/10, both at the end 1/10.

| seed | 5x5 click | switch | 10x7 click | click - switch | hold 10x7 | hold 5x5 | both | best 10x7 | last 10x7 | last 5x5 |
|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| 11 | 2000 | 2000 | - | - | - | False | False | 51.0 | 49.4 | 51.4 |
| 12 | 750 | 750 | - | - | - | False | False | 51.6 | 49.5 | 50.3 |
| 13 | 750 | 750 | 2250 | 1500 | False | False | False | 100.0 | 100.0 | 80.8 |
| 14 | 500 | 500 | 750 | 250 | False | False | False | 100.0 | 100.0 | 27.6 |
| 15 | 1000 | 1000 | 1500 | 500 | True | False | True | 100.0 | 100.0 | 97.5 |
| 16 | 750 | 750 | 1500 | 750 | True | False | False | 100.0 | 100.0 | 60.6 |
| 17 | 1750 | 1750 | 2000 | 250 | True | False | False | 100.0 | 100.0 | 6.1 |
| 18 | 1500 | 1500 | 1750 | 250 | True | False | False | 100.0 | 100.0 | 0.1 |
| 19 | 1500 | 1500 | 1750 | 250 | True | False | False | 100.0 | 100.0 | 26.0 |
| 20 | 1500 | 1500 | 4500 | 3000 | False | False | False | 100.0 | 100.0 | 88.2 |

Prediction (threshold 0.75, committed before the outcomes): 'learns' if the mean train loss of the first 250 steps after the switch is below it. Outcome: 10x7 click within 1000 steps of the switch.

| seed | first loss | prediction | 10x7 click - switch | outcome |
|---:|---:|---|---:|---|
| 11 | 1.047 | collapses | - | not |
| 12 | 0.837 | collapses | - | not |
| 13 | 0.570 | learns | 1500 | not |
| 14 | 0.493 | learns | 250 | learns |
| 15 | 1.045 | collapses | 500 | learns |
| 16 | 1.036 | collapses | 750 | learns |
| 17 | 0.882 | collapses | 250 | learns |
| 18 | 0.423 | learns | 250 | learns |
| 19 | 0.653 | learns | 250 | learns |
| 20 | 0.766 | collapses | 3000 | not |

Predicted 'learns': 3/4 learned; predicted 'collapses': 3/6 learned. Fisher one-sided p = 0.452; wrong predictions: 4. Separation confirmed (p < 0.05): False.

## Table 4a. (c) C, fixed switch at step 5500

C, 5/5 runs. 10x7 click 0/5, hold 10x7 0/5, hold 5x5 0/5, both at the end 0/5.

| seed | 5x5 click | switch | 10x7 click | click - switch | hold 10x7 | hold 5x5 | both | best 10x7 | last 10x7 | last 5x5 |
|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| 1 | - | 5500 | - | - | - | False | False | 53.0 | 52.7 | 49.3 |
| 2 | - | 5500 | - | - | - | False | False | 54.1 | 52.5 | 49.6 |
| 3 | - | 5500 | - | - | - | False | False | 51.8 | 48.5 | 49.9 |
| 4 | - | 5500 | - | - | - | False | False | 52.4 | 50.0 | 50.8 |
| 5 | - | 5500 | - | - | - | False | False | 52.4 | 51.2 | 50.3 |

## Table 4b. (c) C+h with plane 15 zeroed, event switch

Ch15, 5/5 runs. 10x7 click 5/5, hold 10x7 5/5, hold 5x5 0/5, both at the end 4/5.

| seed | 5x5 click | switch | 10x7 click | click - switch | hold 10x7 | hold 5x5 | both | best 10x7 | last 10x7 | last 5x5 |
|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| 1 | 500 | 500 | 750 | 250 | True | False | True | 100.0 | 100.0 | 91.0 |
| 2 | 250 | 250 | 750 | 500 | True | False | True | 100.0 | 100.0 | 91.1 |
| 3 | 250 | 250 | 500 | 250 | True | False | True | 100.0 | 100.0 | 91.0 |
| 4 | 250 | 250 | 500 | 250 | True | False | False | 100.0 | 100.0 | 64.2 |
| 5 | 250 | 250 | 500 | 250 | True | False | True | 100.0 | 100.0 | 91.1 |

## Verdict by the protocol

(a) 10x7 clicks 10/10; (b) separation not confirmed: series 3 (curriculum: soft switch).
Convolution arms with >= 3/5 clicks: C+h without plane 15: series 3 for comparison, over a stop row.

Next: series 3 (curriculum: soft switch).

