# Part 4, series 2'': forgetting of 5x5 with a kept 5x5 share

Protocol: docs/plan_part4.md (series 2''), configs/graph_s2pp.yaml. Gc, school (a) of series 2'; after the 1000-step mix a share of 5x5 batches is kept to the end. Click: first validation >= 95.0%; hold 10x7: every validation after the click >= 90.0%; hold 5x5: every validation from the end of the mix >= 90.0%; both: the last validation >= 90.0% on both sizes. Reference: series 2' (a), share 0, the same seeds.

## k10: share 0.10, 10/10 runs

10x7 click 9/10, hold 10x7 4/10, hold 5x5 8/10, both at the end 9/10.

| seed | switch | 10x7 click | hold 10x7 | hold 5x5 | both | min 5x5 after mix | last 10x7 | last 5x5 | (a): hold 5x5 / both / last 5x5 |
|---:|---:|---:|---|---|---|---:|---:|---:|---|
| 1 | 1000 | 1500 | False | True | True | 96.0 | 100.0 | 100.0 | False / False / 53.8 |
| 2 | 750 | 1000 | False | True | True | 99.8 | 100.0 | 100.0 | False / False / 36.4 |
| 3 | 1500 | 1750 | True | True | True | 100.0 | 100.0 | 100.0 | False / False / 45.1 |
| 4 | 1750 | 2250 | True | True | True | 99.7 | 100.0 | 100.0 | False / False / 62.9 |
| 5 | 500 | - | - | False | False | 43.7 | 50.6 | 49.0 | False / False / 73.2 |
| 6 | 1000 | 1750 | False | True | True | 97.9 | 100.0 | 100.0 | False / False / 56.5 |
| 7 | 1000 | 1750 | False | False | True | 49.6 | 100.0 | 100.0 | True / True / 100.0 |
| 8 | 1250 | 1500 | True | True | True | 99.9 | 100.0 | 100.0 | True / True / 100.0 |
| 9 | 1000 | 1750 | False | True | True | 99.9 | 100.0 | 100.0 | False / False / 80.3 |
| 10 | 1250 | 1500 | True | True | True | 100.0 | 100.0 | 100.0 | False / False / 87.1 |

Paired with (a) on the same seeds (seeds only this variant has / only (a) has / one-sided exact p):

| metric | this | (a) | only this / only (a) | p |
|---|---:|---:|---|---:|
| click10 | 9 | 10 | 0 / 1 | 1 |
| hold10 | 4 | 6 | 1 / 3 | 0.938 |
| hold5 | 8 | 2 | 7 / 1 | 0.0352 |
| both | 9 | 2 | 7 / 0 | 0.00781 |

## k25: share 0.25, 10/10 runs

10x7 click 10/10, hold 10x7 3/10, hold 5x5 7/10, both at the end 10/10.

| seed | switch | 10x7 click | hold 10x7 | hold 5x5 | both | min 5x5 after mix | last 10x7 | last 5x5 | (a): hold 5x5 / both / last 5x5 |
|---:|---:|---:|---|---|---|---:|---:|---:|---|
| 1 | 1000 | 1500 | False | True | True | 100.0 | 100.0 | 100.0 | False / False / 53.8 |
| 2 | 750 | 1000 | False | True | True | 100.0 | 100.0 | 100.0 | False / False / 36.4 |
| 3 | 1500 | 1750 | False | False | True | 83.3 | 100.0 | 100.0 | False / False / 45.1 |
| 4 | 1750 | 2250 | False | True | True | 99.6 | 100.0 | 100.0 | False / False / 62.9 |
| 5 | 500 | 3250 | True | False | True | 88.0 | 100.0 | 100.0 | False / False / 73.2 |
| 6 | 1000 | 1750 | True | True | True | 99.3 | 100.0 | 100.0 | False / False / 56.5 |
| 7 | 1000 | 1750 | True | True | True | 100.0 | 100.0 | 100.0 | True / True / 100.0 |
| 8 | 1250 | 1500 | False | False | True | 50.2 | 97.0 | 100.0 | True / True / 100.0 |
| 9 | 1000 | 1750 | False | True | True | 99.9 | 100.0 | 100.0 | False / False / 80.3 |
| 10 | 1250 | 1500 | False | True | True | 96.3 | 100.0 | 100.0 | False / False / 87.1 |

Paired with (a) on the same seeds (seeds only this variant has / only (a) has / one-sided exact p):

| metric | this | (a) | only this / only (a) | p |
|---|---:|---:|---|---:|
| click10 | 10 | 10 | 0 / 0 | 1 |
| hold10 | 3 | 6 | 2 / 5 | 0.938 |
| hold5 | 7 | 2 | 6 / 1 | 0.0625 |
| both | 10 | 2 | 8 / 0 | 0.00391 |

## Verdict (pre-registered primary: hold 5x5 above (a), p < 0.025 per variant; 10x7 click >= 8 of 10)

- k10: hold 5x5 p = 0.0352 -> not passed; 10x7 click 9/10
- k25: hold 5x5 p = 0.0625 -> not passed; 10x7 click 10/10

Table row: hold 5x5 not above (a) at these shares (p < 0.025 not reached by either variant). Descriptive (not pre-registered): the 5x5 failures after the mix are brief: k10 s7, k25 s3, s5, s8 each have a single validation below 90% (of 71-75 after the mix) and end at 100% on 5x5; k10 s5 is not a hold failure but a run that never clicked (both sizes at chance from the switch on). At the end both sizes >= 90% in 9/10 (k10) and 10/10 (k25) vs 2/10 in (a) (p 0.0078 / 0.0039, uncorrected); hold 10x7 4 and 3 vs 6. Series 3' share rule: neither share passed; the user chose 0.25 (docs/plan_part4.md).

