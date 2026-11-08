Goal of part 4: does a graph network generalise the middle game where the convolution did not?

# Part 4 plan: a graph network for Dot, as a research plan

Status: agreed 2026-11-08 (user); series 1 protocol fixed and run first. Rewritten after stage 0 and (i) of the first plan
(docs/graph_hand.md). Series 1 done: docs/graph_s1.md.
Dot-2 = dot2c gen 16 (runs/dot2c/gen_0016.pt; AZNet2 6x64, 0.51M parameters).

## Rules for every series

- The protocol of a series (configs, seeds, steps, metrics, criterion) is fixed
  in this file and committed before the first run. It does not change during
  the series. Any change, including "one more seed" or "more steps", is a new
  series with its own number and its own entry here.
- Deterministic mode in every run: `torch.use_deterministic_algorithms(True)`,
  cudnn deterministic, fixed random / numpy / CPU / GPU seeds. A series runs on
  one machine only (determinism does not hold across GPUs).
- Seeds are fixed in the config. Seed changes net init and batches; the data
  seed stays 4100 within a series.
- All runs are reported, including failed ones. The probes are read once, after
  training, and never used for selection.
- Limits (CLAUDE.md, user 2026-11-07): GPU memory <= 40%, free RAM >= 12 GB, nice,
  runs over 10 minutes in tmux.

## What is known (from the first plan, docs/graph_hand.md)

- Gc can learn dots + long chains mod 2 on settled positions, but not reliably:
  config (b), 5 deterministic seeds, 5x5 clicks in 2 of 5; 10x7 >= 95% in 1 of 5.
- When it clicks, 10x7 clicks at the same step while training on 5x5 only
  (3500-4000): the transfer comes for free, the 10x7 phase adds nothing.
- After the click both validations are flat to 0.1% for 16000 steps while the
  train loss is not zero: V (tanh) looks saturated. Not checked.
- (i) with the curriculum: no arm above 65% on 10x7 validation, probe 10x7
  C 68.9 / C+h 71.8 / Ga 70.4 / Gc 72.6% (3-seed means); Gc vs C+h p 0.90.
  Seed picked by the single best validation; Ch s3's best point was before the
  switch, so it was a 5x5-only net.

So the open problem comes first: training is a lottery. Until it is not, every
comparison between arms measures luck.

## Series 1: reliable training of the graph net (stage 0, 5x5 only)

Goal: a recipe with which Gc reaches 95% on 5x5 validation in 5 of 5 seeds.

- Task: stage 0 target (sign of dots + long chains mod 2, seen from the mover),
  settled positions from random safe-move playouts, 20000 train / 2000 val, 5x5
  only. 10x7 validation is logged but plays no role here.
- Steps: 18000 per run, the same in every variant (b64 / b512 included). Fixed
  from one timing run of B before the series (2000 steps in 65 s on G16, about 10
  minutes per run; not counted). Validation every 250 steps.
- Protocol: configs/graph_s1.yaml, scripts/graph_s1.py; R1 is chosen by the rule
  written in the config (a factor joins if its click share is above B's; in a
  group of exclusive factors the higher share, then the lower median step).
- Baseline B: the optimiser of stage 0 (b) of 2026-11-08 (note: (b) alternated 5x5 and 10x7
  before its switch; B is 5x5 only) (tanh V, MSE, lr 5e-4, cosine,
  clip 1.0, batch 128).
- Factors, one at a time against B (each variant = B with one change):

| variant | change |
|---|---|
| B | baseline |
| V-sign | V linear, loss on the sign (logistic on the logit) |
| V-clip | V linear, clipped to [-1, 1] in the loss (MSE on clip(v)) |
| init | readout last layer initialised small (std 1e-3), bias 0 |
| lr1e-4 | lr 1e-4 |
| lr1e-3 | lr 1e-3 |
| warmup | linear lr warm-up over the first 5% of steps, then cosine |
| norm | readout sums divided by the node count of their kind |
| b64 | batch 64 |
| b512 | batch 512 |

  10 variants x 5 seeds (1-5) = 50 runs.
- Metrics per run: does 5x5 click (first val >= 95%), the step of the click,
  holding >= 90% to the end, the 5x5 and 10x7 validation curves.
- Freeze diagnosis: at every validation, a histogram of V before and after the
  output nonlinearity on the 5x5 and 10x7 validation sets, and the share with
  |v| > 0.99; gradient norm of the readout. Saturation is confirmed if, after
  the click, the share > 0.9 and the readout gradient is near zero.
- Combination: the factors that raise the click share go together into one
  recipe R1, run on 5 new seeds (6-10). That is fixed now and is part of this
  series. 5 runs.
- Criterion, on the variants: click share first, the median click step second.

| result | reading | next |
|---|---|---|
| R1 clicks in 5 of 5 (seeds 6-10) | recipe found | series 2 with R1 |
| best is 3-4 of 5 | better, not reliable | new series 1b (combinations, more seeds); no series 2 yet |
| no variant above 2 of 5 | not the optimiser: a readout / architecture defect | stop; redesign the readout, then a new series |
| saturation confirmed and V-sign / V-clip fix it | the flat curves were tanh | V linear in all later series and in the CNN arms |

Runs: 55, 10 min each.

## Series 2: transfer to 10x7

Goal: does the 5x5 skill carry to 10x7 without any 10x7 example, and does it
for the convolution too.

- 2a. Gc, recipe R1, 5x5 only, 10 seeds (11-20), 10x7 validation logged.
  Metric: share of seeds with 10x7 >= 95% at the best 5x5 point and holding
  >= 90% to the end.
- 2b. Event curriculum: Gc, R1, the same 10 seeds; switch to 10x7 only when 5x5
  validation >= 95% (not at a fixed step), then 10x7 to the end. Same metric.
- 2c. Control: C (AZNet2) and C+h (AZNet2 + chain planes), recipe R1 where it
  applies (the V and lr factors; the readout factors have no CNN analogue and
  are left as in the CNN), 5x5 only, the same 10 seeds. Same metric.
- Runs: 10 + 10 + 20 = 40.

| result | reading | next |
|---|---|---|
| 2a >= 8 of 10 | transfer is a property of Gc with R1 | series 3, 5x5 then 10x7 by event |
| 2a <= 2 of 10 | 10x7 needs 10x7 data | series 3 with the event curriculum |
| 2b above 2a (Fisher one-sided p < 0.05) | 10x7 phase helps | series 3 with the event curriculum |
| 2b not above 2a | 10x7 phase adds nothing (as seen) | series 3 keeps the simpler of the two |
| C / C+h transfer as often as Gc | transfer is not a graph property | report; series 3 still runs |
| C / C+h do not transfer, Gc does | first evidence of a graph advantage | series 3 |

## Series 3: parity probe on real positions (the former (i))

Goal: does the graph see late parity better than the convolution with chain
planes, on judged positions.

- Data and loss as before: data/parity_hand_{5x5,10x7}.jsonl.gz, 5000 per size,
  loss -log(mass on the correct moves), 10% validation; probes 1000 (5x5) and
  304 (10x7) held out.
- Arms C, C+h, Ga, Gc, recipe R1, curriculum as chosen by series 2, 5 seeds each
  (1-5): 20 runs, 30000 steps.
- Best point by 10x7 validation, and only after the switch to 10x7 (a best point
  before it is not admitted; the lesson of Ch s3). Logged every 500 steps.
- Test per pair (Gc-C, Gc-C+h, Ga-C, Gc-Ga), on the 10x7 probe:
  - all 25 seed pairs, McNemar each, descriptive;
  - primary: bootstrap over seeds and positions (10000 resamples: seeds drawn
    with replacement within each arm, positions with replacement, the same
    positions for both arms), difference of mean accuracy; one-sided 95% bound.
  Gc-C+h is the primary pair; the other three are secondary.
- Runs: 20; then probes once.

| result | reading | next |
|---|---|---|
| Gc - C+h lower bound > 0 and Gc probe > 80% | graph adds something beyond chain planes, late | series 4 |
| Gc - C+h lower bound > 0, probe <= 80% | a small gain, still weak | series 4, report it as weak |
| Gc <= C+h (bound includes 0) | graph gives nothing beyond chain planes late | stop part 4 |
| Gc <= C | representation is not the limit even late | stop part 4 |

## Series 4: middle game (the former (ii))

Goal: does the graph generalise the middle game where the convolution did not.
Only if series 3 says "series 4".

- Start: distil Gc and C+h from Dot-2 on the dot2c replay buffers
  (scripts/distill.py), policy and V; C = Dot-2 itself. Entry check for each
  distilled net: raw Elo 10x7 >= 1400 and V sign agreement with Dot-2 >= 90%. If
  one fails, series 4 is not comparable: stop and report.
- Fine-tune on mid + the 80 rout mistakes, buffers mixed as in
  configs/mid_hand.yaml, 4000 steps, the same split (whole games: 1387 / 538, 199
  decisive held out; mistakes 56 / 24). Two variants for each of Gc, C, C+h:
  **frozen** (body and V frozen, bitwise check) and **lr0.1** (body and V at 0.1x
  lr, self-play V loss kept). Trainable heads within 10% in parameter count.
  Primary share r30.
- Tests on set mid, decisive held out, one-sided sign test:
  **T1** Gc before -> after; **T2** Gc after vs C after; **T3** Gc after vs C+h
  after. Two variants = two tests: p < 0.025 each (Bonferroni, overall p < 0.05).
  The 24 mistakes are descriptive only.
- Confirmation: 2000 new positions at 10-15 safe lines (~700 decisive), fixed in
  advance; same T1. Only if T1 passes.
- Runs: 2 distillations, 2 entry checks, 6 fine-tunes, tests; confirmation set.

| result | reading | next |
|---|---|---|
| distillation misses the entry check | comparison impossible at this cost | stop, or a new series from scratch |
| T1 fails | late parity, not the middle game: the limit is the signal, not the architecture | stop; answer of parts 3 and 4 |
| T1 passes, T2 / T3 do not | fine-tuning helps, no more than for the CNN | report, stop |
| T1, T2, T3 pass, confirmation fails | the first pass was noise or the used set | stop |
| all pass, confirmed | the graph generalises the middle game | self-play on 10x7 as a new series (former (iii)), planned then |

## Time

Measured on G16 (RTX 4060 Laptop 8 GB, limits above): stage 0 (b) 20000 steps
~30 min (mostly the 10x7 phase); (i) 30000 steps C 10, C+h 10, Ga 14, Gc 19 min;
CNN distillation 30k steps 27 min. G18 (RTX 5070, Ryzen 9 9955HX) is not
measured and has no Linux: WSL2 + CUDA setup 0.5-1 day, then by spec about
1.5x G16 on the GPU (estimate). Since a series stays on one machine, G18 can
only take whole series.

| series | code to write | runs | G16 | G18 (estimate) |
|---|---|---:|---:|---:|
| 1 | V-linear, readout init / norm, warm-up, histograms, 5x5-only stage 0: 0.5 day | 55 x 10 min | 9 h | 6 h |
| 2 | event switch, CNN in stage 0: 2 h | 40 | 2a 2 h, 2b 5 h, 2c 3.5 h = 10.5 h | 7 h |
| 3 | seed bootstrap, best-after-switch: 2 h | 20 + probes | 5 h | 3.5 h |
| 4 | graph distillation path: 0.5-1 day | 2 distil + 6 fine-tunes + set | distil 2-4 h, checks 1 h, fine-tune 1 h, confirmation up to 16 h = 20-22 h | 14-15 h (the judge is CPU, faster on 16 cores) |

Total machine time to the end of series 4: about 45 h on G16 (4-5 nights), about
30 h on G18 after its setup. With the code: about one week of calendar time if
every series goes on; stops are likely after series 1 or 3, at about 2-3 days.
