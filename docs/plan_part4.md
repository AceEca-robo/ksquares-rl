Goal of part 4: does a graph network generalise the middle game where the convolution did not?

# Part 4 plan: a graph network for Dot, as a research plan

Status: agreed 2026-11-08 (user); series 1 protocol fixed and run first. Rewritten after stage 0 and (i) of the first plan
(docs/graph_hand.md). Series 1 done: docs/graph_s1.md. Series 2 done 2026-11-09: docs/graph_s2.md (2b 7/10, next series 2'). Series 2' done 2026-11-11 (docs/graph_s2p.md); series 3 protocol 2026-11-12.
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
  runs over 10 minutes in tmux. Series 2 lifts the GPU memory and worker limits for itself only.

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

Protocol rewritten by the user on 2026-11-09, before any run of it: configs/graph_s2.yaml,
scripts/graph_s2.py. Replaces the earlier draft of this section (seeds 11-20, 2c without curriculum).

Goal: does the 5x5 skill carry to 10x7 without any 10x7 example, does an event
curriculum help, and does the convolution do the same at the same school.

- Recipe: B + V-sign, lr of B (5e-4), cosine, clip 1.0, batch 128. This is the
  user's decision: the rule of series 1 gave R1 = B, because V-sign's 5/5 was not
  above B's 5/5. Note: V-sign clicked 5/5 and held, but its V does saturate by the
  metric of series 1 (share |V| > 0.99 is 1.00 at the end of all five runs).
- Limits lifted for this series only (user, 2026-11-09): GPU memory and workers are
  not limited; 4 runs at once. Free RAM >= 12 GB, nice and tmux stay.
- Data: seed 4100; 5x5 20000 train + 2000 val and 10x7 2000 val are those of
  series 1; 10x7 20000 train is new, from the same stream. No judge in this series.
- 20000 steps in total per run, validation of both sizes every 250 steps; deterministic.
- 2a. Gc, 5x5 only, seeds 1-10. 10x7 is validation only.
- 2b. Gc, seeds 1-10. Event curriculum: 5x5 until the first validation with
  5x5 >= 95%, then 10x7 only to the end. A run that never reaches it stays on 5x5.
  One cosine over the 20000 steps, not restarted at the switch.
- 2c. Control: C and C+h, the same recipe (V linear, logistic on the sign, lr; the
  readout factors have no CNN analogue) and the 2b curriculum, seeds 1-5. Symmetry
  augmentation as in all CNN runs of part 4.
- 2d. Diagnosis of the 2b runs without a 10x7 click. V histogram on 10x7 (raw, 20
  bins) at the switch and at the end; readout gradient norm. Medians over the last
  quarter of the logs. Rule: train loss < 0.3: fits the train set without the
  rule. Otherwise readout gradient < 1e-3: frozen. Otherwise: does not find the
  rule (gradient alive, loss not falling). A run that never switched: no 10x7 phase.
- Seeds 1-5 are the seeds V-sign was chosen on in series 1 (user's choice of seeds
  1-10), so the 5x5 click there is not an independent check. Transfer is.
- Runs: 10 + 10 + 10 = 30.

Definitions. Click on 10x7: some validation has 10x7 >= 95%. Hold: every later
validation has 10x7 >= 90%. The share in the criterion counts clicks, as the
question is worded; hold is reported next to it, and if counting click + hold
would give another row of the table, the summary says so.
Fisher: 2b vs 2a, one-sided (2b above 2a), clicks.
"The convolutions click the same": C's share >= 2b's share - 0.2 and at least one
C click; Fisher two-sided is reported, with 5 seeds it has little power.

C+h is not a test of the school here: its hint plane 15 is the stage 0 target
itself (agrees on 1000 of 1000 random positions on 5x5 and on 10x7, measured 2026-11-09). C+h clicking
says only that the CNN can read a plane. The verdict on the school is taken on C alone.

| result | reading | next |
|---|---|---|
| 2b >= 8 of 10 | transfer works | series 3 |
| 2b 4-7 of 10 | partly | series 2' (what gets in the way) |
| 2b <= 3 of 10 | the graph does not transfer | stop with this conclusion |
| C clicks the same as 2b | the school, not the graph | series 3 for comparison, also over the stop row |

## Series 2': what gets in the way of transfer

Protocol set by the user on 2026-11-10 after series 2 (2b 7/10 on 10x7, row "series 2'"),
committed before any run: configs/graph_s2p.yaml, scripts/graph_s2p.py (run loop of
scripts/graph_s2.py with options that are off in series 2). Recipe, data, steps and
validation are those of series 2. Limits lifted (user): GPU memory and workers not
limited, 6 runs at once (8 ran out of GPU memory in the first minutes; restarted).

- (g) first, no new training. 2b seeds whose last 5x5 is below 40% (s1, s2, s3, s4,
  s8, s9) and the reference s7 (91%). Series 2 kept no weights, so these 7 runs are
  reproduced bit-exact (series 2 is deterministic) to get the final weights; the
  validation logs must equal the series 2 logs, otherwise (g) is reported on the
  reproduced nets with that noted. On the 2000 5x5 validation positions: error share
  by the number of long chains (0/1/2/3+), by their total length (0, 3-5, 6-8,
  9-11, 12+) and by the number of loops (0/1/2+); agreement of the net's sign with
  candidate rules "+1 iff (count % 2) == player" for counts of long chains (the
  true rule), chains of 2+, all chains, chains of 1-2, loops, long chains + loops,
  chains + loops, and with the player alone. Descriptive: no test.
- (a) soft switch. Gc, seeds 1-10. After the event (5x5 >= 95%), 1000 steps of a
  mix (batches alternate 5x5 / 10x7), then 10x7 only. Metrics: 10x7 click; hold
  10x7 (every validation after the click >= 90%); hold 5x5 (every validation from
  the end of the mix >= 90%); both (last validation >= 90% on both sizes).
- (b) prediction from the first loss. Plain switch as 2b, new seeds 11-20. The mean
  train loss of the first 250 steps after the switch is written with a prediction
  as soon as it is known: below 0.75 (between 0.53 and 0.97 of series 2) "learns",
  otherwise "collapses". The 10 predictions are committed as soon as all are known,
  before the outcomes (docs/graph_s2p_predictions.json). Outcome "learns": 10x7
  click within 1000 steps of the switch. Separation confirmed: Fisher one-sided
  p < 0.05 on the 2x2 table; if all predictions fall in one class, not testable.
- (c) controls. C with a fixed switch at step 5500 (5x5 until then whatever its
  accuracy, then 10x7), seeds 1-5. C+h with plane 15 (the target) set to 0 in train
  and validation, event switch, seeds 1-5.
- Runs: 7 + 10 + 10 + 5 + 5 = 37. Report: docs/graph_s2p.md, four tables
  ((g), (a), (b), (c)).

| result | reading | next |
|---|---|---|
| (a) 10x7 clicks >= 8 of 10 | the soft switch fixes transfer | series 3, curriculum (a) |
| (a) 4-7, (b) separation confirmed | the spike at the switch predicts the collapse | series 2'' aimed at the spike |
| (a) 4-7, (b) not confirmed | partial transfer, cause not found | stop |
| (a) <= 3 | the soft switch does not help | stop |
| C fixed or C+h without plane 15 clicks >= 3 of 5 on 10x7 | the convolution learns it too | series 3 for comparison, over a stop row |

Caveat found in the smoke test before the series (1500 steps, seed 1): C+h with plane 15
zeroed reached 100% on 5x5 at step 500 and on 10x7 at 750. Plane 14 is the long chain
count / 10 and plane 9 the mover, so the rule is still almost given. The arm is run as
set; a C+h click is read as "the CNN reads plane 14", not as the school.

Hold 5x5 and "both" are reported, not in the table: series 3 is judged on 10x7.

## Series 3: parity probe on real positions (the former (i))

Protocol set by the user on 2026-11-12 after series 2' (row "series 3, curriculum (a)"), committed
before any run: configs/graph_s3.yaml, scripts/graph_s3.py. Replaces the earlier draft of this section.

Goal: does the graph see late parity better than the convolution with chain
planes, on judged positions.

Divergences from the earlier draft and from series 1-2', written before the first run:
- School (a) cannot run on the judge's positions: its event is 5x5 validation >= 95%, and in (i)
  5x5 validation on data/parity_hand never went above 59% (runs/graph_hand3, all 12 runs, mostly
  43-50%). There would be no switch and no "best point after the switch". The V-sign recipe also
  has no role in a policy loss. User's decision (2026-11-12): two stages, below.
- C did not learn the stage 0 rule in series 2' (0/5 even with a fixed switch); with the event
  school it never reaches 95% on 5x5, so its stage 1 is 20000 steps of 5x5 stage 0 without a switch.
- C+h's plane 15 is the stage 0 target; on the judge's positions it still carries the long chain
  parity. Hence the arm C+h14 (plane 15 zeroed, as Ch15 of series 2'; plane 14 = long chain count / 10).
- The earlier draft's secondary pairs (Gc-C, Ga-C, Gc-Ga) are now descriptive; a second test, Gc - C+h14,
  is pre-registered.
- Time: a Gc stage 1 took 2.4 h at 6 runs at once in series 2'; the series is about 10-12 h.

Protocol:
- Arms: C, C+h (17 planes), C+h14 (C+h with plane 15 zeroed in both stages, train, validation and
  probe), Ga, Gc. Seeds 1-5. 25 runs, 6 at once, deterministic, limits lifted (user).
- Stage 1: the school (a) of series 2' on the stage 0 target, unchanged (series 2 data, recipe B +
  V-sign, 20000 steps, 5x5 until validation >= 95%, 1000 steps of alternating batches, then 10x7;
  run loop of scripts/graph_s2.py). Gc s1-s5 stage 1 is series 2' a_Gc_s1..s5 (same config,
  deterministic): copied with its weights, not rerun.
- Stage 2: from the final stage 1 weights, the (i) data and loss: data/parity_hand_{5x5,10x7}, 5000
  per size, 10% validation, loss -log(mass on the correct moves), Adam on all parameters, lr 5e-4,
  cosine over the 30000 steps of stage 2, clip 1.0, batch 128; judge 5x5 positions to step 5500,
  then 10x7 only (as (i)). Symmetry augmentation for the CNN arms. Validation every 500 steps.
- Best point by 10x7 validation, and only after the switch of stage 2 (steps > 5500). The best
  net is saved and probed.
- Diagnostic (user, 2026-11-12), not used for selection: at every validation of stage 2, the V-sign
  accuracy on the stage 0 validation sets (10x7, and 5x5 alongside): does the net keep the chain
  count while it learns to choose the move, or forget it as it forgot 5x5?
- Probes once, after all training: 1000 (5x5), 304 (10x7), held out.
- Primary test: Gc - C+h on the 10x7 probe, bootstrap over seeds and positions (10000 resamples:
  seeds with replacement within each arm, positions with replacement, the same positions for both
  arms), difference of mean accuracy, one-sided 95% lower bound. Second test, pre-registered: Gc - C+h14,
  the same way, at its own 95%. Descriptive: Gc - C, Ga - C, Gc - Ga, C+h - C+h14 (same bootstrap),
  and for every pair the 25 seed pairs' McNemar tests (counts of p < 0.05 each way).

How C's row is read (written before the runs): C enters stage 2 without the rule (its stage 1 never
switches). Gc - C therefore measures architecture and school together: Gc > C is not evidence that
the graph beats the convolution at an equal school, and it is not used for the verdict. The row
"Gc <= C" still stands: if Gc, with the rule learned in stage 1, is not above a C that never learned
it, the school and the graph add nothing late. C's stage 0 diagnostic is expected to stay near 50%.

| result | reading | next |
|---|---|---|
| Gc - C+h lower bound > 0 and Gc probe > 80% | graph adds something beyond chain planes, late | series 4 |
| Gc - C+h lower bound > 0, probe <= 80% | a small gain, still weak | series 4, report it as weak |
| Gc <= C+h (bound includes 0), Gc - C+h14 bound > 0 | graph beats the chain count alone, not the count with the target plane | stop part 4; reported |
| Gc <= C+h and Gc <= C+h14 | graph gives nothing beyond chain planes late | stop part 4 |
| Gc <= C (descriptive bound includes 0 and Gc mean <= C mean) | representation is not the limit even late | stop part 4 |

## Series 2'': forgetting of 5x5

Protocol set by the user on 2026-11-12, committed before any run: configs/graph_s2pp.yaml,
scripts/graph_s2pp.py (run loop of scripts/graph_s2.py, option keep_5x5, off in all earlier series).
Runs after series 3 (queued behind it on the same machine).

Goal: in series 2' (a) 5x5 was held in 2 of 10 runs once training moved to 10x7. Does keeping a
share of 5x5 batches to the end stop the forgetting without costing the 10x7 click?

- Gc, school (a) of series 2' unchanged (series 2 data, recipe B + V-sign, 20000 steps, validation
  every 250, event 5x5 >= 95%, 1000 steps of alternating batches), but after the mix the 5x5 share
  is not zero: k10 = 10% of the batches (every 10th step), k25 = 25% (every 4th step); the rest 10x7.
  Seeds 1-10 in each: 20 runs, 6 at once, deterministic, limits lifted (user).
- Metrics per run (definitions of series 2'): 10x7 click; hold 10x7; hold 5x5 (every validation
  from the end of the mix >= 90%, "5x5 to the end >= 90%"); both (last validation >= 90% on both sizes).
- Comparison with series 2' (a) on the same seeds (10x7 click 10/10, hold 10x7 6/10, hold 5x5 2/10,
  both 2/10): per metric, paired exact one-sided sign test on the seeds where the two differ
  (variant above (a)). Pre-registered primary: hold 5x5, each variant at p < 0.025 (two variants,
  Bonferroni). The other metrics and the 10x7 click (variant below (a)) are descriptive.

| result | reading | next |
|---|---|---|
| hold 5x5 above (a) (p < 0.025) and 10x7 click >= 8 of 10 | a kept 5x5 share stops the forgetting at no cost to transfer | the share becomes part of the school |
| hold 5x5 above (a), 10x7 click <= 7 of 10 | the share holds 5x5 but gets in the way of transfer | report; trade-off |
| hold 5x5 not above (a) | the share does not stop the forgetting (at these shares) | report; the forgetting is not a matter of data |

## Series 3': stage 0 kept as an auxiliary task in stage 2 (DRAFT, not approved)

Draft written 2026-11-12 at the user's request while series 2'' runs; nothing launched. Becomes the
protocol only after the user approves it and the share is fixed from series 2''.

Why: series 3 was paused (2026-11-12) because its diagnostic showed every net that came out of stage 1
with the rule (C+h, C+h14, Gc) losing it within the first 500 steps of stage 2 (stage 0 V-sign
accuracy on 10x7 100% -> ~50%). Stage 1 then adds nothing to the move choice.

- Arms, seeds, stage 1: as series 3 (C, C+h, C+h14, Ga, Gc; seeds 1-5; school (a) on stage 0).
  Stage 1 is deterministic and unchanged, so the stage 1 runs of series 3 (finished ones, and paused
  ones resumed from their checkpoints) are reused with their weights, not rerun.
- Stage 2: as series 3 (judge positions, 30000 steps, 5x5 to 5500 then 10x7, lr 5e-4 cosine, clip 1.0,
  batch 128, Adam on all parameters), but a share s of the steps are stage 0 steps to the end: on those
  steps the batch is 128 stage 0 train positions (series 2 data) and the loss is the stage 0 loss
  (logistic on the sign of V); on the others the policy loss as before. Pattern as series 2'': a step
  is a stage 0 step when floor(step * s) steps up. Stage 0 batches alternate 5x5 / 10x7 (to be confirmed by the user).
  The total stays 30000 steps, so the policy gets (1 - s) x 30000 of them.
- The share s: from series 2'', fixed before the launch by this rule: the smaller of 0.10 / 0.25 whose
  primary test passed (hold 5x5 above (a), p < 0.025) with 10x7 click >= 8 of 10; if neither passed,
  the user chooses before the launch.
- Validation every 500 steps: judge validation of both sizes, stage 0 V-sign accuracy on 5x5 and 10x7.
  Best point by 10x7 judge validation after the switch (steps > 5500), as series 3; V-sign plays no part
  in the choice.
- Gate (user): a run counts in a comparison only if its stage 0 V-sign accuracy on 10x7 is >= 90% at
  its best point. A pair of arms is compared only if both arms have >= 3 of 5 runs passing; the
  bootstrap then draws seeds from the passing runs of each arm. A pair not compared is reported
  as "not comparable" with the counts.
- Tests: as series 3 on the 10x7 probe: primary Gc - C+h (bootstrap over seeds and positions, 10000
  resamples, one-sided 95% lower bound), second pre-registered Gc - C+h14, descriptive Gc - C, Ga - C,
  Gc - Ga, C+h - C+h14, McNemar of the seed pairs. Probes once, after all training.
- Expected before the runs: C and Ga did not learn the rule in stage 1 (C 0/3, Ga 0/2 so far in
  series 3); if the stage 0 steps of stage 2 do not teach it, their pairs fail the gate.
- Runs: 25 stage 2 runs (+ the stage 1 runs not finished in series 3).

| result | reading | next |
|---|---|---|
| primary pair not comparable (Gc or C+h < 3 runs past the gate) | the share does not keep the chain count during stage 2 | stop; report |
| Gc - C+h lower bound > 0 and Gc probe > 80% | graph adds something beyond chain planes, late | series 4 |
| Gc - C+h lower bound > 0, probe <= 80% | a small gain, still weak | series 4, reported as weak |
| Gc <= C+h, Gc - C+h14 bound > 0 | graph beats the chain count alone, not with the target plane | stop part 4; reported |
| Gc <= C+h and Gc <= C+h14 | graph gives nothing beyond chain planes late, even keeping the count | stop part 4 |

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
