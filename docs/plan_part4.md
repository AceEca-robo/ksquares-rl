Goal of part 4: does a graph network generalise the middle game where the convolution did not?

# Part 4 plan: a graph network for Dot, as a research plan

Status: agreed 2026-11-08 (user); series 1 protocol fixed and run first. Rewritten after stage 0 and (i) of the first plan
(docs/graph_hand.md). Series 1 done: docs/graph_s1.md. Series 2 done 2026-11-09: docs/graph_s2.md (2b 7/10, next series 2'). Series 2' done 2026-11-11 (docs/graph_s2p.md); series 3 protocol 2026-11-12.  Series 2'' done 2026-11-12 (docs/graph_s2pp.md; primary not passed). Series 3' approved and done 2026-11-13 (docs/graph_s3p.md): Gc <= C+h, Gc - C+h14 > 0 -> stop part 4. Series 3' accepted by the user 2026-11-14, verdict as in its table; series 5 (how often Gc jumps) protocol 2026-11-14. Series 5 done 2026-11-15 (docs/graph_s5.md): 5 of 20, reproducible by the protocol. Series 6 (stage 1 weights or stage 2 seed) protocol 2026-11-16, done 2026-11-17 (docs/graph_s6.md: 7 of 10 vs 2 of 10, neither row). Series 7 (middle game, evaluation only) done 2026-11-17 (docs/graph_s7.md: no transfer). Series 9 (Gc plays 5x5, run g5) protocol 2026-11-17, done 2026-11-19 (docs/graph_s9.md: stopped at round 117; ladder best gen 49, late gens beat Dot). Series 10 (matches only) done 2026-11-19 (docs/graph_s10.md: Dot gen 84 beats gen 54 -> part 1 note is a measurement error; g5 114 vs Dot 84 62.5 % with MCTS 200, 47.5 % at equal time). Series 11 (10x7, matches only) done 2026-11-19 (docs/graph_s11.md: dot2c gen 16 is the strongest of the candidates, kept). Series 12 (fine-tune on the middle game, Dot-G vs Dot-2 vs C+h) protocol 2026-11-20, done 2026-11-20 (docs/graph_s12.md: (a) no difference seen, (b) Dot-G worse; Dot-G no gain seen).
Dot-2 = dot2c gen 16 (runs/dot2c/gen_0016.pt; AZNet2 6x64, 0.51M parameters).

## Rules for every series

- Best generation (user, 2026-11-19, after series 10): chosen by matches between candidates (the top 3-4 by ladder
  Elo plus the last accepted one, a round robin with games per pair fixed in the protocol), not by the ladder. The
  ladder only for early rounds, while the net does not beat Strong. Before anything is published, the checks of
  CLAUDE.md "Before publishing" apply.

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

## Series 3': stage 0 kept as an auxiliary task in stage 2

Draft written 2026-11-12; approved by the user 2026-11-13 with the decisions below, committed before the
first run: configs/graph_s3p.yaml, scripts/graph_s3.py (option stage2.stage0_share, off in series 3).

Decisions of the user (2026-11-13), replacing the open points of the draft:
- Share s = 0.25, chosen by the user, not by the rule: no share passed the series 2'' primary test
  (k10 p 0.035, k25 p 0.0625). Reason: at 0.25 the 10x7 click was 10/10 and both sizes at the end 10/10;
  at 0.10 one run (s5) never clicked.
- Stage 0 batches in stage 2: 10x7 only, no alternation with 5x5.
- Stage 2 lengthened so the policy still gets 30000 steps: 40000 steps in all, every 4th a stage 0 step.
  The switch (judge 5x5 to 5500) and "best point after the switch" count policy steps; cosine runs over
  the 40000 steps.
- Run conditions (day session, user): 3 runs at once, GPU memory <= 40%, nice -n 10, free RAM >= 12 GB.
  Stage 1 of series 3 had no run in progress at its pause (only stage 2 runs were), so the finished
  stage 1 runs are copied and the 12 missing ones (C s4-5, C+h s3-5, C+h14 s3-5, Ga s3-5) run with the
  same config; the series 3 stage 2 checkpoints are not used (stage 2 differs).

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
  is a stage 0 step when floor(step * s) steps up. Stage 0 batches are 10x7 only (user).
  The total is 40000 steps, so the policy gets 30000 of them (user).
- The share s: the rule (smaller passing share of series 2'') gave none; the user chose 0.25 (above).
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

## Series 5: how often does Gc make the move-choice jump

Protocol by the user, 2026-11-14, committed before the first run: configs/graph_s5.yaml,
scripts/graph_s3.py (option stage2.snapshots, off in series 3 and 3'), scripts/graph_s5.py (check, report).
Series 3' is accepted with its verdict as it stands in the table (stop part 4); this series does not reopen it.

One question: in series 3' one Gc run of five (s1) went from 61.6% to 81.8% on the 10x7 judge validation
right after the switch and ended near 89-90%; the other four stayed at 62-66%. How often does that happen?

- Runs: Gc only, seeds 6-25 (20 runs). Seeds 1-5 are series 3' and are not counted.
- Stage 1 and stage 2: exactly series 3' (the same config values, share 0.25, 40000 stage 2 steps, switch at
  policy step 5500 = total step 7333, validation every 500 steps, best point by 10x7 judge validation at
  policy steps > 5500). No change. tests/test_graph_s3.py compares the two configs key by key.
  Stage 1 of seeds 6-10 exists already as series 2' (a) (the same config, deterministic; series 3' took its
  seeds 1-5 from there in the same way): copied with its weights, not rerun. Seeds 11-25 run stage 1 here.
- Primary metric: the number of the 20 runs whose best 10x7 judge validation after the switch is >= 80%.
  All 20 runs are in the denominator, whatever their stage 1 or their V sign did.
- Reported next to it, not part of the criterion: whether the V sign is kept (stage 0 V-sign accuracy on
  10x7 >= 90% at the best point, the gate of series 3'), as a 2x2 table with the primary.
- Probes (5x5, 10x7): once, after all training; descriptive, never used for selection or for the verdict.
- Kept in addition, for the analysis of the sign loss and of the moment of the jump (no effect on training:
  the weights are written to disk, with no forward pass and no use of a random generator):
  - weights at stage 2 steps 250, 500 and 1000, and the V pre-activations on the stage 0 validation
    sets (2000 positions each of 5x5 and 10x7) computed from those weights after the run has ended;
  - weights at every validation step from the switch to +5000 steps. Read as: total steps 7000-14000,
    i.e. from the last validation before the switch (7000) to switch + 5000 policy steps (policy 10500 =
    total 14000); this also covers +5000 total steps (12333). 15 files per run.
- Run conditions (user, 2026-11-14, corrected before the launch: the machine is shared with the engine
  work today): at most 3 runs at once, GPU memory <= 40%, nice -n 10, no start below 12 GB free RAM,
  as the day session of series 3'. The slot count is stated in the report.
- Check before the launch (scripts/graph_s5.py check): stage 2 of series 3' Gc s1 rerun with the new code
  and this config to step 1500 gives the stored series 3' log exactly (validation, V-sign, losses), and the
  kept step 500 / 1000 weights give the logged V-sign. Passed 2026-11-14.
- Time: Gc stage 2 took 2.6-2.7 h per run at 3 slots in series 3'; stage 1 (20000 steps) is estimated at
  1-1.5 h at 3 slots (15 runs; 5 are copied). 20 runs in 3 slots = 7 rounds, so about 24-28 h from a start at
  about 12:30: expected end 2026-11-15, 12:30-16:30.
  The estimate is refined from the pace of the first runs and noted in docs/overnight.md.

| result (runs of 20 with best 10x7 validation >= 80% after the switch) | reading | next |
|---|---|---|
| >= 5 | the jump is reproducible | reported; the next step is the user's |
| 2-4 | neither | described; the decision is the user's |
| 0-1 | series 3' Gc s1 was chance | part 4 closed |

## Series 6: what decides the jump, the stage 1 weights or the stage 2 seed

Protocol by the user, 2026-11-16, committed before the first run: configs/graph_s6.yaml, scripts/graph_s6.py
(check, run, probe, report), tests/test_graph_s6.py. Series 5 stands as reported (5 of 20, seeds 7 8 9 10 12).

One question: is the jump decided by the stage 1 weights or by the stage 2 seed?

- Stage 2 only, no stage 1 is trained. Stage 2 is the stage 2 of series 5 with no change: the training values of
  configs/graph_s6.yaml equal configs/graph_s5.yaml key by key (tests/test_graph_s6.py), the training code is
  scripts/graph_s3.py run_stage2, not edited. The kept weights (stage2.snapshots) are part of that config and stay.
- Stage 1 weights: the finished series 5 files runs/graph_s5/a_Gc_s<seed>.pt of four seeds, copied (sha256 of
  each file is written to runs/graph_s6/stage1_sha256.json): **8 and 9** (their series 5 run jumped: 92.0, 89.4)
  and **6 and 11** (no jump: 63.4, 64.6). Seeds 6, 8, 9 are the copies of series 2' (a), seed 11 was trained in
  series 5; s8 was retrained on 2026-11-15 and is equal to its copy bit for bit.
- For each of the four: 5 new stage 2 seeds, **101-105**. 20 runs. The stage 2 seed is the one seed
  run_stage2 takes: it seeds torch (CPU and GPU), numpy and python at the start of stage 2 and the batch
  generator, so the batch order of the judge and stage 0 batches and anything else random in stage 2. It does not
  touch the starting weights (loaded from the file), the data, the splits (seed 4100) or the validation sets.
- Reproducibility control, not counted: stage 2 from the weights s8 with the stage 2 seed 8, i.e. the series 5
  run s2_Gc_s8 again, in full. It must equal series 5 bit for bit: the log entry by entry (wall time aside), the
  best point, the weights of the best point, of the last step and of the 18 kept steps, the V pre-activations.
  If it does not, the report says so in its first line and no reading is given before the user has seen it.
- Primary metric: a run jumps if its best 10x7 judge validation after the switch is >= 80% (as series 5).
  Counted per group over the 10 runs of the group: A = weights 8 and 9, B = weights 6 and 11. The four series 5
  runs of these weights and the control are not counted.
- Reading, written before the runs:

| result (jumps of 10 in A = weights 8, 9; of 10 in B = weights 6, 11) | reading | next |
|---|---|---|
| A >= 8 and B <= 2 | the stage 1 weights decide | look for what distinguishes these weights |
| the counts are close: \|A - B\| < 3, i.e. at most 2 | the course of stage 2 decides | change the stage 2 procedure |
| anything else | neither | described; the decision is the user's |

  The two rows cannot both hold. The second row is applied as written also when both counts are low or both high
  (0 and 0, 10 and 9); the report then says so in words next to the row.
- Reported next to it, not part of the reading: V sign kept (>= 90% at the best point), the counts per stage 1
  seed (5 runs each), the early V histograms as Table 2 of series 5.
- Probes (5x5, 10x7): once, after all training; descriptive, never used for selection or for the reading.
- Run conditions: 2 runs at once, GPU memory fraction 0.4 per run, nice -n 10, no start below 12 GB free RAM.
  Series 5 ran 3 at once and measured 42% GPU memory, accepted by the user for that series; CLAUDE.md says 40%,
  and 2 slots keep it (decision of mine; measured alone 49 s per 1000 steps on 5x5 against 125 s at 3 slots, so
  the third slot adds little). The slot count is no part of the training computation (series 2' ran at 6
  slots and the 3-slot retrain was bit-equal); the control tests this again.
- Check before the launch (scripts/graph_s6.py check, scratch directory): the control to step 1500 gives the
  stored series 5 log of s2_Gc_s8 exactly and the kept weights of steps 250, 500, 1000 bit for bit; the same
  weights with the stage 2 seed 101 give the same step 0 validation and differ at every later entry.
  Passed 2026-11-16.
- Checkpoints every 500 steps and the stop command as usual: touch runs/graph_s6/STOP (the run directories
  runs/graph_s6/w<seed> and runs/graph_s6/control see it through a link).
- Order: the control first, then stage 2 seed by stage 2 seed over the four weights (101 for 8, 9, 6, 11, then
  102, ...), so that a stopped series has the groups filled evenly.
- Time: 21 stage 2 runs. Series 5 took 2.3 h per run at 3 slots; at 2 slots estimated 1.6-2.0 h per run, so
  17-21 h from the start. Refined from the pace of the first runs in docs/overnight.md.

## Series 7: does the jump carry over to the middle game (evaluation only)

Protocol by the user, 2026-11-17, committed before the run: configs/graph_s7.yaml, scripts/graph_s7.py (overlap,
eval, report), tests/test_graph_s7.py. No training: finished nets only, read on fixed labelled sets.

One question: do the Gc nets that jumped on the late 10x7 judge positions also choose better moves in the middle game?

- Nets (best point weights, the .pt the probes used; none is selected by anything in this series):
  - **Gc with a jump**: all 9 jump runs of series 6: weights s8 with stage 2 seeds 101-105, s9 with 104 and 105,
    s6 with 103 and 104 (runs/graph_s6/w<W>/s2_Gc_s<T>.pt).
  - **Gc without a jump**: 5 of the 11 no-jump runs of series 6, one or two per weights, fixed here as the lowest
    stage 2 seeds: s9 / 101, s9 / 102, s6 / 101, s6 / 102, s11 / 101 (decision of mine: the user asked for "one or
    two of each weights"; lowest seeds, so no choice by result).
  - **C+h** s1-s5 of series 3' (runs/graph_s3p/s2_Ch_s<seed>.pt; all five passed the V-sign gate there).
  - **Dot-2 without thinking** (the hand: policy argmax over legal lines), dot2c gen 16, recomputed here on the same
    positions with the code of scripts/mid_judge_eval.py nets (parity_hand.evaluator); not taken from old reports.
    The report also says whether the recomputed moves equal the moves stored in runs/mid_judge/nets_mid.json.
  - Gc and C+h move: argmax of the policy over legal lines (graph_hand.correct_rate, as the judge validation and
    the probes of series 3'-6).
- Sets (labels of part 3 (c), runs/mid_judge, configs/mid_judge.yaml; not relabelled):
  1. **mid**: the 1925 solved positions at 10-15 safe lines (labels_mid.jsonl). Metric: share of correct moves
     (a move that keeps the best result) on the **decisive** positions (not every move has the same result: 721).
     All 1925 reported alongside.
  2. **chart**: the solved positions of labels_chart.jsonl at 0-18 safe lines, the same per safe-line buckets as
     docs/who_knows.png (a point is drawn where a bucket has >= 10 decisive positions). Descriptive.
- Overlap check, before the reading and written in the report: the mid and chart positions against every position
  the nets were trained or selected on: the judge's 10x7 train, validation and probe (data/parity_hand via
  configs/parity_hand.yaml, the split of series 3'-6) and the stage 0 10x7 train and validation (data.npz of
  series 2, copied into series 3', 5 and 6; sha256 compared). Key: drawn lines only and the full 17-plane encoding,
  minimum over the 4 grid symmetries (as scripts/graph_s3p_checks.py). 5x5 positions cannot coincide with 10x7
  ones. Dot-2 is not in this check: these positions come from its own self-play games (MCTS-200), so they are
  its own distribution; that is a property of the comparison, said in the report.
  If any position is shared, the report says so and gives the reading on the set without the shared positions
  as well (the reading itself is taken on the full set as fixed here).
- Primary, written before the run: **D = mean accuracy of the 9 jump nets on the decisive mid positions minus the
  accuracy of Dot-2 without thinking on the same positions.** Bootstrap over positions and nets: 10000 resamples,
  positions with replacement (the same draw for the nets and for Dot-2), the 9 nets with replacement, seed fixed in
  the config; two-sided 95% percentile interval.

| result | reading |
|---|---|
| D > 0 and the interval's lower end > 0 | the jump carries over to the middle game |
| the interval contains 0 | no transfer to the middle game |
| the interval's upper end < 0 | no transfer; Gc with a jump is below Dot-2 without thinking (said in words) |

- Alongside, the same difference and interval, not part of the reading: Gc without a jump (5 nets) - Dot-2,
  C+h (5 nets) - Dot-2. Reference only (user): MCTS-200 96.0%, intuition 90.7%; docs/mid_judge.md gives 95.6% and
  91.0% on the 721 decisive positions; the report shows both from the stored moves of runs/mid_judge/nets_mid.json
  on exactly these positions (MCTS and intuition are not rerun).
- Report: docs/graph_s7.md: table of nets, the primary with its row, the two other differences, mid by safe lines,
  chart by safe lines, and the map "who knows what" with a new line for Gc with a jump (mean of the 9 nets):
  docs/who_knows_s7.png (docs/who_knows.png stays as part 3 made it).
- Load: inference only, one process (the user allowed 2 slots), GPU memory fraction 0.4, nice -n 10. A few minutes.

## Series 9: the graph net plays 5x5 (AlphaZero run g5)

Protocol by the user, 2026-11-17, with the user's amendment of the same day (before the run), committed before the
run: configs/g5.yaml, scripts/graph_s9.py, tests/test_graph_s9.py; the AlphaZero code (scripts/az_train.py,
scripts/az_eval.py, agents/az_net.py) extended as below. Series 8 does not exist (the user's numbering).

Goal: a finished graph network that plays (no deadline).

1. **Gc in az_train and MCTS.** Gc (agents/graph_net.py, configs/graph_gc.yaml: 8 layers, width 50, pool 16,
   segments; 0.50M parameters) already returns (policy over lines, V, score) like AZNet, so the Evaluator
   (batched), the Python and C trees, self-play, gating and the ladder take it unchanged. Additions:
   - `net.compile: true`: the Evaluator runs the graph net through torch.compile in fp32 on CUDA instead of eager
     fp16 autocast. Same net, same weights; measured on a batch of 192 5x5 positions 4.9 ms against 10.9 ms (eager
     fp16) and 11.4 ms (eager fp32); compiled outputs equal eager fp32 to 1.5e-7 (eager fp16 differs from fp32 by
     5e-4). Tested (test_compiled_evaluator_equals_eager_fp32). Batching was already there (192 games per task).
   - Seeds independent of the worker count (user): `selfplay.tasks: 24` splits the 2500 games into 24 tasks
     whatever the number of workers, each with its seed from the self-play generator, and the buffer gets the
     results in task order; `phase_seeds: true` gives every phase its own generator [seed, gen, phase], so a run
     resumed after any phase continues exactly. Tested on CPU (test_resume_and_worker_count: a run stopped after
     two phases and resumed, and a run with 1 worker instead of 2, end with the same weights, buffer and gating
     results bit for bit). On the GPU runs are not bit-reproducible anyway (atomic adds), but the worker count
     does not enter the games' seeds.
   - Python and C trees give the same search on Gc (test_c_tree_matches_python_on_gc, 1 and 8 leaves per tree).
   - 8 symmetries: `train.symmetries` as az5b (a random one of the 8 per sample; the graph is equivariant by
     construction, so it changes nothing for Gc but stays as in az5b). Checkpoint every phase (state, trainer,
     buffer, best, gen_XXXX.pt) and the stop command (touch runs/g5/STOP) as az5b.
   - Optional `limits.gpu_memory_fraction` per process (self-play workers, trainer, evaluator), for the user's
     "2 workers, 40%".
2. **Measured before the run** (2026-11-17, machine otherwise idle; 5x5, MCTS sims [100, 300], 192 games per
   worker; random-init nets, about 8.9k evaluations per game): AZNet (Dot's net) 1 worker 30.7k evals/s; Gc eager
   1 worker 9.1k, 2 workers 12.4k; Gc compiled 6 workers 19.0k evals/s (compilation about 4 min once per worker
   process); 8 workers 19.5k (+2.6%, within noise), so 6 workers stay and leave room for the evaluator. Training 1100 steps: Gc 64 s, AZNet 43 s. A round (2500 games,
   about 22M evaluations, + 1100 steps + 100 gating games) is therefore about 20-25 min at 6 workers and about
   35-40 min at 2 workers; well under 2 h. The amendment removes the 2 h gate anyway (run at any round time).
3. **Run g5**: Gc from scratch, configs/az5b.yaml unchanged except the net and the plumbing above. Gating 55 of 100.
   Stop: **6 rejected generations in a row only** (amendment: no 48 h limit, no Elo stop); replaced during round 16
   by the plateau rule below (second amendment). Changes against az5b,
   each in configs/g5.yaml and checked key by key by test_config_is_az5b: net (Gc, compile), no init_from,
   run_dir, selfplay.workers (6 today, user) and selfplay.tasks, phase_seeds, stop, eval.opponents (+ easy),
   probes: false (the az5 probe maps), update_best: false (runs/best, the app's net, is not touched), after_eval
   (probe + report), limits, bridge_eval, series9. Note: az5b was a continuation of az5 (lr 0.01 = half of az5's,
   score head, sims [100, 300]); g5 starts from scratch with these values, as the user asked ("az5b unchanged
   except the net").
4. **Evaluation of every accepted generation** (scripts/az_eval.py in the background, as az5b): 100 games per
   opponent with MCTS 200 and without thinking (policy argmax) against random, greedy, easy, medium, hard, strong
   (node budget 23000 as az5b); parity probe 5x5 on the 1000 positions of docs/parity_probe_positions.json
   (policy argmax, V sign = V child balanced, MCTS 200), as docs/parity_curves.md (scripts/graph_s9.py probe).
5. **Comparison, fixed now** (descriptive; no pass / fail row was given):
   - Dot = az5 (rounds 1-31) then az5b (rounds 32-119); round = generation (accepted or not), as in az_train.
     Dot's values come from its stored evaluations (runs/az5/eval, runs/az5b/eval) and parity curves
     (runs/parity_probe/curves.json).
   - By rounds: the first round whose accepted generation has Hard >= 90% with MCTS 200; the first round with V sign
     on the probe >= 75%. The user quotes Dot 17 and 8; the stored data give round 11 for both (az5 gen 11: MCTS vs
     hard 91%, V child 76.3%). Both are reported.
   - By time: hours of training = summed self-play + train + gate seconds of the generations (scripts/elo_vs_time.py's
     definition; evaluations run alongside and are not counted). Elo of the last accepted, evaluated generation
     within 12, 24, 48 h, and the best Elo so far, against Dot at the same hours. Dot trained 28.4 h in total
     (az5 11.4 h + az5b 17.0 h), so at 48 h Dot's value is its final one (said in the table). Elo here on the
     opponents of both ladders (random, greedy, medium, hard, strong: Dot's ladder had no easy); the g5 ladder table
     (docs/g5_ladder.md) also has the Elo with easy.
   - Best g5 = the accepted generation with the highest MCTS Elo (common opponents); a tie goes to the later one.
   - Matches, best g5 against Dot gen 54 (runs/az5b/gen_0054.pt), 200 games each, first mover alternating, the
     first 4 lines at temperature 1 (gating's temp_moves; otherwise two nets without thinking would play two games),
     then the best move: (a) MCTS 200 both; (b) without thinking both (amendment); (c) equal time per move: Dot 200
     sims, g5 the sims that take the same mean time per move in single-game search (leaves_per_tree 8, as the app)
     on 200 probe positions, measured on the idle machine (one correction step).
   - 20 games over the bridge against the real KSquares Hard (amendment; scripts/az_vs_hard.py, MCTS 200, seed 11,
     as Dot's 18/20).
6. **App** (amendment): the best g5 goes into the app as a 5x5 net whatever its strength (net list, thoughts
   panel, net-vs-net mode g5 against Dot), with an honest caption of its ladder strength; screenshot; a GitHub
   release with the weights.
- **Amendment to the stop rule** (user, 2026-11-17 20:25, written during round 16, before any plateau was in sight;
  rounds 1-15 done, best MCTS Elo 1816): the "6 rejects in a row" rule never fired in Dot's 117 rounds (longest
  streak 5), so the run stops normally when both hold: (1) at least 117 rounds done (as many as Dot, so that the
  comparison at equal rounds is complete); (2) the best MCTS ladder Elo (common opponents, the best-g5 definition)
  grew by at most 20 over the last 20 rounds. Then the protocol goes on (matches, bridge, app, release). 6 rejects
  in a row stays as a fallback. Hard cap: 250 rounds; if reached, stop and write to the user. Best g5 is still the
  generation with the highest MCTS Elo. Implementation (configs/g5.yaml stop: min_rounds 117, plateau_window 20,
  plateau_gain 20, max_rounds 250, elo_opponents; az_train Run.plateau / plateau_rule, tested by
  test_plateau_rule): the evaluation runs behind training, so the rule is judged at R = the last round up to which
  every accepted generation is evaluated (usually the current round or the one before it): R >= 117 and
  best Elo up to R - best Elo up to R-20 <= 20. The trainer checks it before each round, so the round in progress
  when it fires is not started. The amendment took effect by a safe stop and an exact resume.
- **Second amendment** (user, 2026-11-18 15:55, written during round 61; rounds 1-60 done, best MCTS Elo 2127 at
  gen 45, gens 58 and 60 lower with MCTS: 1999, 1954, while raw Elo kept rising to 1861):
  - Final matches: besides the best generation by MCTS Elo, also the best by Elo without thinking and the last
    accepted generation; each plays Dot gen 54 in the same three modes (MCTS 200 both, without thinking both, equal
    time per move), 200 games each. A generation that is several picks plays once (labels joined). The equal-time
    sims are calibrated once (all picks are the same net). Best g5 for the app is still the highest MCTS Elo.
  - Gens 45-60 (accepted ones) are evaluated again on the same ladder with 300 games per cell instead of 100 (raw and
    MCTS 200, all six opponents, fresh seeds [seed, gen, 7, 300]), in the background next to the training, to tell
    the drop with MCTS from noise (`scripts/graph_s9.py recheck`, runs/g5/eval300, a report table with 95 %
    parametric bootstrap intervals). Decided here: for picking the match generations (and the app's best g5) a
    generation's Elo is its 300-games value where it exists, else the 100-games one; the stop rule keeps the
    100-games evaluations of the running evaluator (rule fixed in the first amendment).
- **Third amendment** (user, 2026-11-18 20:10, during round 70): diagnosis of the MCTS drop (the 300-games
  re-evaluation showed it is real: MCTS Elo 2072-2122 at gens 45-51, 1961-2036 at gens 54-60, raw flat at ~1850).
  Description only; the series' verdict does not change. Gens 45, 49, 54, 60 and the last accepted one (when the
  diagnosis reaches it): (1) entropy of the raw policy and its mass on the top move on 1000 self-play positions,
  the same for every generation; (2) MCTS 50 / 200 / 800 against strong, 200 games each: does strength grow with
  the number of simulations at the late generations as at gen 49; (3) V sign accuracy on the parity probe (stored)
  and on 1000 middle-game self-play positions. Decided here: positions drawn once (seed 13) from the replay buffer
  after round 70 (games of the nets of rounds ~67-70), middle game = 20-40 of 60 lines drawn; the V sign truth on
  self-play positions is the game's result for the mover (not an exact value; the probe has exact labels). Run in
  the background under nice with 3 opponent processes; the load is set so that a self-play round slows by at most
  20 % (baseline 1350-1420 s; the 300-games re-evaluation with 6 processes cost about 20 %). A separate section of
  docs/graph_s9.md (`scripts/graph_s9.py diag`, runs/g5/diag).
- Load: today (user) 6 self-play workers, or 8 if measured faster with memory to spare; no GPU memory limit, no
  nice. When the user says "2 workers, 40%": stop at the next phase (runs/g5/STOP), set workers 2 and
  gpu_memory_fraction per process so that the sum stays <= 40%, nice -n 10, resume (exact by phase_seeds). Every
  change goes into series9.slots of configs/g5.yaml and the report.
- Report: docs/graph_s9.md (rebuilt after every evaluation: rounds table, curves next to Dot in
  docs/graph_s9.png, the comparisons as they become available); a daily summary to the user with the curve by
  rounds next to Dot's.

## Series 10: was Dot gen 54 the best Dot? (matches only)

User, 2026-11-19, after series 9 (where the ladder's best g5, gen 49, lost to Dot gen 54 while the late gens 112 / 114
beat it). Question: was Dot gen 54 the best Dot, or did the ladder err there too. No training.
- Matches, 200 games each, first mover alternating, the first 4 lines at temperature 1 (nets), then the best move;
  modes MCTS 200 both, without thinking both, equal time per move:
  1. Dot's last accepted generation (az5b gen 84, round 114) against Dot gen 54.
  2. The Dot with the best Elo without thinking (az5b gen 52, round 82: 1724 on the common opponents) against gen 54.
  3. g5 gen 114 against the strongest Dot of items 1-2.
  4. g5 gen 114 against g5 gen 49, for completeness.
- Decided here: equal time differs only between g5 and Dot (g5 58 sims = Dot 200, the series 9 calibration on this
  machine); between two nets of one architecture (items 1, 2, 4) it is MCTS 200 both and is not played again. The
  "strongest Dot by items 1-2" = a candidate whose MCTS-200 score against gen 54 has its 95 % Wilson interval above
  50 % (of two, the higher score); none -> gen 54 (then item 3 replays series 9's match with a new seed). "Beats"
  below means the MCTS 200 mode (the ladder's and the app's mode); the other modes are described.
- Reading, fixed now: if the late Dot (item 1) beats gen 54 (interval above 50 %), the graph vs convolution
  comparison is counted by item 3 and part 1's "Dot did not grow after round 54" is marked a measurement error. If
  not, the comparison stays as in series 9 (if only the best-raw Dot of item 2 beats gen 54, item 3 is played against
  it and reported, but part 1's note stands).
- Descriptive, separate section: round robin of 6 players (g5 49, g5 114, Dot 54, Dot 84; Hard and Strong as on the
  ladder), MCTS 200 for the nets, 100 games per pair: a table, to show that strength here is not one line.
- Seeds [10, item, mode]; scripts/graph_s10.py, configs/graph_s10.yaml, tests/test_graph_s10.py; docs/graph_s10.md.

## Series 11: is dot2c gen 16 the strongest 10x7 Dot? (matches only)

User, 2026-11-19, after series 10 (the ladder picked a weaker Dot on 5x5). dot2c gen 16, the app's 10x7 network, was
picked by the 10x7 ladder. Matches, 200 games each, MCTS 200 both, first mover alternating, the first 4 lines at
temperature 1: gen 16 against dot2c's accepted neighbours 14, 10, 9 (16 is dot2c's last accepted generation, so
all neighbours are earlier) and against dot2s gen 25 (dot2s best by the ladder) and dot2s gen 39 (dot2s last
accepted, the app's release network); "the best dot2s" is ambiguous after series 10, so both are played.
Reading, fixed now: a challenger whose score against gen 16 has its 95 % Wilson interval above 50 % is stronger;
of several, the highest score. If one is stronger, it replaces gen 16 as the app's 10x7 network and README notes
it; otherwise gen 16 stays. scripts/graph_s11.py, configs/graph_s11.yaml, tests/test_graph_s11.py;
docs/graph_s11.md.

## Series 12: the same judge answers, graph net against convolution (fine-tune on the middle game)

User, 2026-11-20: the experiment promised in the part 3 video ("the same eighty judge answers: will the graph net learn
what the convolution did not"). Protocol fixed here and committed before any run; a change = a new series. The voice
of the part 4 video's scene 6 waits for the result.
- Data, as mid_hand of part 3 plus the 80 mistakes: set mid (runs/mid_judge/labels_mid.jsonl, 1925 solved positions
  at 10-15 safe lines) and the 80 mistakes of runs/rout_mistakes. Splits are the existing ones, unchanged and the same
  for every net: mid by game, 70/30 (scripts/mid_hand.py `mistakes`, seed 3031: 1387 train / 538 held out, 199 of
  them decisive); the mistakes 56 / 24 (scripts/rout_mistakes.py `mistakes`, seed 3031; one mistake per rout game, so
  this split is by game too). The held-out part takes no part in training or in any choice (no early stopping, no
  checkpoint selection: the weights after the last step are evaluated).
- Overlap, checked and written in the report before the reading: held-out positions (mid and mistakes) against the
  fine-tune train positions (mid train + mistakes train), the dot2c replay buffers mixed into the batches (10x7) and
  the nets' own earlier training / selection sets (series 6 and 3' stage 0 and judge sets, as series 7); key = drawn
  lines, minimum over the grid symmetries; game ids of held-out and train disjoint. If a held-out position is shared,
  the reading is also given without the shared positions, and that version is the one read.
- Nets, 10x7: **Dot-G** = Gc of series 6 from the s8 weights, stage 2 seeds 101, 102, 103 (runs/graph_s6/w8/
  s2_Gc_s10{1,2,3}.pt, all three jump); **Dot-2** = dot2c gen 16; **C+h** = series 3' s1-s3 (runs/graph_s3p/
  s2_Ch_s{1,2,3}.pt), the control "convolution with the hint".
- Fine-tune, the same for every net, hyperparameters of Dot-2's mid_hand r30 (configs/mid_hand.yaml), not tuned for
  Dot-G: 4000 steps, Adam lr 0.0005, batch 384, of it 30 % judge positions (mid train and mistakes train pooled, drawn
  uniformly; loss -log policy mass on the judge's best moves) and 70 % dot2c replay positions (5x5 / 7x7 / 10x7 at
  0.1 / 0.3 / 0.6, loss cross-entropy to their MCTS policy), grid symmetries on, no clip, no schedule. The same seed
  for every net (3031), so every net sees the same batches in the same order; BatchNorm in eval mode (statistics as
  trained). Two variants:
  **(a) frozen**: only the move-choice (policy) head learns: AZNet2 (Dot-2, C+h) p_conv, g_conv, p_g, p_out (7361
  parameters); Gc p_out (5101). Body, V and score frozen; checked after training: those tensors unchanged and V bit
  for bit equal on the held-out positions. The plan's "heads within 10 % in parameter count" does not hold (Gc's
  head is 69 % of AZNet2's); not changed, because the series fine-tunes the nets as they are; written in the report.
  **(b) lr0.1**: every parameter learns, the policy head at lr 0.0005, the rest (body, V, score) at 0.00005. Decided
  here, against series 4's plan: no V loss is kept, because the V heads have different targets (Dot-2: the game's
  result from self-play; Gc and C+h: the stage 0 parity sign), so no V loss could be the same for all; the loss is
  the one of (a).
- Metric: share of correct moves without thinking (policy argmax over legal lines keeps the judge's best result) on
  the **199 held-out decisive mid positions**, before and after. Descriptive: all 538 held-out mid positions, the 24
  held-out mistakes, train accuracy.
- Reading, fixed now, for each variant separately (two rows; if they differ, both are reported, no joint verdict):
  1. Primary: D = mean of the 3 Dot-G after - Dot-2 after on the same 199 positions; bootstrap over positions (the
     same draw for both) and over the Dot-G nets, 10000 resamples, seed 12, two-sided 95 % percentile interval.
     Interval above 0: "the graph net learns the middle game better"; contains 0: "no difference seen"; below 0:
     "worse".
  2. Gain of each net group against itself before (Dot-G 3 nets, Dot-2, C+h 3 nets): after - before, positions and
     nets resampled (paired), the same interval.
  3. Reference on the same positions: Dot-2 with MCTS 200 (the stored moves of runs/mid_judge/nets_mid.json, part 3
     (c)). Descriptive: C+h after against Dot-2 after and against Dot-G after, the same interval.
- Caveat for the report: the positions come from Dot-2's own games (self-play, part 3 (c); the mistakes from its games
  against Hard), and the 70 % replay share is Dot-2's own MCTS targets: for Dot-2 an anchor, for Dot-G and C+h new
  data. Both favour Dot-2.
- Code check before the run: the trainer with Dot-2, variant (a) and mid only (no mistakes) must reproduce
  runs/mid_hand/r30.pt (same held-out moves; weights equal or within GPU nondeterminism, written).
- Resources: one process, GPU memory <= 40 %, nice -n 10, in tmux. scripts/graph_s12.py, configs/graph_s12.yaml,
  tests/test_graph_s12.py; docs/graph_s12.md; rows for scene 6 in docs/video_claims_part4.md by the result.

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
