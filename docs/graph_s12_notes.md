## Conclusions (2026-11-20)

1. Primary, read by the protocol for each variant: **(a) head only: no difference seen**, D = -4.2 points [-9.9, +1.5];
   **(b) everything at 0.1x lr: the graph net learns the middle game worse**, D = -6.2 [-10.6, -2.2]. The rows differ,
   so there is no joint verdict. In neither variant does Dot-G after fine-tuning beat Dot-2 after fine-tuning: the
   promise of part 3 ("will the graph net learn what the convolution did not") is answered no on this test.
2. Gains against themselves: Dot-G +0.3 [-5.0, +5.4] in (a) and +2.3 [-2.7, +7.0] in (b), no gain seen in either;
   Dot-2 +2.0 [-2.5, +6.5] in (a) (as mid_hand's r30, a gain not proven) and +6.0 [+2.5, +9.5] in (b); C+h gains most
   from the lowest start, +6.5 [+0.3, +13.6] and +13.4 [+6.4, +21.1], and in (b) ends level with Dot-G (-0.3
   [-4.5, +3.5]). Before any fine-tuning Dot-G and Dot-2 are not told apart on these 199 positions (-2.5 [-7.2, +2.0];
   series 7 had -2.4 [-4.7, -0.1] for the 9 jump nets on all 721 decisive mid positions).
3. Thinking stays far ahead: Dot-2 with MCTS 200 is right on 98.0 % of the same positions; the best fine-tuned net
   without thinking (Dot-2 (b)) 93.0 %.
4. Observation (not pre-registered): for Dot-G the replay share is not an anchor. Its replay policy loss averages
   193-674 over the first 250 steps (cross-entropy to Dot-2's MCTS policy; Dot-2 2.37, C+h 4.3-6.7): Gc's policy, trained in stage 2 only on
   "mass on the judge's best moves", has very large logits and puts almost no mass on Dot-2's moves elsewhere. In (a)
   its 5101-parameter head spends the run on that loss (down to 7-20) and its judge train accuracy does not rise
   (91.1 -> 89.8, 90.4 -> 90.4, 90.2 -> 89.7 %); in (b) it rises 1.3-3.4 points (Dot-2 9.5, C+h 8.6-11). So the test
   measured "the graph net fine-tuned on Dot-2's mixture", which is what the protocol fixed and the caveat names; it
   does not separate "the graph net cannot learn the middle game" from "the mixture pulls it towards Dot-2's
   policy". A check that could tell them apart (a judge-only fine-tune, no replay share, for every net) was not run:
   it would be a new series.
5. The 24 held-out mistakes (descriptive): 8-33 % for every net and state; too few to read. Dot-G is right on more
   of all 80 mistakes before any fine-tuning (32.5-36.2 %) than Dot-2 (11.2 %) or C+h (8.8-11.2 %): the mistakes were Dot-2's own wrong moves,
   so Dot-2's raw policy is wrong there by selection.
6. Data: split by game (mid 542 / 232 games, mistakes 56 / 24 rout games), the held-out part shares no position with
   any training or selection set (0 of 538 + 24, drawn-line key over the symmetries, including 485896 dot2c replay
   positions); no selection on the held-out part (weights after the last step), hyperparameters from Dot-2's mid_hand
   r30 (the trainer reproduces runs/mid_hand/r30.pt bit for bit), not tuned for Dot-G. Accuracy in fp32 on the CPU, as
   part 3: on the GPU one Dot-2 position flips by a logit tie (10.6402 vs 10.6399), which would give 86.4 instead of
   86.9 % before.
