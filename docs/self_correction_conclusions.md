## Conclusions (2026-11-04)

1. **What the judge finds.** 3000 games gave 200 losses to judge (50 per
   arm); a mistake was found in 81 (40%), 118 hit the wall at 60 s. Against
   Hard the found mistakes are mostly late: median 0-1 safe lines (range
   0-12), 38 of 81 lose a *drawn* position, i.e. close losses in the loony
   endgame where MCTS 50 / 200 misplays a capture, not the middle-game
   routs of docs/blowouts.md. In self-play the losses are decided early:
   87 of 100 are walls at 15-25+ safe lines, the 12 found are at 6-25.
   So the cheap 60 s judge mostly harvests a different kind of mistake
   from the one that makes the routs.
2. **Fine-tuning learns the positions, not the skill.** The policy head
   memorises the 73 training mistakes (argmax correct 33% -> 96-100%), on
   the 8 held-out ones 1 -> 3 correct in every variant. V is bitwise
   unchanged (the frozen check).
3. **Evaluation: no variant reduces routs; more mistake weight hurts.**
   The 200-game rout count is noisy: base gives 5-11 on six game sets
   (7.8 per 200 over 1200 games; 11 in part 3 (a) was one of them). Over
   the six sets: m10 7.7 per 200 (= base, losses 75 vs 77), m30 10.8 and
   m50 11.7 (losses 123 / 127, +60%). Raw Elo: m10 1457 (base 1450), m30
   1354, m50 1328; probe policy +2 points (69 vs 67.4%), MCTS 92-93% as
   base; bridge m10 20-0, m30 18-1-1, m50 16-4 with one rout (base 19-0-1).
4. **Item 4 (the self-play loop) not run:** its condition (best variant
   under 6 routs of 200, clearly below base) is not met; m10, the best, is
   indistinguishable from base. With 5-11 routs per 200 for the same net,
   "< 6 of 200" cannot be read from one 200-game match anyway: about 1200
   games per net are needed to see a change of ~3 per 200.
5. Next, if at all: the targets must be the middle-game mistakes behind
   the routs (docs/blowouts.md: lines 70-83, 7-20 safe lines, minutes of
   judge time each), many more of them, and a check on held-out positions
   before playing games.
