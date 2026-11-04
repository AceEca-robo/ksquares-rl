## Conclusions

1. **Collection.** 80 middle-game mistakes (7-25 safe lines) from 248 routs in 5500 games (target cut from
   150 to 80). The judge finds an in-range mistake in about a third of the routs; 107 routs (43%) end at a
   600 s wall, mostly at 15-25 safe lines, so the in-range set leans to 7-20 safe lines (only 1 at 21-25).
   56 routs were lost later, in the endgame (below 7 safe lines).
2. **Generalisation check: failed by the rule set beforehand** (held-out rise with p < 0.05). The fine-tune
   memorises the training mistakes (12.5% -> 89-100%), held out goes 2/24 -> 5/24 (r10) and 6/24 (r30):
   net +3 / +4 positions, sign test p 0.23 / 0.14. The direction is up, but 24 positions cannot tell it
   from noise; it is not evidence that the hand generalises the middle game.
3. Per the plan, no evaluation games were played (routs on 1200 games and the bridge skipped): the
   conclusion is **"the hand does not generalise the middle game"** at this data size. A larger held-out
   set would need far more judge time (~1 mistake per 70 games, ~6 h per 80).
