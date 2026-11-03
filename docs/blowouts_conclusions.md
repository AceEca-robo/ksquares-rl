## Conclusions (2026-11-03)

1. **Reproduced.** Seeds 11-200 replayed as the app plays them give the
   same 17 losses of 190 (seed 121 again 1:69); every loss is a rout
   (1:69 ... 17:53).
2. **Found in 6 of 17.** In each of them the position before Dot-2's
   move is a *win* for Dot-2 and the line it played turns it into a loss
   (one move from won to lost, no slow drift). The mistakes come at lines
   71-83 with 7-18 safe lines left: the late middle game, exactly the
   range where the hand drops (docs/mid_judge.md: 53-85% at 15-18 lines).
   Seed 121: the mistake is at line 79 (16 safe lines), four lines before
   the exact solver first saw the loss (line 83) and while V was still
   positive.
3. **Wall in 11 of 17.** The judge (30 s) proves the loss back to lines
   73-85 and then runs out of time at lines 70-79 with 8-23 safe lines;
   the mistake is there or earlier. The walls at 8 and 11 safe lines
   (seeds 141, 131) are unusual: in self-play the judge solves ~100% at
   that s; why these are harder is not checked yet.
4. So the routs are not lost in the endgame: the net loses a won game
   with one middle-game move, then Hard takes the chains. A teacher for
   this range (the judge labels of docs/mid_judge.md at 10-15 lines,
   longer limits for 16-20) is the next step.
