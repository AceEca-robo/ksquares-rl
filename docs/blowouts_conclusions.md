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
3. **Wall: 8 of 17 after the 600 s rerun.** At 30 s the judge stopped in
   11 games; at 600 s those positions took 19-82 s (seed 144: 503 s). With 600 s from the wall backwards three more mistakes are
   found (seeds 47, 102, 156: won -> lost with one move at 13-20 safe
   lines, exactly the 30 s wall position). The other 8 move the wall back
   by one Dot-2 move each and stop at 13-27 safe lines (lines 68-72): the
   loss is proven there, the mistake is at or before that point. Seeds 131
   and 141, the walls at 8-11 safe lines, now stop at 14 and 13: the
   safe-line count falls by 3-5 over a single move pair there.
4. **Together:** in 9 of 17 routs the decisive mistake is one move from a
   won position, at lines 70-83 with 7-20 safe lines; in the other 8 the
   loss is proven back to 13-27 safe lines. The routs are lost in the
   middle game, not the endgame; then Hard takes the chains. A teacher for
   this range (the judge labels of docs/mid_judge.md at 10-15 lines,
   longer limits for 16-20) is the next step.
