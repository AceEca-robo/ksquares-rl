## Conclusions (2026-11-03, stopped before item 3 was finished)

1. **Correct.** C == Python reference == exact search on small boards
   (tests); on 10x7 every move's result sign equal on 5000/5000 late
   judge-set positions, median 0.03 s per position for all moves.
2. **Reach.** Solves (every move labelled) all positions with up to 10
   safe lines in seconds; at 15: 72% / 75% / 88% within 10 s / 60 s /
   10 min; at 20: 5% / 20% / 50% (solved ones take 4-10 min); at 30:
   nothing. Time grows roughly tenfold per 5 safe lines past 10. So the
   judge covers the late middle game (10-15 lines), not 20-40.
3. **Chart set** (0-18 safe lines, 1510 solved positions, 630 decisive):
   on decisive positions hand 89%, intuition 93%, MCTS-200 96%, MCTS-2000
   96.5%; the hand drops to 53-85% at 11-18 safe lines where MCTS stays
   at 86-100%. Formula judge: right whenever it answers, but on self-play
   positions it answers only at 0-1 safe lines (+1 ply needs every child
   settled). Exact Python judge (5 s): reaches ~8 safe lines.
4. **Not done:** item 3 (2000 positions at 10-15 safe lines, MCTS-2000 and
   hand checked on them) was stopped at 1400/2000 labelled, before the nets
   were run (runs/mid_judge/labels_mid.jsonl; `label --set mid` resumes).
