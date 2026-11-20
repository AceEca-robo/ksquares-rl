### Conclusions of series 12b (2026-11-20)

1. Primary, by the protocol: **(a) head only: no difference seen**, D = -3.0 [-7.5, +1.3]; **(b) all at 0.1x lr: the
   graph net learns the middle game worse**, D = -6.0 [-11.4, -0.8]. The same rows as series 12. With both series
   "not better", the rule fixed before 12b gives the scene 6 sentence: «те же ответы судьи графовой сети помогли не
   больше, чем свёрточной».
2. Series 12's open point (note 4: did Dot-2's replay share hold Dot-G back?) is now checked: without the share Dot-G
   after is 86.9 / 83.9 % against Dot-2's 89.9 / 89.9 %, the same readings. The mixture was not what kept Dot-G behind
   on this test.
3. The examples themselves: in (b) every net learns all 1443 training answers (100 %), Dot-G included, and the
   held-out positions do not follow (Dot-G -0.5 [-4.4, +3.9], Dot-2 +3.0 [-1.5, +7.5]): memorised, not generalised.
   In (a) the head learns them partly: Dot-G 90.6 -> 94.1 %, Dot-2 90.2 -> 98.0 %, C+h 86.3 -> 91.9 %.
4. Gains against themselves: Dot-G +2.5 [-0.7, +5.7] (a), -0.5 (b), not shown; Dot-2 +3.0 in both, not shown; only
   C+h gains (+8.7 [+1.8, +16.9], +10.7 [+2.2, +19.8]), from the lowest start (73.7 %), to below Dot-2.
5. Limits: 199 held-out positions from Dot-2's own games (they favour Dot-2, the caveat of series 12); three Dot-G
   nets from one set of stage 1 weights; one step count, fixed from series 12, not tuned for anyone.
