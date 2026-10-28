## Conclusions (2026-10-28 morning)

**Answer to the goal question: partly.** What transfers from 5x5 to 10x7
without training is the *policy*: local tactics and the ordinary
"keep playing safe lines" instinct. What does not transfer is the *value*,
i.e. counting the chain parity of a bigger board.

1. **The policy transfers and is above Medium.** Dot-2 without search on
   10x7: Elo 1369 (medium 1237, hard 1420). It beats greedy/easy 96%,
   medium 47%, hard 37%, strong and strong4 62-64% (strong is weak on
   10x7, see the ladder). 7x7: Elo 1551 (raw), level with Hard (51%).
   The untrained Dot-2 loses every game to greedy: the gain comes from the
   5x5 training, not from the architecture.
2. **The value head does not transfer, and search makes things worse.**
   Value after the move: 5x5 88% (wins 86 / losses 89), 7x7 73% (49 / 97),
   10x7 67% (35 / 99). On the bigger boards the value head calls almost
   every move a loss: it sees losing moves, not winning ones. MCTS, which
   trusts V, is then below the raw policy against everything stronger than
   medium (10x7: 47% vs hard, 40% vs strong, 39% vs strong4; raw 37 / 62 /
   64%). MCTS Elo 1322 < raw 1369.
3. **Parity probe: no sign of parity understanding on 10x7 beyond
   heuristics.** On positions with 1-2 safe lines the policy is right in
   69%, but greedy and Hard (no parity knowledge, "play the safe line")
   score 66% and 64%. On 5x5, the same kind of positions: policy 68%,
   MCTS 97%. On 10x7 MCTS 67% (search adds nothing). 7x7 lies between
   (policy 57%, MCTS 62%; Hard 53%, strong 93%).
4. **Against the real Hard over the bridge (10x7):** 3-17 with 8 leaves
   per round (the app's search), 6-14 with 1 leaf. Virtual loss with a
   pessimistic V drives the search away from good moves; the simulator gives
   the same (1/12 and 4/12 in the same protocol), so it is the search, not
   the bridge. In batched games (1 leaf) vs the sim Hard: 47% of 200.
   7x7, 8 leaves: 9-11.
5. **Caveats.** The 10x7 parity set is 304 positions (the judge solves few
   parity positions on 10x7: 15 s per position, 1-2 safe lines only); its
   positions come from games of ladder agents, not of the net. The judge's
   time limit ran on a loaded machine, so the set leans to positions that
   solve fast. strong at 14.4k nodes is not a top anchor on 10x7 (ties Hard);
   strong4 is only a little above Hard (66%).

What this means for (c): training on 10x7 should above all calibrate V.
dot2c's first generation with 10x7 self-play (gen 9) already gives MCTS
90% vs hard and 95% vs strong (Elo 1682 vs ~1490 before), while its raw
policy stays where it was (Elo ~1430-1460).
