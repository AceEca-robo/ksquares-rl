## Conclusions (2026-11-03)

1. **Exact where it should be.** No safe line left: value, result and
   move 100% (10x7: 2060 positions, 5x5: 1502; double-cross decisions
   10386 / 5780, all right; tests: 333/333 small positions).
2. **An approximation with safe lines left.** A sacrifice made now cuts
   the board differently from F. On the hard judge sets the formula's move
   keeps the result in 70% (10x7) / 57% (5x5); with one real ply first
   (`ply_values`) 84% / 74% — above raw Dot-2 (67%) and a1 (77%) on 10x7,
   below its MCTS (92%).
3. **Earlier than the exact judge on 10x7, not in the middle game.** On
   10x7 the formula answers 95% of positions with 1-2 safe lines (exact
   35%) and 61% at 3-5 (exact 5%); median first answer at 3 safe lines vs
   0. On 5x5 the exact judge reaches further (9 vs 2). Middle game
   (20-40 safe lines): 0% on both the self-play set (5500) and the ladder
   games; the safe phase is never fixed there. Coverage of the middle
   game: none.
4. **Use.** A cheap late teacher for 10x7 positions the exact judge cannot
   reach (1-5 safe lines), not a middle-game teacher. A middle-game signal
   still needs search: real-board search over sacrifices with the formula
   as the s = 0 leaf.
