## Notes (2026-11-19)

- Item 1 decides the reading: Dot's last accepted generation (az5b gen 84) beats gen 54, 68.0 % [61.2, 74.1] with
  MCTS 200 and 64.5 % without thinking. The best-raw Dot (gen 52) loses to gen 54 (42 %). So the strongest Dot is
  gen 84, ranked low by the ladder (MCTS Elo 2005 against gen 54's 2164). The ladder erred for Dot as for g5. Part 1's
  "Dot did not grow after round 54" is a measurement error (marked in docs/results.md).
- Item 3, the graph vs convolution comparison from now on: g5 gen 114 against Dot gen 84 scores 62.5 % [55.6, 68.9]
  with MCTS 200 both and 73.5 % [67.0, 79.1] without thinking. At equal time per move (g5 58 sims, Dot 200) it scores
  47.5 % [40.7, 54.4]: no difference. With the same number of simulations the graph net is stronger. At the same time
  per move it is level, because one g5 simulation costs about 3x.
- Item 4: g5 gen 114 beats g5 gen 49 93 % with MCTS 200, 66.5 % without thinking.
- Round robin: the four nets do line up, consistently in every pair (g5 114 > Dot 84 > Dot 54 > g5 49; the round
  robin repeats items 1, 3 and series 9 within the noise). What does not line up is the ladder's view of them. Hard
  loses to every net (0-2 %). Strong scores 16-21 % against every net, with g5 gen 49, the weakest net, doing best
  against it (84 %). With 100 games a cell is about +-8 points, so Strong cannot separate the nets at all. The
  ladder's Elo at the top comes from the strong column alone, so it ranks generations by noise and by
  strong-specific play, not by strength against each other. The Bradley-Terry Elo anchored on scripted agents cannot
  rank nets that all beat those agents.
- Not changed by this series: the app's Dot network is still gen 54 (configs/app.yaml best.gen); switching it is
  the user's decision.
