## Outcome (2026-11-19)

- The run stopped by the plateau rule (first amendment) after round 117 (16:27): best MCTS Elo of the running
  100-games evaluation 2127 (gen 45), +0 over the last 20 rounds. 117 rounds, 67 accepted, 50.8 h of training
  (self-play + train + gate), 6 workers throughout, no other load change.
- By rounds, g5 reached both milestones first: Hard >= 90 % with MCTS at round 6 (Dot 11 in the data, 17 as the user
  quoted), V sign >= 75 % at round 10 (Dot 11; user: 8). By hours: 12 h 1966 / 1673 against Dot 1870 / 1614;
  24 h 2096 / 1856 against 2151 / 1713; 48 h 2077 / 1825 against Dot's final 2005 / 1684 (best 2164).
- Ladder: MCTS Elo rose to about 2100-2120 by gens 45-51 (300-games re-evaluation), then settled at about 1950-2050;
  raw Elo rose to about 1850 and stayed (Dot's best raw 1724). The drop with MCTS on the ladder is real (third
  table) and is not explained by the diagnosis: the policy sharpens a little, V gets better, the gain from more
  simulations against strong gets smaller at gens 54-60.
- Best g5 by the protocol (highest MCTS Elo, 300-games values where they exist) = gen 49. Against the real
  KSquares Hard over the bridge it won 20 of 20 (Dot 18 of 20).
- Head to head against Dot gen 54 the ladder order does not hold: gen 49 loses with MCTS 200 (40 %) and at equal
  time (31 %; g5 needs about 3x the time per simulation, 58 sims = Dot's 200), and wins only without thinking
  (61.5 %). The late generations beat Dot in every mode: gen 112 74 / 78.5 / 65 %, gen 114 68 / 84.5 / 61.5 %
  (MCTS 200 / without thinking / equal time; 200 games, about +-7 points). The ladder's opponents (scripted agents up
  to strong) and Dot rank the late g5 generations differently; the "MCTS drop" is a drop against the ladder, not
  against Dot.
- App (amendment, whatever the strength): g5 gen 49 is a release network (configs/app.yaml release_generations,
  v0.4), offered in "play" and "vs KSquares", every accepted g5 generation in "network vs itself"; the menu shows
  its ladder caption, the bridge and the head-to-head result. Screenshot: docs/app_g5_vs_dot.png (g5 gen 49 against
  az5b gen 54, thoughts panel of g5). The app runs the graph net eagerly (no torch.compile wait at start).
