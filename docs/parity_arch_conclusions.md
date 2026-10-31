## Conclusions (2026-10-31)

**No.** Neither architecture arm reaches the plan's criteria (raw 10x7 probe
>= 80% *and* raw Elo >= 1520 *and* routs < 3%); both play worse than Dot-2
with search.

1. **The teacher.** MCTS-2000 of Dot-2 is right in 94.4% [91-96] of the
   held-out late positions (MCTS-200: 92.4%) and 89.6% of the judge's
   validation split. Ten times the search buys ~2 points late; in the middle
   game it cannot be checked, and it is a soft teacher there: the raw
   policy agrees with its move in ~33% before and after fine-tuning, for
   all three nets (many near-equal moves; nothing learnable showed up).
2. **The hint works late, not in the middle.** b2 (chain planes) has the
   best raw parity probe of part 3 on 10x7: 82.6% [78-86] (base 67.4%, a2
   87.5% but on a whole-net fine-tune); b1 (pooling everywhere, 96 ch)
   78.0%. On 5x5 b2 62.5%, b1 56.3% (base 55.1%). Middle-game agreement
   with the teacher does not move, V sign vs the teacher 70-74% (base 68%).
3. **Search gets worse.** On the probe, MCTS-200 drops from 92.4% to 86.8%
   (b2) and 75.7% (b1): V after a winning move is right in only 20-23%
   (base 77%), the value heads became pessimistic. Hence routs at MCTS 50:
   base 11, b1 42, b2 46 of 200; bridge 15-5 for both (4 routs each; base
   19-0-1). Raw Elo: 10x7 1418 / 1434 (base 1450), 5x5 1563 / 1670 (1695).
4. **Self-play did not repair it.** 10 rounds per arm, no candidate passed
   the 55% gate (b1 18-48%, b2 21-46%), so the final nets are the
   fine-tuned ones. dot2c itself after gen 16 also failed every gate
   (51, 18, 49, 19, 47, 32%): this net sits on a plateau where 1000 games
   per round do not move it.
5. **BatchNorm cost two tries** (log of 2026-10-31): fine-tuning in eval
   mode gave weights that self-play training (train mode) could not keep
   (candidate 19-81 vs its start); training in train mode with separate
   batches let the net read "this batch is late positions" from the batch
   statistics (judge loss 0.16, nothing in eval mode). What worked: one
   mixed batch per step and board size, statistics recomputed on
   self-play positions before and after.
6. **For later.** The value damage, not the policy, is what costs games
   here: the judge / teacher fine-tune should keep V frozen (a1-style for
   V) or weight the self-play value loss much higher. And the middle-game
   signal needs a teacher that is actually right there (deeper exact
   search on smaller boards, or Dot-2 MCTS at 20k+ with a check of its
   stability), not one that agrees with the student a third of the time.
