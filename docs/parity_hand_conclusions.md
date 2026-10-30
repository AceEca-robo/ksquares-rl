## Conclusions (2026-10-30)

**Partly.** The hand can learn to see parity in late positions from the
judge's answers, but that does not fix the routs, which start earlier.

1. **The probe rises.** The policy head alone (a1) goes from 67% to 77% on
   the held-out 10x7 set (greedy 66%, Hard 64%) and from 55% to 65% on
   5x5; it does not reach the 80% criterion. With the body at 0.1x lr (a2)
   the policy is right in 87.5% [83-91] on 10x7 and 75.5% on 5x5 without
   search, which meets the criterion: the body has room for the count once
   it is trained on it. The held-out sets come from the same generator as
   the training sets (other games, same agents, same 1-2 safe lines on
   10x7), so this is generalisation to new positions of the same kind.
2. **Ordinary play: a1 harmless, a2 not quite.** Raw Elo on 10x7: base
   1450, a1 1467, a2 1488 (+38, about the error). On 5x5: 1695 / 1668 /
   1633: a2 loses 62, more than the ~±40 error. a2 also moved V: on 10x7
   V after a winning move is right in 56% (base 77%), so its value head
   became more pessimistic despite the self-play value loss.
3. **Routs do not move.** 200 games vs the simulator's Hard at MCTS 50:
   base 188-12 (11 routs), a1 184-14-2 (10), a2 176-20-4 (14). Over the
   bridge (MCTS 200) all three go 19-1 / 19-0-1, no routs. As the log of
   2026-10-29 showed, the losing mistake is in the middle game, where the
   judge solves nothing on 10x7 and so the training set has no positions
   (median 80 undrawn lines of 157, 1-2 safe lines left).
4. **For (b).** The question is no longer "can a head express parity" (it
   can, at least late), but "can the net see it before the chains are
   fixed". A middle-game signal is needed: (b2)'s chain planes give the
   count directly; for training data of that phase the judge's limit
   would have to be raised by a lot or replaced by the net's own MCTS
   at many simulations as a softer teacher.
