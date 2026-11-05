# The hand on the judge's labelled middle game

`scripts/mid_hand.py check`, configs/mid_hand.yaml. Net: Dot-2 (dot2c gen 16). Set mid (docs/mid_judge.md): 1925 solved positions at 10-15 safe lines, split by game: 1387 train, 538 held out (199 decisive). Policy head only, 4000 steps, lr 0.0005, batch 384 with the variant's share of judge positions, the rest dot2c replay positions. Body, value and score frozen: weights unchanged, V bitwise equal on 538 positions.

Accuracy = raw policy argmax keeps the best result. Up / down: held-out positions that became right / wrong; p: one-sided sign test. Rule set beforehand: a variant up with p < 0.05 (then routs on 1200 games and the bridge).

| net | train % | held out % | held out, decisive % | up | down | p |
|---|---:|---:|---:|---:|---:|---:|
| base | 93.4 | 95.2 | 86.9 | | | |
| r10 | 94.1 | 95.7 | 88.4 | 6 | 3 | 0.254 |
| r30 | 95.7 | 96.5 | 90.5 | 11 | 4 | 0.059 |

Generalises: **no**, by the rule. No evaluation games.

## Conclusions

1. The check fails, narrowly: r30 misses the threshold (p 0.059). The direction is up, and on a set this size that is not proof of no effect.
2. Unlike the 80 mistakes, this set is not memorised either: train rises only 2.3 points, so the head was held by its self-play share or lacks capacity for it. Not separated here.
3. Not rerun with other shares or step counts to pass the threshold: that would make the p-value meaningless. A fair next check would be a new held-out set, fixed in advance.
