# Part 4 video (RU): claims check before the storyboard

Rule: CLAUDE.md "Before publishing". Every claim and number of the user's voice text (2026-11-19, ten scenes), its
source file, the value recomputed from the saved logs / results (not copied from a summary), and a status:
**checked** (data and, where a comparison is claimed, an interval that excludes a draw), **observation**
(descriptive, not pre-registered, or no direct comparison; may go in only with a spoken caveat), **not checked**,
or **does not match** (wording stronger than the data or a wrong number; a fix is proposed).

Names: Dot-G = Gc (agents/graph_net.py, configs/graph_gc.yaml: 8 layers, width 50, segments) in every scene; the g5
run uses the same net (configs/g5.yaml). Simple graph net = Ga (same code, `segments: false`). Convolution = C
(AZNet2, as Dot-2). Dot = az5 / az5b; Dot-2 = dot2c gen 16.

Recomputed here: series 1 click widths from runs/graph_s1/{B,R1}_s*.json; stage 1 end state of the series 6 weights
from runs/graph_s5/a_Gc_s{6,8,9,11}.json; match shares and 95 % Wilson intervals from runs/graph_s10/results.json;
equal-time calibration from runs/g5/match.json; exact test of jumps by weights (series 6 table 2).

## Scene 1. Other eyes

| claim | source | value | status |
|---|---|---|---|
| "last time we ended with a promise: give Dot other eyes" | docs/video_script_part3_ru.md, scene 10 | part 3 promised Dot-2 graph eyes *and* a test: "the same 80 judge answers, will it learn what the convolution did not" | observation: the paraphrase is fair, but the 80-answers test was not run in part 4 (see note A) |
| Dot looks through a small window, like at a picture | AZNet / AZNet2: 3x3 convolutions | — | checked (metaphor) |
| we found an error in our own first video | docs/graph_s10.md, README "Correction" | Dot gen 84 beats gen 54 68.0 % [61.2, 74.1] | checked |

## Scene 2. How Dot-G sees

| claim | source | value | status |
|---|---|---|---|
| boxes, lines and each chain as a separate object with a length | agents/graph_net.py: segment nodes, SEG_FEATS (length / 10, loop, chain of 1 / 2 / 3+, ends) | — | checked |

## Scene 3. First lesson: counting

| claim | source | value | status |
|---|---|---|---|
| task: who has to open the first chain, end-game positions | plan_part4.md series 1 (stage 0 target: sign of dots + long chains mod 2, settled positions) | — | checked |
| three nets, each from scratch | graph_s1, graph_s2 (2c), graph_s2p (cC), graph_s3p (stage 1) | — | checked |
| convolution stayed at the coin however long we trained it | graph_s2.md 2c (5 runs), graph_s2p.md table 4a (5), graph_s3p.md C s1 (5); 20000 steps each | 0 of 15 left chance: best 10x7 51.1-54.1 %, 5x5 49-52 % | checked |
| simple graph net (cells and lines, no chain object): the coin too | overnight 2026-11-07 run 1 (1 run); graph_s3p.md Ga stage 1 (5 runs, 20000 steps) | 0 of 5 (+1): 10x7 50.6 % at the end of stage 1, never clicked on 5x5 | checked |
| Dot-G clicks: from coin to almost no errors in a few hundred steps | runs/graph_s1/{B,R1}_s*.json (validation every 250 steps) | last validation <= 60 % to first >= 95 %: 250 steps in 8 runs, 500 in 2 (B s2, R1 s8); after the click 96-99 % | checked |
| ten runs from different starts, the click all ten times | graph_s1.md: B seeds 1-5 5/5, R1 (= B) seeds 6-10 5/5 | 10 of 10 (5x5 >= 95 %); also 10/10 in 2a, 2b, 2' (a) with V-sign | checked |

## Scene 4. School

| claim | source | value | status |
|---|---|---|---|
| shown 10x7 with no example from it | graph_s2.md 2a (5x5 only, 20000 steps) | — | checked |
| "it again answered at random" | graph_s2.md 2a | best 10x7 per run 50.6, 50.6, 50.6, 51.5, 57.2, 60.7, 68.5, 75.0, 76.0, 86.1 %; median of the last values 50.3 % | **does not match**: 4 of 10 were at the coin, the others 57-86 % (wrong, but not random). Fix: "и большая доска ей не далась" or "и на ней снова ошибалась" |
| ten tries, none successful | graph_s2.md 2a | 0 of 10 reached 95 % | checked |
| school of three classes: 5x5 until almost no errors, a short mix, then 10x7 only | graph_s2p.md (a): event 5x5 >= 95 %, mix 1000 steps, then 10x7 | — | checked |
| after the school all ten of ten mastered the big board | graph_s2p.md table 2 | 10x7 click 10/10, last 10x7 100.0 % in all 10 (hold >= 90 % after the click 6/10: brief dips) | checked (against 2a 0/10; 2b vs 2a Fisher one-sided p 0.0016) |
| by the end of the third class Dot-G forgets the small board | graph_s2p.md table 2 | last 5x5 < 90 % in 8 of 10 (36-87 %); 2 kept it | checked, but "forgets" is 8 of 10: say "почти всегда" / "как правило" |
| every fourth task from the small board -> it remembers both | graph_s2pp.md k25 | both sizes >= 90 % at the end 10/10 vs 2/10 without the share (paired exact p 0.0039, not pre-registered); the pre-registered primary (5x5 held >= 90 % at every validation after the mix) not passed: 7/10, p 0.0625; the 3 failures are one dip each and end at 100 % | **observation**: true at the end of training, not throughout, and not the series' primary. "Лечится это просто" is stronger than the data. Fix below |
| "every fourth" = share 0.25 | configs/graph_s2pp.yaml | 0.25 of the batches after the mix | checked |

## Scene 5. From counting to the move

| claim | source | value | status |
|---|---|---|---|
| trained to choose the move on end-games with the judge's answers | graph_s3p / graph_s5 stage 2 (judge positions) | — | checked |
| most nets choose the move about as the ordinary convolution does | graph_s5.md (non-jump Gc probe 10x7 69.7-76.3 %), graph_s3p.md (C 64.8-72.0 %), parity_arch.md (Dot-2 raw 67.4 %) | — | checked (roughly; descriptive) |
| about every fourth makes a jump | graph_s5.md primary | 5 of 20 (25 %, Wilson [11, 47]); with series 3' 6 of 25 | checked |
| then finds the right move in nine end-games of ten, at once, without thinking | graph_s5.md (policy argmax) | jump nets: probe 10x7 90.8-95.4 %, best validation 86.2-92.0 % | checked |
| four nets from the school, each trained **six** times anew | graph_s6.md | **five** runs per weights (stage 2 seeds 101-105); the sixth would be the series 5 run, which the protocol does not count (the weights were picked by it) | **does not match** |
| one jumped six of six, one never, two "two or three of six" | graph_s6.md table 2 | protocol (5 runs): s8 5/5, s11 0/5, s9 2/5, s6 2/5. "6/6, 0/6, 3/6, 2/6" only if the series 5 run is added | **does not match**: use five. Fix below |
| "changing only the order of the tasks" | configs/graph_s6.yaml: stage 2 seed = torch / numpy / python seed of stage 2 | batch order and every other random draw of stage 2; same stage 1 weights | checked ("порядок задач" is a fair simplification) |
| so it is mostly not luck during training but what the net came out of school with | graph_s6.md: protocol primary 7/10 vs 2/10, "neither row, described" | weights matter: 5,2,2,0 of 5, exact test of equal jump rates p 0.015 (after the fact); but the series 5 label does not predict it (s9 "jump" 2/5 = s6 "no jump" 2/5) | **observation**; "в основном" is a quantity the data do not give. Fix below |
| graduates look the same, all count chains without errors | runs/graph_s5/a_Gc_s{6,8,9,11}.json | 10x7 100.0 % in all four; 5x5 at the end of stage 1: 56.6 / 100.0 / 80.4 / 79.5 % (these nets went through the school without the 1/4 share) | **does not match** as said: true on the big board only. Fix: "все без ошибок считают цепочки на большой доске" |
| what differs inside, we do not know | — | — | checked (open) |

## Scene 6. The middle game

| claim | source | value | status |
|---|---|---|---|
| part 3: Dot-2 loses games by one mistake in the middle | video_script_part3_ru.md scene 2, docs/blowouts.md | about routs at MCTS 50: the judge found a one-move mistake in 10 of 17 routs, all at 7-20 safe lines | checked if said as "проигрывает разгромом"; "проигрывает партии" generalises |
| "those very positions from the middle game" | graph_s7.md: mid = 1925 judge-solved positions at 10-15 safe lines from Dot-2 self-play (part 3 (c)) | not the rout mistakes; the part 3 middle-game test positions | wording: "позиции из середины партии, на которых мы проверяли Dot-2 в третьей части" |
| jump nets: 81 % correct moves | graph_s7.md primary | 81.2 % (9 nets, 721 decisive positions) | checked |
| Dot-2 without thinking: 84 | graph_s7.md | 83.6 % | checked |
| the jump did not carry over | graph_s7.md | D = -2.4 points [-4.7, -0.1] | checked; no overlap with training data (0 of 1925) |
| Dot-G learned to solve end-games, not to understand the game | interpretation of the above | — | observation (fine as a sentence of the narrator) |

## Scene 7. Then let it play

| claim | source | value | status |
|---|---|---|---|
| until then Dot-G had not played a game | series 1-7: supervised only | — | checked |
| self-play from scratch on 5x5 "exactly on the same conditions" | configs/g5.yaml vs az5b.yaml, tests/test_graph_s9.py | g5 = az5b settings (score head, sims 100-300, lr 0.01). But Dot's rounds 1-31 were az5 (no score head, 200 sims, lr 0.02); az5b started from az5 gen 29 | **does not match** for "в точности": Dot itself was not trained on one set of conditions. Fix: "на тех же условиях, на которых училась Dot" (drop "в точности") + caveat in scene 7 line about round 11 |
| two days, 117 rounds | graph_s9.md | 117 rounds, 67 accepted, 50.8 h of training (2026-11-17 to 11-19 16:27) | checked |
| beats the built-in opponent confidently from round 6; first Dot needed 11 | graph_s9.md "Comparison by rounds" | Hard >= 90 % with MCTS 200: g5 round 6, Dot round 11 (az5) | **observation**: the data match, but Dot's round 11 was under az5 settings, not g5's; also by hours g5 was not faster (24 h: MCTS Elo 2096 vs Dot 2151) |
| against the real KSquares on hard: 20 of 20 | runs/g5/bridge.json, graph_s9.md | gen 49: 20 wins of 20 (Dot gen 54: 18 of 20) | checked |

## Scene 8. The broken ruler

| claim | source | value | status |
|---|---|---|---|
| strength was measured by the ladder: built-in opponents and the judge | part 1 vocabulary (strong = судья) | — | checked |
| by the ladder Dot-G peaked by round 49, then weakened | graph_s9.md (300-games re-evaluation) | best MCTS Elo gen 49 2122; gen 60 1961; the drop vs the ladder is real | checked |
| late Dot-G against the ladder's best: 93 of 100 | runs/graph_s10/results.json item 4 | g5 114 vs 49, MCTS 200: 186-14, 93.0 % [88.6, 95.8] | checked |
| "there was no decline at all" | as above + 300-games re-evaluation | the decline against the ladder's opponents is real; head to head there is none | **does not match**: say "Спада в силе не было — он был только на лестнице" |
| part 1 said: after round 54 it stopped growing, only learned to beat itself | video_script_ru.md scene 7, README | quoted in README "Correction" | checked |
| that was untrue: late Dot beats gen 54 in two games of three | results.json item 1 | 136-64, 68.0 % [61.2, 74.1] (raw 64.5 % [57.7, 70.8]) | checked |
| the ruler erred: when everyone beats the built-in opponents, the ladder stops telling who is stronger | graph_s10.md round robin | Hard scores 0-2 %, Strong 16-21 % against all four nets; Strong's order of the nets differs from the head-to-head order | checked (descriptive, 100 games per pair). Note: the nets beat the judge too; "когда всех на лестнице обыгрывают все" is more exact |
| now the best net is chosen by matches only | CLAUDE.md, plan_part4.md rules | — | checked |

## Scene 9. The match

| claim | source | value | status |
|---|---|---|---|
| Dot-G after 114 rounds vs the strongest Dot, 84 rounds | results.json item 3; "strongest" = dot_last (gen 84) | — | checked. Note: gen 84 is generation 84 (round 114 of az5b); "прошла 84 раунда" is the same loose wording as in part 1 (gen = round) |
| 200 variants each: 62 of 100 | results.json 3:mcts | 125-75, 62.5 % [55.6, 68.9] | checked ("62" rounds 62.5 down; "шестьдесят две с половиной" or "больше шестидесяти" is exact) |
| both without thinking: 73 of 100 | results.json 3:raw | 147-53, 73.5 % [67.0, 79.1] | checked (same rounding) |
| Dot-G looks three times longer | runs/g5/match.json equal_time | 200 sims, single-game search as in the app: g5 102.8 ms, Dot 33.8 ms per move = 3.04x (this GPU) | checked |
| at equal time Dot-G gets about sixty variants | runs/g5/match.json | 58 sims = 34.8 ms | checked |
| "in this match the strengths are equal" | results.json 3:equal_time | 95-105, 47.5 % [40.7, 54.4] | **does not match** strictly: the interval contains 50 but also Dot up to +9 points; "equal" is not shown. Fix: "и в этом матче разницы не видно" |
| the graph net sees better, the convolution counts faster | the three rows above | — | checked as a summary; "равны" as above |

## Scene 10. Summary

| claim | source | value | status |
|---|---|---|---|
| 1: how a net sees decides what it can learn; Dot-G learned to count chains; the ordinary net from scratch could not | scene 3 | Gc 10/10, C 0/15, Ga 0/5 | checked |
| 2: a new board size via school, repeat the old or it is forgotten | scene 4 | school: 10/10 vs 0/10; repeating: observation (as scene 4) | checked / observation |
| 3: Dot-G learns to play faster | graph_s9.md | by rounds yes (6 vs 11, caveat of scene 7); by hours no (24 h: 2096 vs 2151 MCTS Elo) | **does not match** as general: "за меньшее число раундов" |
| 3: sees the move better | results.json 3:raw | 73.5 % [67, 79] | checked |
| 3: each look costs three times more | match.json | 3.04x | checked |
| 3: at equal time the two nets are equal | results.json 3:equal_time | 47.5 % [40.7, 54.4] | **does not match** strictly: "при равном времени разницы мы не увидели" |
| 4: the ruler can lie, check it by a direct match | series 10 | — | checked |
| both nets are in the app, play each and watch them play each other | configs/app.yaml (best = az5b 84, release g5 114, v0.4); launcher "self" mode (gen_a / gen_b), docs/app_g5_114_vs_dot.png | — | checked |
| code, experiments and the correction to part 1 are on GitHub | README "Correction", commits 5a1973c, e1d3379 | — | checked |

## Notes

A. Part 3 ended with a concrete promise: graph eyes for Dot-2 and the test "the same eighty judge answers, does it
   learn what the convolution did not". Part 4 did not run that test (series 7 evaluated the jump nets on the middle
   game; nobody was trained on the 80 mistakes). Scene 1 paraphrases the promise as "other eyes" and does not say
   the 80-answers test was left out. Not wrong, but a viewer of part 3 may wait for it.
B. Scene 9 counts are shares of 200 games (no ties on 5x5): 62.5 and 73.5 per hundred.
C. Equal time was calibrated on this machine (RTX 4060 Laptop, single-game search with 8 leaves per tree, as in the
   app). On another device the ratio can differ.
D. Data overlap (rule 5): series 7 checked 0 of 1925 mid positions shared with any training / selection set; the
   judge 10x7 sets of series 3 / 3' / 5 / 6 are disjoint from the stage 0 data (graph_s3p.md data checks). Series 10
   and 9 use games, not position sets.
