# Part 4 video (RU): claims check before the storyboard

Rule: CLAUDE.md "Before publishing". Every claim and number of the user's voice text (2026-11-19, ten scenes), its
source file, the value recomputed from the saved logs / results (not copied from a summary), and a status:
**checked** (data and, where a comparison is claimed, an interval that excludes a draw), **observation**
(descriptive, not pre-registered, or no direct comparison; may go in only with a spoken caveat), **not checked**,
or **does not match** (wording stronger than the data or a wrong number; a fix is proposed).

Revision 2 (2026-11-19, after the user accepted every fix): the rows below are for the final text of
docs/video_script_part4_ru.md; the quoted wording is the new one, the old wording is given where it changed.
Every row is now **checked** or **observation**.

Revision 3 (2026-11-20, after series 12): rows for scene 6 from the series 12 result added; the scene 6 sentence
"до этого опыта мы не дошли" no longer matches and has to be replaced. Only scene 6 changes; the voice of scene 6 is
not redone yet (user: not before the result; now it is in).

Revision 4 (2026-11-20, after series 12b): rows for series 12b and the one sentence for scene 6 from both series,
by the rule fixed before 12b.

Names: Dot-G = Gc (agents/graph_net.py, configs/graph_gc.yaml: 8 layers, width 50, segments) in every scene; the g5
run uses the same net (configs/g5.yaml). Simple graph net = Ga (same code, `segments: false`). Convolution = C
(AZNet2, as Dot-2). Dot = az5 / az5b; Dot-2 = dot2c gen 16.

Recomputed here: series 1 click widths from runs/graph_s1/{B,R1}_s*.json; stage 1 end state of the series 6 weights
from runs/graph_s5/a_Gc_s{6,8,9,11}.json; match shares and 95 % Wilson intervals from runs/graph_s10/results.json;
equal-time calibration from runs/g5/match.json; exact test of jumps by weights (series 6 table 2).

## Scene 1. Other eyes

| claim | source | value | status |
|---|---|---|---|
| "last time we ended with a promise: give Dot other eyes" | docs/video_script_part3_ru.md, scene 10 | part 3 promised Dot-2 graph eyes *and* a test: "the same 80 judge answers, will it learn what the convolution did not" | checked: the paraphrase is fair; that the 80-answers test was not run is now said in scene 6 (note A) |
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
| "и большая доска ей не далась" (was: "снова отвечала наугад") | graph_s2.md 2a | best 10x7 per run 50.6, 50.6, 50.6, 51.5, 57.2, 60.7, 68.5, 75.0, 76.0, 86.1 %; median of the last values 50.3 % | checked: none reached 95 % |
| ten tries, none successful | graph_s2.md 2a | 0 of 10 reached 95 % | checked |
| school of three classes: 5x5 until almost no errors, a short mix, then 10x7 only | graph_s2p.md (a): event 5x5 >= 95 %, mix 1000 steps, then 10x7 | — | checked |
| after the school all ten of ten mastered the big board | graph_s2p.md table 2 | 10x7 click 10/10, last 10x7 100.0 % in all 10 (hold >= 90 % after the click 6/10: brief dips) | checked (against 2a 0/10; 2b vs 2a Fisher one-sided p 0.0016) |
| "Dot-G почти всегда забывает маленькую доску" | graph_s2p.md table 2 | last 5x5 < 90 % in 8 of 10 (36-87 %); 2 kept it | checked |
| "Помогает простое средство: … каждую четвёртую задачу давать со старой, маленькой доски. Тогда к концу учёбы она помнит обе доски." | graph_s2pp.md k25 | both sizes >= 90 % at the end 10/10 vs 2/10 without the share (paired exact p 0.0039, not pre-registered); the pre-registered primary (5x5 held >= 90 % at every validation after the mix) not passed: 7/10, p 0.0625; the 3 failures are one dip each and end at 100 % | **observation**: the text now says "к концу учёбы", which is what the data show |
| "every fourth" = share 0.25 | configs/graph_s2pp.yaml | 0.25 of the batches after the mix | checked |

## Scene 5. From counting to the move

| claim | source | value | status |
|---|---|---|---|
| trained to choose the move on end-games with the judge's answers | graph_s3p / graph_s5 stage 2 (judge positions) | — | checked |
| most nets choose the move about as the ordinary convolution does | graph_s5.md (non-jump Gc probe 10x7 69.7-76.3 %), graph_s3p.md (C 64.8-72.0 %), parity_arch.md (Dot-2 raw 67.4 %) | — | checked (roughly; descriptive) |
| about every fourth makes a jump | graph_s5.md primary | 5 of 20 (25 %, Wilson [11, 47]); with series 3' 6 of 25 | checked |
| then finds the right move in nine end-games of ten, at once, without thinking | graph_s5.md (policy argmax) | jump nets: probe 10x7 90.8-95.4 %, best validation 86.2-92.0 % | checked |
| four nets from the school, each trained five times anew | graph_s6.md | five runs per weights (stage 2 seeds 101-105); the series 5 run is not counted | checked |
| one jumped all five times, one never, two "по два раза из пяти" | graph_s6.md table 2 | s8 5/5, s11 0/5, s9 2/5, s6 2/5 | checked |
| "changing only the order of the tasks" | configs/graph_s6.yaml: stage 2 seed = torch / numpy / python seed of stage 2 | batch order and every other random draw of stage 2; same stage 1 weights | checked ("порядок задач" is a fair simplification) |
| "многое зависит не от везения во время учёбы, а от того, какой сеть вышла из школы" | graph_s6.md: protocol primary 7/10 vs 2/10, "neither row, described" | weights matter: 5,2,2,0 of 5, exact test of equal jump rates p 0.015 (after the fact); the series 5 label does not predict it (s9 "jump" 2/5 = s6 "no jump" 2/5) | **observation** ("многое", not "в основном") |
| "все без ошибок считают цепочки на большой доске" | runs/graph_s5/a_Gc_s{6,8,9,11}.json | 10x7 100.0 % in all four (5x5: 56.6 / 100.0 / 80.4 / 79.5 %, not claimed) | checked |
| what differs inside, we do not know | — | — | checked (open) |

## Scene 6. The middle game

| claim | source | value | status |
|---|---|---|---|
| "Dot-2 проигрывает разгромом из-за одной ошибки в середине партии" | video_script_part3_ru.md scene 2, docs/blowouts.md | the judge found a one-move mistake in 10 of 17 routs, all at 7-20 safe lines | checked |
| "позиции из середины партии, на которых проверяли Dot-2 в третьей части" | graph_s7.md: mid = 1925 judge-solved positions at 10-15 safe lines from Dot-2 self-play (part 3 (c), docs/mid_judge.md) | — | checked |
| jump nets: 81 % correct moves | graph_s7.md primary | 81.2 % (9 nets, 721 decisive positions) | checked |
| Dot-2 without thinking: 84 | graph_s7.md | 83.6 % | checked |
| the jump did not carry over | graph_s7.md | D = -2.4 points [-4.7, -0.1] | checked; no overlap with training data (0 of 1925) |
| Dot-G learned to solve end-games, not to understand the game | interpretation of the above | — | observation (fine as a sentence of the narrator) |
| "в прошлый раз мы обещали дать графовой сети те же восемьдесят ответов судьи. До этого опыта мы не дошли: сначала нужно было узнать, видит ли она середину вообще" | video_script_part3_ru.md scene 10; plan_part4.md (no series trains on the 80 mistakes) | — | **does not match after series 12** (2026-11-20): the experiment is now done (docs/graph_s12.md); the sentence has to be replaced by the rows below; the scene 6 voice is to be rewritten from them |

Series 12 (docs/graph_s12.md, runs/graph_s12/eval.json; recomputed by `scripts/graph_s12.py report` from the saved
fine-tuned weights): the promised test, run 2026-11-20 under a protocol committed before the run (33ed4d3). Rows for
the new scene 6 voice; numbers are % correct moves without thinking on the 199 held-out decisive mid positions.

| claim (for the new voice) | source | value | status |
|---|---|---|---|
| we gave Dot-G the same judge answers (middle-game positions and the 80 mistakes) and fine-tuned Dot-2 and the convolution with the hint in exactly the same way | plan_part4.md series 12; graph_s12.md header | 1387 mid + 56 mistakes train, 4000 steps, the same batches for every net, hyperparameters of Dot-2's mid_hand | checked |
| the test positions were never seen in training | graph_s12.md Overlap | split by game; 0 of 538 + 24 held-out positions shared with any training / selection set | checked |
| "графовая сеть не научилась середине лучше свёрточной" (neither variant above Dot-2) | graph_s12.md primary | (a) D -4.2 [-9.9, +1.5]; (b) D -6.2 [-10.6, -2.2] | checked (no interval above 0) |
| "хуже" (Dot-G learns worse than Dot-2) | graph_s12.md primary | only (b) excludes 0; (a) "no difference seen" | checked only with the variant: "when the whole net learns, worse; when only the move choice learns, no difference seen". Unqualified "хуже": does not match |
| Dot-G after: 85 / 87, Dot-2 after: 89 / 93 | graph_s12.md primary | (a) 84.8 vs 88.9; (b) 86.8 vs 93.0 | checked |
| before fine-tuning: Dot-G 84, Dot-2 87 | graph_s12.md gains table | 84.4 vs 86.9; difference -2.5 [-7.2, +2.0] | checked as numbers; "Dot-2 was ahead before" is not shown on these 199 positions (series 7 shows it on 721) |
| Dot-G did not get better from the lesson | graph_s12.md gains | +0.3 [-5.0, +5.4] (a), +2.3 [-2.7, +7.0] (b) | checked as "no gain seen"; "learned nothing" is stronger than the data |
| Dot-2 got better when the whole net learned | graph_s12.md gains | +6.0 [+2.5, +9.5] (b); (a) +2.0 [-2.5, +6.5] not shown | checked for (b) only |
| the convolution with the hint gained the most and caught up with Dot-G | graph_s12.md gains, descriptive | C+h +6.5 [+0.3, +13.6] / +13.4 [+6.4, +21.1]; (b) Dot-G - C+h -0.3 [-4.5, +3.5] | gains: checked; "caught up" = no difference seen, observation |
| with thinking Dot-2 is still far ahead: 98 | graph_s12.md reference | 98.0 % (MCTS 200, stored moves of part 3 (c), 199 of 199 found) | checked (reference, not a test) |
| caveat to be spoken: the positions come from Dot-2's own games, which favours Dot-2 | graph_s12.md Caveat | — | required with the scene 6 sentence |
| (series 12 only) 70 % of each lesson was Dot-2's own self-play targets | graph_s12.md notes 4 | Dot-G's loss on that share 193-674 at the start vs Dot-2 2.4 | no longer needed as a caveat: series 12b without that share gives the same readings (rows below) |

Series 12b (graph_s12.md section "Series 12b", runs/graph_s12b/eval.json; protocol committed before the run, 6b43074):
the same, every lesson judge answers only, 1198 steps (as many judge answers as in series 12).

| claim (for the new voice) | source | value | status |
|---|---|---|---|
| we repeated it with the judge's answers alone, without Dot-2's games | plan_part4.md series 12b | 1198 steps x 384 judge positions, no replay | checked |
| the result is the same: Dot-G is not better | graph_s12.md, 12b primary | (a) D -3.0 [-7.5, +1.3]; (b) D -6.0 [-11.4, -0.8] | checked (no interval above 0); "worse" only for (b), as in series 12 |
| Dot-G after 87 / 84, Dot-2 after 90 / 90 | 12b primary | (a) 86.9 vs 89.9; (b) 83.9 vs 89.9 | checked |
| every net learned the training answers by heart, but did not get better on new positions | 12b gains table | (b): all nets 100 % on the 1443 training answers; held-out gains Dot-G -0.5 [-4.4, +3.9], Dot-2 +3.0 [-1.5, +7.5] | checked for Dot-G and Dot-2 ("no gain seen"); C+h did gain (+10.7 [+2.2, +19.8]), so "no net got better" does not match |
| **scene 6 sentence from both series**: «те же ответы судьи графовой сети помогли не больше, чем свёрточной» | graph_s12.md "Both series"; rule fixed before 12b | 12: -4.2 [-9.9, +1.5], -6.2 [-10.6, -2.2]; 12b: -3.0 [-7.5, +1.3], -6.0 [-11.4, -0.8] | checked (none of the four intervals is above 0); with the spoken caveat about Dot-2's games |

## Scene 7. Then let it play

| claim | source | value | status |
|---|---|---|---|
| until then Dot-G had not played a game | series 1-7: supervised only | — | checked |
| "на тех же условиях, на которых училась Dot. Правда, первые раунды Dot шли на чуть других настройках" | configs/g5.yaml vs az5b.yaml, az5.yaml; tests/test_graph_s9.py | g5 = az5b settings (score head, sims 100-300, lr 0.01); Dot's rounds 1-31 were az5 (no score head, 200 sims, lr 0.02) | checked (the caveat is spoken) |
| two days, 117 rounds | graph_s9.md | 117 rounds, 67 accepted, 50.8 h of training (2026-11-17 to 11-19 16:27) | checked |
| beats the built-in opponent confidently from round 6; first Dot needed 11 | graph_s9.md "Comparison by rounds" | Hard >= 90 % with MCTS 200: g5 round 6, Dot round 11 (az5) | **observation**: the data match; the settings caveat is spoken just before; by hours g5 was not faster (24 h: 2096 vs 2151), and the text claims rounds only |
| against the real KSquares on hard: 20 of 20 | runs/g5/bridge.json, graph_s9.md | gen 49: 20 wins of 20 (Dot gen 54: 18 of 20) | checked |

## Scene 8. The broken ruler

| claim | source | value | status |
|---|---|---|---|
| strength was measured by the ladder: built-in opponents and the judge | part 1 vocabulary (strong = судья) | — | checked |
| by the ladder Dot-G peaked by round 49, then weakened | graph_s9.md (300-games re-evaluation) | best MCTS Elo gen 49 2122; gen 60 1961; the drop vs the ladder is real | checked |
| late Dot-G against the ladder's best: 93 of 100 | runs/graph_s10/results.json item 4 | g5 114 vs 49, MCTS 200: 186-14, 93.0 % [88.6, 95.8] | checked |
| "Спада в силе не было: он был только на лестнице" | as above + 300-games re-evaluation | the decline against the ladder's opponents is real; head to head the late net wins 93 % | checked |
| part 1 said: after round 54 it stopped growing, only learned to beat itself | video_script_ru.md scene 7, README | quoted in README "Correction" | checked |
| that was untrue: late Dot beats gen 54 in two games of three | results.json item 1 | 136-64, 68.0 % [61.2, 74.1] (raw 64.5 % [57.7, 70.8]) | checked |
| the ruler erred: when everyone beats the built-in opponents, the ladder stops telling who is stronger | graph_s10.md round robin | Hard scores 0-2 %, Strong 16-21 % against all four nets; Strong's order of the nets differs from the head-to-head order | checked (descriptive, 100 games per pair). Note: the nets beat the judge too; "когда всех на лестнице обыгрывают все" is more exact |
| now the best net is chosen by matches only | CLAUDE.md, plan_part4.md rules | — | checked |

## Scene 9. The match

| claim | source | value | status |
|---|---|---|---|
| Dot-G after 114 rounds vs the strongest Dot, 84 rounds | results.json item 3; "strongest" = dot_last (gen 84) | — | checked. Note: gen 84 is generation 84 (round 114 of az5b); "прошла 84 раунда" is the same loose wording as in part 1 (gen = round) |
| 200 variants each: "шестьдесят две с половиной из ста" | results.json 3:mcts | 125-75, 62.5 % [55.6, 68.9] | checked |
| both without thinking: "семьдесят три с половиной" | results.json 3:raw | 147-53, 73.5 % [67.0, 79.1] | checked |
| Dot-G looks three times longer | runs/g5/match.json equal_time | 200 sims, single-game search as in the app: g5 102.8 ms, Dot 33.8 ms per move = 3.04x (this GPU) | checked |
| at equal time Dot-G gets about sixty variants | runs/g5/match.json | 58 sims = 34.8 ms | checked |
| "И в этом матче разницы не видно" | results.json 3:equal_time | 95-105, 47.5 % [40.7, 54.4] | checked (the interval contains 50) |
| the graph net sees better, the convolution counts faster | the three rows above | — | checked as a summary |

## Scene 10. Summary

| claim | source | value | status |
|---|---|---|---|
| 1: how a net sees decides what it can learn; Dot-G learned to count chains; the ordinary net from scratch could not | scene 3 | Gc 10/10, C 0/15, Ga 0/5 | checked |
| 2: a new board size via school, repeat the old or it is forgotten | scene 4 | school: 10/10 vs 0/10; repeating: observation (as scene 4) | checked / observation |
| 3: "Dot-G учится играть за меньшее число раундов" | graph_s9.md | 6 vs 11 rounds (caveat of scene 7) | observation (as scene 7) |
| 3: sees the move better | results.json 3:raw | 73.5 % [67, 79] | checked |
| 3: each look costs three times more | match.json | 3.04x | checked |
| 3: "при равном времени разницы между двумя сетями мы не увидели" | results.json 3:equal_time | 47.5 % [40.7, 54.4] | checked |
| 4: the ruler can lie, check it by a direct match | series 10 | — | checked |
| both nets are in the app, play each and watch them play each other | configs/app.yaml (best = az5b 84, release g5 114, v0.4); launcher "self" mode (gen_a / gen_b), docs/app_g5_114_vs_dot.png | — | checked |
| code, experiments and the correction to part 1 are on GitHub | README "Correction", commits 5a1973c, e1d3379 | — | checked |

## Notes

A. Part 3 ended with a concrete promise: graph eyes for Dot-2 and the test "the same eighty judge answers, does it
   learn what the convolution did not". Part 4 did not run that test (series 7 evaluated the jump nets on the middle
   game; nobody was trained on the 80 mistakes). Scene 6 now says so. Series 12 (2026-11-20) ran the test: the scene 6
   rows after "Series 12" replace that sentence; the voice of scene 6 is to be redone from them.
B. Scene 9 counts are shares of 200 games (no ties on 5x5): 62.5 and 73.5 per hundred, said so in the voice.
C. Equal time was calibrated on this machine (RTX 4060 Laptop, single-game search with 8 leaves per tree, as in the
   app). On another device the ratio can differ.
D. Data overlap (rule 5): series 7 checked 0 of 1925 mid positions shared with any training / selection set; the
   judge 10x7 sets of series 3 / 3' / 5 / 6 are disjoint from the stage 0 data (graph_s3p.md data checks). Series 10
   and 9 use games, not position sets.
