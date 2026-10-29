# Video script, part 2 (English version)

Translation of `docs/video_script_part2_ru.md` (the agreed Russian cut);
screen notes are there. Names: Dot (the part 1 network, "old Dot"), Dot-2
(new architecture, trained on 5×5 and moved to 10×7), Dot-3 (from scratch
on 10×7). The voice says "thinking" / "going through the options" for
search and never says "Elo" or "policy". Voice: edge-tts en-US-GuyNeural.

## Scene 1. A bigger board (0:03–0:38)

**Voice:** «Last time, Dot learned to play Dots and Boxes on a small five
by five board and beat the strongest opponent in KSquares. But the real
board in the game is bigger: ten by seven. Seventy boxes instead of
twenty-five, more chains, and longer ones. The question is simple: if
Dot understood the game, will she understand it on the big board? Or
did she just memorize the small one?»

## Scene 2. Dot-2 (0:38–1:18)

**Voice:** «First, the problem. Old Dot physically cannot play on
another board: her outputs are nailed to twenty-five boxes. So we built
a new one, Dot-2. Dot-2 looks at the board like a picture, and she
doesn't care what size it is. Old Dot became Dot-2's teacher: first
Dot-2 took over everything the old one knew, then she kept learning on
the five by five board by herself. Then, without showing her a single
big game, we sat her down at the ten by seven board.»

## Scene 3. Moving without preparation (1:18–2:03)

**Voice:** «And here is what happened. Dot-2 draws safe lines with
confidence, doesn't give boxes away, and almost always beats the greedy
opponent. This is the hand: the first impulse of where to move. On the
big board, Dot-2 plays stronger than KSquares' medium level and a little
weaker than hard. Not bad for a network that has never seen this board.
But then things get strange. We let Dot-2 think, going through the
options, the way Dot the champion did on the small board. And Dot-2
starts playing worse.»

## Scene 4. Intuition lies (2:03–2:58)

**Voice:** «Remember intuition? It is the other half of Dot: it looks at
a position and says who is winning. On the small board, intuition was
rarely wrong. On the big board, intuition broke in a strange way: it
sees a loss almost everywhere. Intuition still recognizes losing moves,
but it recognizes only one winning move in three. Why? A big game is
decided by the parity of the chains, and on the big board there are more
of them and they are harder to count. Intuition learned to count on five
by five, and on ten by seven it simply panics.»

## Scene 5. Thinking hurts (2:58–3:48)

**Voice:** «When Dot-2 thinks, she goes through the options and at the
end of each one asks intuition: so, is it good? If intuition answers
"bad" to everything, Dot-2 has nothing to hold on to, and she just runs
from one move to another. Worse: she can walk away from the move the
hand chose correctly. In the end, Dot-2 is weaker with thinking than
without it. And against the real KSquares on hard: three wins out of
twenty. The lesson of part one was: intuition knows who is winning, and
thinking turns that into moves. The lesson of part two: if intuition
lies, thinking amplifies the lie.»

## Scene 6. One round cures it (3:48–4:38)

**Voice:** «Let's cure it. Dot-2 keeps learning from her own games, but
now the boards grow step by step: first seven by seven, then ten by
seven. And here is the first round in which Dot-2 played a thousand
games on the big board. Look at the jump. The hand barely changed: the
dashed line runs flat. But the solid line, Dot-2 with thinking, jumps
two steps up the ladder in a single round, and against hard: ninety
percent wins. All it took was to calibrate intuition: to show it how big
games really end. Everything else, Dot-2 already knew.»

## Scene 7. 19 out of 20 (4:38–5:13)

**Voice:** «After six hours of training on the big board, here she is.
Dot-2 against the real KSquares, hard level, ten by seven board. Twenty
games: nineteen wins, one draw, not a single loss. Before, it was three
wins out of twenty. The same network, the same thinking: only now
intuition tells the truth.»

## Scene 8. From scratch vs experience (5:13–6:08)

**Voice:** «But maybe the small board isn't needed at all? Let's check.
We train Dot-3: from a blank slate, straight on ten by seven, with the
same settings as Dot-2. In the first rounds she loses to everyone:
random lines. Then she catches up with KSquares' hard level, and gets
stuck there: new versions can no longer beat the old one, and training
stops. The best Dot-3 wins twelve games out of twenty against hard.
Dot-2, who went from the small board to the big one: nineteen, and twice
as fast. The small board is not a toy. It is a school.»

## Scene 9. Play on the big board (6:08–6:33)

**Voice:** «All of this is in the app. Pick the ten by seven board and
play against Dot, watch how she thinks, or set her against KSquares. You
can also train your own Dot right on the big board. The link is in the
description.»

## Scene 10. What next (6:33–6:53)

**Voice:** «One question is left. We cured intuition. But can we grow
the hand, so that Dot sees the parity of the chains at once, without
going through the options? That's for next time.»
