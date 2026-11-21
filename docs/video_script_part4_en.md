# Video script, part 4 (English version)

Translation of `docs/video_script_part4_ru.md` (the published Russian cut, claims check revision 5); screen notes are
there, the screen texts are translated through `video/p4_i18n.py` (P4_LANG=en). Words: Dot, Dot-2 and Dot-G ("she",
as in parts 1-3), thinking / without thinking, going through options, intuition, the judge ("it"), the built-in
opponent, round, school (three grades), the ladder, the ruler. The voice never says policy, value, MCTS or Elo. Board
sizes in words: five by five, ten by seven. Voice: edge-tts en-US-GuyNeural. Every number and every wording has the
strength of the Russian one: docs/video_claims_part4.md, section "English cut".

## Scene 1. Other eyes (0:03–0:37)

**Voice:** «Last time we ended with a promise: to give Dot other eyes. Our Dot looks at the board through a small
window, like at a picture. But the chains that decide a game are not a picture. They are links: which box is
connected to which, and what will pull what after it. There are networks that see exactly that, the links. They are
called graph networks. We built one and named her Dot-G. This video is about what she learned, where she fell short
of our hopes, and how along the way we found a mistake in our own first video.»

## Scene 2. How Dot-G sees (0:37–1:01)

**Voice:** «For Dot-G the board is not a grid of dots, but a diagram. Boxes, the lines between them and, most
importantly, chains: every chain is a separate object for her, and it has a length. The ordinary Dot has to guess
where a chain is, piecing it together from bits in her window. Dot-G gets the chain whole. The question was whether
that would help.»

## Scene 3. First lesson: counting (1:01–1:55)

**Voice:** «We started with a single task. A position from the end of a game, and you have to say just one thing:
who will have to open the first chain. We gave this task to three networks, each starting from scratch. The first: an
ordinary convolutional network, built the same way as Dot. It stayed at the level of a coin toss, however long we
trained it. The second: a simple graph network. It sees the board as a diagram of boxes and lines, but it doesn't see
a chain as a whole, so it too has to put it together piece by piece. The same result, a coin toss. The third: Dot-G,
the graph network for which every chain is a separate object. And at some point she clicks: from a coin toss to an
almost error-free answer in a few hundred training steps. We ran her ten times from different starts, and the click
happened all ten times.»

## Scene 4. School (1:55–2:51)

**Voice:** «But that's on the small board, five by five. We showed Dot-G the big board, ten by seven, without giving
her a single example from it, and the big board didn't work out for her. Ten tries, not one successful. So we set up
a school with three grades. First grade: only the small board, until Dot-G answers almost without mistakes. Second
grade, a short one: tasks from the small and the big board mixed together. Third grade: only the big board. After
this school, all ten out of ten mastered the big board. But it turned out that by the end of third grade Dot-G almost
always forgets the small board. A simple remedy helps: in third grade, every fourth task comes from the old, small
board. Then by the end of school she remembers both boards. Just like people: what you don't repeat, you forget.»

## Scene 5. From counting to the move (2:51–3:52)

**Voice:** «Knowing who has to open the first chain is not yet knowing the right move. We started teaching Dot-G to
choose the move in endgames, with the judge's answers. And we saw something strange. Most networks choose the move
about as well as an ordinary convolutional network does. But roughly every fourth one suddenly makes a jump and
starts finding the right move in nine endgames out of ten, at once, without thinking. We set out to find why some are
lucky and others are not. We took four networks that had finished school, and taught each of them to choose the move
five times over, changing only the order of the tasks. One network made the jump all five times. Another one never
did. Two more, twice out of five. So a lot depends not on luck during training, but on what kind of network came out
of school. On the outside the graduates are the same: all of them count chains on the big board without mistakes.
But inside they are different, and how exactly, we don't know yet.»

## Scene 6. The middle of the game (3:52–5:07)

**Voice:** «This looked like what we were looking for last time: a network that sees the move at once. The main test
was still ahead. In the third part we found that Dot-2 loses in a rout because of a single mistake in the middle of
the game. We took the middle-game positions on which we had tested Dot-2, and showed them to the networks with the
jump. The result: eighty-one percent right moves. Dot-2 without thinking gets eighty-four on the same positions. The
jump did not carry over to the middle of the game. Then we ran the experiment we promised last time: we gave both
networks the same answers from the judge for the middle of the game, and started teaching them. When the whole
network learned, both learned these answers by heart, every single one. But on new positions they had not seen, we
see no gain for either of them. When Dot-2's own games were mixed in with the judge's answers, she improved, while for
Dot-G we see no gain there either. The positions, though, came from Dot-2's games, so they are more familiar to her.
So the conclusion is a careful one: the same answers from the judge helped the graph network no more than the
convolutional one. Dot-G learned to solve endgames, but in the middle of the game the other eyes gave no advantage.»

## Scene 7. Then let her play (5:07–5:43)

**Voice:** «Up to this point Dot-G had not played a single game, she had only solved tasks. So we did what we did with
the very first Dot: we set Dot-G to play against herself, from scratch, on the small board, under the same conditions
Dot learned under. Though Dot's first rounds ran with slightly different settings. Two days, one hundred and seventeen
rounds. She started beating the game's built-in opponent confidently in round six. The first Dot needed eleven rounds
for that. Against the real KSquares on the hard level, Dot-G won twenty games out of twenty.»

## Scene 8. The broken ruler (5:43–6:38)

**Voice:** «And then something happened that we did not expect. We had always measured strength with the ladder: the
built-in opponents and the judge. By the ladder, Dot-G peaked by round forty-nine, and then started getting weaker.
We decided to check, and set up a match: the late Dot-G against the one the ladder called the best. The late one won
ninety-three games out of a hundred. There was no decline in strength: it was only on the ladder. Then we checked the
first Dot. In the first video we said that after round fifty-four she stopped growing and only learned to beat
herself. That turned out to be untrue: the late Dot beats that round fifty-four Dot in two games out of three. It
wasn't the networks that were wrong. The ruler was wrong: when everyone beats the built-in opponents, the ladder can
no longer tell who is stronger. Now we choose the best network by matches only.»

## Scene 9. The match (6:38–7:33)

**Voice:** «And now the main comparison. On one side, Dot-G after one hundred and fourteen rounds. On the other, the
strongest Dot: the one that went through eighty-four rounds. If both think the same way, going through two hundred
options per move, Dot-G wins sixty-two and a half games out of a hundred. If both move at once, without thinking,
seventy-three and a half. But this comes at a price. To evaluate one option, a network has to look at the position
once. Dot-G looks three times longer, because taking the board apart into chains and links is harder than sliding a
window over it. So we played one more match, with equal time per move: while Dot goes through two hundred options,
Dot-G manages about sixty. And in this match we see no difference. The graph network sees better. The convolutional
one counts faster.»

## Scene 10. Summary (7:33–8:28)

**Voice:** «What we learned. First: how a network sees the board decides what it can learn. Dot-G got chains as
ready-made objects and learned to count them on her own. The ordinary network starting from scratch could not do
that. Second: a new board size has to be taught gradually, through school, and the old one has to be repeated, or it
is forgotten. Third: Dot-G learns to play in fewer rounds and sees the move better, but every look costs her three
times as much, and at equal time we saw no difference between the two networks. And fourth, about ourselves: the
ruler you measure strength with can lie, and you have to check it with a direct match. Both networks are in the app:
you can play each of them and watch them play each other. All the code, all the experiments and our correction to
the first part are on GitHub. The next game will be something completely different: not on a board. See you.»
