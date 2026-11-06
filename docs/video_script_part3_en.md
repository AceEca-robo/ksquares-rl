# Video script, part 3 (English version)

Translation of `docs/video_script_part3_ru.md` (the agreed Russian cut);
screen notes are there. Words: Dot-2 ("she", as in part 2), intuition,
thinking / without thinking, options, the judge ("it"), the built-in
opponent, the box, the "who knows what" map, Berlekamp. The voice never says
policy, value, MCTS, Elo or parity. Voice: edge-tts en-US-GuyNeural.

## Scene 1. The question (0:03–1:11)

**Voice:** «Last time we tuned the intuition, and Dot-2 started beating the
built-in opponent on the big board. But she only beats it when she thinks:
before every move she goes through two hundred options. If we don't let her
think, one look at the board and she moves, she plays noticeably worse. Why?
There is one thing Dot-2 can't see without thinking. At the end of a game the
board is cut into chains, and everything depends on who has to open the first
one. That takes counting. When she thinks, Dot-2 counts it. Without thinking,
she doesn't see it. So here is our question: can Dot-2 learn to understand the
game, instead of recounting it every time? To see the right move at once, the
way an experienced player does. We made five attempts. And this is no idle
question: when Dot-2 thinks only a little, at the app's fastest setting, about
one game in ten ends in a rout. Not a loss by a couple of boxes. A real rout.»

## Scene 2. Where the mistake is (1:11–2:00)

**Voice:** «Here is one of those games. At this moment Dot-2 holds a win. We
know that for certain: the judge says so, a program that calculates the
position all the way to the end. Dot-2 draws one line, and the game is lost.
Her intuition notices nothing, and raises the alarm only a few moves later.
After that the built-in opponent simply collects the chains. We took the routs
apart from the end backwards. The judge found the mistake in ten of the
seventeen. Each time it is a single move: a win before it, a loss after it. And
each time in the middle of the game, not at the end. In the others the judge
couldn't reach the mistake, but there too the game was lost long before the
finish.»

## Scene 3. Attempt 1: the judge as teacher (2:00–2:43)

**Voice:** «The plan is simple: let Dot-2 learn from the judge. We show her
thousands of positions with the right moves marked. And she learns. On new late
positions, without thinking, she is right not sixty-seven percent of the time
any more, but eighty-seven. At a single glance. Wonderful. We look at the routs,
and they are the same. Why? This judge only answers close to the end of the
game. That's where Dot-2 learned. But the mistake lives earlier, in the middle,
and there the judge is silent. So we need a teacher for the middle of the
game.»

## Scene 4. Attempt 2: a new network (2:43–3:54)

**Voice:** «Second attempt: change the network itself. All our networks are
convolutional: they look at the board like a picture, through a small window,
and search it for patterns: "here is a box with three sides", "here is a safe
line". The same convolutional networks recognize faces and road signs. But
chains are long and the window is small: to count who has to open the first one,
you need to see the whole board at once. So we gave the network a view of the
entire board, and a direct hint: which chains are on the board, and who opens
the first one. Without thinking, Dot-2 really does get better. But move choice
and intuition live in one body. Pull on one, and the other breaks: the intuition
starts seeing losses where there are none, and thinking relies on it. There are
four times as many routs. All right, let's freeze the intuition and train only
the move choice. Dot-2 learns, her game stays intact, but the routs are the
same. The hint tells the network what it already knew about the end of the
game, and changes nothing in the middle.»

## Scene 5. Attempt 3: a judge by formula (3:54–4:52)

**Voice:** «Maybe the judge is too primitive? It goes through variations, but
chains have mathematics. The mathematician Berlekamp worked it out back in the
last century. The rule goes like this: when the safe lines run out, count the
long chains on the board. If their number is even, one player wins, if it's odd,
the other. No need to go through moves, just count the chains. Third attempt: a
judge built on this formula. It doesn't search, it counts. And once the board is
cut into chains, it is always exact. But in the middle of the game the board
isn't cut yet: lines are still going down, chains are only forming, and there is
nothing to count. There this judge didn't answer a single time. The formula is
good when everything is already decided, and we need it earlier. There's no way
around searching.»

## Scene 6. Attempt 4: a strong judge (4:52–5:31)

**Voice:** «Fourth attempt: put the two together. An honest search on the real
board, and the formula wherever the board is cut. This is a strong judge, and it
gets into the middle of the game. We tested it on late positions: not one
disagreement with exact calculation. But it has a wall. At fifteen safe lines a
minute is usually enough. The more lines, the more sharply the time grows. And
at thirty safe lines it didn't solve a single position. The middle itself is
beyond its strength.»

## Scene 7. Who knows what (5:31–8:09)

**Voice:** «But now we can draw a map: who knows what over the course of a game.
Horizontally, the game: on the left the first line, on the right the last. Vertically, how
often a player makes the right move. But first, where the numbers on this map
come from at all. To say "Dot-2 moved correctly", you have to know which move is
correct. Only the judge knows that, and only where it managed to finish
calculating. So the whole map splits into three zones. The start of the game.
The judge doesn't reach this far: there are too many positions. There is nothing
to check Dot-2 against, so we honestly leave this place empty. The middle of the
game is a dark box. This is where Dot-2 makes the very mistake that leads to a
rout. And this is also where the judge doesn't answer yet. That's why the box
holds a question mark: we know the mistake lives here, but we can't point a
finger at it. The endgame, from twenty safe lines to the end of the game. Here
the judge answers, and here we have numbers. Four lines. Blue is Dot-2 without
thinking, a move at a single glance. Coming out of the box, Dot-2 without
thinking gets every fourth or fifth move wrong, and in some positions every
second one. By the end of the game, almost never. Orange is Dot-2's intuition,
her estimate of who is winning. The intuition is a little more accurate than the
move without thinking. Green is Dot-2 with thinking, two hundred options per
move. The green line stays above the others: on average ninety-six percent right
moves, and it sags only right by the box. The distance from the blue line to the
green one is what going through options gives Dot-2. And white is the judge. The
white line has a different meaning: not how often the judge is right, but what
share of positions the judge could solve at all. At the very end of the game the
judge solves everything. The closer to the box, the fewer positions it manages
to finish. At the edge of the box the white line falls to fifteen percent. That
is the wall: beyond it the judge is silent, and without the judge we can't check
Dot-2, or her intuition, or her thinking. So inside the box Dot-2 doesn't know
how to move. And nobody can check where she went wrong.»

## Scene 8. Attempt 5: learning from her own mistakes (8:09–8:57)

**Voice:** «The last attempt. If the judge reaches only part of the middle, let's
collect exactly what it can reach: Dot-2's own real mistakes. Five and a half
thousand games, and the catch is eighty mistakes. Each one takes minutes of the
judge's calculation, and many routs run into the same wall. We train Dot-2 on
these mistakes, and on everything the judge managed to label in the middle. Now
the main test: positions she has never seen. There it got a little better. There
is a gain, but we couldn't prove it isn't chance: too few answers from the judge.
To see a result you need many times more, and each one costs minutes of
calculation.»

## Scene 9. The answer (8:57–9:54)

**Voice:** «So, the answer. Can Dot-2 learn to understand the game without
thinking? Yes, but it's expensive. Expensive means this: every example for the
middle of the game costs minutes of the judge's calculation, and thousands of
examples are needed. Eighty examples took eight and a half hours. At that rate a
thousand is four and a half days, and five thousand is twenty-two days, for one
network. Where there are examples, Dot-2 learns. Where there are none, she
doesn't. Thinking is cheaper: two hundred options per move take five hundredths
of a second on a graphics card, or a quarter of a second on a processor, and give
ninety-six percent right moves wherever that can be checked. That's why in the
app Dot-2 goes through two hundred options by default. In the settings you can
choose fifty: moves get faster, but fifty is exactly where we found the routs.»

## Scene 10. What next (9:54–11:27)

**Voice:** «Two games and three parts are behind us, and in each we learned
something new. All three parts are one convolutional network and one method,
AlphaZero: a network plus a search. The network sees, the search counts. Where
you need to see, she is strong. Where you need to count, the search counts, and
so far we haven't managed to replace it with a network. But remember what the
examples for the middle of the game cost: five and a half thousand games, hours
of the judge's calculation, and only eighty answers. For a convolutional network
that's too few: to learn a rule, it needs thousands of examples. And there is
nowhere to get more, because the judge runs into the wall. So we need a network
for which eighty is enough. A convolutional network looks at the board through a
window, like a picture. But chains aren't a picture, they are a graph: nodes and
the links between them. And there are networks that see graphs directly. They
don't have to guess where a chain is, they get it ready-made, and the rule of who
opens the first one is almost arithmetic for them. In the next part we'll give
Dot-2 such eyes and test it: the same eighty answers from the judge, will she
learn what the convolution didn't? The answer "expensive" is a result too: now
we can see where the border runs, and from which side to storm it. You can play
Dot in the app, the link is in the description. See you.»
