# Video script (English version)

The main theme is not "the network won" but "how the network thinks": the
value head knows who is winning; the policy does not know which move wins;
search closes the gap.

Pipeline as in pacman-rl/video: `video/storyboard_en.yaml`,
`video/video_build.py`, edge-tts en-US-GuyNeural (`video/voice_en_<N>.mp3`,
durations in `video/voice_en.json`), soft SRT subtitles, 1920x1080. Output
in `~/YouTube/ksquares-rl/`. Screen directions are the same as in
`docs/video_script_ru.md`; only the voice differs. Guy reads ~180 words a
minute (Pacman video), so the English text is longer for the same slots.

| scene | time | length | words |
|---|---|---|---:|
| 1. What this is | 0:00–0:40 | 0:40 | 88 |
| 2. The game and its trick | 0:40–2:05 | 1:25 | 174 |
| 3. How it was built | 2:05–3:05 | 1:00 | 154 |
| 4. AlphaZero in brief | 3:05–3:55 | 0:50 | 111 |
| 5. The ladder | 3:55–4:55 | 1:00 | 98 |
| 6. How the network thinks | 4:55–7:25 | 2:30 | 343 |
| 7. The plateau | 7:25–8:15 | 0:50 | 111 |
| 8. The app | 8:15–8:55 | 0:40 | 88 |
| 9. What next | 8:55–9:35 | 0:40 | 77 |

## Scene 1. What this is (0:00–0:40)

**Voice:** "This is KSquares — dots and boxes from the KDE games
collection. Red is a neural network. Blue is the game's own strongest
level, Hard. The network learned with the AlphaZero method: no strategy
rules, only games against itself. Two nights and the day in between, on a
laptop graphics card. But this video is not about the fact that it wins.
It's about how it thinks. Because inside the network there turned out to
be two parts, and one of them knows something the other one doesn't."

## Scene 2. The game and its trick (0:40–2:05)

**Voice:** "The rules in thirty seconds. Players take turns drawing a line
between two neighbouring dots. Close a box, and it's yours — and you move
again. Most boxes wins. At first everyone plays safe lines: lines that
don't give the opponent a third side of a box. But safe moves run out,
and the board ends up cut into chains. Now someone has to open a chain.
The opponent takes almost all of it, but hands the last two boxes back —
with a single line. That's the double-cross. After taking those two boxes,
the player who opened has to move again — and open the next chain. So the
player in control gets nearly everything. And control is decided by
parity: how many long chains there will be, and who has to open the first
one. The fight for it happens earlier, while safe moves are still left.
KSquares Hard knows the double-cross and always plays it right. But on
parity it guesses right only half the time — like a greedy player."

## Scene 3. How it was built (2:05–3:05)

**Voice:** "We didn't rewrite the game. KSquares got a small bridge in its
source: the game sends out the board over a socket and takes a move back.
Next to it there's a Python simulator with the same line numbering:
thousands of games per second on one core. Then we ported all three of
the game's computer levels — Easy, Medium and Hard — into the simulator,
down to the random number generator. Twelve hundred games were played
twice, in the simulator and in the real game, and all one hundred and
thirty thousand moves matched. So the network could train against Hard
without a window on the screen. To measure it we also needed something
stronger: an alpha-beta solver that beats Hard in ninety-four games out of
a hundred. As in my previous project, the code was written by the Claude
Code agent, mostly at night; the whole decision log is in the repository."

## Scene 4. AlphaZero in brief (3:05–3:55)

**Voice:** "The network is small: half a million parameters. In goes the
whole board; out come two heads. The policy is a first impression: which
lines look good. The value is a single number: who it thinks will win.
On its own the network plays by the policy, instantly. With search, before
every move it builds a tree of two hundred variations: it tries the moves
the policy likes and asks the value head how they end. The network plays
thousands of games against itself and learns from them, and each new
generation has to beat the previous one in at least fifty-five games out
of a hundred. Otherwise it's thrown away."

## Scene 5. The ladder (3:55–4:55)

**Voice:** "To measure strength we built a ladder. A random player is zero
on the Elo scale. Greedy is about nine hundred. Medium is twelve hundred,
Hard almost fifteen hundred, the solver eighteen hundred. The best
generation, number fifty-four, without search lands between Hard and the
solver: it beats Hard in eighty-three games out of a hundred. With a
two-hundred-variation search it's at twenty-one sixty-four: it beats Hard
almost every time and the solver in eighty-nine games out of a hundred.
Search adds almost five hundred Elo points. The interesting part is where
those five hundred points come from."

## Scene 6. How the network thinks (4:55–7:25)

**Voice:** "We collected a thousand positions where only parity matters:
nothing to capture, safe moves still left, and the line you choose decides
who has to open the first long chain. The solver worked out the right
answer for each one, all the way to the end. Then we tested every
generation of the network on them in three ways. First, the policy: the
move without search. By generation three it already reaches Hard's level
— about half the moves right — and then stays there for dozens of
generations. Second, the value head. We play every possible move and ask
the network: who is winning now? Guessing the sign at random gets half of
them right, and up to generation seven the network does no better. At
generation eight it jumps; by generation eleven it's right three times out
of four, and later up to eighty-eight percent. The network knows who will
win. But the same network, picking a move by itself, doesn't know which
move leads to the win. It seems to come down to what each head learns
from. The value learns from game results, and in these positions the
result is parity. The policy learns to imitate the search, and to pick the
right line out of thirty similar ones it would have to ask, for each of
them: and what happens next? Even the simplest version of that works:
play each move and pick the one after which the value head says "I'm
winning". That alone gets over eighty percent — no tree at all. And the
network with a two-hundred-variation search climbs right behind the value
head and reaches ninety percent. The solver is at ninety-five. You can see
it on the heat map: the policy is smeared over the whole board, while the
search visits pile up on the winning move. Only in the last thirty
generations did the policy start to catch up: from fifty to fifty-nine
percent, and that's not noise. But with search it's ninety. The parity
work is still done by the search."

## Scene 7. The plateau (7:25–8:15)

**Voice:** "The last night: forty-seven generations. Thirteen passed the
gate; each one beat the one before, sometimes sixty-five to thirty-five.
But against fixed opponents — Hard and the solver — strength stopped
growing after generation fifty-four. The last accepted one, eighty-four,
beat its predecessor by a wide margin and at the same time lost to the
solver more often than any other generation that night. The network was
learning to beat itself, not the game. A gate that only compares with the
previous version can't see that. So the app runs not the latest
generation but the best one on the ladder — fifty-four. In the morning we
stopped training by hand."

## Scene 8. The app (8:15–8:55)

**Voice:** "You can watch all of this yourself. The app has three modes:
the network against the real KSquares, you against the network, and the
network against itself. On the right are its thoughts: the policy heat
map, the search visits with the main line, a who-is-winning bar, the long
chains on the board. And late in the game a solver checks every move: does
the network see the win or not. Here's generation one against generation
fifty-four, and below it, how the who-is-winning estimate moves during the
game."

## Scene 9. What next (8:55–9:35)

**Voice:** "What next. KSquares plays on a ten-by-seven board by default,
but our network only knows five by five: its heads are tied to the board
size. Make them convolutional, and one network could play on any board.
And the main question is still open: can the policy itself learn to see
parity — or is that forever the search's job? The code, the experiment log
and the app are on GitHub; the link is in the description."

## End card

`github.com/AceEca-robo/ksquares-rl` (created before publishing) · "KSquares 21.12.3 © Matt Williams et al., GPL-2+".
