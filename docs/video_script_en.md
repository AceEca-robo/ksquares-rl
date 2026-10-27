# Video script (English version)

The English cut of `docs/video_script_ru.md`, in the same tone: the
network is the hero, Dot; the value head is her intuition, the strong
solver is the judge, KSquares Hard is the built-in opponent, a generation
is a round, the simulator is the training hall. Every voice is sentence for
sentence the Russian one, so the cuts of `video/storyboard_ru.yaml` carry
over to `video/storyboard_en.yaml` (`video/storyboard_en.py`). Screen
directions: as in the Russian script.

Pipeline: `video/video_build.py`, edge-tts en-US-GuyNeural
(`video/voice_en_<N>.mp3`, durations in `video/voice_en.json`), soft SRT
subtitles, 1920×1080. Thumbnail C: "Dot knows who wins. / Just not how."

## Scene 1. Meet Dot (0:03–0:46)

**Voice:** "Let's teach a neural network to play KSquares — dots and boxes
from the KDE games. First thing: give the future champion a name. Let's
call her Dot. Nobody explained any strategy to Dot: for a day and a half
she played against herself on an ordinary laptop and learned from her own
games. Here is the result. Dot plays red; blue is the strongest opponent
built into the game. Dot wins. But this video is not about winning. It is
about Dot having an intuition — and her intuition knows something Dot
herself does not."

## Scene 2. The game (0:46–1:27)

**Voice:** "The rules are simple. Players take turns drawing a line between
two neighbouring dots. Whoever closes a box takes it and moves again. Most
boxes wins. At first everyone plays safe lines: lines that do not let the
opponent close a box. But safe moves run out, and the board ends up cut
into chains. Now someone has to open a chain. The opponent eats almost all
of it, but hands the last two boxes back — with a single line. That is
called the double-cross."

## Scene 2p. Pause: the double-cross (1:27–1:31)

**Screen:** no voice. The double-cross up close.

## Scene 2a. The secret of the game (1:31–2:10)

**Voice:** "Having taken those two boxes, the player who opened the chain
has to move again — and open the next one. So whoever holds control gets
nearly everything. But control has to be fought for early, while safe
moves are still left: you have to count who gets the last one. The
built-in opponent knows the double-cross and always plays it. But counting
who gets the last safe move is beyond it — it guesses, like a coin toss.
And that is exactly what Dot has to learn."

## Scene 3. The training hall (2:10–2:58)

**Voice:** "So that Dot could play millions of games, we did not rewrite
the game itself — we added a small bridge to it: the game shows the board
and takes a move. And next to it we built a training hall: a copy of the
game in Python, thousands of games a second. We moved the built-in
opponents into the hall too — so precisely that they play the very same
moves there as in the real game. And one more: the judge, a program that
calculates the game to the very end and always knows the right move. The
judge beats the built-in opponent almost every time. We need it not as a
rival, but as a teacher with the answer key. All the code, like last time,
was written by Claude Code; the log of its decisions is in the
repository."

## Scene 4. How Dot learns (2:58–3:44)

**Voice:** "Dot is built simply. She looks at the board and gives two
answers. The first: where would I go. The second, from her intuition: who
is winning here. Dot plays thousands of games against herself, then learns
from them, and that is one round. After every round the new Dot plays a
match against the old one. If she wins by a margin, she stays. If not, she
is thrown away. And so, round after round, Dot gets stronger. This is the
AlphaZero method — the one DeepMind used to teach a computer go and chess.
Only we have a laptop."

## Scene 5. The ladder (3:44–4:15)

**Voice:** "We measured Dot's strength with a ladder: a random player at
the bottom, then a greedy one, then the game's three levels, and the judge
at the top. If Dot moves at once, without thinking, she is already above
the built-in opponent. And if you let her think — go through the options —
she climbs above the judge. Where does that difference come from? This is
where it gets interesting."

## Scene 6. Dot and her intuition (4:15–5:50)

**Voice:** "Dot glances at the board and says right away where to go. And
Dot also has an intuition: she knows who is winning here. We collected a
thousand positions where everything hinges on one thing: who gets the last
safe move. The judge knows the right answer for each. When choosing a
move, Dot is wrong half the time. And she does not improve, round after
round. And the intuition? At first it is wrong too. But around round ten —
click. From then on Dot almost always says correctly who will win. And
here is the strange part. Dot knows who wins. But ask her "with which
move?" — and she answers at random. The intuition sees how the game will
end. But Dot does not know which of thirty similar moves leads there. How
do you make Dot friends with her intuition? Let Dot think: go through the
moves and ask the intuition about each one: and now, who is winning? And
Dot solves nine problems out of ten. Almost like the judge. That is how
AlphaZero works: intuition plus a little thinking gives a strength that
neither has on its own."

## Scene 7. The ceiling (5:50–6:18)

**Voice:** "On the last night Dot went through almost fifty more rounds.
Each new version beat the one before. But against the game and the judge,
her strength stopped growing. Dot had learned to beat herself — not the
game. We stopped the training: there was no one left to learn from. What
went into the app is not the last Dot, but the strongest one."

## Scene 8. Play Dot (6:18–6:48)

**Screen:** as in Russian, plus a closing frame of the app's new mode
"Train your own Dot" (`docs/app_train.png`, English).

**Voice:** "You can play Dot yourself. The app has four modes: Dot
against the real KSquares, you against Dot, Dot against an early
version of herself — and training your own Dot. On the right you see what she is thinking: the
options she goes through, and a bar — who, in her view, is winning right
now. Here is Dot from round one against Dot from round fifty-four. Below —
how her confidence changes over the game."

## Scene 9. What next (6:48–7:21)

**Voice:** "For now Dot can only play on a five-by-five board. The next
step is to teach her any board. And an open question: can Dot be taught to
see what her intuition sees — or will she always have to think? Everything
is open: the code, Dot, the app and the training log — on GitHub. Clone it
and run install.sh — and you can play Dot. And if you have a graphics
card, one button in the app starts the training, and you will watch Dot
learn from scratch right before your eyes."

## End card

`github.com/AceEca-robo/ksquares-rl` · "KSquares 21.12.3 © Matt Williams et al., GPL-2+".
