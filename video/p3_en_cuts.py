#!/usr/bin/env python3
"""English cut of part 3: the cuts from the voice's sentence starts.

    .venv/bin/python video/p3_en_cuts.py   # video/storyboard_part3_en.yaml, video/p3/times_en.json
"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from video_anim import sentence_starts
from video_voice import scenes

SCRIPT = os.path.join(ROOT, "docs/video_script_part3_en.md")
TEXT = {str(k): (ti, t) for k, ti, _, t in scenes(SCRIPT, "en")}
TAIL = 0.7


def at(n, words):
    """Start of the sentence beginning with these words, on the clip's clock; the scene's end."""
    s, end = sentence_starts("en", str(n), SCRIPT, os.path.join(HERE, f"p3/voice_en_{n}.mp3"))
    sents = [x for x in re.split(r"(?<=[.!?…])\s+", TEXT[str(n)][1].strip()) if x]
    assert len(s) == len(sents), (n, len(s), len(sents))
    return next(a for a, x in zip(s, sents) if x.startswith(words)), end


def cut(n, parts):
    """parts: (fragment text with {s}, first words of the next fragment or None)"""
    lines = [f"  - n: {n}", f'    title: "{TEXT[str(n)][0]}"', f"    voice: p3/voice_en_{n}.mp3", "    fragments:"]
    t = 0.0
    for frag, nxt in parts:
        a, end = at(n, nxt) if nxt else (None, None)
        if nxt is None:
            _, end = at(n, "")
            a = end + TAIL
        lines.append(f"      - {{{frag.format(s=round(a - t, 1))}}}")
        t = round(a, 1)
    return lines + [""]


times = {}
f8, _ = at(8, "Five and a half")
times["funnel"] = [round(f8, 1), round(f8 + 2.0, 1), round(f8 + 3.3, 1)]
s10, _ = at(10, "But remember")
times["funnel10"] = [3.1, 4.8, 6.2]
w10, _ = at(10, "A convolutional network looks")
g, _ = at(10, "But chains")
times["graph"] = round(g - w10 + 1.0, 1)
json.dump(times, open(os.path.join(HERE, "p3/times_en.json"), "w"))

a1 = at(1, "At the end of a game")[0]
a4 = at(4, "Without thinking")[0]
sp1 = round(24 / a1, 2)
sp4 = round(15 / at(4, "So we gave")[0], 2)
out = ["""# English cut of the part 3 video (docs/video_script_part3_en.md), written by video/p3_en_cuts.py.
#
#   .venv/bin/python video/video_build.py --lang en --storyboard video/storyboard_part3_en.yaml \\
#       --script docs/video_script_part3_en.md --out video/p3/ksquares_rl_part3_en.mp4 \\
#       --srt video/p3/ksquares_rl_part3_en.srt --build video/build_p3_en

size: [1920, 1080]
fps: 25
crf: 18
voice_at: 0.5
min_tail: 0.7
fade: 0.4
title:
  seconds: 3
  image: p3/card_title_en.png
end:
  seconds: 6
  lines: ["github.com/AceEca-robo/ksquares-rl", "",
          "KSquares 21.12.3 © Matt Williams et al., GPL-2+"]

scenes:"""]
scenes_ = [
 (1, [(f"file: p3/two_boards_en.mp4, seconds: {{s}}, speed: {sp1}", "At the end of a game"), ("file: p3/card_chains_en.png, seconds: {s}", "So here is"),
      ("file: p3/card_question_en.png, seconds: {s}", "And this is"), ('file: p3/game121_end_en.mp4, seconds: {s}, caption: "1:69 · 50 options per move · 17 routs of 190"', None)]),
 (2, [("file: p3/game121_en.mp4, seconds: {s}", "We took the routs"), ("file: p3/card_blowouts_en.png, seconds: {s}", None)]),
 (3, [('file: p3/card_teacher_en.png, seconds: {s}, caption: "positions with 1–2 safe lines"', "Wonderful"), ("file: p3/card_timeline_en.png, seconds: {s}", None)]),
 (4, [(f"file: p3/conv_window_en.mp4, seconds: {{s}}, speed: {sp4}", "Without thinking"), ("file: p3/card_shared_body_en.png, seconds: {s}", "There are four times"), ("file: p3/card_frozen_en.png, seconds: {s}", None)]),
 (5, [("file: p3/card_formula_en.png, seconds: {s}", "But in the middle"), ('file: p3/card_formula_back_en.png, seconds: {s}, caption: "middle of the game, 5500 positions: no answers"', None)]),
 (6, [("file: p3/card_tree_en.png, seconds: {s}", "But it has a wall"), ("file: p3/card_wall_en.png, seconds: {s}", None)]),
 (7, [("file: p3/map_en.mp4, seconds: {s}", None)]),
 (8, [("file: p3/funnel_en.mp4, seconds: {s}", "Now the main test"), ("file: p3/card_mistakes_held_en.png, seconds: {s}", "There is a gain"), ("file: p3/card_mid_held_en.png, seconds: {s}", None)]),
 (9, [("file: p3/card_map_6_en.png, seconds: {s}", "That's why in the app"), ("file: p3/menu_10x7_200_en.png, seconds: {s}", None)]),
 (10, [("file: p3/card_map_5_en.png, seconds: {s}", "But remember"), ("file: p3/funnel10_en.mp4, seconds: {s}", "A convolutional network looks"),
       ("file: p3/window_graph_en.mp4, seconds: {s}", "The answer"), ("file: p3/card_end_en.png, seconds: {s}", None)]),
]
for n, parts in scenes_:
    out += cut(n, parts)
open(os.path.join(HERE, "storyboard_part3_en.yaml"), "w", encoding="utf-8").write("\n".join(out))
print(times, sp1, sp4)
