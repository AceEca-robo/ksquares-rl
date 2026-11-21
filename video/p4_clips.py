#!/usr/bin/env python3
"""Animated clips of the part 4 video (docs/video_script_part4_ru.md), timed on the voice's sentence starts.

    .venv/bin/python video/p4_clips.py   # video/p4/window_graph.mp4 (scene 1), video/p4/click.mp4 (scene 3)

window_graph: the part 3 scheme "window or graph" (video/p3_clips.window_vs_graph), the graph appearing on the
sentence "Это связи". click: the scene 3 curves (video/p4_cards.click data); the convolution's curves appear with
"Первая", the simple graph net's with "Вторая", Dot-G's are drawn left to right from "И она в какой-то момент".
The clip starts at the sentence "Первая" (the storyboard cut).
P4_LANG=en: the same clips on the English voice (docs/video_script_part4_en.md, video/p4/voice_en_<N>.mp3),
written as *_en.mp4.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from p4_i18n import LANG, L, suffix  # noqa: E402  (first: sets P3_LANG)
from video_anim import sentence_starts  # noqa: E402
from video_voice import scenes  # noqa: E402

OUT = os.path.join(HERE, "p4")
FPS = 25
SCRIPT = os.path.join(ROOT, f"docs/video_script_part4_{LANG}.md")


def sentences(n):
    text = next(t for k, _, _, t in scenes(SCRIPT, LANG) if str(k) == str(n))
    return [s for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s]


def starts(n):
    s, end = sentence_starts(LANG, str(n), SCRIPT, os.path.join(OUT, f"voice_{LANG}_{n}.mp3"))
    assert len(s) == len(sentences(n)), (n, len(s), len(sentences(n)))
    return s, end


def at(n, prefix):
    s, _ = starts(n)
    return s[next(i for i, t in enumerate(sentences(n)) if t.startswith(prefix))]


def write(name, frames):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080",
                          "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18",
                          os.path.join(OUT, name.replace(".mp4", suffix() + ".mp4"))], stdin=subprocess.PIPE)
    for im in frames:
        p.stdin.write(im.convert("RGB").tobytes())
    p.stdin.close()
    p.wait()
    print(name)


def click_frames(t0, t_c, t_ga, t_gc, seconds, draw=6.0):
    """Times in the scene; the clip starts at t0."""
    import p4_cards as C
    from video_cards import page
    cur = C.P.click_curves()
    C.P.click_curves = lambda: cur                      # read once
    base = {}
    for f in range(round(seconds * FPS)):
        t = t0 + f / FPS
        keys = tuple(k for k, tt in (("c", t_c), ("ga", t_ga)) if t >= tt)
        prog = min(1.0, max(0.0, (t - t_gc) / draw))
        if prog == 0 or prog == 1:
            key = (keys, prog)
            if key not in base:
                base[key] = frame(C, page, cur, keys, prog)
            yield base[key]
        else:
            yield frame(C, page, cur, keys, prog)


def frame(C, page, cur, keys, prog):
    full = {k: (v if k in keys else []) for k, v in cur.items() if k != "gc"}
    full["gc"] = [[(s, v) for s, v in run if s <= 6000 * prog] for run in cur["gc"]] if prog > 0 else []
    C.P.click_curves = lambda: full
    img = C.click(1.0)
    if prog == 0:                                       # the Dot-G box texts only once it clicks
        from video_cards import font
        d = __import__("PIL.ImageDraw", fromlist=["Draw"]).Draw(img)
        d.rectangle((1520, 280, 1920, 510), fill=(17, 19, 24))
    return img


def main():
    from p3_clips import window_vs_graph
    write("window_graph.mp4", window_vs_graph(seconds=40, at=at(1, L("Это связи", "They are links"))))
    s, end = starts(3)
    t0 = at(3, L("Первая", "The first"))
    write("click.mp4", click_frames(t0, t0, at(3, L("Вторая", "The second")),
                                    at(3, L("И она в какой-то момент", "And at some point")), end - t0 + 3))


if __name__ == "__main__":
    main()
