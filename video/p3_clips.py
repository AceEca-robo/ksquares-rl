#!/usr/bin/env python3
"""Two schematic clips of the part 3 video (docs/video_script_part3_ru.md).

    .venv/bin/python video/p3_clips.py   # video/p3/two_boards.mp4, video/p3/conv_window.mp4

two_boards (scene 1): the position of game seed 121 after line 60. Left: the raw
policy's move at once. Right: the same position, the policy's top lines flicker
while a counter runs to 200, then the same kind of highlight. The flicker is an
illustration of search, not a recording of it.
conv_window (scene 4): a small window slides over the board and cannot hold a
long chain; then the frame covers the whole board and the hint appears.
"""
import os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "scripts")]
from p3_i18n import T, suffix
from video_cards import DIM, FAINT, FG, LATO, arrow, font, page
from video_thumbnail import BLUE, PANEL, RED, YELLOW
from video_cards import C_POLICY
from p3_game121 import M, seg
from sim.board import Board, geometry

OUT = os.path.join(HERE, "p3"); FPS = 25; K = 60
G = geometry(10, 7)


def policy():
    import torch
    from agents.az_net import Codec, N_PLANES_2, make_net
    from _common import load_config
    net = make_net(load_config(os.path.join(ROOT, "configs/dot2c.yaml")))
    net.load_state_dict(torch.load(os.path.join(ROOT, "runs/dot2c/gen_0016.pt"), map_location="cpu")); net.eval()
    b = Board(10, 7)
    for ln in M[:K]:
        b.step(ln)
    c = Codec(10, 7, planes=N_PLANES_2)
    x = torch.from_numpy(c.encode_boards([b]).reshape(1, -1, c.rows, c.cols))
    with torch.no_grad():
        p = net(x)[0].flatten()
    legal = [ln for ln in range(len(p)) if ln not in set(M[:K])]
    return sorted(legal, key=lambda ln: -float(p[ln]))


def board(d, x0, y0, s, hi=None, hi_col=YELLOW):
    b = Board(10, 7); who = {}
    for ln in M[:K]:
        who[ln] = b.player; b.step(ln)
    for i, o in enumerate(b.owner):
        if o in (0, 1):
            x, y = i % 10, i // 10
            d.rectangle((x0 + x * s + 3, y0 + y * s + 3, x0 + (x + 1) * s - 3, y0 + (y + 1) * s - 3),
                        fill=tuple(int(c * .55) for c in (RED if o == 0 else BLUE)))
    for ln, p in who.items():
        a, e = seg(G, ln)
        d.line((x0 + a[0] * s, y0 + a[1] * s, x0 + e[0] * s, y0 + e[1] * s), fill=RED if p == 0 else BLUE, width=4)
    if hi is not None:
        a, e = seg(G, hi)
        d.line((x0 + a[0] * s, y0 + a[1] * s, x0 + e[0] * s, y0 + e[1] * s), fill=hi_col, width=10)
    for y in range(8):
        for x in range(11):
            d.ellipse((x0 + x * s - 5, y0 + y * s - 5, x0 + x * s + 5, y0 + y * s + 5), fill=FG)


def two_boards(top, seconds=24):
    best = top[0]; cand = top[:8]
    SEARCH = 9.0
    for f in range(int(seconds * FPS)):
        t = f / FPS
        img, d = page()
        d.text((110, 150), T("без размышлений"), font=font(46), fill=C_POLICY)
        d.text((1010, 150), T("с размышлениями"), font=font(46), fill=YELLOW)
        s = 78
        board(d, 110, 280, s, best if t > 0.6 else None)
        d.text((110, 900), T("взглянула и сразу ходит") if t > 0.6 else "", font=font(36, LATO), fill=DIM)
        done = t > 1 + SEARCH
        n = 200 if done else int(max(0, t - 1) / SEARCH * 200)
        hi = top[0] if done else (cand[int(t * 7) % len(cand)] if t > 1 else None)
        board(d, 1010, 280, s, hi, YELLOW if done else FAINT)
        d.text((1010, 900), T("вариантов:") + f" {n}", font=font(36, LATO), fill=DIM)
        yield img


def conv_window(seconds=26):
    s, x0, y0 = 90, 130, 240
    chain = [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (7, 4), (7, 3), (8, 3)]
    SLIDE = 15.0
    for f in range(int(seconds * FPS)):
        t = f / FPS
        img, d = page(T("Что видит сеть"))
        for x, y in chain:
            d.rectangle((x0 + x * s + 8, y0 + y * s + 8, x0 + (x + 1) * s - 8, y0 + (y + 1) * s - 8), fill=PANEL)
        for a, e in zip(chain, chain[1:]):
            d.line((x0 + (a[0] + .5) * s, y0 + (a[1] + .5) * s, x0 + (e[0] + .5) * s, y0 + (e[1] + .5) * s), fill=C_POLICY, width=10)
        for y in range(8):
            for x in range(11):
                d.ellipse((x0 + x * s - 5, y0 + y * s - 5, x0 + x * s + 5, y0 + y * s + 5), fill=FAINT)
        tx = 1160
        if t < SLIDE:
            u = t / SLIDE * 3
            k = int(u); r = u - k
            path = [(0, 4), (7, 4), (2, 1), (7, 1)]
            a, e = path[k], path[min(k + 1, 3)]
            wx, wy = a[0] + (e[0] - a[0]) * r, a[1] + (e[1] - a[1]) * r
            d.rounded_rectangle((x0 + wx * s, y0 + wy * s, x0 + (wx + 3) * s, y0 + (wy + 3) * s), 12, outline=YELLOW, width=7)
            d.multiline_text((tx, 330), T("свёрточное окошко:\nмаленькие узоры"), font=font(44), fill=YELLOW, spacing=12)
            if t > 8:
                d.multiline_text((tx, 520), T("цепочка длиннее окошка"), font=font(38, LATO), fill=FG, spacing=10)
        else:
            d.rounded_rectangle((x0 - 24, y0 - 24, x0 + 10 * s + 24, y0 + 7 * s + 24), 16, outline=YELLOW, width=7)
            d.multiline_text((tx, 330), T("взгляд на всё поле"), font=font(44), fill=YELLOW)
            if t > SLIDE + 1.5:
                d.text((tx, 450), "+", font=font(70), fill=FG)
                d.multiline_text((tx, 560), T("подсказка:\nкакие цепочки,\nкому открывать первую"), font=font(40, LATO), fill=FG, spacing=10)
        yield img


def funnel(seconds=31, at=(11.6, 13.6, 14.9), fade=0.4):
    """Scene 8: 5500 -> 248 -> 80, each number appears at its time on the voice."""
    from PIL import Image
    steps = []
    for k in range(4):
        img, d = page(T("Попытка 5: на своих ошибках"))
        if k == 0:
            d.text((960, 460), T("настоящие ошибки самой Dot-2"), font=font(60), fill=FG, anchor="mm")
        for i, (n, l) in enumerate((("5500", T("партий")), ("248", T("разгромов")), ("80", T("ошибок в середине")))[:k]):
            x = 340 + i * 620
            d.text((x, 420), n, font=font(120), fill=YELLOW if i == 2 else FG, anchor="mm")
            d.text((x, 520), l, font=font(40, LATO), fill=DIM, anchor="mm")
            if i: arrow(d, (x - 450, 420), (x - 200, 420))
        steps.append(img.convert("RGB"))
    for f in range(round(seconds * FPS)):
        t = f / FPS
        k = sum(t >= a for a in at)
        a = (t - at[k - 1]) / fade if k else 1
        yield Image.blend(steps[k - 1], steps[k], a) if k and a < 1 else steps[k]


def window_vs_graph(seconds=32, at=4.4, fade=0.4):
    """Scene 10, schematic: the window on the board; then, at the word "граф",
    chains as a graph: junctions are nodes, chains are edges with their length."""
    from PIL import Image
    steps = []
    for k in range(2):
        img, d = page(T("Окошко или граф"))
        x0, y0, c = 110, 330, 70
        for i in range(11):
            for j in range(8):
                d.ellipse((x0 + i * c - 5, y0 + j * c - 5, x0 + i * c + 5, y0 + j * c + 5), fill=FAINT)
        d.rectangle((x0 + 2 * c - 8, y0 + 2 * c - 8, x0 + 4 * c + 8, y0 + 4 * c + 8), outline=YELLOW, width=7)
        d.text((x0 + 5 * c, y0 + 7 * c + 70), T("свёрточная сеть: окошко"), font=font(44, LATO), fill=FG, anchor="mm")
        if k:
            nodes = [(1140, 420), (1480, 340), (1780, 470), (1240, 700), (1560, 640), (1800, 780)]
            edges = [(0, 1, "8", RED), (1, 2, "5", C_POLICY), (0, 3, "3", YELLOW), (1, 4, "10", RED),
                     (3, 4, "7", C_POLICY), (4, 5, "4", YELLOW), (2, 4, "3", RED)]
            for a, b, n, col in edges:
                (p, q), (r, s) = nodes[a], nodes[b]
                d.line((p, q, r, s), fill=col, width=12)
            for x, y in nodes:
                d.ellipse((x - 26, y - 26, x + 26, y + 26), fill=FG)
            for a, b, n, col in edges:
                (p, q), (r, s) = nodes[a], nodes[b]
                x, y = (p + r) / 2, (q + s) / 2
                d.ellipse((x - 34, y - 34, x + 34, y + 34), fill=PANEL)
                d.text((x, y), n, font=font(40), fill=FG, anchor="mm")
            d.text((1460, y0 + 7 * c + 70), T("графовая сеть: цепочки готовыми"), font=font(44, LATO), fill=FG, anchor="mm")
        steps.append(img.convert("RGB"))
    for f in range(round(seconds * FPS)):
        a = (f / FPS - at) / fade
        yield steps[1] if a >= 1 else steps[0] if a <= 0 else Image.blend(steps[0], steps[1], a)


def write(name, frames):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080",
                          "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18",
                          os.path.join(OUT, name.replace(".mp4", suffix() + ".mp4"))], stdin=subprocess.PIPE)
    for im in frames:
        p.stdin.write(im.convert("RGB").tobytes())
    p.stdin.close(); p.wait(); print(name)


if __name__ == "__main__":
    write("two_boards.mp4", two_boards(policy()))
    write("conv_window.mp4", conv_window())
    if T("партий") == "партий":
        write("funnel.mp4", funnel())
        write("funnel10.mp4", funnel(seconds=25, at=(3.1, 4.8, 6.2)))
        write("window_graph.mp4", window_vs_graph())
    else:                               # English: times from the voice (sentence 3 of scene 8, etc.)
        import json
        a = json.load(open(os.path.join(OUT, "times_en.json")))
        write("funnel.mp4", funnel(at=tuple(a["funnel"])))
        write("funnel10.mp4", funnel(seconds=25, at=tuple(a["funnel10"])))
        write("window_graph.mp4", window_vs_graph(at=a["graph"]))
