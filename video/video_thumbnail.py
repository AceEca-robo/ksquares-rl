#!/usr/bin/env python3
"""YouTube thumbnails for the KSquares video (docs/video_script_ru.md).

Text: "Dot знает, кто выиграет. / Но не знает, как." (en: "The network
knows who will win. / But not how."), white on top, yellow below, Lato
Black on a dark blurred board, as in the Pacman thumbnails
(pacman-rl/video/video_thumbnail.py).

Boards, both real:
- CHAIN_GAME: KSquares Hard vs Hard in the simulator (sim/ksquares_ai.py,
  GlibcRand(3)) played until no safe line is left: five long chains
  6 6 4 4 4, no box taken yet. Each chain gets its own colour. Layout b
  adds gen 54's value for that position (runs/best/best.pt), checked
  against the exact result (agents/strong.py) before it is drawn.
- PROBE: parity probe position #172 from docs/probes/az5b/gen_54.json
  (one winning line out of 31): the policy as a heat map (its argmax is
  wrong) and the value's sign after every legal move (right for the one
  winning line and 27 of the 30 losing ones).

    .venv/bin/python video/video_thumbnail.py            # layouts a, b, c; ru + en

Writes video/thumb/thumb_{layout}_{lang}.png (1280x720), _1080.png
(1920x1080) and _320.png (320x180, the readability check); the chosen
layout (FINAL, c) also goes to docs/thumb_{layout}_{lang}.png (in git).
"""

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from sim.board import Board  # noqa: E402

FONT = "/usr/share/fonts/truetype/lato/Lato-Black.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/lato/Lato-Bold.ttf"
W, H = 1920, 1080
BG = (17, 19, 24)
PANEL = (27, 30, 38)
YELLOW = (255, 214, 0)
WHITE = (255, 255, 255)
RED = (236, 94, 80)           # KSquares / app colours: first player red, second blue
BLUE = (88, 150, 245)
GREEN = (80, 200, 120)
LINE = (215, 218, 226)        # drawn line
GHOST = (48, 52, 62)          # undrawn line
CHAIN_COLOURS = [(255, 184, 0), (0, 196, 178), (170, 120, 255), (255, 96, 150), (120, 205, 80)]

CHAIN_GAME = [33, 13, 23, 26, 16, 7, 36, 52, 53, 14, 0, 29, 11, 41, 56, 49, 1, 20, 4, 37,
              18, 38, 3, 46, 27, 55, 45, 43, 10, 54, 30]
PROBE = (os.path.join(ROOT, "docs/probes/az5b/gen_54.json"),
         os.path.join(ROOT, "docs/probes/positions.json"), 172)

TEXT = {
    "ru": ("Dot знает, кто выиграет.", "Но не знает, как.",
           "Dot выбирает ход: какой?", "интуиция после каждого хода", "кто выиграет"),
    "en": ("Dot knows who wins.", "Just not how.",
           "Dot picks a move: which?", "intuition after each move", "who wins"),
}


# ---------------------------------------------------------------- board geometry

def line_ends(i, w):
    """Dot coordinates (col, row) of line i in KSquares numbering."""
    r, c = divmod(i, 2 * w + 1)
    if c < w:
        return (c, r), (c + 1, r)
    return (c - w, r), (c - w, r + 1)


def chain_boxes(board):
    """Lists of boxes, one per chain or loop (free boxes with two undrawn
    sides, joined through undrawn lines), longest first."""
    geo, sides = board.geo, board.sides
    link = {b for b in range(geo.n_boxes) if sides[b] == 2}
    seen, out = set(), []
    for b in sorted(link):
        if b in seen:
            continue
        group, stack = [], [b]
        seen.add(b)
        while stack:
            x = stack.pop()
            group.append(x)
            for ln in geo.box_lines[x]:
                if board.lines[ln]:
                    continue
                for o in geo.line_boxes[ln]:
                    if o != x and o in link and o not in seen:
                        seen.add(o)
                        stack.append(o)
        out.append(group)
    return sorted(out, key=len, reverse=True)


class BoardPainter:
    def __init__(self, img, board, x0, y0, cell):
        self.img, self.b, self.x0, self.y0, self.cell = img, board, x0, y0, cell
        self.d = ImageDraw.Draw(img)

    def xy(self, p):
        return self.x0 + p[0] * self.cell, self.y0 + p[1] * self.cell

    def box_rect(self, b, inset):
        w = self.b.geo.w
        x, y = self.xy((b % w, b // w))
        return (x + inset, y + inset, x + self.cell - inset, y + self.cell - inset)

    def boxes(self, colours, alpha=150):
        over = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        for b, col in colours.items():
            d.rounded_rectangle(self.box_rect(b, self.cell * 0.09), radius=self.cell * 0.08,
                                fill=col + (alpha,))
        self.img.alpha_composite(over)

    def lines(self, colour_of=None, width=None):
        """Drawn lines light, undrawn ones dim or coloured by colour_of(i)."""
        width = width or max(4, round(self.cell * 0.075))
        w = self.b.geo.w
        over = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        for i in range(self.b.geo.n_lines):
            a, z = (self.xy(p) for p in line_ends(i, w))
            if self.b.lines[i]:
                d.line((a, z), fill=LINE + (255,), width=width)
            else:
                col = colour_of(i) if colour_of else None
                if col is None:
                    d.line((a, z), fill=GHOST + (255,), width=max(2, width // 3))
                else:
                    d.line((a, z), fill=col, width=round(width * 1.6))
        self.img.alpha_composite(over)
        r = width * 0.95
        for y in range(self.b.geo.h + 1):
            for x in range(w + 1):
                cx, cy = self.xy((x, y))
                self.d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=LINE)

    def ring(self, i, colour, r=None):
        a, z = (self.xy(p) for p in line_ends(i, self.b.geo.w))
        cx, cy = (a[0] + z[0]) / 2, (a[1] + z[1]) / 2
        r = r or self.cell * 0.2
        self.d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=colour, width=max(4, round(r * 0.2)))
        return cx, cy, r

    def label(self, b, s, size, colour=WHITE):
        x0, y0, x1, y1 = self.box_rect(b, 0)
        f = ImageFont.truetype(FONT, size)
        self.d.text(((x0 + x1) / 2, (y0 + y1) / 2), s, font=f, fill=colour, anchor="mm",
                    stroke_width=max(2, size // 12), stroke_fill=(10, 10, 14))


# ---------------------------------------------------------------- positions

def chain_board():
    b = Board(5, 5)
    for m in CHAIN_GAME:
        b.step(m)
    return b


def probe_position():
    snap_path, pos_path, idx = PROBE
    snap = next(p for p in json.load(open(snap_path))["positions"] if p["probe_index"] == idx)
    pos = next(p for p in json.load(open(pos_path))["positions"] if p["probe_index"] == idx)
    b = Board(5, 5)
    for m in pos["moves"]:
        b.step(m)
    assert b.player == snap["player"]
    return b, snap


def chain_value(board):
    """gen 54's V for the side to move, and the exact best result for it."""
    import yaml
    import torch

    from agents.az_net import Evaluator, make_net
    from agents.strong import Engine, exact_move_values

    rc = yaml.safe_load(open(os.path.join(ROOT, "configs/az5b.yaml")))
    net = make_net(rc)
    net.load_state_dict(torch.load(os.path.join(ROOT, "runs/best/best.pt"), map_location="cpu"))
    _, v = Evaluator(net, "cpu")([board])
    vals = exact_move_values(Engine(board.geo.w, board.geo.h), board, None)
    best = max(vals.values())
    return float(v[0]), best


# ---------------------------------------------------------------- drawing helpers

def fit_font(text, max_w, size, path=FONT):
    while size > 20:
        f = ImageFont.truetype(path, size)
        if f.getlength(text) <= max_w:
            return f
        size -= 2
    return f


def text(canvas, s, cx, cy, max_w, size, fill, anchor="mm", path=FONT):
    f = fit_font(s, max_w, size, path)
    stroke = max(4, f.size // 11)
    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).text((cx, cy), s, font=f, fill=(0, 0, 0, 220), anchor=anchor,
                              stroke_width=stroke * 2, stroke_fill=(0, 0, 0, 220))
    canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(stroke)))
    ImageDraw.Draw(canvas).text((cx, cy), s, font=f, fill=fill, anchor=anchor, stroke_width=stroke,
                                stroke_fill=(8, 8, 12))
    return f


def background(board, colours):
    """The chain board, huge, blurred and darkened."""
    c = Image.new("RGBA", (W, H), BG + (255,))
    cell = 300
    p = BoardPainter(c, board, (W - 5 * cell) // 2 + 260, (H - 5 * cell) // 2 + 40, cell)
    p.boxes(colours, alpha=110)
    p.lines()
    c = c.filter(ImageFilter.GaussianBlur(10))
    return Image.fromarray((np.asarray(c).astype(np.float32) * [0.32, 0.32, 0.32, 1]).astype(np.uint8))


def panel(canvas, box, radius=28):
    sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    x0, y0, x1, y1 = box
    ImageDraw.Draw(sh).rounded_rectangle((x0, y0 + 10, x1, y1 + 10), radius, fill=(0, 0, 0, 200))
    canvas.alpha_composite(sh.filter(ImageFilter.GaussianBlur(16)))
    ImageDraw.Draw(canvas).rounded_rectangle(box, radius, fill=PANEL + (255,))


def chain_colours(board):
    groups = [g for g in chain_boxes(board) if len(g) >= 3]
    return groups, {b: CHAIN_COLOURS[k % len(CHAIN_COLOURS)] for k, g in enumerate(groups) for b in g}


def chain_panel(canvas, board, x0, y0, cell, labels=True):
    pad = cell * 0.45
    panel(canvas, (x0 - pad, y0 - pad, x0 + 5 * cell + pad, y0 + 5 * cell + pad))
    groups, colours = chain_colours(board)
    p = BoardPainter(canvas, board, x0, y0, cell)
    p.boxes(colours)
    p.lines()
    if labels:
        for g in groups:
            # the label goes in the chain's middle box
            p.label(sorted(g)[len(g) // 2], str(len(g)), round(cell * 0.5))
    return p


def value_bar(canvas, x0, y0, w, h, v_red, caption):
    """Who-is-winning bar as in the app: red share = (1 + V_red) / 2."""
    d = ImageDraw.Draw(canvas)
    split = x0 + w * (1 + v_red) / 2
    d.rounded_rectangle((x0, y0, x1 := x0 + w, y0 + h), h // 2, fill=BLUE)
    d.rounded_rectangle((x0, y0, split, y0 + h), h // 2, fill=RED)
    d.line((x0 + w / 2, y0 - 8, x0 + w / 2, y0 + h + 8), fill=WHITE, width=4)
    f = ImageFont.truetype(FONT_BOLD, round(h * 0.75))
    d.text(((x0 + x1) / 2, y0 + h + 18), caption, font=f, fill=(200, 204, 214), anchor="mt")


# ---------------------------------------------------------------- layouts

def layout_a(lang, board, colours, **_):
    """The chain board on the left; the two lines stacked on the right."""
    c = background(board, colours)
    cell = 150
    chain_panel(c, board, 120, (H - 5 * cell) // 2, cell)
    t1, t2 = TEXT[lang][:2]
    right_x, right_w = 120 + 5 * cell + 140, W - (120 + 5 * cell + 140) - 70
    words = t1.rstrip(".").split(" ")
    cut = 2                                          # «Dot знает,» / «кто выиграет.»
    l1, l2 = " ".join(words[:cut]), " ".join(words[cut:]) + "."
    size = min(fit_font(l1, right_w, 150).size, fit_font(l2, right_w, 150).size)
    text(c, l1, right_x, 290, right_w, size, WHITE, anchor="lm")
    text(c, l2, right_x, 290 + size * 1.15, right_w, size, WHITE, anchor="lm")
    text(c, t2, right_x, 290 + size * 2.75, right_w, size, YELLOW, anchor="lm")
    return c


def layout_b(lang, board, colours, v_red, **_):
    """Pacman-style: line 1 on top, the chain board with the value bar in
    the middle, line 2 in yellow at the bottom."""
    c = background(board, colours)
    t1, t2, *_rest, who = TEXT[lang]
    text(c, t1, W // 2, 110, W - 100, 120, WHITE)
    cell = 120
    bx = 230
    by = 225
    chain_panel(c, board, bx, by, cell)
    vx = bx + 5 * cell + 150
    vw = W - vx - 120
    panel(c, (vx - 50, by - 54, vx + vw + 50, by + 5 * cell + 54))
    d = ImageDraw.Draw(c)
    f = ImageFont.truetype(FONT, 92)
    d.text((vx, by + 40), f"V = {v_red:+.2f}", font=f, fill=WHITE, anchor="lm")
    value_bar(c, vx, by + 160, vw, 58, v_red, who)
    f2 = ImageFont.truetype(FONT_BOLD, 52)
    note = {"ru": "длинных цепочек: 5", "en": "long chains: 5"}[lang]
    d.text((vx, by + 400), note, font=f2, fill=(200, 204, 214), anchor="lm")
    q = ImageFont.truetype(FONT, 110)
    move = {"ru": "ход: ?", "en": "move: ?"}[lang]
    d.text((vx, by + 5 * cell - 50), move, font=q, fill=YELLOW, anchor="ls")
    text(c, t2, W // 2, 975, W - 140, 150, YELLOW)
    return c


def layout_c(lang, board, colours, probe, snap, **_):
    """Two copies of probe #172: the policy heat map (argmax wrong, marked
    "?") and the value's sign after every move (green = the mover wins,
    red = loses; the winning line ringed)."""
    c = background(board, colours)
    t1, t2, cap_pol, cap_val, _who = TEXT[lang]
    text(c, t1, W // 2, 105, W - 100, 120, WHITE)
    cell = 110
    gap = 230
    x_left = (W - 2 * 5 * cell - gap) // 2
    x_right = x_left + 5 * cell + gap
    y0 = 300
    pad = cell * 0.4
    for x in (x_left, x_right):
        panel(c, (x - pad, y0 - pad - 76, x + 5 * cell + pad, y0 + 5 * cell + pad))
    pol = {int(k): v for k, v in snap["policy"].items()}
    top = max(pol.values())
    argmax = max(pol, key=pol.get)
    p = BoardPainter(c, probe, x_left, y0, cell)
    p.lines(lambda i: YELLOW + (round(40 + 215 * (pol[i] / top) ** 0.6),) if i in pol else None)
    cx, cy, r = p.ring(argmax, WHITE, cell * 0.24)
    f = ImageFont.truetype(FONT, round(cell * 0.42))
    ImageDraw.Draw(c).text((cx, cy - r - 6), "?", font=f, fill=WHITE, anchor="mb",
                           stroke_width=4, stroke_fill=(8, 8, 12))
    cv = {int(k): v for k, v in snap["child_v"].items()}
    q = ImageDraw.Draw(c)
    p2 = BoardPainter(c, probe, x_right, y0, cell)
    p2.lines(lambda i: ((GREEN if cv[i] > 0 else RED) + (round(70 + 185 * min(1, abs(cv[i]))),))
             if i in cv else None)
    for i in snap["correct"]:
        p2.ring(i, WHITE, cell * 0.24)
    fc = ImageFont.truetype(FONT_BOLD, 44)
    for x, cap in ((x_left, cap_pol), (x_right, cap_val)):
        q.text((x + 5 * cell / 2, y0 - pad - 20), cap, font=fc, fill=(220, 222, 230), anchor="mm")
    text(c, t2, W // 2, 985, W - 140, 140, YELLOW)
    return c


LAYOUTS = {"a": layout_a, "b": layout_b, "c": layout_c}
FINAL = "c"                                         # picked 2026-10-26


def main():
    board = chain_board()
    _, colours = chain_colours(board)
    v, best = chain_value(board)
    # V is for the side to move; the bar shows the first player's (red) view
    v_red = v if board.player == 0 else -v
    assert (v > 0) == (best > 0), f"gen 54 V {v:+.2f} disagrees with the exact result {best:+d}"
    print(f"chain position: mover {board.player}, V {v:+.3f}, exact best margin {best:+d}")
    probe, snap = probe_position()
    out = os.path.join(HERE, "thumb")
    os.makedirs(out, exist_ok=True)
    for name, fn in LAYOUTS.items():
        for lang in TEXT:
            img = fn(lang, board=board, colours=colours, v_red=v_red, probe=probe, snap=snap)
            img = img.convert("RGB")
            img.save(os.path.join(out, f"thumb_{name}_{lang}_1080.png"))
            small = img.resize((1280, 720), Image.LANCZOS)
            small.save(os.path.join(out, f"thumb_{name}_{lang}.png"))
            if name == FINAL:
                small.save(os.path.join(ROOT, "docs", f"thumb_{name}_{lang}.png"))
            img.resize((320, 180), Image.LANCZOS).save(os.path.join(out, f"thumb_{name}_{lang}_320.png"))
            print("wrote", name, lang)


if __name__ == "__main__":
    main()
