#!/usr/bin/env python3
"""YouTube thumbnails for the part 2 video (docs/video_script_part2_ru.md).

Text: "Dot переезжает / на большую доску" (en: "Dot moves / to the big
board"), white then yellow, Lato Black, as the part 1 thumbnails
(video/video_thumbnail.py, whose drawing helpers this reuses). The board
is real: KSquares Hard vs Hard on 10x7 in the simulator (sim/ksquares_ai.py,
GlibcRand(SEED)) played until no safe line is left, no box taken yet; each
chain of 3+ boxes gets its own colour and its length.

    .venv/bin/python video/p2_thumbnail.py

Writes video/p2/thumbnail_{lang}.png (1280x720) and _320.png (readability).
"""

import os
import sys

from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from sim.board import Board  # noqa: E402
from sim.ksquares_ai import GlibcRand, make_ai  # noqa: E402
from video_thumbnail import (BG, CHAIN_COLOURS, H, W, WHITE, YELLOW, BoardPainter,  # noqa: E402
                             chain_boxes, fit_font, panel, text)

SEEDS = range(1, 200)
TEXT = {"ru": ("Dot переезжает", "на большую доску"), "en": ("Dot moves", "to the big board")}


def safe(board, ln):
    return all(board.sides[b] < 2 for b in board.geo.line_boxes[ln])


def chain_position(seed):
    """Hard vs Hard on 10x7 until no safe line is left (None if a box was taken)."""
    b = Board(10, 7)
    ai = make_ai("hard", GlibcRand(seed))
    while any(safe(b, ln) for ln in b.legal_moves()):
        before = sum(s == 4 for s in b.sides)
        b.step(ai.choose_line([bool(x) for x in b.lines], 10, 7))
        if sum(s == 4 for s in b.sides) != before:
            return None
    return b


def pick():
    """The position whose chains of 3+ cover the most boxes, at most 7 of them,
    none over 10 boxes (a few long chains read better than many short ones
    or two that fill the board)."""
    best = None
    for s in SEEDS:
        b = chain_position(s)
        if b is None:
            continue
        gs = [g for g in chain_boxes(b) if len(g) >= 3]
        n = sum(map(len, gs))
        if len(gs) <= 7 and max(map(len, gs), default=0) <= 10 and (best is None or n > best[0]):
            best = (n, s, b, [len(g) for g in gs])
    print(f"seed {best[1]}: chains {best[3]}")
    return best[2]


def colour_chains(board, groups):
    """Colours so that touching chains differ (greedy, longest first)."""
    w = board.geo.w
    where = {b: k for k, g in enumerate(groups) for b in g}
    out = []
    for k, g in enumerate(groups):
        near = {where[o] for b in g for o in (b - 1, b + 1, b - w, b + w)
                if o in where and where[o] < k and (abs(o - b) == w or o // w == b // w)}
        used = {out[j] for j in near}
        out.append(next(i for i in range(len(CHAIN_COLOURS)) if i not in used))
    return {b: CHAIN_COLOURS[out[k]] for k, g in enumerate(groups) for b in g}


def render(lang, board):
    groups = [g for g in chain_boxes(board) if len(g) >= 3]
    colours = colour_chains(board, groups)
    c = Image.new("RGBA", (W, H), BG + (255,))
    # the same board huge, blurred and darkened behind
    back = Image.new("RGBA", (W, H), BG + (255,))
    bp = BoardPainter(back, board, -200, -150, 260)
    bp.boxes(colours, alpha=110)
    bp.lines()
    back = back.filter(ImageFilter.GaussianBlur(10))
    c = Image.blend(c, back, 0.35)
    cell = 88
    x0, y0 = W - 10 * cell - 110, (H - 7 * cell) // 2 + 30
    pad = cell * 0.45
    panel(c, (x0 - pad, y0 - pad, x0 + 10 * cell + pad, y0 + 7 * cell + pad))
    p = BoardPainter(c, board, x0, y0, cell)
    p.boxes(colours)
    p.lines()
    for g in groups:
        p.label(sorted(g)[len(g) // 2], str(len(g)), round(cell * 0.5))
    t1, t2 = TEXT[lang]
    left_w = x0 - pad - 90
    lines = [(w, WHITE) for w in t1.split(" ", 1)] + [(w, YELLOW) for w in t2.rsplit(" ", 1)]
    size = min(fit_font(s, left_w, 190).size for s, _ in lines)
    y = H / 2 - size * 1.1 * (len(lines) - 1) / 2
    for s, col in lines:
        text(c, s, 60, y, left_w, size, col, anchor="lm")
        y += size * 1.1
    return c.convert("RGB")


def main():
    board = pick()
    out = os.path.join(HERE, "p2")
    os.makedirs(out, exist_ok=True)
    for lang in TEXT:
        img = render(lang, board)
        img.resize((1280, 720), Image.LANCZOS).save(os.path.join(out, f"thumbnail_{lang}.png"))
        img.resize((320, 180), Image.LANCZOS).save(os.path.join(out, f"thumbnail_{lang}_320.png"))
        print("wrote", lang)


if __name__ == "__main__":
    main()
