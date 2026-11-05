#!/usr/bin/env python3
"""YouTube thumbnail of the part 3 video (docs/video_script_part3_ru.md).

Text: "Понять, а не пересчитать", as the part 2 thumbnail (video/p2_thumbnail.py,
whose helpers this reuses). The board is real: game seed 121 of runs/blowouts
(the 1:69 rout of the video) at the last position with no safe line left and
no box taken yet by the player to move, chains of 3+ coloured with their length.

    .venv/bin/python video/p3_thumbnail.py   # video/p3/thumbnail_ru.png, thumbnail_ru_320.png
"""
import os, sys
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "scripts")]
from p2_thumbnail import colour_chains, safe
from p3_game121 import M
from sim.board import Board
from video_thumbnail import BG, H, W, WHITE, YELLOW, BoardPainter, chain_boxes, fit_font, panel, text


def position():
    """The board at the first moment no safe line is left."""
    b = Board(10, 7)
    for ln in M:
        if not any(safe(b, x) for x in b.legal_moves()):
            return b
        b.step(ln)
    raise SystemExit("no such position")


def render(b):
    groups = [g for g in chain_boxes(b) if len(g) >= 3]
    print("chains", sorted((len(g) for g in groups), reverse=True))
    colours = colour_chains(b, groups)
    c = Image.new("RGBA", (W, H), BG + (255,))
    cell = 88
    x0, y0 = W - 10 * cell - 110, (H - 7 * cell) // 2 + 30
    pad = cell * 0.45
    panel(c, (x0 - pad, y0 - pad, x0 + 10 * cell + pad, y0 + 7 * cell + pad))
    p = BoardPainter(c, b, x0, y0, cell)
    p.boxes(colours); p.lines()
    for g in groups:
        p.label(sorted(g)[len(g) // 2], str(len(g)), round(cell * 0.5))
    lines = [("Понять,", WHITE), ("а не", WHITE), ("пересчитать", YELLOW)]
    left = x0 - pad - 90
    size = min(fit_font(s, left, 190).size for s, _ in lines)
    y = H / 2 - size * 1.1
    for s, col in lines:
        text(c, s, 60, y, left, size, col, anchor="lm")
        y += size * 1.1
    return c.convert("RGB")


if __name__ == "__main__":
    img = render(position())
    img.resize((1280, 720), Image.LANCZOS).save(os.path.join(HERE, "p3", "thumbnail_ru.png"))
    img.resize((320, 180), Image.LANCZOS).save(os.path.join(HERE, "p3", "thumbnail_ru_320.png"))
