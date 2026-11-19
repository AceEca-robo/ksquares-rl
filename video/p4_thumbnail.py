#!/usr/bin/env python3
"""YouTube thumbnail of the part 4 video: Dot-G's graph of chains (yellow) against Dot's window on the board (blue),
under the title.

    .venv/bin/python video/p4_thumbnail.py     # video/p4/thumbnail_{ru,en}.png (1280x720) and _320
"""
import os
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from video_cards import font  # noqa: E402
from video_thumbnail import BG, BLUE, CHAIN_COLOURS, H, PANEL, W, WHITE, YELLOW  # noqa: E402

TEXT = {"ru": ("Dot-G видит лучше.", "Dot считает быстрее."), "en": ("Dot-G sees better.", "Dot thinks faster.")}


def render(lang):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    a, b = TEXT[lang]
    size = 150
    while max(d.textlength(s, font=font(size)) for s in (a, b)) > W - 140:
        size -= 2
    d.text((70, 50), a, font=font(size), fill=YELLOW)
    d.text((70, 50 + size * 1.2), b, font=font(size), fill=BLUE)
    top = int(110 + size * 2.4)
    # left: graph of chains
    nodes = [(130, top + 120), (470, top + 40), (800, top + 170), (250, top + 420), (600, top + 380), (880, top + 470)]
    edges = [(0, 1, "8"), (1, 2, "5"), (0, 3, "3"), (1, 4, "10"), (3, 4, "7"), (4, 5, "4"), (2, 4, "3")]
    for i, (p, q, _) in enumerate(edges):
        d.line(nodes[p] + nodes[q], fill=CHAIN_COLOURS[i % len(CHAIN_COLOURS)], width=22)
    for x, y in nodes:
        d.ellipse((x - 34, y - 34, x + 34, y + 34), fill=WHITE)
    for p, q, n in edges:
        x, y = (nodes[p][0] + nodes[q][0]) / 2, (nodes[p][1] + nodes[q][1]) / 2
        d.ellipse((x - 46, y - 46, x + 46, y + 46), fill=PANEL)
        d.text((x, y), n, font=font(52), fill=WHITE, anchor="mm")
    # right: dots and a small window
    c, x0, y0 = 105, 1080, top + 10
    for j in range(6):
        for i in range(8):
            d.ellipse((x0 + i * c - 11, y0 + j * c - 11, x0 + i * c + 11, y0 + j * c + 11), fill=(120, 126, 140))
    d.rounded_rectangle((x0 + 2 * c - 26, y0 + 1 * c - 26, x0 + 4 * c + 26, y0 + 3 * c + 26), 18, outline=BLUE, width=14)
    return img


if __name__ == "__main__":
    for lang in TEXT:
        img = render(lang)
        img.resize((1280, 720), Image.LANCZOS).save(os.path.join(HERE, "p4", f"thumbnail_{lang}.png"))
        img.resize((320, 180), Image.LANCZOS).save(os.path.join(HERE, "p4", f"thumbnail_{lang}_320.png"))
        print(lang)
