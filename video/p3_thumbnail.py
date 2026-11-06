#!/usr/bin/env python3
"""YouTube thumbnail of the part 3 video: the "who knows what" map (video/p3_map.py)
under the title. The curves are the real data: green Dot-2 with thinking, blue
without, white the judge's share solved, and the dark box with the question mark.

    .venv/bin/python video/p3_thumbnail.py             # video/p3/thumbnail_ru.png, thumbnail_ru_320.png
    P3_LANG=en .venv/bin/python video/p3_thumbnail.py  # the same with _en
"""
import os, sys
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p3_i18n import LANG, T
from p3_map import DATA, SOLVED
from video_cards import C_POLICY, PLUS, font
from video_thumbnail import BG, H, W, WHITE, YELLOW

X0, B0, B1, X1, Y0, Y1 = 60, 60, 760, W - 70, 430, H - 60
fx = lambda s: B1 + (20 - s) / 20 * (X1 - B1)
fy = lambda v: Y1 - v / 100 * (Y1 - Y0)


def render():
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    size = 118
    while max(d.textlength(T(s), font=font(size)) for s in ("Что знает нейросеть —", "и где кончается знание")) > W - 120:
        size -= 2
    d.text((60, 70), T("Что знает нейросеть —"), font=font(size), fill=WHITE)
    d.text((60, 70 + size * 1.25), T("и где кончается знание"), font=font(size), fill=YELLOW)
    d.rectangle((B0, Y0, B1, Y1), fill=(5, 6, 9), outline=(70, 75, 90), width=4)
    d.text(((B0 + B1) / 2, (Y0 + Y1) / 2), "?", font=font(400), fill=YELLOW, anchor="mm")
    for k, col in ((0, C_POLICY), (2, PLUS)):
        pts = [(fx(s), fy(v[k])) for s, v in sorted(DATA.items(), reverse=True)]
        d.line(pts, fill=col, width=16, joint="curve")
    pts = [(fx(s), fy(v)) for s, v in sorted(SOLVED.items(), reverse=True)]
    d.line(pts, fill=WHITE, width=16, joint="curve")
    return img


if __name__ == "__main__":
    img = render()
    img.resize((1280, 720), Image.LANCZOS).save(os.path.join(HERE, "p3", f"thumbnail_{LANG}.png"))
    img.resize((320, 180), Image.LANCZOS).save(os.path.join(HERE, "p3", f"thumbnail_{LANG}_320.png"))
