#!/usr/bin/env python3
"""Stills for the part 2 video (docs/video_script_part2_ru.md), 1920x1080
png in video/p2/, in the style of video_cards.py (part 1).

    .venv/bin/python video/p2_cards.py            # video/p2/card_*.png
    .venv/bin/python video/p2_cards.py --lang en  # the same with _en

Numbers: docs/transfer_conclusions.md (scenes 3-5), docs/results.md dot2c /
dot2s (6-8); the Elo-against-hours curves are read from the runs as
scripts/elo_vs_time.py does.
"""

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from video_cards import (C_MCTS, C_POLICY, C_VALUE, DIM, FAINT, FG, GRID, LATO, MINUS, PLUS,  # noqa: E402
                         arrow, box, font, hline_label, page)
from video_thumbnail import BG, BLUE, PANEL, RED, YELLOW  # noqa: E402

OUT = os.path.join(HERE, "p2")
MEDIUM, HARD = 1237, 1420          # 10x7 ladder anchors

T = {
    "ru": {
        "boards_title": "Доска побольше",
        "b5": "5×5: 25 квадратов, 60 линий", "b10": "10×7: 70 квадратов, 157 линий",
        "net_title": "Dot-2: одна сеть на любую доску",
        "net_old": "старая Dot\nвыходы прибиты\nк 25 квадратам\n→ только 5×5",
        "net_new": "Dot-2\nодин узор вокруг каждой линии\n+ взгляд на всю доску:\nсколько цепочек, кто впереди",
        "net_steps": ["учитель: старая Dot (5×5)", "Dot-2 доучилась сама на 5×5", "→ 10×7 без единой большой партии"],
        "tr_title": "10×7 без подготовки: Dot-2 ходит сразу, без перебора",
        "tr_rows": [("жадный", 96), ("Medium", 47), ("Hard", 37)],
        "tr_win": "побед из 100",
        "tr_elo": "Эло без перебора: 1369   ·   Medium 1237   ·   Hard 1420",
        "in_title": "Интуиция на большой доске",
        "in_a": "угадывает исход хода", "in_b": "узнаёт выигрышные ходы",
        "in_note": "проигрышные ходы узнаёт по-прежнему (89 → 97 → 99 %)",
        "se_title": "Размышления, которые верят интуиции",
        "se_lose": "проигрыш", "se_good": "верный ход",
        "se_rows": [("Dot-2 ходит сразу", "Эло 1369"), ("Dot-2 думает", "Эло 1322"),
                    ("думает, против Hard", "3 из 20")],
        "rd_title": "Dot-2 с переездом: 5×5 → 7×7 → 10×7",
        "rd_x": "раунд", "rd_mcts": "думает", "rd_raw": "ходит сразу",
        "rd_jump": "первый раунд на 10×7:\n1489 → 1682,\nпротив Hard 62 → 90 %",
        "rd_mark": "партии 10×7",
        "w_title": "Против настоящего KSquares, сложный уровень, 10×7",
        "w_big": "19 побед · 1 ничья · 0 поражений",
        "w_before": "до обучения на большой доске: 3 победы из 20",
        "sc_title": "Dot-2 (школа) против Dot-3 (с нуля): сила по часам",
        "sc_x": "часы обучения", "sc_c": "Dot-2: 5×5 → 10×7", "sc_s": "Dot-3: с нуля на 10×7",
        "sc_note": "сплошная — думает, пунктир — ходит сразу; часы школы 5×5 не считаны",
        "sc_tab": [("Dot-2, раунд 16", "Эло 1755", "19–0–1", "6 ч"),
                   ("Dot-3, раунд 39", "Эло 1413", "12–8", "15 ч")],
        "t_top": "Нейросеть Dot, часть 2",
        "t_big": "Переезд на большую доску",
        "t_sub": "5×5 → 10×7: интуиция врёт, размышления вредят, школа побеждает",
        "q_title": "Что дальше",
        "q_text": "Интуицию вылечили.\nА рука — без перебора — почти не выросла:\nЭло 1369 → 1457",
        "q_ask": "можно ли научить руку видеть чётность цепочек?",
    },
    "en": {
        "boards_title": "A bigger board",
        "b5": "5×5: 25 boxes, 60 lines", "b10": "10×7: 70 boxes, 157 lines",
        "net_title": "Dot-2: one network for any board",
        "net_old": "old Dot\noutputs nailed\nto 25 boxes\n→ 5×5 only",
        "net_new": "Dot-2\nthe same pattern around every line\n+ a look at the whole board:\nhow many chains, who is ahead",
        "net_steps": ["teacher: old Dot (5×5)", "Dot-2 trained on by herself on 5×5", "→ 10×7 without a single big game"],
        "tr_title": "10×7 unprepared: Dot-2 moves at once, no thinking",
        "tr_rows": [("greedy", 96), ("Medium", 47), ("Hard", 37)],
        "tr_win": "wins out of 100",
        "tr_elo": "Elo without thinking: 1369   ·   Medium 1237   ·   Hard 1420",
        "in_title": "Intuition on the big board",
        "in_a": "guesses a move's outcome", "in_b": "recognises winning moves",
        "in_note": "still recognises losing moves (89 → 97 → 99 %)",
        "se_title": "Thinking that trusts intuition",
        "se_lose": "losing", "se_good": "the right move",
        "se_rows": [("Dot-2 moves at once", "Elo 1369"), ("Dot-2 thinks", "Elo 1322"),
                    ("thinks, against Hard", "3 of 20")],
        "rd_title": "Dot-2 moves house: 5×5 → 7×7 → 10×7",
        "rd_x": "round", "rd_mcts": "thinks", "rd_raw": "moves at once",
        "rd_jump": "first round on 10×7:\n1489 → 1682,\nagainst Hard 62 → 90 %",
        "rd_mark": "10×7 games",
        "w_title": "Against the real KSquares, Hard, 10×7",
        "w_big": "19 wins · 1 draw · 0 losses",
        "w_before": "before training on the big board: 3 wins out of 20",
        "sc_title": "Dot-2 (school) vs Dot-3 (from scratch): strength by hours",
        "sc_x": "hours of training", "sc_c": "Dot-2: 5×5 → 10×7", "sc_s": "Dot-3: from scratch on 10×7",
        "sc_note": "solid: thinks, dashed: moves at once; the 5×5 school hours not counted",
        "sc_tab": [("Dot-2, round 16", "Elo 1755", "19–0–1", "6 h"),
                   ("Dot-3, round 39", "Elo 1413", "12–8", "15 h")],
        "t_top": "The Dot network, part 2",
        "t_big": "Moving to the big board",
        "t_sub": "5×5 → 10×7: intuition lies, thinking hurts, school wins",
        "q_title": "What next",
        "q_text": "Intuition is cured.\nBut the hand — without thinking — barely grew:\nElo 1369 → 1457",
        "q_ask": "can the hand learn to see chain parity?",
    },
}


def save(img, name, lang):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}{'' if lang == 'ru' else '_en'}.png")
    img.save(path)
    print("wrote", path)


def dots(d, x, y, w, h, c, r=7, col=FG):
    for i in range(w + 1):
        for j in range(h + 1):
            d.ellipse((x + i * c - r, y + j * c - r, x + i * c + r, y + j * c + r), fill=col)


def card_boards(lang):
    t = T[lang]
    img, d = page(t["boards_title"])
    dots(d, 180, 330, 5, 5, 90)
    dots(d, 860, 330, 10, 7, 90)
    d.text((405, 860), t["b5"], font=font(36, LATO), fill=FG, anchor="mt")
    d.text((1310, 1000), t["b10"], font=font(36, LATO), fill=YELLOW, anchor="mt")
    save(img, "card_boards", lang)


def card_net(lang):
    t = T[lang]
    img, d = page(t["net_title"])
    box(d, (96, 220, 820, 600), t["net_old"], size=42, outline=MINUS)
    box(d, (980, 220, 1824, 600), t["net_new"], size=40, outline=PLUS)
    arrow(d, (830, 410), (970, 410))
    d.line((300, 650, 620, 970), fill=MINUS, width=10)
    d.line((620, 650, 300, 970), fill=MINUS, width=10)
    for i, s in enumerate(t["net_steps"]):
        d.text((1010, 680 + i * 70), f"{i + 1}. {s}", font=font(38, LATO), fill=FG)
    save(img, "card_net", lang)


def bars(d, x0, y0, rows, colour, width=1300, h=90, gap=50, size=40):
    for i, (name, v) in enumerate(rows):
        y = y0 + i * (h + gap)
        d.text((x0, y + h / 2), name, font=font(size, LATO), fill=FG, anchor="lm")
        bx = x0 + 330
        d.rounded_rectangle((bx, y, bx + width, y + h), 14, fill=PANEL)
        d.rounded_rectangle((bx, y, bx + width * v / 100, y + h), 14, fill=colour)
        d.text((bx + width * v / 100 + 20, y + h / 2), f"{v} %", font=font(size), fill=FG, anchor="lm")


def card_transfer(lang):
    t = T[lang]
    img, d = page(t["tr_title"])
    d.text((96, 190), t["tr_win"], font=font(32, LATO), fill=DIM)
    bars(d, 96, 260, t["tr_rows"], RED)
    d.text((960, 840), t["tr_elo"], font=font(40), fill=YELLOW, anchor="mm")
    save(img, "card_transfer", lang)


def card_intuition(lang):
    t = T[lang]
    img, d = page(t["in_title"])
    groups = ((t["in_a"], (88, 73, 67), C_VALUE), (t["in_b"], (86, 49, 35), MINUS))
    x0, y1, bw = 200, 860, 150
    for gi, (label, vals, col) in enumerate(groups):
        gx = x0 + gi * 860
        for i, (b, v) in enumerate(zip(("5×5", "7×7", "10×7"), vals)):
            x = gx + i * (bw + 60)
            top = y1 - v / 100 * 520
            d.rounded_rectangle((x, top, x + bw, y1), 12, fill=col)
            d.text((x + bw / 2, top - 14), f"{v} %", font=font(40), fill=FG, anchor="mb")
            d.text((x + bw / 2, y1 + 16), b, font=font(34, LATO), fill=DIM, anchor="mt")
        d.text((gx + (3 * bw + 120) / 2, 240), label, font=font(42), fill=col, anchor="mm")
    d.text((960, 990), t["in_note"], font=font(32, LATO), fill=DIM, anchor="mm")
    save(img, "card_intuition", lang)


def card_search(lang):
    t = T[lang]
    img, d = page(t["se_title"])
    root = (560, 250)
    kids = [(260, 470), (560, 470), (860, 470)]
    leaves = [(x + dx, 720) for x, _ in kids for dx in (-100, 0, 100)]
    for k in kids:
        d.line((root, k), fill=DIM, width=5)
    for i, lf in enumerate(leaves):
        d.line((kids[i // 3], lf), fill=FAINT, width=4)
    for i, lf in enumerate(leaves):
        d.ellipse((lf[0] - 30, lf[1] - 30, lf[0] + 30, lf[1] + 30), fill=MINUS)
    d.text((560, 790), t["se_lose"] + " · " + t["se_lose"] + " · " + t["se_lose"] + " …",
           font=font(34, LATO), fill=MINUS, anchor="mt")
    for k in kids:
        d.ellipse((k[0] - 36, k[1] - 36, k[0] + 36, k[1] + 36), fill=PANEL, outline=DIM, width=4)
    d.ellipse((kids[2][0] - 40, kids[2][1] - 40, kids[2][0] + 40, kids[2][1] + 40), outline=PLUS, width=8)
    d.text((kids[2][0] + 60, kids[2][1]), t["se_good"], font=font(32, LATO), fill=PLUS, anchor="lm")
    d.ellipse((root[0] - 40, root[1] - 40, root[0] + 40, root[1] + 40), fill=YELLOW)
    y = 320
    for name, v in t["se_rows"]:
        d.rounded_rectangle((1150, y, 1824, y + 130), 18, fill=PANEL)
        d.text((1180, y + 65), name, font=font(34, LATO), fill=FG, anchor="lm")
        d.text((1800, y + 65), v, font=font(40), fill=YELLOW, anchor="rm")
        y += 160
    save(img, "card_search", lang)


def dot2c_points():
    """[(gen, raw, mcts)] of dot2c: gen 0 = dot2a gen 1 (docs/transfer.md)."""
    from elo_vs_time import curve
    from _common import load_config
    pts = [(0, 1369, 1322)]
    for g, _h, m, r in curve(load_config(os.path.join(ROOT, "configs/dot2c.yaml"))):
        pts.append((g, r, m))
    return pts


def card_round(lang):
    t = T[lang]
    img, d = page(t["rd_title"])
    pts = dot2c_points()
    x0, y0, x1, y1 = 220, 200, 1500, 900
    lo, hi = 1200, 1800

    def yy(e):
        return y1 - (e - lo) / (hi - lo) * (y1 - y0)

    def xx(g):
        return x0 + g / 16 * (x1 - x0)
    for e in range(lo, hi + 1, 100):
        d.line((x0, yy(e), x1, yy(e)), fill=GRID, width=2)
        d.text((x0 - 18, yy(e)), str(e), font=font(28), fill=DIM, anchor="rm")
    for g in (0, 4, 9, 12, 16):
        d.text((xx(g), y1 + 14), str(g), font=font(28), fill=DIM, anchor="mt")
    d.text(((x0 + x1) / 2, y1 + 60), t["rd_x"], font=font(30), fill=DIM, anchor="mt")
    d.rectangle((xx(8.5), y0, x1, y1), fill=(30, 34, 46))
    d.text((xx(8.5) + 14, y0 + 10), t["rd_mark"], font=font(28, LATO), fill=DIM)
    for name, e in (("Medium", MEDIUM), ("Hard", HARD)):
        for xa in range(x0, x1, 28):
            d.line((xa, yy(e), xa + 14, yy(e)), fill=DIM, width=3)
        hline_label(d, x1, yy(e), f"{name} {e}", DIM)
    for idx, col, label in ((2, C_MCTS, t["rd_mcts"]), (1, C_POLICY, t["rd_raw"])):
        p = [(xx(r[0]), yy(r[idx])) for r in pts]
        if idx == 1:
            for a, b in zip(p, p[1:]):
                n = int(math.dist(a, b) / 24) + 1
                for k in range(0, n, 2):
                    d.line((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n,
                            a[0] + (b[0] - a[0]) * (k + 1) / n, a[1] + (b[1] - a[1]) * (k + 1) / n), fill=col, width=5)
        else:
            d.line(p, fill=col, width=7, joint="curve")
        for q in p:
            d.ellipse((q[0] - 8, q[1] - 8, q[0] + 8, q[1] + 8), fill=col)
        d.text((x1 + 16, p[-1][1] + (-36 if idx == 2 else 70)), label, font=font(30), fill=col, anchor="lm")
    j = next(r for r in pts if r[0] == 9)
    d.ellipse((xx(9) - 16, yy(j[2]) - 16, xx(9) + 16, yy(j[2]) + 16), outline=YELLOW, width=6)
    d.multiline_text((xx(9) - 40, yy(j[2]) - 40), t["rd_jump"], font=font(34), fill=YELLOW, anchor="rd",
                     align="right", spacing=8)
    save(img, "card_round", lang)


def card_wins(lang):
    t = T[lang]
    img, d = page(t["w_title"])
    d.text((960, 470), t["w_big"], font=font(84), fill=YELLOW, anchor="mm")
    d.text((960, 640), t["w_before"], font=font(40, LATO), fill=DIM, anchor="mm")
    save(img, "card_wins", lang)


def card_scratch(lang):
    from elo_vs_time import curve
    from _common import load_config
    t = T[lang]
    img, d = page(t["sc_title"])
    runs = [(curve(load_config(os.path.join(ROOT, f"configs/{n}.yaml"))), col, lab)
            for n, col, lab in (("dot2c", BLUE, t["sc_c"]), ("dot2s", (240, 130, 60), t["sc_s"]))]
    x0, y0, x1, y1 = 200, 190, 1500, 760
    lo, hi, hmax = 0, 1800, 16

    def yy(e):
        return y1 - (e - lo) / (hi - lo) * (y1 - y0)

    def xx(h):
        return x0 + h / hmax * (x1 - x0)
    for e in range(0, hi + 1, 300):
        d.line((x0, yy(e), x1, yy(e)), fill=GRID, width=2)
        d.text((x0 - 18, yy(e)), str(e), font=font(26), fill=DIM, anchor="rm")
    for h in range(0, hmax + 1, 2):
        d.text((xx(h), y1 + 12), str(h), font=font(26), fill=DIM, anchor="mt")
    d.text(((x0 + x1) / 2, y1 + 50), t["sc_x"], font=font(28), fill=DIM, anchor="mt")
    for name, e in (("Medium", MEDIUM), ("Hard", HARD)):
        for xa in range(x0, x1, 28):
            d.line((xa, yy(e), xa + 14, yy(e)), fill=DIM, width=3)
        hline_label(d, x1, yy(e), f"{name}", DIM)
    for pts, col, lab in runs:
        pm = [(xx(h), yy(m)) for _g, h, m, _r in pts]
        pr = [(xx(h), yy(r)) for _g, h, _m, r in pts]
        for a, b in zip(pr, pr[1:]):
            n = int(math.dist(a, b) / 20) + 1
            for k in range(0, n, 2):
                d.line((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n,
                        a[0] + (b[0] - a[0]) * (k + 1) / n, a[1] + (b[1] - a[1]) * (k + 1) / n), fill=col, width=4)
        d.line(pm, fill=col, width=7, joint="curve")
        for q in pm:
            d.ellipse((q[0] - 7, q[1] - 7, q[0] + 7, q[1] + 7), fill=col)
        ly = yy(600) if col == BLUE else yy(450)
        d.line((xx(8.5), ly, xx(9.5), ly), fill=col, width=7)
        d.text((xx(9.8), ly), lab, font=font(32), fill=col, anchor="lm")
    d.text((x0, 850), t["sc_note"], font=font(28, LATO), fill=DIM)
    y = 910
    for i, row in enumerate(t["sc_tab"]):
        col = BLUE if i == 0 else (240, 130, 60)
        for x, s in zip((200, 760, 1150, 1450), row):
            d.text((x, y + i * 70), s, font=font(36, LATO) if x == 200 else font(40), fill=col)
    save(img, "card_scratch", lang)


def card_question(lang):
    t = T[lang]
    img, d = page(t["q_title"])
    dots(d, 1210, 300, 10, 7, 60, r=6, col=FAINT)
    d.text((1510, 510), "?", font=font(300), fill=YELLOW, anchor="mm")
    d.multiline_text((96, 330), t["q_text"], font=font(46, LATO), fill=FG, spacing=20)
    d.text((96, 760), t["q_ask"], font=font(48), fill=YELLOW)
    save(img, "card_question", lang)


def card_title(lang):
    t = T[lang]
    img, d = page()
    dots(d, 1180, 560, 10, 7, 62, r=7, col=FAINT)
    dots(d, 260, 640, 5, 5, 62, r=7, col=DIM)
    arrow(d, (620, 800), (1120, 800), col=YELLOW, w=8)
    d.text((96, 120), t["t_top"], font=font(44, LATO), fill=DIM)
    d.text((96, 190), t["t_big"], font=font(110), fill=YELLOW)
    d.text((96, 360), t["t_sub"], font=font(44, LATO), fill=FG)
    save(img, "card_title", lang)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", choices=sorted(T), default="ru")
    a = p.parse_args()
    for f in (card_title, card_boards, card_net, card_transfer, card_intuition, card_search, card_round, card_wins,
              card_scratch, card_question):
        f(a.lang)


if __name__ == "__main__":
    main()
