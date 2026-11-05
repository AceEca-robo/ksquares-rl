#!/usr/bin/env python3
"""Stills for the part 3 video (docs/video_script_part3_ru.md), video/p3/card_*.png.
Numbers: docs/blowouts.md, parity_hand.md, parity_arch.md, chain_solver.md,
mid_judge.md, rout_mistakes.md, mid_hand.md; the map from docs/mid_judge.md's
'chart' table and runs/mid_judge/labels_chart.jsonl.
"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from video_cards import C_MCTS, C_POLICY, C_VALUE, DIM, FAINT, FG, GRID, LATO, MINUS, PLUS, arrow, box, font, page
from video_thumbnail import BLUE, PANEL, RED, YELLOW
OUT = os.path.join(HERE, "p3")


def save(img, name):
    os.makedirs(OUT, exist_ok=True); img.save(f"{OUT}/{name}.png"); print(name)


def dots(d, x, y, w, h, c, r=6, col=FAINT):
    for j in range(h + 1):
        for i in range(w + 1):
            d.ellipse((x + i * c - r, y + j * c - r, x + i * c + r, y + j * c + r), fill=col)


def pair(d, x, label, a, b, col, y1=860, top=300, sa=None, sb=None):
    """before / after bars"""
    bw, H = 190, y1 - top
    for i, (v, c, s) in enumerate(((a, FAINT, sa), (b, col, sb))):
        xx = x + i * (bw + 50); t = y1 - v / 100 * H
        d.rounded_rectangle((xx, t, xx + bw, y1), 10, fill=c)
        d.text((xx + bw / 2, t - 14), s or f"{v} %", font=font(46), fill=FG, anchor="mb")
    d.text((x + bw + 25, y1 + 24), label, font=font(36, LATO), fill=DIM, anchor="mt")


def title():
    img, d = page()
    dots(d, 1180, 560, 10, 7, 62, r=7)
    d.text((96, 120), "Нейросеть Dot, часть 3", font=font(44, LATO), fill=DIM)
    d.text((96, 190), "Понять, а не пересчитать", font=font(96), fill=YELLOW)
    d.text((96, 360), "пять попыток научить Dot-2 видеть правильный ход сразу", font=font(44, LATO), fill=FG)
    save(img, "card_title")


def question():
    img, d = page("Наш вопрос")
    dots(d, 1210, 300, 10, 7, 60)
    d.text((1510, 510), "?", font=font(300), fill=YELLOW, anchor="mm")
    d.multiline_text((96, 330), "Думая, Dot-2 считает,\nкому открывать первую цепочку.\nБез размышлений она этого не видит.", font=font(46, LATO), fill=FG, spacing=20)
    d.text((96, 760), "понимать игру, а не пересчитывать её каждый раз?", font=font(46), fill=YELLOW)
    save(img, "card_question")


def blowouts():
    img, d = page("17 разгромов: где судья нашёл ошибку")
    found = [20, 47, 48, 59, 81, 102, 121, 145, 156, 196]
    wall = [29, 41, 53, 98, 131, 141, 144]
    for i, s in enumerate(sorted(found + wall)):
        x, y = 150 + (i % 6) * 280, 260 + (i // 6) * 170
        f = s in found
        d.rounded_rectangle((x, y, x + 240, y + 130), 14, fill=PANEL, outline=YELLOW if f else None, width=4)
        d.text((x + 120, y + 42), f"партия {s}", font=font(32, LATO), fill=FG if f else DIM, anchor="mm")
        d.text((x + 120, y + 92), "ошибка найдена" if f else "судья не дотянулся", font=font(23, LATO), fill=YELLOW if f else DIM, anchor="mm")
    d.text((960, 830), "10 из 17: один ход от выигрыша к проигрышу, 7–20 безопасных линий", font=font(42), fill=YELLOW, anchor="mm")
    d.text((960, 900), "в остальных 7 партия проиграна ещё раньше: 13–29 безопасных линий", font=font(34, LATO), fill=DIM, anchor="mm")
    save(img, "card_blowouts")


def teacher():
    img, d = page("Попытка 1: Dot-2 учится у судьи")
    pair(d, 240, "верные ходы без размышлений", 67, 87, C_POLICY, y1=800)
    pair(d, 1130, "разгромов из 200", 11, 14, RED, y1=800, top=560, sa="11", sb="14")
    d.text((1370, 330), "разгромы те же", font=font(50), fill=YELLOW, anchor="mm")
    save(img, "card_teacher")


def timeline():
    img, d = page("Судья видит только конец партии")
    x0, x1, y = 160, 1760, 520
    d.rounded_rectangle((x0, y, x1, y + 90), 14, fill=PANEL)
    d.rounded_rectangle((1560, y, x1, y + 90), 14, fill=C_POLICY)
    d.text((1660, y + 140), "судья отвечает", font=font(38), fill=C_POLICY, anchor="mm")
    d.rounded_rectangle((980, y - 18, 1250, y + 108), 14, outline=RED, width=6)
    d.text((1115, y - 70), "здесь ошибка", font=font(44), fill=RED, anchor="mm")
    d.text((x0, y + 120), "начало партии", font=font(34, LATO), fill=DIM)
    d.text((1115, y + 150), "судья молчит", font=font(38, LATO), fill=DIM, anchor="mm")
    save(img, "card_timeline")


def body(d, lock=False, bad=True):
    box(d, (120, 420, 520, 620), "подсказка: цепочки,\nкому открывать первую", outline=YELLOW, size=36, colour=YELLOW)
    arrow(d, (530, 520), (660, 520))
    box(d, (670, 400, 1030, 640), "общее тело", size=46)
    arrow(d, (1040, 470), (1210, 360)); arrow(d, (1040, 570), (1210, 680))
    box(d, (1220, 260, 1720, 440), "выбор хода", outline=C_POLICY, size=46, colour=C_POLICY)
    box(d, (1220, 600, 1720, 780), "интуиция" + (" (заморожена)" if lock else ""), outline=RED if bad else C_VALUE, size=40, colour=RED if bad else C_VALUE)


def shared():
    img, d = page("Попытка 2: взгляд на всё поле и подсказка")
    body(d)
    d.text((1470, 225), "без размышлений: 67 → 83 %", font=font(36), fill=FG, anchor="mm")
    d.text((1470, 830), "видит проигрыш там, где его нет", font=font(34, LATO), fill=DIM, anchor="mm")
    d.text((960, 930), "разгромов 11 → 46 из 200  ·  настоящий противник 19–0–1 → 15–5", font=font(42), fill=YELLOW, anchor="mm")
    save(img, "card_shared_body")


def frozen():
    img, d = page("Заморозили интуицию")
    body(d, lock=True, bad=False)
    d.text((1470, 225), "без размышлений: 67 → 75 %", font=font(36), fill=FG, anchor="mm")
    d.text((960, 930), "разгромов 12 из 200  ·  настоящий противник 19–1: игра цела, разгромы те же", font=font(38), fill=YELLOW, anchor="mm")
    save(img, "card_frozen")


def chains(d, x, y, c, full):
    rows = [[(0, 0, 5, 0), (5, 0, 5, 3)], [(0, 2, 4, 2)], [(0, 4, 3, 4), (3, 4, 3, 6)], [(5, 5, 9, 5)], [(7, 1, 9, 1), (9, 1, 9, 3)]]
    cols = [RED, C_POLICY, C_VALUE, PLUS, YELLOW]
    for r, col in zip(rows if full else rows[:2], cols):
        for a, b, e, f in r:
            d.line((x + a * c, y + b * c, x + e * c, y + f * c), fill=col, width=14)


def chains_card():
    img, d = page("Конец партии: доска разрезана на цепочки")
    dots(d, 180, 260, 9, 6, 80); chains(d, 180, 260, 80, True)
    d.multiline_text((1060, 400), "всё решает,\nкому придётся\nоткрыть первую", font=font(56), fill=YELLOW, spacing=20)
    save(img, "card_chains")


def formula():
    img, d = page("Попытка 3: судья по формуле")
    dots(d, 180, 260, 9, 6, 80); chains(d, 180, 260, 80, True)
    d.multiline_text((1060, 340), "безопасных линий\nне осталось,\nдоска разрезана\nна цепочки:", font=font(46, LATO), fill=FG, spacing=16)
    d.text((1060, 680), "100 % верно", font=font(76), fill=YELLOW)
    save(img, "card_formula")


def formula_back():
    img, d = page("Попытка 3: судья по формуле")
    dots(d, 180, 260, 9, 6, 80); chains(d, 180, 260, 80, False)
    d.multiline_text((1060, 340), "в середине партии\nдоска ещё не разрезана:\nсчитать нечего", font=font(46, LATO), fill=FG, spacing=16)
    d.text((1060, 640), "ответов нет", font=font(76), fill=RED)
    save(img, "card_formula_back")


def tree():
    img, d = page("Попытка 4: сильный судья")
    root = (960, 250)
    for i in range(3):
        a = (480 + i * 480, 480)
        d.line(root + a, fill=DIM, width=4)
        for j in range(3):
            b = (a[0] - 150 + j * 150, 700)
            d.line(a + b, fill=DIM, width=3)
            d.rounded_rectangle((b[0] - 55, 700, b[0] + 55, 770), 10, fill=YELLOW)
            d.text((b[0], 735), "f", font=font(38), fill=PANEL, anchor="mm")
        d.ellipse((a[0] - 24, a[1] - 24, a[0] + 24, a[1] + 24), fill=C_POLICY)
    d.ellipse((root[0] - 30, root[1] - 30, root[0] + 30, root[1] + 30), fill=FG)
    d.text((960, 880), "перебор на настоящей доске, на листьях формула", font=font(44), fill=FG, anchor="mm")
    save(img, "card_tree")


def wall():
    img, d = page("Стена: сколько позиций судья решает")
    rows = [("безопасных линий", "за минуту", "за 10 минут"), ("10", "100 %", "100 %"), ("15", "75 %", "88 %"), ("20", "20 %", "50 %"), ("30", "0", "0")]
    for i, r in enumerate(rows):
        y = 260 + i * 125
        if i: d.line((260, y - 28, 1660, y - 28), fill=GRID, width=2)
        for x, v in zip((460, 1000, 1460), r):
            d.text((x, y + 30), v, font=font(62) if i else font(36, LATO), fill=DIM if i == 0 else (RED if r[0] == "30" else FG), anchor="mm")
    d.text((960, 930), "позиции из партий Dot-2: 20 на лимит в минуту, 8 на лимит в 10 минут", font=font(30, LATO), fill=DIM, anchor="mm")
    save(img, "card_wall")


def map_cards():
    t = open(f"{ROOT}/docs/mid_judge.md").read()
    t = t[t.index("Labelled set 'chart'"):]
    t = t[t.index("Decisive positions by safe lines"):]
    data = {}
    for m in re.finditer(r"^\| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|", t, re.M):
        s, n, h, i, c = map(int, m.groups())
        if n >= 10: data[s] = (h, i, c)
    rs = [json.loads(x) for x in open(f"{ROOT}/runs/mid_judge/labels_chart.jsonl")]
    solved = {}
    for s in range(21):
        a = [r for r in rs if r["key"][0] == s]
        if a: solved[s] = 100 * sum(r["labels"] is not None for r in a) / len(a)
    X0, X1, Y0, Y1 = 170, 1560, 240, 880
    fx = lambda s: X0 + s / 24 * (X1 - X0); fy = lambda v: Y1 - v / 100 * (Y1 - Y0)
    S = [("Dot-2 без\nразмышлений", C_POLICY, 0), ("интуиция", C_VALUE, 1), ("Dot-2 с\nразмышлениями", PLUS, 2)]

    def draw(stage):
        img, d = page("Кто что знает")
        for v in (0, 25, 50, 75, 100):
            d.line((X0, fy(v), X1, fy(v)), fill=GRID, width=2)
            d.text((X0 - 16, fy(v)), f"{v}", font=font(28, LATO), fill=DIM, anchor="rm")
        for s in (0, 5, 10, 15, 20):
            d.text((fx(s), Y1 + 16), str(s), font=font(30, LATO), fill=DIM, anchor="mt")
        d.text((fx(10), Y1 + 62), "безопасных линий осталось  →  к середине партии", font=font(32, LATO), fill=DIM, anchor="mt")
        d.text((X0, 180), "% верных ходов на решающих позициях", font=font(28, LATO), fill=DIM)
        if stage >= 5 and stage != 6:
            d.rectangle((fx(20), Y0, X1, Y1), fill=(10, 11, 15))
            d.text(((fx(20) + X1) / 2, 500), "?", font=font(170), fill=YELLOW, anchor="mm")
        for name, col, k in S:
            if stage < k + 1 or (stage == 6 and k != 2): continue
            pts = [(fx(s), fy(v[k])) for s, v in sorted(data.items())]
            d.line(pts, fill=col, width=5, joint="curve")
            for p in pts: d.ellipse((p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7), fill=col)
            d.ellipse((1600, 270 + k * 90, 1630, 300 + k * 90), fill=col)
            d.multiline_text((1645, 262 + k * 90), name, font=font(30, LATO), fill=FG, spacing=4)
        if stage in (4, 5):
            pts = [(fx(s), fy(v)) for s, v in sorted(solved.items())]
            d.line(pts, fill=FG, width=4)
            for p in pts: d.rectangle((p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7), fill=FG)
            d.rectangle((1600, 545, 1630, 575), fill=FG)
            d.text((1645, 560), "судья", font=font(34, LATO), fill=FG, anchor="lm")
            d.multiline_text((1600, 590), "доля решённых\nпозиций,\nне точность", font=font(26, LATO), fill=DIM, spacing=6)
        if stage == 6:
            d.text((960, 1020), "200 вариантов: 96 % верных ходов там, где судья может проверить", font=font(36), fill=YELLOW, anchor="mm")
        save(img, f"card_map_{stage}")
    for st in range(7): draw(st)


def mistakes():
    img, d = page("Попытка 5: на своих ошибках")
    for i, (n, l) in enumerate((("5500", "партий"), ("248", "разгромов"), ("80", "ошибок в середине"))):
        x = 340 + i * 620
        d.text((x, 420), n, font=font(120), fill=YELLOW if i == 2 else FG, anchor="mm")
        d.text((x, 520), l, font=font(40, LATO), fill=DIM, anchor="mm")
        if i: arrow(d, (x - 450, 420), (x - 200, 420))
    save(img, "card_mistakes")


def mistakes_held():
    img, d = page("Знакомые ошибки Dot-2 запоминает, а новые?")
    pair(d, 240, "знакомые ошибки", 12.5, 100, C_POLICY, y1=800, sa="12 %", sb="100 %")
    pair(d, 1130, "новые ошибки", 8.3, 25, C_POLICY, y1=800, sa="2 из 24", sb="6 из 24")
    d.text((960, 950), "так мало, что от шума не отличить", font=font(44), fill=YELLOW, anchor="mm")
    save(img, "card_mistakes_held")


def mid_held():
    img, d = page("Большой набор судьи: 1925 позиций")
    pair(d, 240, "решающие позиции из новых партий", 86.9, 90.5, C_POLICY, y1=780, top=260, sa="87 %", sb="90,5 %")
    d.text((1320, 340), "стало верно: 11", font=font(58), fill=PLUS, anchor="mm")
    d.text((1320, 430), "стало неверно: 4", font=font(58), fill=RED, anchor="mm")
    d.multiline_text((1320, 590), "порог, заданный заранее,\nне пройден\n(партии не игрались)", font=font(40, LATO), fill=DIM, anchor="mm", align="center", spacing=12)
    d.text((960, 950), "прибавка есть, доказать её не смогли", font=font(44), fill=YELLOW, anchor="mm")
    save(img, "card_mid_held")


def end():
    img, d = page()
    d.ellipse((930, 430, 990, 490), fill=YELLOW)
    d.text((960, 640), "часть 4: графовая сеть вместо свёрточной", font=font(64), fill=FG, anchor="mm")
    save(img, "card_end")


for f in (title, question, chains_card, blowouts, teacher, timeline, shared, frozen, formula, formula_back, tree, wall, map_cards, mistakes, mistakes_held, mid_held, end):
    f()
