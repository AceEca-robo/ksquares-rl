#!/usr/bin/env python3
"""Stills of the part 4 video (docs/video_script_part4_ru.md), video/p4/card_*.png.

    .venv/bin/python video/p4_cards.py

Every number comes from video/p4_data.py (run files and reports); claims: docs/video_claims_part4.md.
Screen words follow the video's vocabulary: размышления, интуиция, судья, встроенный противник, раунд, школа.
P4_LANG=en writes the English cut (video/p4/card_*_en.png, docs/video_script_part4_en.md): L(ru, en) per text.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p4_i18n import LANG, L, suffix  # noqa: E402  (first: sets P3_LANG)
import p4_data as P  # noqa: E402
from video_cards import C_POLICY, DIM, FAINT, FG, GRID, LATO, PLUS, arrow, box, font, page  # noqa: E402
from video_thumbnail import BLUE, CHAIN_COLOURS, PANEL, RED, YELLOW  # noqa: E402

OUT = os.path.join(HERE, "p4")
C_G = YELLOW                    # Dot-G
C_D = BLUE                      # Dot / Dot-2
C_OTHER = (150, 155, 170)


def save(img, name):
    os.makedirs(OUT, exist_ok=True)
    img.save(f"{OUT}/{name}{suffix()}.png")
    print(name)


def pct(v, nd=0):
    return L(f"{v:.{nd}f}".replace(".", ",") + " %", f"{v:.{nd}f}%")


def of(a, b):
    return L(f"{a} из {b}", f"{a} of {b}")


# ------------------------------------------------------------------ helpers

def dots(d, x, y, n, m, c, r=6, col=FAINT):
    for j in range(m + 1):
        for i in range(n + 1):
            d.ellipse((x + i * c - r, y + j * c - r, x + i * c + r, y + j * c + r), fill=col)


class Plot:
    """A plot area: data (x, y) -> pixels, with a light grid."""
    def __init__(self, d, rect, xr, yr):
        self.d, self.r, self.xr, self.yr = d, rect, xr, yr

    def p(self, x, y):
        x0, y0, x1, y1 = self.r
        return (x0 + (x - self.xr[0]) / (self.xr[1] - self.xr[0]) * (x1 - x0),
                y1 - (y - self.yr[0]) / (self.yr[1] - self.yr[0]) * (y1 - y0))

    def grid(self, ys, xs, ylab=str, xlab=str, size=28):
        x0, y0, x1, y1 = self.r
        for v in ys:
            _, y = self.p(self.xr[0], v)
            self.d.line((x0, y, x1, y), fill=GRID, width=2)
            self.d.text((x0 - 16, y), ylab(v), font=font(size, LATO), fill=DIM, anchor="rm")
        for v in xs:
            x, _ = self.p(v, self.yr[0])
            self.d.text((x, y1 + 14), xlab(v), font=font(size, LATO), fill=DIM, anchor="mt")
        self.d.line((x0, y1, x1, y1), fill=DIM, width=3)

    def line(self, pts, col, w=5, dot=0):
        q = [self.p(x, y) for x, y in pts]
        self.d.line(q, fill=col, width=w, joint="curve")
        for x, y in q if dot else []:
            self.d.ellipse((x - dot, y - dot, x + dot, y + dot), fill=col)


# ------------------------------------------------------------------ scene 1-2

def title():
    img, d = page()
    d.text((96, 120), L("Нейросеть Dot, часть 4", "Neural network Dot, part 4"), font=font(44, LATO), fill=DIM)
    d.text((96, 190), L("Dot-G: другие глаза", "Dot-G: other eyes"), font=font(110), fill=YELLOW)
    d.text((96, 380), L("графовая сеть видит цепочки целиком", "a graph network sees chains whole"), font=font(48, LATO), fill=FG)
    graph_sketch(d, 1080, 560, 1.0)
    save(img, "card_title")


def graph_sketch(d, ox, oy, k):
    nodes = [(0, 0), (300, -90), (560, 40), (110, 260), (400, 210), (640, 330)]
    edges = [(0, 1, "8"), (1, 2, "5"), (0, 3, "3"), (1, 4, "10"), (3, 4, "7"), (4, 5, "4"), (2, 4, "3")]
    P_ = [(ox + x * k, oy + y * k) for x, y in nodes]
    for i, (a, b, n) in enumerate(edges):
        d.line(P_[a] + P_[b], fill=CHAIN_COLOURS[i % len(CHAIN_COLOURS)], width=12)
    for x, y in P_:
        d.ellipse((x - 24, y - 24, x + 24, y + 24), fill=FG)
    for a, b, n in edges:
        x, y = (P_[a][0] + P_[b][0]) / 2, (P_[a][1] + P_[b][1]) / 2
        d.ellipse((x - 32, y - 32, x + 32, y + 32), fill=PANEL)
        d.text((x, y), n, font=font(36), fill=FG, anchor="mm")


def sees():
    """Scene 2: the board as a picture (left) and as Dot-G's scheme (right): boxes, lines, chains with a length."""
    img, d = page(L("Как видит Dot-G", "How Dot-G sees"))
    c, x0, y0 = 120, 150, 300
    chains = [[(0, 0), (1, 0), (2, 0), (3, 0)], [(4, 0), (4, 1), (4, 2)], [(0, 2), (1, 2), (2, 2), (2, 3), (2, 4)],
              [(3, 3), (4, 3), (4, 4), (3, 4)]]
    for k, ch in enumerate(chains):
        for x, y in ch:
            d.rectangle((x0 + x * c + 10, y0 + y * c + 10, x0 + (x + 1) * c - 10, y0 + (y + 1) * c - 10),
                        fill=tuple(int(v * .45) for v in CHAIN_COLOURS[k]))
    dots(d, x0, y0, 5, 5, c, r=7, col=FG)
    d.text((x0 + 2.5 * c, y0 + 5 * c + 50), L("обычная Dot: картинка, окошко 3×3", "ordinary Dot: a picture, a 3×3 window"), font=font(34, LATO), fill=DIM, anchor="mm")
    d.rounded_rectangle((x0 + c, y0 + c, x0 + 4 * c, y0 + 4 * c), 12, outline=C_D, width=6)
    arrow(d, (870, 600), (1010, 600))
    # scheme
    sx, sy = 1060, 300
    lab = L(["цепочка, длина 4", "цепочка, длина 3", "цепочка, длина 5", "петля, длина 4"],
            ["chain, length 4", "chain, length 3", "chain, length 5", "loop, length 4"])
    for k, t in enumerate(lab):
        y = sy + k * 140
        d.ellipse((sx, y, sx + 90, y + 90), fill=CHAIN_COLOURS[k])
        d.text((sx + 45, y + 45), t.split()[-1], font=font(40), fill=PANEL, anchor="mm")
        d.text((sx + 120, y + 45), t, font=font(38, LATO), fill=FG, anchor="lm")
        for j in range(3):
            bx = sx + 560 + j * 70
            d.rectangle((bx, y + 25, bx + 40, y + 65), outline=C_OTHER, width=3)
            d.line((sx + 90, y + 45, bx, y + 45), fill=FAINT, width=2) if j == 0 else None
    d.text((sx + 380, sy + 4 * 140 + 30), L("клетки и линии — тоже узлы схемы", "boxes and lines are nodes of the diagram too"), font=font(32, LATO), fill=DIM, anchor="mm")
    d.text((sx + 380, sy - 50), L("Dot-G: схема, цепочка — отдельный предмет", "Dot-G: a diagram, a chain is a separate object"), font=font(36), fill=C_G, anchor="mm")
    save(img, "card_sees")


# ------------------------------------------------------------------ scene 3

def three_nets():
    img, d = page(L("Задача: кому придётся открыть первую цепочку", "Task: who will have to open the first chain"))
    cols = L([("обычная свёрточная", "как Dot: окошко", C_OTHER), ("простая графовая", "клетки и линии,\nцепочки собирает сама", C_OTHER),
              ("Dot-G", "цепочка —\nотдельный предмет", C_G)],
             [("ordinary convolutional", "like Dot: a window", C_OTHER), ("simple graph", "boxes and lines,\nputs chains together itself", C_OTHER),
              ("Dot-G", "a chain is\na separate object", C_G)])
    for i, (a, b, col) in enumerate(cols):
        x = 160 + i * 560
        d.rounded_rectangle((x, 330, x + 480, 720), 22, fill=PANEL, outline=col, width=4)
        d.text((x + 240, 410), a, font=font(40), fill=col, anchor="mm")
        d.multiline_text((x + 240, 560), b, font=font(38, LATO), fill=FG, anchor="mm", align="center", spacing=12)
    d.text((960, 850), L("все три начинают с нуля, позиции из конца партии, доска пять на пять",
                         "all three start from scratch, positions from the end of a game, five by five board"), font=font(38, LATO), fill=DIM, anchor="mm")
    save(img, "card_three")


def click(progress=1.0, img_d=None):
    """Scene 3: right answers on the small board by training step; progress 0..1 draws the curves left to right."""
    img, d = img_d or page(L("Щелчок: верные ответы по шагам обучения", "The click: right answers by training step"))
    cur = P.click_curves()
    pl = Plot(d, (220, 230, 1500, 880), (0, 6000), (30, 100))
    pl.grid((50, 75, 100), (0, 1000, 2000, 3000, 4000, 5000, 6000), ylab=lambda v: pct(v),
            xlab=lambda v: f"{v}")
    d.text((860, 960), L("шаги обучения", "training steps"), font=font(30, LATO), fill=DIM, anchor="mm")
    xmax = 6000 * progress
    for key, col, w in (("c", C_OTHER, 4), ("ga", (110, 115, 130), 4), ("gc", C_G, 4)):
        for run in cur[key]:
            pts = [(s, v) for s, v in run if s <= xmax]
            if len(pts) > 1:
                pl.line(pts, col, w)
    d.text((1530, 300), "Dot-G:", font=font(40), fill=C_G)
    d.text((1530, 350), of(10, 10), font=font(64), fill=C_G)
    d.text((1530, 450), L("щёлкнули", "clicked"), font=font(36, LATO), fill=C_G)
    d.text((1530, 620), L("свёрточная: 0 из 5", "convolutional: 0 of 5"), font=font(32, LATO), fill=C_OTHER)
    d.text((1530, 670), L("простая графовая:", "simple graph:"), font=font(32, LATO), fill=C_OTHER)
    d.text((1530, 712), of(0, 5), font=font(32, LATO), fill=C_OTHER)
    d.text((1530, 790), L("монетка = 50 %", "coin toss = 50%"), font=font(28, LATO), fill=DIM)
    return img


# ------------------------------------------------------------------ scene 4

def school():
    img, d = page(L("Школа из трёх классов", "A school with three grades"))
    cls = L([("1 класс", "только маленькая доска,\nпока не ответит\nпочти без ошибок", "5×5"),
             ("2 класс", "короткий:\nмаленькая и большая\nвперемешку", "5×5 + 10×7"),
             ("3 класс", "только большая доска\n\n+ каждая четвёртая задача\nс маленькой", "10×7")],
            [("grade 1", "only the small board,\nuntil it answers\nalmost without mistakes", "5×5"),
             ("grade 2", "a short one:\nsmall and big board\nmixed together", "5×5 + 10×7"),
             ("grade 3", "only the big board\n\n+ every fourth task\nfrom the small one", "10×7")])
    for i, (a, b, s) in enumerate(cls):
        x = 140 + i * 580
        d.rounded_rectangle((x, 260, x + 500, 760), 22, fill=PANEL, outline=YELLOW if i == 2 else None, width=4)
        d.text((x + 250, 320), a, font=font(50), fill=YELLOW, anchor="mm")
        d.text((x + 250, 400), s, font=font(40, LATO), fill=FG, anchor="mm")
        d.multiline_text((x + 250, 580), b, font=font(34, LATO), fill=FG, anchor="mm", align="center", spacing=10)
        if i:
            arrow(d, (x - 70, 510), (x - 10, 510))
    save(img, "card_school")


def school_table(stage=3):
    s = P.school()
    img, d = page(L("Большая доска, десять запусков", "The big board, ten runs"))
    rows = [(L("освоила большую доску", "mastered the big board"), L("без школы", "no school"), of(s['no_school'], 10),
             L("после школы", "after school"), of(s['school'], 10)),
            (L("помнит обе доски к концу учёбы", "remembers both boards at the end of school"), L("без повторения", "no repeating"),
             of(s['both_plain'], 10), L("каждая 4-я — с маленькой", "every 4th from the small one"), of(s['both_k25'], 10))]
    for i, (what, a, va, b, vb) in enumerate(rows[:1 if stage < 3 else 2]):
        y = 300 + i * 300
        d.text((140, y), what, font=font(44), fill=FG)
        d.text((200, y + 110), a, font=font(34, LATO), fill=DIM)
        d.text((640, y + 100), va, font=font(64), fill=RED)
        if stage == 1:
            continue
        arrow(d, (940, y + 135), (1080, y + 135))
        d.text((1120, y + 110), b, font=font(34, LATO), fill=DIM)
        d.text((1560, y + 100), vb, font=font(64), fill=PLUS)
    if stage == 3:
        d.text((960, 960), L("по дороге маленькая доска иногда ненадолго проседает (3 из 10)",
                                 "along the way the small board sometimes dips for a while (3 of 10)"), font=font(30, LATO), fill=DIM, anchor="mm")
    save(img, "card_school_table" + ("" if stage == 3 else str(stage)))


# ------------------------------------------------------------------ scene 5

def cloud(full=True):
    pts = P.jump_cloud()
    img, d = page(L("20 сетей учатся выбирать ход на концовках", "20 networks learn to choose the move in endgames"))
    pl = Plot(d, (260, 240, 1500, 880), (0, 21), (50, 100))
    pl.grid((50, 60, 70, 80, 90, 100), (), ylab=lambda v: pct(v))
    d.text((880, 940), L("20 сетей после школы (каждая точка — одна сеть)", "20 networks after school (each dot is one network)"), font=font(30, LATO), fill=DIM, anchor="mm")
    d.text((120, 170), L("верные ходы на новых концовках большой доски, без размышлений (лучшая точка обучения)",
                                 "right moves in new endgames of the big board, without thinking (best point of training)"), font=font(30, LATO), fill=DIM)
    order = sorted(pts, key=lambda p: p[1])
    for i, (seed, v) in enumerate(order):
        x, y = pl.p(i + 1, v)
        col = C_G if v >= 80 and full else C_OTHER
        d.ellipse((x - 16, y - 16, x + 16, y + 16), fill=col)
    jumps = sum(v >= 80 for _, v in pts)
    if not full:
        save(img, "card_cloud0")
        return
    d.text((1540, 330), L("скачок:", "jump:"), font=font(40), fill=C_G)
    d.text((1540, 385), of(jumps, len(pts)), font=font(64), fill=C_G)
    d.text((1540, 600), L("остальные —", "the others are"), font=font(34, LATO), fill=C_OTHER)
    d.text((1540, 645), L("как обычная", "like an ordinary"), font=font(34, LATO), fill=C_OTHER)
    d.text((1540, 690), L("свёрточная сеть", "convolutional network"), font=font(34, LATO), fill=C_OTHER)
    save(img, "card_cloud")


def four():
    rows = P.four_nets()
    img, d = page(L("Четыре выпускника, каждый учится заново пять раз", "Four graduates, each learns anew five times"))
    for i, (w, j, n) in enumerate(sorted(rows, key=lambda r: -r[1])):
        y = 290 + i * 160
        d.text((200, y + 40), L(f"выпускник {i + 1}", f"graduate {i + 1}"), font=font(42, LATO), fill=FG, anchor="lm")
        for k in range(n):
            x = 620 + k * 150
            hit = k < j
            d.ellipse((x, y, x + 84, y + 84), fill=C_G if hit else PANEL, outline=None if hit else C_OTHER, width=4)
        d.text((1430, y + 40), of(j, n), font=font(52), fill=C_G if j else C_OTHER, anchor="lm")
    d.text((960, 960), L("жёлтый — скачок; меняется только порядок задач", "yellow is a jump; only the order of the tasks changes"), font=font(32, LATO), fill=DIM, anchor="mm")
    save(img, "card_four")


# ------------------------------------------------------------------ scene 6

def map_g():
    """Scene 6: the part 3 map (video/p3_map.py, final state) plus the Dot-G line with a jump on the same positions."""
    import p3_map as M
    line, g_mid, d_mid = P.mid_and_chart()
    img = M.layers()[-1]
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    a, b = M.fx(15.4), M.fx(9.6)
    d.rectangle((a, M.Y0, b, M.Y1), outline=YELLOW, width=4)
    d.text(((a + b) / 2, M.Y1 + 150), L("позиции проверки: 10–15 безопасных линий", "test positions: 10–15 safe lines"), font=font(30), fill=YELLOW, anchor="mm")
    M.curve(d, [(M.fx(s), M.fy(v)) for s, v in sorted(line.items(), reverse=True)], C_G)
    x, y = 760 + 0 * 520, 62 + 2 * 52
    d.ellipse((x, y, x + 28, y + 28), fill=C_G)
    d.text((x + 42, y + 14), L("Dot-G со скачком", "Dot-G with a jump"), font=font(28, LATO), fill=FG, anchor="lm")
    save(img, "card_map_g")


def mid_bars():
    _, g, dd = P.mid_and_chart()
    img, d = page(L("Середина партии: те же позиции из третьей части", "Middle of the game: the same positions as in part 3"))
    y1, top, bw = 820, 260, 300
    for i, (lab, v, col) in enumerate(((L("Dot-G со скачком", "Dot-G with a jump"), g, C_G), (L("Dot-2 без размышлений", "Dot-2 without thinking"), dd, C_D))):
        x = 520 + i * 560
        t = y1 - v / 100 * (y1 - top)
        d.rounded_rectangle((x, t, x + bw, y1), 12, fill=col)
        d.text((x + bw / 2, t - 16), pct(v, 0), font=font(64), fill=FG, anchor="mb")
        d.text((x + bw / 2, y1 + 30), lab, font=font(36, LATO), fill=FG, anchor="mt")
    d.text((960, 970), L("верные ходы на 721 решающей позиции; скачок на середину не перенёсся",
                         "right moves on 721 decisive positions; the jump did not carry over to the middle"), font=font(34), fill=YELLOW, anchor="mm")
    save(img, "card_mid_bars")


# ------------------------------------------------------------------ scenes 7-8

def rounds_chart(name, title_, show_dot=True, mark_peak=False, mark_hard=False, note=None):
    G, D, g_hard, d_hard = P.ladder_rows()
    img, d = page(title_)
    pl = Plot(d, (230, 240, 1600, 860), (0, 120), (1000, 2300))
    pl.grid((1200, 1500, 1800, 2100), (1, 20, 40, 60, 80, 100, 120), ylab=lambda v: "", xlab=str)
    d.text((915, 940), L("раунд", "round"), font=font(32, LATO), fill=DIM, anchor="mm")
    d.text((120, 165), L("сила по лестнице, с размышлениями (выше — сильнее)", "strength by the ladder, with thinking (higher is stronger)"), font=font(30, LATO), fill=DIM)
    if show_dot:
        pl.line([(r["round"], r["mcts_elo"]) for r in D], C_D, 5, dot=5)
        x, y = pl.p(D[-1]["round"], D[-1]["mcts_elo"])
        d.text((x + 16, y + 40), "Dot", font=font(40), fill=C_D, anchor="lm")
    pl.line([(r["round"], r["mcts_elo"]) for r in G], C_G, 5, dot=5)
    x, y = pl.p(G[-1]["round"], G[-1]["mcts_elo"])
    d.text((x + 16, y - 30), "Dot-G", font=font(40), fill=C_G, anchor="lm")
    if mark_hard:
        for rr, col, rows, dy in ((g_hard, C_G, G, -120), (d_hard, C_D, D, 70)):
            r = next(r for r in rows if r["round"] == rr)
            x, y = pl.p(rr, r["mcts_elo"])
            d.ellipse((x - 14, y - 14, x + 14, y + 14), outline=col, width=5)
            d.text((x + 30, y + dy), L(f"раунд {rr}", f"round {rr}"), font=font(34), fill=col, anchor="lm")
        d.multiline_text((1660, 560), L("встроенный\nпротивник:\n90 из 100\nи больше —\nс раунда", "built-in\nopponent:\n90 of 100\nor more\nfrom round"), font=font(32, LATO), fill=FG, spacing=8)
        d.text((1660, 800), f"Dot-G: {g_hard}", font=font(40), fill=C_G)
        d.text((1660, 850), f"Dot: {d_hard}", font=font(40), fill=C_D)
    if mark_peak:
        best = max(G, key=lambda r: r["mcts_elo"])
        x, y = pl.p(best["round"], best["mcts_elo"])
        d.ellipse((x - 16, y - 16, x + 16, y + 16), outline=FG, width=5)
        d.text((x, y - 40), L(f"пик лестницы: раунд {best['round']}", f"ladder peak: round {best['round']}"), font=font(34), fill=FG, anchor="mb")
        last = G[-1]
        x, y = pl.p(last["round"], last["mcts_elo"])
        d.ellipse((x - 16, y - 16, x + 16, y + 16), outline=FG, width=5)
        d.text((x - 20, y + 50), L(f"раунд {last['round']}", f"round {last['round']}"), font=font(34), fill=FG, anchor="rt")
    if note:
        d.text((960, 1020), note, font=font(34), fill=YELLOW, anchor="mm")
    save(img, name)


def bridge_card():
    b = P.bridge()
    wins = sum(1 for g in b["games"] if g["net"] > g["hard"])
    img, d = page(L("Против настоящего KSquares", "Against the real KSquares"))
    d.text((960, 430), of(wins, len(b['games'])), font=font(220), fill=C_G, anchor="mm")
    d.text((960, 640), L("Dot-G, раунд 49, против сложного уровня игры", "Dot-G, round 49, against the game's Hard level"), font=font(44, LATO), fill=FG, anchor="mm")
    d.text((960, 720), L(f"по {b['sims']} вариантов на ход", f"{b['sims']} options per move"), font=font(36, LATO), fill=DIM, anchor="mm")
    save(img, "card_bridge")


def matches(stage=2):
    r = P.s10()
    sc = lambda x: 100 * (x[0] + x[2] / 2) / sum(x[:3])
    img, d = page(L("Прямые матчи, по 200 партий", "Direct matches, 200 games each"))
    rows = [(L("Dot-G: раунд 114 против раунда 49", "Dot-G: round 114 vs round 49"), sc(r["4:mcts"]),
             L("лестница звала лучшим раунд 49", "the ladder called round 49 the best"), C_G),
            (L("Dot: раунд 84 против раунда 54", "Dot: round 84 vs round 54"), sc(r["1:mcts"]),
             L("в первом ролике: «после 54-го не росла»", "in the first video: \"no growth after round 54\""), C_D)]
    for i, (a, v, b, col) in enumerate(rows[:stage]):
        y = 300 + i * 320
        d.text((140, y), a, font=font(48), fill=col)
        d.text((140, y + 80), b, font=font(34, LATO), fill=DIM)
        d.text((1780, y + 40), of(f"{v:.0f}", 100), font=font(90), fill=col, anchor="rm")
    d.text((960, 960), L("оба думают по 200 вариантов на ход", "both think, 200 options per move"), font=font(32, LATO), fill=DIM, anchor="mm")
    save(img, "card_matches" + ("" if stage == 2 else "1"))


def round_robin():
    r = P.s10()
    ps = ["g5_49", "g5_114", "dot54", "dot_last", "hard", "strong"]
    lab = {"g5_49": "Dot-G 49", "g5_114": "Dot-G 114", "dot54": "Dot 54", "dot_last": "Dot 84",
           "hard": L("встроенный", "built-in"), "strong": L("судья", "judge")}
    col = {"g5_49": C_G, "g5_114": C_G, "dot54": C_D, "dot_last": C_D, "hard": C_OTHER, "strong": C_OTHER}

    def v(p, q):
        if f"rr:{p}:{q}" in r:
            x = r[f"rr:{p}:{q}"]
            return 100 * (x[0] + x[2] / 2) / sum(x[:3])
        x = r[f"rr:{q}:{p}"]
        return 100 - 100 * (x[0] + x[2] / 2) / sum(x[:3])
    img, d = page(L("Круговой турнир: очки строки против столбца", "Round robin: score of the row against the column"))
    x0, y0, cw, ch = 420, 230, 240, 110
    for j, q in enumerate(ps):
        d.text((x0 + j * cw + cw / 2, y0), lab[q], font=font(32), fill=col[q], anchor="mm")
    for i, p in enumerate(ps):
        y = y0 + 80 + i * ch
        d.text((x0 - 30, y + ch / 2 - 10), lab[p], font=font(34), fill=col[p], anchor="rm")
        for j, q in enumerate(ps):
            cx = x0 + j * cw
            if p == q:
                d.text((cx + cw / 2, y + ch / 2 - 10), "—", font=font(34), fill=FAINT, anchor="mm")
                continue
            val = v(p, q)
            hl = p in ("hard", "strong") and q not in ("hard", "strong")
            if hl:
                d.rounded_rectangle((cx + 20, y + 4, cx + cw - 20, y + ch - 24), 10, fill=(60, 40, 40))
            d.text((cx + cw / 2, y + ch / 2 - 10), f"{val:.0f}", font=font(40 if hl else 34), fill=RED if hl else FG, anchor="mm")
    d.text((960, 1000), L("встроенный противник и судья проигрывают всем четырём сетям почти одинаково: различить их не могут",
             "the built-in opponent and the judge lose to all four networks almost equally: they cannot tell them apart"),
           font=font(30), fill=YELLOW, anchor="mm")
    save(img, "card_round_robin")


# ------------------------------------------------------------------ scene 9-10

def final_match(stage=3):
    r = P.s10()
    et = P.equal_time()
    sc = lambda x: 100 * (x[0] + x[2] / 2) / sum(x[:3])
    img, d = page(L("Dot-G (раунд 114) против Dot (раунд 84), очки Dot-G", "Dot-G (round 114) vs Dot (round 84), Dot-G's score"))
    y1, top, bw = 800, 250, 330
    cols = [(L("по 200 вариантов\nу обеих", "200 options\neach"), sc(r["3:mcts"])),
            (L("обе без размышлений", "both without thinking"), sc(r["3:raw"])),
            (L(f"равное время на ход:\nDot-G {et['g5_sims']}, Dot 200", f"equal time per move:\nDot-G {et['g5_sims']}, Dot 200"), sc(r["3:equal_time"]))]
    _, half = Plot(d, (0, top, 10, y1), (0, 1), (0, 100)).p(0, 50)
    for i, (lab, v) in enumerate(cols[:stage]):
        x = 260 + i * 520
        t = y1 - v / 100 * (y1 - top)
        d.rounded_rectangle((x, t, x + bw, y1), 12, fill=C_G if i < 2 else (180, 160, 60))
        d.text((x + bw / 2, t - 16), pct(v, 1), font=font(64), fill=FG, anchor="mb")
        d.multiline_text((x + bw / 2, y1 + 30), lab, font=font(34, LATO), fill=FG, anchor="ma", align="center", spacing=8)
    d.line((200, half, 1760, half), fill=FG, width=3)
    d.text((1780, half), "50", font=font(32, LATO), fill=FG, anchor="lm")
    if stage == 3:
        d.text((960, 1010), L("200 партий в каждом матче; при равном времени разницы не видно",
                                  "200 games in each match; at equal time no difference is seen"), font=font(34), fill=YELLOW, anchor="mm")
    save(img, "card_final_match" + ("" if stage == 3 else str(stage)))


def cost():
    et = P.equal_time()
    img, d = page(L("Цена одного взгляда", "The price of one look"))
    if LANG != "ru":
        d.text((120, 165), "time per move", font=font(30, LATO), fill=DIM)
    for i, (lab, ms, col) in enumerate(((L("Dot: 200 вариантов", "Dot: 200 options"), et["dot_s"] * 1000, C_D),
                                        (L("Dot-G: 200 вариантов", "Dot-G: 200 options"), et["g5_200_s"] * 1000, C_G),
                                        (L(f"Dot-G: {et['g5_sims']} вариантов", f"Dot-G: {et['g5_sims']} options"), et["g5_s"] * 1000, C_G))):
        y = 330 + i * 190
        d.text((140, y + 40), lab, font=font(42, LATO), fill=FG, anchor="lm")
        d.rounded_rectangle((700, y, 700 + ms * 9, y + 80), 12, fill=col)
        d.text((720 + ms * 9, y + 40), L(f"{ms:.0f} мс на ход", f"{ms:.0f} ms"), font=font(40), fill=FG, anchor="lm")
    d.text((960, 960), L(f"один взгляд Dot-G стоит в {et['g5_200_s'] / et['dot_s']:.0f} раза дороже",
                         f"one look of Dot-G costs {et['g5_200_s'] / et['dot_s']:.0f} times as much"), font=font(40), fill=YELLOW, anchor="mm")
    save(img, "card_cost")


def summary(stage=4):
    img, d = page(L("Что мы узнали", "What we learned"))
    items = L(["Как сеть видит доску, решает, чему она научится:\nDot-G сама научилась считать цепочки.",
               "Новую доску — через школу, а старое надо повторять.",
               "Dot-G учится за меньшее число раундов и видит ход лучше, но взгляд втрое дороже:\nпри равном времени разницы не видно.",
               "Линейка может врать: силу проверяем прямым матчем."],
              ["How a network sees the board decides what it can learn:\nDot-G learned to count chains on her own.",
               "A new board through school, and the old one has to be repeated.",
               "Dot-G learns in fewer rounds and sees the move better,\nbut a look costs three times as much: at equal time no difference is seen.",
               "The ruler can lie: we check strength with a direct match."])
    y = 250
    for i, t in enumerate(items[:stage]):
        d.text((150, y), f"{i + 1}", font=font(70), fill=YELLOW)
        d.multiline_text((260, y + 10), t, font=font(40, LATO), fill=FG, spacing=12)
        y += 170 if "\n" not in t else 220
    save(img, "card_summary" + ("" if stage == 4 else str(stage)))


def promise():
    """Scene 6: the experiment promised in part 3 (series 12 / 12b): the same judge answers for both nets."""
    S = P.s12()
    img, d = page(L("Опыт, обещанный в третьей части", "The experiment promised in part 3"))
    box(d, (120, 380, 760, 640), L(f"одни и те же\nответы судьи\n{plural(S['n_train'], 'позиция', 'позиции', 'позиций')} середины партии",
                                        f"the same\njudge's answers\n{S['n_train']} middle-game positions"), size=40,
        outline=(90, 96, 115))
    box(d, (1160, 300, 1680, 460), "Dot-G", size=56, colour=C_G, outline=C_G)
    box(d, (1160, 560, 1680, 720), "Dot-2", size=56, colour=C_D, outline=C_D)
    arrow(d, (780, 480), (1130, 390))
    arrow(d, (780, 540), (1130, 630))
    d.text((960, 880), L(f"проверка: {plural(S['n_held'], 'новая позиция', 'новые позиции', 'новых позиций')}, которых сети не видели",
             f"test: {S['n_held']} new positions the networks had not seen"), font=font(42, LATO),
           fill=FG, anchor="mm")
    d.text((960, 950), L("позиции — из партий Dot-2", "the positions come from Dot-2's games"), font=font(34, LATO), fill=DIM, anchor="mm")
    save(img, "card_promise")


def plural(n, one, few, many):
    w = one if n % 10 == 1 and n % 100 != 11 else few if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else many
    return f"{n} {w}"


def gain_text(g):
    d, lo, hi = g
    s = lambda v: L(f"{v:+.1f}".replace(".", ","), f"{v:+.1f}").replace("-", "\u2212")
    word = gain_word(g)
    return s(d), L(f"[{s(lo)}; {s(hi)}]", f"[{s(lo)}, {s(hi)}]"), L(*word)


def gain_word(g):
    """(ru, en): the reading of a gain with its 95 % interval."""
    _, lo, hi = g
    return (("прибавила", "improved") if lo > 0 else ("стала хуже", "got worse") if hi < 0
            else ("прироста не видно", "no gain seen"))


def s12b(stage=1):
    """Scene 6, end: series 12b variant (b) (judge answers only, the whole net learns): the training answers learned
    by heart, no gain on the new positions; stage 2 adds the series 12 (b) row (with Dot-2's games)."""
    S = P.s12()
    img, d = page(L("Учится вся сеть, одни ответы судьи", "The whole network learns, judge's answers only"))
    d.text((96, 150), L("серия 12б, вариант (б)", "series 12b, variant (b)"), font=font(36, LATO), fill=DIM)
    nets = (("Dot-G", C_G), ("Dot-2", C_D))
    # left: the training answers after fine-tuning
    d.text((460, 250), L("выученные ответы", "learned answers"), font=font(48), fill=FG, anchor="mm")
    y1, top, bw = 700, 340, 200
    for i, (n, col) in enumerate(nets):
        v = S["12b"][n]["train_lr01"]
        x = 230 + i * 260
        t = y1 - v / 100 * (y1 - top)
        d.rounded_rectangle((x, t, x + bw, y1), 12, fill=col)
        d.text((x + bw / 2, t - 12), pct(v, 0), font=font(56), fill=FG, anchor="mb")
        d.text((x + bw / 2, y1 + 20), n, font=font(40, LATO), fill=col, anchor="mt")
    d.text((460, 800), L(f"{plural(S['n_train'], 'ответ', 'ответа', 'ответов')} судьи, которые сеть видела",
                      f"{S['n_train']} judge's answers the network saw"), font=font(30, LATO), fill=DIM, anchor="mm")
    # right: the gain on the new positions, with its 95 % interval around zero
    d.text((1400, 250), L("новые позиции", "new positions"), font=font(48), fill=FG, anchor="mm")
    x0, x1, lim = 1080, 1720, 12.0
    px = lambda v: x0 + (v + lim) / (2 * lim) * (x1 - x0)
    d.text((px(0), 715), "0", font=font(30, LATO), fill=DIM, anchor="mt")
    for i, (n, col) in enumerate(nets):
        g = S["12b"][n]["lr01"]
        big, iv, word = gain_text(g)
        y = 410 + i * 190
        d.text((x0 - 30, y), n, font=font(40, LATO), fill=col, anchor="rm")
        d.line((px(0), y - 34, px(0), y + 34), fill=DIM, width=3)
        d.line((px(g[1]), y, px(g[2]), y), fill=col, width=8)
        for v in g[1:]:
            d.line((px(v), y - 18, px(v), y + 18), fill=col, width=6)
        d.ellipse((px(g[0]) - 14, y - 14, px(g[0]) + 14, y + 14), fill=col)
        d.text((px(0), y - 44), f"{big}  {iv}", font=font(32, LATO), fill=FG, anchor="mb")
        d.text((px(0), y + 44), word, font=font(34), fill=FG, anchor="mt")
    d.text((1400, 800), L(f"прирост к себе до учёбы, {plural(S['n_held'], 'позиция', 'позиции', 'позиций')}, интервал 95 %",
                       f"gain over itself before learning, {S['n_held']} positions, 95% interval"), font=font(30, LATO), fill=DIM,
           anchor="mm")
    if stage == 2:
        parts = []
        for n in ("Dot-2", "Dot-G"):
            big, iv, word = gain_text(S["12"][n]["lr01"])
            parts.append(f"{n} {big} {iv} — {word}")
        d.text((960, 900), L("с партиями Dot-2 (серия 12, вариант (б)):", "with Dot-2's games (series 12, variant (b)):"), font=font(36, LATO), fill=YELLOW, anchor="mm")
        d.text((960, 960), ";   ".join(parts), font=font(36, LATO), fill=FG, anchor="mm")
    else:
        assert all(S["12b"][n]["train_lr01"] == 100 and gain_word(S["12b"][n]["lr01"])[0] == "прироста не видно"
                   for n, _ in nets)
        d.text((960, 930), L("все ответы выучены наизусть, на новых позициях прироста не видно ни у одной",
                         "all answers learned by heart; on new positions no gain is seen for either"),
               font=font(36), fill=YELLOW, anchor="mm")
    save(img, "card_s12b" + ("" if stage == 1 else "2"))


def end():
    img, d = page()
    d.text((960, 430), L("Dot-G видит лучше. Dot считает быстрее.", "Dot-G sees better. Dot counts faster."), font=font(72), fill=YELLOW, anchor="mm")
    d.text((960, 580), L("обе сети — в приложении (релиз v0.4)", "both networks are in the app (release v0.4)"), font=font(48, LATO), fill=FG, anchor="mm")
    d.text((960, 660), L("код, опыты и поправка к первой части — на GitHub", "code, experiments and the correction to part 1 are on GitHub"), font=font(40, LATO), fill=DIM, anchor="mm")
    save(img, "card_end")


def main():
    title(); sees(); three_nets(); save(click(), "card_click"); school(); cloud(False); cloud(); four()
    for k in (1, 2, 3):
        school_table(k)
    matches(1); final_match(1); final_match(2); summary(1); summary(2); summary(3)
    rounds_chart("card_rounds0", L("Dot-G учится играть рядом с Dot", "Dot-G learns to play next to Dot"))
    map_g(); mid_bars()
    rounds_chart("card_rounds", L("Dot-G учится играть рядом с Dot", "Dot-G learns to play next to Dot"), mark_hard=True)
    bridge_card()
    rounds_chart("card_peak", L("Лестница: пик и спад", "The ladder: peak and decline"), show_dot=False, mark_peak=True)
    matches(); round_robin(); final_match(); cost(); summary(); promise(); s12b(1); s12b(2); end()


if __name__ == "__main__":
    main()
