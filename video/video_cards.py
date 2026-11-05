#!/usr/bin/env python3
"""Stills for scenes 3-9 of the video (docs/video_script_ru.md), 1920x1080
png in video/, dark like the app and the scene 2 animation.

    .venv/bin/python video/video_cards.py            # video/card_*.png, shot_*.png
    .venv/bin/python video/video_cards.py --lang en  # the same with _en

Terminal shots are the real output of the commands, run here (as in
pacman-rl/video/video_shots.py). Charts and tables read the repo's data:
runs/parity_probe/curves.json and eval.json (scene 6), docs/az5b_ladder.md
(scenes 5 and 7), docs/results.md anchors.
"""

import argparse
import json
import os
import re
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from video_thumbnail import BG, BLUE, FONT, FONT_BOLD, GREEN, PANEL, RED, W, H, WHITE, YELLOW  # noqa: E402

LATO = "/usr/share/fonts/truetype/lato/Lato-Regular.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FG = (225, 228, 235)
DIM = (140, 146, 160)
FAINT = (70, 75, 90)
GRID = (44, 48, 62)
TERM, BAR, PATH = (24, 27, 36), (36, 40, 52), (90, 170, 255)
PLUS, MINUS = (80, 200, 120), (235, 64, 52)
# the scene 6 curves (the plot docs/parity_curves.png, same order)
C_POLICY, C_VALUE, C_GREEDY, C_MCTS = (90, 160, 255), (255, 170, 60), (120, 205, 120), (240, 90, 90)

T = {
    "ru": {
        "term": "Терминал",
        "bridge": "Терминал — мост в исходниках KSquares (git diff --stat от исходного архива)",
        "bench": "Терминал — скорость тренировочного зала (results.md)",
        "verify": "Терминал — встроенные противники в зале против настоящей игры (results.md)",
        "strong": "Терминал — судья против встроенного противника (results.md)",
        "log": "Терминал — журнал тренировки (overnight.md)",
        "az_title": "Как Dot учится: одна сеть, два ответа",
        "az_in": "доска 5×5\n11×11 клеток:\nлинии, квадраты,\nсчёт",
        "az_net": "Dot\nResNet 6×64\n0,51 млн\nпараметров",
        "az_pol": "«куда бы я пошла»\n60 чисел — по линии",
        "az_val": "интуиция\n«кто выиграет»\n−1 … +1",
        "az_loop": ["Dot играет сама с собой тысячи партий", "учится на этих партиях",
                    "матч: новая Dot ≥ 55 из 100 против прежней", "→ следующий раунд"],
        "az_mcts": "Dot думает: перебор\n200 вариантов\nперед каждым ходом",
        "elo_title": "Лестница силы (Эло, 5×5)",
        "elo_net_raw": "Dot, раунд 54, ходит сразу",
        "elo_net_mcts": "Dot, раунд 54, думает",
        "random": "случайный", "greedy": "жадный", "solver": "судья", "hard": "встроенный противник",
        "curves_title": "1000 позиций: кому достанется последний безопасный ход",
        "policy": "Dot выбирает ход", "value": "интуиция: кто выиграет",
        "greedy_v": "интуиция на ход вперёд", "mcts": "Dot думает",
        "x_label": "раунды, прошедшие отбор: первая тренировка 1–29, затем вторая 1–84",
        "run1": "первая тренировка", "run2": "вторая тренировка",
        "chance": "случайный ход",
        "tail_title": "1000 позиций: доля верных ходов",
        "tail_head": ["", "ходит сразу", "думает"],
        "tail_row": "Dot, раунд {g}",
        "night_title": "Последняя ночь: раунды 38–84",
        "night_head": ["раунд", "матч против\nпредыдущей", "Эло (думает)", "против\nсудьи"],
        "night_note": "каждая принятая Dot обыгрывала предыдущую, но против игры и судьи — потолок",
        "ladder_title": "Сила Dot по раундам (вторая тренировка)",
        "ladder_x": "раунд",
        "head_title": "Почему только 5×5",
        "head_now": "сейчас: выходы сети\nпривязаны к сетке 11×11\n→ размер зашит",
        "head_next": "дальше: свёрточные выходы\n→ любое поле, хоть 10×7",
    },
    "en": {
        "term": "Terminal",
        "bridge": "Terminal — the bridge in KSquares' source (git diff --stat from the upstream tarball)",
        "bench": "Terminal — the training hall's speed (results.md)",
        "verify": "Terminal — the built-in opponents in the hall vs the real game (results.md)",
        "strong": "Terminal — the judge vs the built-in opponent (results.md)",
        "log": "Terminal — the training log (overnight.md)",
        "az_title": "How Dot learns: one network, two answers",
        "az_in": "5×5 board\n11×11 grid:\nlines, boxes,\nscore",
        "az_net": "Dot\nResNet 6×64\n0.51M\nparameters",
        "az_pol": "\"where would I go\"\n60 numbers, one per line",
        "az_val": "intuition\n\"who will win\"\n−1 … +1",
        "az_loop": ["Dot plays thousands of games against herself", "learns from those games",
                    "match: the new Dot ≥ 55 of 100 vs the old one", "→ next round"],
        "az_mcts": "Dot thinks: goes through\n200 options\nbefore every move",
        "elo_title": "Strength ladder (Elo, 5×5)",
        "elo_net_raw": "Dot, round 54, moves at once",
        "elo_net_mcts": "Dot, round 54, thinks",
        "random": "random", "greedy": "greedy", "solver": "the judge", "hard": "built-in opponent",
        "curves_title": "1000 positions: who gets the last safe move",
        "policy": "Dot picks a move", "value": "intuition: who will win",
        "greedy_v": "intuition one move ahead", "mcts": "Dot thinks",
        "x_label": "rounds that passed the match: first training 1–29, then second 1–84",
        "run1": "first training", "run2": "second training",
        "chance": "random move",
        "tail_title": "1000 positions: share of right moves",
        "tail_head": ["", "moves at once", "thinks"],
        "tail_row": "Dot, round {g}",
        "night_title": "The last night: rounds 38–84",
        "night_head": ["round", "match vs\nprevious", "Elo (thinks)", "vs\nthe judge"],
        "night_note": "each accepted Dot beat the previous one, but against the game and the judge: a ceiling",
        "ladder_title": "Dot's strength per round (second training)",
        "ladder_x": "round",
        "head_title": "Why 5×5 only",
        "head_now": "now: the network's outputs\nare tied to the 11×11 grid\n→ the size is baked in",
        "head_next": "next: convolutional outputs\n→ any board, 10×7 too",
    },
}


def font(size, path=FONT):
    return ImageFont.truetype(path, size)


def page(title=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    if title:
        d.text((96, 64), title, font=font(64), fill=YELLOW)
    return img, d


def save(img, name, lang):
    path = os.path.join(HERE, f"{name}{'' if lang == 'ru' else '_en'}.png")
    img.save(path)
    print("wrote", path)


# ---------------------------------------------------------------- terminal shots

def run(cmd):
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "COLUMNS": "118"})
    return [l.expandtabs(8) for l in (r.stdout + r.stderr).rstrip("\n").splitlines()]


def terminal(name, lang, cmd, title, size=28, out=None, cut=118):
    img, d = page()
    f, fb = font(size, MONO), font(size, MONO_BOLD)
    x0, y0, x1, y1 = 60, 50, W - 60, H - 50
    d.rounded_rectangle((x0, y0, x1, y1), 16, fill=TERM)
    d.rounded_rectangle((x0, y0, x1, y0 + 52), 16, fill=BAR)
    d.rectangle((x0, y0 + 36, x1, y0 + 52), fill=BAR)
    for i, c in enumerate(((235, 64, 52), (255, 204, 0), (80, 200, 120))):
        d.ellipse((x0 + 22 + i * 34, y0 + 16, x0 + 42 + i * 34, y0 + 36), fill=c)
    d.text(((x0 + x1) / 2, y0 + 26), title, font=font(24, MONO), fill=DIM, anchor="mm")
    lh = int(size * 1.36)
    y = y0 + 80
    x = x0 + 28
    for part, col, ff in (("~/ksquares-rl", PATH, fb), ("$ ", FG, f), (cmd, FG, f)):
        d.text((x, y), part, font=ff, fill=col)
        x += d.textlength(part, font=ff)
    y += lh
    for line in (out if out is not None else run(cmd)):
        if y > y1 - lh:
            d.text((x0 + 28, y), "…", font=f, fill=DIM)
            break
        line = line[:cut]
        x = x0 + 28
        m = re.match(r"^(.*\|\s*\d+ )(\+*)(-*)$", line)
        if m:
            for part, col in ((m.group(1), FG), (m.group(2), PLUS), (m.group(3), MINUS)):
                d.text((x, y), part, font=f, fill=col)
                x += d.textlength(part, font=f)
        else:
            d.text((x, y), line, font=f, fill=FG)
        y += lh
    save(img, name, lang)


def shots(lang):
    t = T[lang]
    root = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()[:7]
    terminal("shot_bridge", lang, f"git diff --stat {root} HEAD -- game/ && grep -rc 'RL bridge' game/src | grep -v ':0'",
             t["bridge"], size=36, cut=80)
    terminal("shot_bench", lang, "sed -n '/^## Simulator speed/,/^The real game/p' docs/results.md | grep -E '^\\| (board|---|5x5)'",
             t["bench"], size=34)
    terminal("shot_verify", lang, "sed -n '/^## KSquares AI port/,/^## Strong/p' docs/results.md | grep -E '^\\||^1200|^130' | cut -c1-110",
             t["verify"], size=24, cut=110)
    terminal("shot_strong", lang, "sed -n '/^## Strong reference/,/^## Ladder/p' docs/results.md | grep -E '^\\|'",
             t["strong"], size=26)
    terminal("shot_log", lang, "grep '^## ' docs/overnight.md", t["log"], size=36)


# ---------------------------------------------------------------- scene 4: AlphaZero

def box(d, rect, text, fill=PANEL, outline=None, size=40, colour=FG):
    d.rounded_rectangle(rect, 22, fill=fill, outline=outline, width=4 if outline else 0)
    cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
    d.multiline_text((cx, cy), text, font=font(size), fill=colour, anchor="mm", align="center", spacing=10)


def arrow(d, a, b, col=DIM, w=6):
    d.line((a, b), fill=col, width=w)
    import math
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for s in (-1, 1):
        d.line((b, (b[0] - 26 * math.cos(ang + s * 0.45), b[1] - 26 * math.sin(ang + s * 0.45))), fill=col, width=w)


def card_az(lang):
    t = T[lang]
    img, d = page(t["az_title"])
    box(d, (96, 300, 476, 620), t["az_in"], size=38)
    box(d, (620, 340, 1060, 580), t["az_net"], size=42, outline=(90, 96, 115))
    box(d, (1220, 210, 1820, 430), t["az_pol"], size=38, outline=YELLOW, colour=YELLOW)
    box(d, (1220, 490, 1820, 710), t["az_val"], size=38, outline=C_VALUE, colour=C_VALUE)
    arrow(d, (476, 460), (612, 460))
    arrow(d, (1060, 420), (1212, 330))
    arrow(d, (1060, 500), (1212, 600))
    # the loop and the search
    y = 790
    d.rounded_rectangle((96, y, 1100, 1020), 22, fill=PANEL)
    for i, s in enumerate(t["az_loop"]):
        d.text((130, y + 30 + i * 50), f"{i + 1}. {s}" if i < 3 else f"    {s}", font=font(36, LATO), fill=FG)
    box(d, (1220, y, 1820, 1020), t["az_mcts"], size=38, outline=C_MCTS, colour=WHITE)
    save(img, "card_az", lang)


# ---------------------------------------------------------------- scene 5: Elo ladder

ANCHORS = [("random", 0), ("greedy", 905), ("Easy", 888), ("Medium", 1235), ("Hard", 1488), ("solver", 1815)]


def ladder_rows():
    """[(gen, gating %, raw Elo, mcts Elo, mcts vs strong %)] from docs/az5b_ladder.md."""
    rows = []
    for line in open(os.path.join(ROOT, "docs/az5b_ladder.md")):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 14 and c[0].isdigit():
            rows.append((int(c[0]), int(c[1]), int(c[7]), int(c[13]), int(c[12])))
    return rows


def card_elo(lang):
    t = T[lang]
    img, d = page(t["elo_title"])
    gen54 = next(r for r in ladder_rows() if r[0] == 54)
    x0, x1, ytop, ybot = 300, 1500, 220, 980
    lo, hi = -100, 2300

    def y(e):
        return ybot - (e - lo) / (hi - lo) * (ybot - ytop)
    for e in range(0, 2400, 500):
        d.line((x0, y(e), x1, y(e)), fill=GRID, width=2)
        d.text((x0 - 24, y(e)), str(e), font=font(32), fill=DIM, anchor="rm")
    names = {"random": t["random"], "greedy": t["greedy"], "solver": t["solver"], "Hard": t["hard"]}
    for i, (n, e) in enumerate(ANCHORS):
        key = n in ("random", "Hard", "solver")
        col = WHITE if key else DIM
        xx = x0 + 60 + (420 if n == "Easy" else 0)
        d.ellipse((xx - 14, y(e) - 14, xx + 14, y(e) + 14), fill=col)
        d.text((xx + 32, y(e)), f"{names.get(n, n)}  {e}", font=font(44 if key else 34), fill=col, anchor="lm")
    for e, label, col in ((gen54[2], t["elo_net_raw"], YELLOW), (gen54[3], t["elo_net_mcts"], C_MCTS)):
        xx = x0 + 760
        d.ellipse((xx - 18, y(e) - 18, xx + 18, y(e) + 18), fill=col)
        d.text((xx + 40, y(e)), f"{label}  {e}", font=font(44), fill=col, anchor="lm")
    arrow(d, (x0 + 760, y(gen54[2]) - 30), (x0 + 760, y(gen54[3]) + 34), col=C_MCTS)
    save(img, "card_elo", lang)


# ---------------------------------------------------------------- scene 6: the curves

def curves_data():
    res = json.load(open(os.path.join(ROOT, "runs/parity_probe/curves.json")))
    base = json.load(open(os.path.join(ROOT, "runs/parity_probe/eval.json")))
    keys = sorted(res, key=lambda k: (k.split("/")[0] != "az5", int(k.split("/")[1])))
    n = len(res[keys[0]]["policy"])
    rows = [(100 * sum(res[k]["policy"]) / n, res[k]["v_child"], 100 * sum(res[k]["v_greedy"]) / n,
             100 * sum(res[k]["mcts"]) / n) for k in keys]
    lines = {"hard": 100 * sum(base["hard"]) / n, "strong": 100 * sum(base["strong"]) / n,
             "chance": 100 * sum(base["_chance"]) / n}
    return keys, rows, lines


def chart_frame(d, rect, keys, xlab, run_names):
    x0, y0, x1, y1 = rect
    for v in range(0, 101, 20):
        yy = y1 - v / 100 * (y1 - y0)
        d.line((x0, yy, x1, yy), fill=GRID, width=2)
        d.text((x0 - 18, yy), f"{v}%", font=font(30), fill=DIM, anchor="rm")
    split = sum(1 for k in keys if k.startswith("az5/"))
    xs = [x0 + (i + 0.5) / len(keys) * (x1 - x0) for i in range(len(keys))]
    xm = (xs[split - 1] + xs[split]) / 2
    d.line((xm, y0, xm, y1), fill=FAINT, width=3)
    d.text(((x0 + xm) / 2, y1 + 30), run_names[0], font=font(34), fill=DIM, anchor="mm")
    d.text(((xm + x1) / 2, y1 + 30), run_names[1], font=font(34), fill=DIM, anchor="mm")
    d.text(((x0 + x1) / 2, y1 + 80), xlab, font=font(30), fill=DIM, anchor="mm")
    for i, k in enumerate(keys):
        g = k.split("/")[1]
        if g in ("1", "8", "29", "54", "84") or (k.startswith("az5/") and g == "3"):
            d.text((xs[i], y1 + 4), g, font=font(22), fill=FAINT, anchor="mt")
    return xs


def hline_label(d, x1, y, text, col, below=False):
    """A reference line's label right of the chart, or above (below) the
    line's end when it does not fit there (the longer Russian names)."""
    f = font(30)
    if x1 + 16 + d.textlength(text, font=f) <= W - 24:
        d.text((x1 + 16, y), text, font=f, fill=col, anchor="lm")
    else:
        d.text((x1, y + 8), text, font=f, fill=col, anchor="rt") if below else \
            d.text((x1, y - 8), text, font=f, fill=col, anchor="rb")


def card_curves(lang):
    t = T[lang]
    keys, rows, lines = curves_data()
    series = [(t["policy"], C_POLICY, 0, 6), (t["value"], C_VALUE, 1, 6), (t["greedy_v"], C_GREEDY, 2, 4),
              (t["mcts"], C_MCTS, 3, 6)]
    rect = (220, 190, 1500, 810)
    for upto in range(0, 5):
        img, d = page(t["curves_title"])
        xs = chart_frame(d, rect, keys, t["x_label"], (t["run1"], t["run2"]))
        x0, y0, x1, y1 = rect

        def yy(v):
            return y1 - v / 100 * (y1 - y0)
        for name, v, col, dash in ((t["hard"], lines["hard"], DIM, 1), (t["solver"], lines["strong"], DIM, 1),
                                   (t["chance"], lines["chance"], FAINT, 1)):
            for xa in range(x0, x1, 28):
                d.line((xa, yy(v), xa + 14, yy(v)), fill=col, width=3)
            hline_label(d, x1, yy(v), f"{name} {v:.0f}%", col, below=True)
        for k, (name, col, idx, w) in enumerate(series[:upto]):
            pts = [(x, yy(r[idx])) for x, r in zip(xs, rows)]
            d.line(pts, fill=col, width=w, joint="curve")
            for p in pts:
                d.ellipse((p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5), fill=col)
            last = rows[-1][idx]
            d.text((150 + (k % 2) * 880, 950 + (k // 2) * 56), f"● {name}: {rows[0][idx]:.0f} → {last:.0f}%",
                   font=font(36), fill=col)
        save(img, f"card_curves_{upto}", lang)


def card_tail(lang):
    t = T[lang]
    img, d = page(t["tail_title"])
    res = json.load(open(os.path.join(ROOT, "runs/parity_probe/eval.json")))
    n = len(res["_chance"])
    rows = [(t["hard"], 100 * sum(res["hard"]) / n, None)]
    for g in (33, 54, 73, 84):
        r = res[f"az5b/{g}"]
        rows.append((t["tail_row"].format(g=g), 100 * sum(r["raw"]) / n, 100 * sum(r["mcts"]) / n))
    rows.append((t["solver"], 100 * sum(res["strong"]) / n, None))
    xs = (160, 1000, 1500)
    y = 230
    for x, h in zip(xs, t["tail_head"]):
        d.text((x, y), h, font=font(40), fill=DIM, anchor="la" if x == 160 else "ra")
    for i, (name, raw, mc) in enumerate(rows):
        y = 320 + i * 100
        if name.endswith(("54", "84")):
            d.rounded_rectangle((120, y - 14, 1560, y + 76), 14, fill=PANEL)
        d.text((xs[0], y), name, font=font(52), fill=FG)
        d.text((xs[1], y), f"{raw:.1f}%", font=font(52), fill=C_POLICY, anchor="ra")
        if mc is not None:
            d.text((xs[2], y), f"{mc:.1f}%", font=font(52), fill=C_MCTS, anchor="ra")
    save(img, "card_tail", lang)


# ---------------------------------------------------------------- scene 7: the night

def card_night(lang):
    t = T[lang]
    img, d = page(t["night_title"])
    rows = [r for r in ladder_rows() if r[0] >= 38]
    xs = (200, 640, 1080, 1460)
    y = 200
    for x, h in zip(xs, t["night_head"]):
        d.multiline_text((x, y), h, font=font(34), fill=DIM, anchor="ma", align="center")
    best = max(r[3] for r in rows)
    for i, (g, gate, raw, mc, vs) in enumerate(rows):
        y = 310 + i * 52
        hl = g in (54, 84)
        if hl:
            d.rounded_rectangle((90, y - 6, 1620, y + 46), 10, fill=PANEL)
        col = YELLOW if g == 54 else (C_MCTS if g == 84 else FG)
        for x, v in zip(xs, (str(g), f"{gate}%", str(mc), f"{vs}%")):
            d.text((x, y), v, font=font(38), fill=col, anchor="ma")
    d.text((96, 1010), t["night_note"], font=font(34), fill=DIM)
    save(img, "card_night", lang)


def card_ladder(lang):
    """MCTS Elo and raw Elo per accepted az5b generation, with the anchors."""
    t = T[lang]
    img, d = page(t["ladder_title"])
    rows = ladder_rows()
    x0, y0, x1, y1 = 220, 200, 1560, 880
    lo, hi = 1400, 2250

    def yy(e):
        return y1 - (e - lo) / (hi - lo) * (y1 - y0)

    def xx(g):
        return x0 + (g - 1) / 83 * (x1 - x0)
    for e in range(1400, 2300, 100):
        d.line((x0, yy(e), x1, yy(e)), fill=GRID, width=2)
        d.text((x0 - 18, yy(e)), str(e), font=font(28), fill=DIM, anchor="rm")
    for g in (1, 20, 38, 54, 84):
        d.text((xx(g), y1 + 14), str(g), font=font(28), fill=DIM, anchor="mt")
    d.text(((x0 + x1) / 2, y1 + 60), t["ladder_x"], font=font(30), fill=DIM, anchor="mt")
    for name, e in ((t["hard"], 1488), (t["solver"] if lang == "ru" else "strong", 1815)):
        for xa in range(x0, x1, 28):
            d.line((xa, yy(e), xa + 14, yy(e)), fill=DIM, width=3)
        hline_label(d, x1, yy(e), f"{name} {e}", DIM)
    for idx, col, label in ((3, C_MCTS, t["mcts"]), (2, C_POLICY, t["policy"])):
        pts = [(xx(r[0]), yy(r[idx])) for r in rows]
        d.line(pts, fill=col, width=6, joint="curve")
        for (g, *_), p in zip(rows, pts):
            r = 11 if g == 54 else 6
            d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=YELLOW if g == 54 else col)
        d.text((x1 + 16, pts[-1][1] + (-30 if idx == 3 else 30)), label, font=font(30), fill=col, anchor="lm")
    xm = xx(38)
    d.line((xm, y0, xm, y1), fill=FAINT, width=3)
    save(img, "card_ladder", lang)


# ---------------------------------------------------------------- scene 9: the heads

def card_heads(lang):
    t = T[lang]
    img, d = page(t["head_title"])
    box(d, (96, 260, 900, 620), t["head_now"], size=44, outline=MINUS)
    box(d, (1020, 260, 1824, 620), t["head_next"], size=44, outline=PLUS)
    arrow(d, (910, 440), (1010, 440))
    # a 5x5 and a 10x7 grid side by side
    for (x, y, w, h, c) in ((330, 720, 5, 5, 46), (1160, 720, 10, 7, 46)):
        for i in range(w + 1):
            d.line((x + i * c, y, x + i * c, y + h * c), fill=DIM, width=3)
        for j in range(h + 1):
            d.line((x, y + j * c, x + w * c, y + j * c), fill=DIM, width=3)
    save(img, "card_heads", lang)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", choices=sorted(T), default="ru")
    a = ap.parse_args()
    shots(a.lang)
    for f in (card_az, card_elo, card_curves, card_tail, card_night, card_ladder, card_heads):
        f(a.lang)


if __name__ == "__main__":
    main()
