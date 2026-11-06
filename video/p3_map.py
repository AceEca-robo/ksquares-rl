#!/usr/bin/env python3
"""The "who knows what" map of the part 3 video (scenes 7, 9, 10).

    .venv/bin/python video/p3_map.py   # video/p3/map_ru.mp4, card_map_5.png, card_map_6.png

The X axis is the game, first line to last. Three zones: start (empty, the judge
does not reach it), middle (20-40 safe lines, dark box) and endgame (20 to 0 safe
lines, the lines of docs/who_knows.png with the axis reversed). The X scale
is to scale only inside the endgame. Each part appears when the voice names it:
EVENTS gives the sentence of scene 7 that starts it. Needs video/p3/voice_ru_7.mp3.
"""
import json, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from PIL import Image
from video_anim import sentence_starts
from video_cards import C_POLICY, C_VALUE, DIM, FAINT, FG, GRID, LATO, PLUS, font, page
from video_thumbnail import YELLOW
OUT = os.path.join(HERE, "p3"); FPS = 25; TAIL = 1.0; FADE = 0.4; WIPE = 1.6
SCRIPT = os.path.join(ROOT, "docs/video_script_part3_ru.md")

X0, B0, B1, X1, Y0, Y1 = 150, 450, 880, 1820, 250, 860     # plot left, box, plot right; plot top, bottom
BOX = (10, 11, 15)
WHITE = (255, 255, 255)
fx = lambda s: B1 + (20 - s) / 20 * (X1 - B1)
fy = lambda v: Y1 - v / 100 * (Y1 - Y0)


def load():
    """{safe lines: (hand, intuition, search)} on decisive positions, n >= 10; {safe lines: % solved}"""
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
    return data, solved


DATA, SOLVED = load()


def curve(d, pts, col, square=False):
    d.line(pts, fill=col, width=5, joint="curve")
    for x, y in pts:
        (d.rectangle if square else d.ellipse)((x - 7, y - 7, x + 7, y + 7), fill=col)


def legend(d, i, col, text):
    x, y = (760 + (i % 2) * 520, 62 + (i // 2) * 52)
    d.ellipse((x, y, x + 28, y + 28), fill=col)
    d.text((x + 42, y + 14), text, font=font(28, LATO), fill=FG, anchor="lm")


def zone(d, x0, x1, text, col=FG):
    d.multiline_text(((x0 + x1) / 2, Y0 - 12), text, font=font(29), fill=col, anchor="md", align="center", spacing=4)


def x_axis(d):
    d.line((X0, Y1, X1, Y1), fill=DIM, width=3)
    d.polygon(((X1 + 30, Y1), (X1 + 6, Y1 - 10), (X1 + 6, Y1 + 10)), fill=DIM)
    d.text((X0, Y1 + 56), "первая линия", font=font(30, LATO), fill=DIM, anchor="lt")
    d.text((X1, Y1 + 56), "последняя линия", font=font(30, LATO), fill=DIM, anchor="rt")
    for s, x in ((40, B0), (20, fx(20)), (15, fx(15)), (10, fx(10)), (5, fx(5)), (0, fx(0))):
        d.text((x, Y1 + 14), str(s), font=font(28, LATO), fill=DIM, anchor="mt")
    d.text((fx(10), Y1 + 56), "безопасных линий осталось", font=font(28, LATO), fill=DIM, anchor="mt")
    d.text((fx(10), Y1 + 100), "игроки: до 18 линий, дальше мало позиций для проверки", font=font(24, LATO), fill=FAINT, anchor="mt")


def y_axis(d):
    for v in (0, 25, 50, 75, 100):
        if v: d.line((X0, fy(v), X1, fy(v)), fill=GRID, width=2)
        d.text((X0 - 16, fy(v)), str(v), font=font(28, LATO), fill=DIM, anchor="rm")
    d.text((X0 - 80, 148), "% верных ходов", font=font(28, LATO), fill=DIM)


def zone_start(d):
    zone(d, X0, B0, "начало партии")
    d.text(((X0 + B0) / 2, 530), "судья", font=font(30, LATO), fill=DIM, anchor="mm")
    d.text(((X0 + B0) / 2, 572), "не достаёт", font=font(30, LATO), fill=DIM, anchor="mm")


def zone_box(d):
    d.rectangle((B0, Y0, B1, Y1), fill=BOX, outline=FAINT, width=2)
    zone(d, B0, B1, "середина\n20–40 безопасных линий")


def question(d):
    d.text(((B0 + B1) / 2, 555), "?", font=font(190), fill=YELLOW, anchor="mm")


def zone_end(d):
    zone(d, B1, X1, "концовка\nот 20 линий до конца")


def player(k):
    def f(d):
        curve(d, [(fx(s), fy(v[k])) for s, v in sorted(DATA.items(), reverse=True)],
              (C_POLICY, C_VALUE, PLUS)[k])
        legend(d, k, (C_POLICY, C_VALUE, PLUS)[k],
               ("Dot-2 без размышлений", "интуиция", "Dot-2 с размышлениями")[k])
    return f


def judge(d):
    curve(d, [(fx(s), fy(v)) for s, v in sorted(SOLVED.items(), reverse=True)], WHITE, square=True)
    legend(d, 3, WHITE, "судья: доля решённых")


# (first words of the sentence, what appears, fade or wipe: 1 left to right, -1 right to left)
EVENTS = [
    ("По горизонтали", x_axis, 0), ("По вертикали", y_axis, 0),
    ("Начало партии", zone_start, 0), ("Середина партии", zone_box, 0), ("Поэтому в коробке", question, 0),
    ("Концовка", zone_end, 0),
    ("Синяя", player(0), 1), ("Оранжевая", player(1), 1), ("Зелёная —", player(2), 1), ("И белая", judge, -1),
]


def layers():
    """images[k]: the map with the first k events drawn"""
    img, d = page("Кто что знает")
    out = [img.copy()]
    for _, f, _ in EVENTS:
        f(d)
        out.append(img.copy())
    return out


def sentences():
    from video_voice import scenes
    text = next(t for k, _, _, t in scenes(SCRIPT, "ru") if str(k) == "7")
    return [s for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s]


def frames(imgs, starts, end):
    sents = sentences()
    at = [starts[next(i for i, s in enumerate(sents) if s.startswith(w))] for w, _, _ in EVENTS]
    print({w: round(t, 1) for (w, _, _), t in zip(EVENTS, at)})
    idle = {k: im.tobytes() for k, im in enumerate(imgs)}
    for i in range(round((end + TAIL) * FPS)):
        t = i / FPS
        k = sum(a <= t for a in at)
        if k == 0:
            yield idle[0]; continue
        wipe = EVENTS[k - 1][2]; p = (t - at[k - 1]) / (WIPE if wipe else FADE)
        if p >= 1:
            yield idle[k]; continue
        if not wipe:
            yield Image.blend(imgs[k - 1], imgs[k], p).tobytes(); continue
        im = imgs[k - 1].copy()
        cut = B1 + (X1 + 10 - B1) * p
        box = (B1 - 10, 0, int(cut), 1080) if wipe > 0 else (int(X1 + 10 - (X1 + 10 - B1 + 10) * p), 0, X1 + 10, 1080)
        im.paste(imgs[k].crop(box), box[:2])
        # the legend appears at once, outside the wiped area
        im.paste(imgs[k].crop((0, 0, 1920, 170)), (0, 0))
        yield im.tobytes()


def main():
    imgs = layers()
    imgs[-1].save(f"{OUT}/card_map_5.png")
    img, d = page("Кто что знает")
    x_axis(d); y_axis(d); zone_start(d); zone_box(d); question(d); zone_end(d); player(2)(d)
    d.text((960, 1040), "200 вариантов: 96 % верных ходов там, где судья может проверить", font=font(34), fill=YELLOW, anchor="mm")
    img.save(f"{OUT}/card_map_6.png")
    s, end = sentence_starts("ru", "7", SCRIPT, f"{OUT}/voice_ru_7.mp3")
    assert len(s) == len(sentences()), (len(s), len(sentences()))
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080",
                          "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18",
                          f"{OUT}/map_ru.mp4"], stdin=subprocess.PIPE)
    for f in frames(imgs, s, end):
        p.stdin.write(f)
    p.stdin.close(); p.wait()


if __name__ == "__main__":
    main()
