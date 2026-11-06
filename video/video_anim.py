#!/usr/bin/env python3
"""Scene 2 of the video (docs/video_script_ru.md): the rules, the chains and
the double-cross, animated on a board in the app's colours.

    .venv/bin/python video/video_anim.py            # video/anim_2.mp4, anim_2p.mp4, anim_2a.mp4
    .venv/bin/python video/video_anim.py --lang en  # anim_*_en.mp4

The position is the thumbnail's (video_thumbnail.CHAIN_GAME: KSquares Hard
vs Hard in the simulator up to the end of the safe lines, chains 6 6 4 4 4);
the endgame after it is played by the exact solver (agents/strong.py) for
both sides, so every capture and every double-cross is the best move. Events
are timed to the sentences of the scene's voice (video/voice_<lang>_<n>.mp3;
sentence ends snapped to the voice's pauses), so the voice has to exist first.
"""

import argparse
import os
import re
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from sim.board import Board  # noqa: E402
from video_thumbnail import (BG, BLUE, CHAIN_COLOURS, CHAIN_GAME, FONT, FONT_BOLD, GHOST,  # noqa: E402
                             LINE, PANEL, RED, W, H, YELLOW, chain_boxes, line_ends)
from video_voice import duration, scenes  # noqa: E402

FPS = 25
VOICE_AT = 0.5          # as in the storyboard
TAIL = 3.0              # picture after the voice (the storyboard cuts it)
DIM = (150, 156, 170)

TEXT = {
    "ru": {
        "red": "красные", "blue": "синие",
        "rules_line": "ход — одна линия",
        "rules_box": "замкнул квадрат — он твой,\nи ходишь ещё раз",
        "rules_win": "больше квадратов —\nпобеда",
        "safe": "безопасные линии:\nне дают сопернику\nзамкнуть квадрат",
        "chains": "безопасные кончились:\nдоска нарезана на цепочки",
        "open": "синим приходится\nоткрыть цепочку",
        "allbut2": "красные берут всё, кроме двух,\nи отдают два одной линией",
        "dc": "двойной крест",
        "dc_take": "два квадрата одной линией —\nи снова ходить",
        "next": "и открыть следующую цепочку",
        "control": "у красных контроль:\nони получают почти всё",
        "parity": "бороться за контроль — заранее:\nсчитать, кому достанется\nпоследний безопасный ход",
        "early": "",
        "hard": "встроенный противник",
        "hard_dc": "двойной крест: 100 %",
        "hard_par": "кому последний ход: 50 %",
        "hard_note": "как монетка (линия наугад: 49 %)",
    },
    "en": {
        "red": "red", "blue": "blue",
        "rules_line": "a move is one line",
        "rules_box": "close a box: it's yours,\nand you move again",
        "rules_win": "most boxes\nwins",
        "safe": "safe lines:\nno box gets a third side",
        "chains": "safe lines are gone:\nthe board is chains",
        "open": "blue has to\nopen a chain",
        "allbut2": "red takes all but two\nand hands those over with one line",
        "dc": "the double-cross",
        "dc_take": "two boxes with one line —\nand move again",
        "next": "and opens the next chain",
        "control": "red is in control:\nred gets nearly everything",
        "parity": "fight for control early:\ncount who gets\nthe last safe move",
        "early": "",
        "hard": "built-in opponent",
        "hard_dc": "double-cross: 100%",
        "hard_par": "who gets the last move: 50%",
        "hard_note": "a coin toss (a random line: 49%)",
    },
}

# the rules on a 2x2 board: (player, line); box 0 is closed by red's 5
MINI = [(0, 0), (1, 9), (0, 11), (1, 2), (0, 7), (1, 3), (0, 5), (0, 6)]


# ---------------------------------------------------------------- timing

def pick(cands, ests):
    """len(ests) of the sorted cands, in order, with the least squared
    distance to ests (dynamic programming)."""
    n, m = len(cands), len(ests)
    INF = float("inf")
    cost = [[INF] * (n + 1) for _ in range(m + 1)]   # cost[i][j]: ests[:i] from cands[:j]
    cost[0] = [0.0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(i, n + 1):
            cost[i][j] = min(cost[i][j - 1], cost[i - 1][j - 1] + (cands[j - 1] - ests[i - 1]) ** 2)
    out, j = [], n
    for i in range(m, 0, -1):
        while cost[i][j] == cost[i][j - 1]:
            j -= 1
        out.append(cands[j - 1])
        j -= 1
    return out[::-1]


def sentence_starts(lang, n, script=None, mp3=None):
    """Start of every sentence of scene n's voice, in seconds of the clip
    (the voice starts at VOICE_AT): boundaries by text length, snapped to the
    nearest pause of the voice."""
    script = script or os.path.join(ROOT, "docs", {"ru": "video_script_ru.md", "en": "video_script_en.md"}[lang])
    text = next(t for k, _, _, t in scenes(script, lang) if str(k) == n)
    sents = [s for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s]
    mp3 = mp3 or os.path.join(HERE, f"voice_{lang}_{n}.mp3")
    total = duration(mp3)
    r = subprocess.run(["ffmpeg", "-i", mp3, "-af", "silencedetect=noise=-35dB:d=0.18", "-f", "null", "-"],
                       capture_output=True, text=True)
    st = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    en = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    pauses = [(a, b) for a, b in zip(st, en) if a > 0.1 and b < total - 0.1]
    weights = [len(s) + 8 for s in sents]
    ests = [total * sum(weights[:i + 1]) / sum(weights) for i in range(len(sents) - 1)]
    # edge-tts pauses ~1 s between sentences (more when voiced one sentence
    # at a time) and seldom that long inside one: the boundaries are the
    # long pauses, picked in order closest to the estimates by text length
    long = [b for a, b in pauses if b - a > 0.9]
    if len(long) >= len(ests):
        return [VOICE_AT + t for t in [0.0] + pick(long, ests)], VOICE_AT + total
    out = [0.0]
    for est in ests:
        near = [p for p in pauses if abs((p[0] + p[1]) / 2 - est) < 1.5 and p[1] > out[-1] + 0.5]
        out.append(min(near, key=lambda p: abs((p[0] + p[1]) / 2 - est))[1] if near else est)
    return [VOICE_AT + t for t in out], VOICE_AT + total


# ---------------------------------------------------------------- the game

def endgame():
    """[(player, line, boxes)] from the chain position to the end, both
    sides by the exact solver; among equally good moves the lowest line
    (blue's first move then opens the 4-chain at the top right)."""
    from agents.strong import Engine, exact_move_values

    b = chain_board()
    eng = Engine(5, 5)
    seq = []
    while not b.done:
        vals = exact_move_values(eng, b, None)
        m = min(k for k, v in vals.items() if v == max(vals.values()))
        p = b.player
        seq.append((p, m, b.step(m)))
    return seq


def chain_board():
    b = Board(5, 5)
    for m in CHAIN_GAME:
        b.step(m)
    return b


# ---------------------------------------------------------------- drawing

class Frame:
    def __init__(self, lang):
        self.t = TEXT[lang]
        self.f_cap = ImageFont.truetype(FONT, 58)
        self.f_big = ImageFont.truetype(FONT, 96)
        self.f_score = ImageFont.truetype(FONT_BOLD, 46)
        self.f_small = ImageFont.truetype(FONT_BOLD, 40)
        self.f_len = ImageFont.truetype(FONT, 64)

    def board(self, img, board, x0, y0, cell, last=None, chains=None, glow=(), chain_alpha=0.0,
              labels=False):
        """Lines, boxes (owner's colour), unclaimed chain boxes in their
        chain colour at chain_alpha, the last line in its player's colour,
        glow: lines drawn yellow and thick."""
        w, h = board.geo.w, board.geo.h
        over = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)

        def xy(p):
            return x0 + p[0] * cell, y0 + p[1] * cell

        inset = cell * 0.08
        for b in range(board.geo.n_boxes):
            bx, by = xy((b % w, b // w))
            rect = (bx + inset, by + inset, bx + cell - inset, by + cell - inset)
            if board.owner[b] in (0, 1):
                d.rounded_rectangle(rect, cell * 0.08, fill=(RED if board.owner[b] == 0 else BLUE) + (230,))
            elif chains and b in chains and chain_alpha > 0:
                d.rounded_rectangle(rect, cell * 0.08, fill=chains[b] + (round(150 * chain_alpha),))
        lw = max(5, round(cell * 0.06))
        for i in range(board.geo.n_lines):
            a, z = (xy(p) for p in line_ends(i, w))
            if i in glow:
                d.line((a, z), fill=YELLOW + (255,), width=round(lw * 2.2))
            elif last and i == last[1]:
                d.line((a, z), fill=(RED if last[0] == 0 else BLUE) + (255,), width=round(lw * 1.8))
            elif board.lines[i]:
                d.line((a, z), fill=LINE + (255,), width=lw)
            else:
                d.line((a, z), fill=GHOST + (255,), width=max(2, lw // 3))
        r = lw * 1.1
        for y in range(h + 1):
            for x in range(w + 1):
                cx, cy = xy((x, y))
                d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=LINE + (255,))
        if labels and chains and chain_alpha > 0.5:
            for g in chain_boxes(board):
                if len(g) >= 3 and all(board.owner[b] not in (0, 1) for b in g):
                    b = sorted(g)[len(g) // 2]
                    bx, by = xy((b % w, b // w))
                    d.text((bx + cell / 2, by + cell / 2), str(len(g)), font=self.f_len,
                           fill=(255, 255, 255, round(255 * min(1, (chain_alpha - 0.5) * 2))),
                           anchor="mm", stroke_width=5, stroke_fill=(10, 10, 14, 255))
        img.alpha_composite(over)

    def panel(self, img, score, caption=None, big=None, card=None):
        d = ImageDraw.Draw(img)
        x0, x1 = 1080, 1840
        d.rounded_rectangle((x0, 165, x1, 915), 28, fill=PANEL)
        # score: red left, blue right
        for k, (name, col) in enumerate(((self.t["red"], RED), (self.t["blue"], BLUE))):
            cx = x0 + 60 + k * 360
            d.ellipse((cx, 222, cx + 36, 258), fill=col)
            d.text((cx + 56, 240), f"{name}  {score[k]}", font=self.f_score, fill=(235, 237, 242), anchor="lm")
        d.line((x0 + 40, 300, x1 - 40, 300), fill=(52, 56, 68), width=2)
        y = 360
        width = x1 - x0 - 100
        if big:
            f = fitted(big, width, 96)
            d.multiline_text((x0 + 50, y), big, font=f, fill=YELLOW, spacing=10)
            y += round(f.size * 1.25 * (big.count("\n") + 1) + 40)
        if caption:
            d.multiline_text((x0 + 50, y), caption, font=fitted(caption, width, 54), fill=(235, 237, 242),
                             spacing=16)
        if card:
            title, rows, note = card
            y = 620 if (big or caption) else 360
            d.text((x0 + 50, y), title, font=self.f_score, fill=DIM)
            for i, (s, col) in enumerate(rows):
                d.text((x0 + 50, y + 70 + i * 70), s, font=fitted(s, x1 - x0 - 100, 54), fill=col)
            if note:
                d.text((x0 + 50, y + 70 + len(rows) * 70 + 10), note, font=self.f_small, fill=DIM)


def fitted(text, width, size):
    """The font of `size` shrunk until every line of `text` fits `width`."""
    while size > 24:
        f = ImageFont.truetype(FONT, size)
        if max(f.getlength(line) for line in text.split("\n")) <= width:
            return f
        size -= 2
    return f


def ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------- scenes

class Clip:
    """Timed line events on a board plus a caption track."""

    def __init__(self, board, x0, y0, cell, length):
        self.start = board
        self.x0, self.y0, self.cell = x0, y0, cell
        self.length = length
        self.moves = []        # (t, player, line)
        self.caps = []         # (t, dict(caption=..., big=..., card=...))
        self.chains = None     # (t_in, colours)
        self.glows = []        # (t0, t1, lines)

    def at(self, t):
        b = self.start.clone()
        last = None
        for tm, p, ln in self.moves:
            if tm > t:
                break
            assert b.player == p, (tm, p, ln)
            b.step(ln)
            last = (p, ln)
        cap = {}
        for tm, c in self.caps:
            if tm <= t:
                cap = c
        glow = set()
        for t0, t1, lines in self.glows:
            if t0 <= t < t1:
                glow |= set(lines)
        return b, last, cap, glow


def render(clip, fr, path):
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset",
                             "medium", "-crf", "18", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for k in range(round(clip.length * FPS)):
        t = k / FPS
        b, last, cap, glow = clip.at(t)
        img = Image.new("RGBA", (W, H), BG + (255,))
        alpha, colours = 0.0, None
        if clip.chains:
            t_in, colours = clip.chains
            alpha = ease((t - t_in) / 0.8)
        fr.board(img, b, clip.x0, clip.y0, clip.cell, last=last, chains=colours, glow=glow,
                 chain_alpha=alpha, labels=True)
        fr.panel(img, b.score, **cap)
        proc.stdin.write(img.convert("RGB").tobytes())
    proc.stdin.close()
    proc.wait()
    print("wrote", path, f"{clip.length:.1f} s")


def chain_colours(board):
    groups = [g for g in chain_boxes(board) if len(g) >= 3]
    return {b: CHAIN_COLOURS[k % len(CHAIN_COLOURS)] for k, g in enumerate(groups) for b in g}


def spread(t0, t1, n):
    return [t0 + (t1 - t0) * (i + 0.5) / n for i in range(n)]


def scene_2(lang, fr, seq, out):
    s, end = sentence_starts(lang, "2")
    assert len(s) == 9, f"scene 2: {len(s)} sentences, the timeline expects 9"
    t = TEXT[lang]
    # the rules on 2x2, then the 5x5 game (one clip each, joined)
    length = end + TAIL
    mini = Clip(Board(2, 2), 160 + 75, 165 + 75, 300, s[4])
    times = spread(s[1], s[2], 5) + [s[2] + 0.2, s[2] + 1.2, s[2] + 2.4]
    assert times[-1] < s[3], "sentence 3 is too short for the capture"
    mini.moves = [(tm, p, ln) for tm, (p, ln) in zip(times, MINI)]
    mini.caps = [(0, {}), (s[1], {"caption": t["rules_line"]}), (s[2] + 1.2, {"caption": t["rules_box"]}),
                 (s[3], {"big": t["rules_win"]})]
    mini.glows = [(s[2] + 1.2, s[2] + 2.2, [5])]

    big = Clip(Board(5, 5), 160, 165, 150, length - s[4])
    off = s[4]
    moves = [(tm - off, p, ln) for tm, (p, ln) in
             zip(spread(s[4], s[5] - 0.3, len(CHAIN_GAME)), replay_players(CHAIN_GAME))]
    blue_open, *rest = seq
    take = [m for m in rest[:2]]                 # red's two captures
    deal = rest[2]                               # red's double-dealing line
    moves.append((s[6] + 1.0 - off, *blue_open[:2]))
    for tm, m in zip(spread(s[7], s[7] + 2.4, 2), take):
        moves.append((tm - off, *m[:2]))
    moves.append((s[7] + 3.4 - off, *deal[:2]))
    big.moves = moves
    big.chains = (s[5] - off, chain_colours(chain_board()))
    big.caps = [(0, {"caption": t["safe"]}), (s[5] - off, {"caption": t["chains"]}),
                (s[6] - off, {"caption": t["open"]}), (s[7] - off, {"caption": t["allbut2"]}),
                (s[8] - off, {"big": t["dc"], "caption": t["allbut2"]})]
    big.glows = [(s[7] + 3.4 - off, length, [deal[1]])]
    parts = [os.path.join(HERE, "build_anim", f"2_{k}.mp4") for k in "ab"]
    render(mini, fr, parts[0])
    render(big, fr, parts[1])
    concat(parts, out)
    return seq[4:]


def replay_players(moves):
    b = Board(5, 5)
    out = []
    for m in moves:
        out.append((b.player, m))
        b.step(m)
    return out


def scene_2p(lang, fr, seq, out):
    """No voice: blue takes the two boxes with one line (the double-cross)."""
    t = TEXT[lang]
    b = chain_board()
    for p, m, _ in seq[:4]:
        b.step(m)
    dc = seq[4]
    assert dc[2] == 2, "the 5th move of the endgame should take two boxes"
    clip = Clip(b, 160, 165, 150, 4.0)
    clip.moves = [(0.9, *dc[:2])]
    clip.chains = (-10, chain_colours(chain_board()))
    clip.caps = [(0, {"big": t["dc"]}), (0.9, {"big": t["dc"], "caption": t["dc_take"]})]
    clip.glows = [(0.9, 4.0, [dc[1]])]
    render(clip, fr, out)


def scene_2a(lang, fr, seq, out):
    s, end = sentence_starts(lang, "2а" if lang == "ru" else "2a")
    assert len(s) == 6, f"scene 2a: {len(s)} sentences, the timeline expects 6"
    t = TEXT[lang]
    b = chain_board()
    for p, m, _ in seq[:5]:
        b.step(m)
    rest = seq[5:]
    clip = Clip(b, 160, 165, 150, end + TAIL)
    clip.chains = (-10, chain_colours(chain_board()))
    times = [s[0] + 2.6] + spread(s[1], s[3] + 1.5, len(rest) - 1)
    clip.moves = [(tm, p, ln) for tm, (p, ln, _) in zip(times, rest)]
    card = (t["hard"], [(t["hard_dc"], (120, 220, 140))], None)
    card2 = (t["hard"], [(t["hard_dc"], (120, 220, 140)), (t["hard_par"], RED)], t["hard_note"])
    if t["early"]:      # parity and "early" in two sentences
        caps = [(s[2], {"caption": t["parity"]}), (s[3], {"caption": t["early"]}),
                (s[4], {"card": card}), (s[5], {"card": card2})]
    else:               # parity and "early" in one sentence, the card from s3
        caps = [(s[2], {"caption": t["parity"]}), (s[3], {"card": card}), (s[4], {"card": card2})]
    clip.caps = [(0, {"caption": t["dc_take"]}), (s[0] + 2.2, {"caption": t["next"]}),
                 (s[1], {"caption": t["control"]})] + caps
    render(clip, fr, out)


def concat(parts, out):
    lst = os.path.join(HERE, "build_anim", "concat.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out],
                   check=True)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lang", choices=sorted(TEXT), default="ru")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "build_anim"), exist_ok=True)
    fr = Frame(a.lang)
    seq = endgame()
    sfx = "" if a.lang == "ru" else "_en"
    scene_2(a.lang, fr, seq, os.path.join(HERE, f"anim_2{sfx}.mp4"))
    scene_2p(a.lang, fr, seq, os.path.join(HERE, f"anim_2p{sfx}.mp4"))
    scene_2a(a.lang, fr, seq, os.path.join(HERE, f"anim_2a{sfx}.mp4"))


if __name__ == "__main__":
    main()
