#!/usr/bin/env python3
"""Seed 121 of runs/blowouts (dot2c gen 16, 50 sims vs the real Hard, 1:69):
the intuition (raw V, Dot-2's view) by line, and two clips drawn from the moves.

    .venv/bin/python video/p3_game121.py   # video/p3/game121_v.json, game121_ru.mp4, game121_end.mp4
"""
import json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "scripts")]
from PIL import ImageDraw
from video_cards import DIM, FAINT, FG, font, page, LATO
from video_thumbnail import BLUE, PANEL, RED, YELLOW
from sim.board import Board

OUT = os.path.join(HERE, "p3"); FPS = 25; MISTAKE = 79
G = next(g for g in map(json.loads, open(os.path.join(ROOT, "runs/blowouts/games.jsonl"))) if g["seed"] == 121)
M = G["moves"]


def values():
    import torch
    from agents.az_net import Codec, N_PLANES_2, make_net
    from _common import load_config
    net = make_net(load_config(os.path.join(ROOT, "configs/dot2c.yaml")))
    net.load_state_dict(torch.load(os.path.join(ROOT, "runs/dot2c/gen_0016.pt"), map_location="cpu")); net.eval()
    c = Codec(10, 7, planes=N_PLANES_2)
    out = {}; b = Board(10, 7)
    for k, ln in enumerate(M):
        if not b.done:
            x = torch.from_numpy(c.encode_boards([b]).reshape(1, -1, c.rows, c.cols))
            with torch.no_grad():
                _, v, _ = net(x)
            out[k] = round(float(v.flatten()[0]) * (1 if b.player == 0 else -1), 3)
        b.step(ln)
    return out


X0, Y0, S = 150, 190, 100
def seg(geo, ln):
    st = 2 * geo.w + 1; r, q = ln // st, ln % st
    if q < geo.w:
        return (q, r), (q + 1, r)
    return (q - geo.w, r), (q - geo.w, r + 1)


def frame(k, judge, V):
    """k lines drawn."""
    img, d = page()
    b = Board(10, 7); own = {}
    g = b.geo if hasattr(b, "geo") else None
    from sim.board import geometry
    g = geometry(10, 7); drawn = {}
    for ln in M[:k]:
        p = b.player; before = list(b.owner) if hasattr(b, "owner") else None
        b.step(ln); drawn[ln] = p
    for i, o in enumerate(b.owner):
        if o in (0, 1):
            x, y = i % 10, i // 10
            d.rectangle((X0 + x * S + 6, Y0 + y * S + 6, X0 + (x + 1) * S - 6, Y0 + (y + 1) * S - 6),
                        fill=tuple(int(c * .55) for c in (RED if o == 0 else BLUE)))
    for ln, p in drawn.items():
        a, e = seg(g, ln)
        last = k and ln == M[k - 1]
        d.line((X0 + a[0] * S, Y0 + a[1] * S, X0 + e[0] * S, Y0 + e[1] * S),
               fill=YELLOW if last else (RED if p == 0 else BLUE), width=10 if last else 6)
    for y in range(8):
        for x in range(11):
            d.ellipse((X0 + x * S - 7, Y0 + y * S - 7, X0 + x * S + 7, Y0 + y * S + 7), fill=FG)
    sc = [sum(1 for o in b.owner if o == s) for s in (0, 1)]
    bx = 1290
    d.text((bx, 70), "Dot-2", font=font(44), fill=RED)
    d.text((bx + 420, 70), f"{sc[0]} : {sc[1]}", font=font(56), fill=FG)
    d.text((bx, 200), f"линия {k}", font=font(36, LATO), fill=DIM)
    d.text((bx, 270), "судья", font=font(36, LATO), fill=DIM)
    if judge:
        d.text((bx, 320), judge, font=font(64), fill=(120, 205, 120) if judge == "выигрыш" else RED)
    d.text((bx, 460), "интуиция: кто выигрывает", font=font(36, LATO), fill=DIM)
    W = 520; y = 530; mid = bx + W / 2
    d.rounded_rectangle((bx, y, bx + W, y + 50), 8, fill=PANEL)
    v = V.get(k, V[max(V)])
    d.rectangle((min(mid, mid + v * W / 2), y, max(mid, mid + v * W / 2), y + 50), fill=RED if v > 0 else BLUE)
    d.line((mid, y - 8, mid, y + 58), fill=FG, width=3)
    d.text((bx, y + 66), "Hard", font=font(30, LATO), fill=DIM)
    d.text((bx + W, y + 66), "Dot-2", font=font(30, LATO), fill=DIM, anchor="ra")
    d.text((bx + W / 2, y + 120), f"{v:+.2f}".replace(".", ",").replace("-", "−"), font=font(56), fill=FG, anchor="ma")
    return img


def clip(name, seq):
    """seq: (k, judge, seconds)."""
    with tempfile.TemporaryDirectory() as t:
        lst = []
        for i, (k, j, s) in enumerate(seq):
            f = f"{t}/{i}.png"; frame(k, j, V).save(f)
            lst += [f"file '{f}'", f"duration {s}"]
        lst.append(lst[-2])
        open(f"{t}/l.txt", "w").write("\n".join(lst))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"{t}/l.txt", "-r", str(FPS),
                        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18", f"{OUT}/{name}"], check=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pv = os.path.join(OUT, "game121_v.json")
    if os.path.exists(pv):
        V = {int(k): v for k, v in json.load(open(pv)).items()}
    else:
        V = values(); json.dump(V, open(pv, "w"))
    print({k: V[k] for k in (77, 79, 80, 83, 88, 98) if k in V})
    seq = [(MISTAKE, "выигрыш", 7.0), (MISTAKE + 1, "проигрыш", 4.0)]
    seq += [(k, "проигрыш", 1.0) for k in range(MISTAKE + 2, 100)]
    seq += [(k, "проигрыш", 0.12) for k in range(100, len(M) + 1)]
    clip("game121_ru.mp4", seq)
    clip("game121_end.mp4", [(k, None, 0.12) for k in range(70, len(M) + 1)] + [(len(M), None, 6.0)])
