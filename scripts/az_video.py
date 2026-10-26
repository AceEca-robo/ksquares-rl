"""Record a game AlphaZero net + MCTS vs KSquares Hard in the real game.

The game runs over the RL bridge with a visible window on the X display
(``DISPLAY``, default :0); Hard is the game's own computer player, with
``KSQUARES_RL_AI_DELAY_MS`` so its moves can be followed. The window is
placed and raised with xdotool and recorded with ffmpeg (x11grab), then
scaled to ``--width`` pixels into an H.264 mp4.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time

import numpy as np
import torch

from _common import ROOT, load_config

from agents.az_mcts import choose_move
from agents.az_net import Evaluator, make_net
from agents.az_play import make_mcts
from env.bridge import KSquaresBridge


def sh(*args) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/az5b.yaml"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--net-first", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--move-delay", type=float, default=0.9, help="seconds before each net move")
    ap.add_argument("--ai-delay-ms", type=int, default=900)
    ap.add_argument("--display", default=os.environ.get("DISPLAY", ":0"))
    ap.add_argument("--size", default="560x640", help="window size")
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--out", default=str(ROOT / "docs/az5_vs_hard.mp4"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    w, h = cfg["board"]
    net = make_net(cfg)
    net.load_state_dict(torch.load(ROOT / a.checkpoint, map_location="cpu"))
    ev = Evaluator(net, cfg["device"])
    rng = np.random.default_rng(a.seed)
    mcts = make_mcts(ev, cfg["mcts"], rng, leaves_per_tree=cfg["mcts"]["leaves_per_tree"])

    env = {"DISPLAY": a.display, "QT_QPA_PLATFORM": "xcb", "KSQUARES_RL_AI_DELAY_MS": str(a.ai_delay_ms)}
    if os.environ.get("XAUTHORITY"):
        env["XAUTHORITY"] = os.environ["XAUTHORITY"]
    xenv = dict(os.environ, DISPLAY=a.display)
    net_seat = 0 if a.net_first else 1
    with KSquaresBridge(env=env) as br:
        pid = br._res["proc"].pid
        ai = [True, True]
        ai[net_seat] = False
        st = br.reset(w, h, ai=ai, difficulty=2, seed=a.seed)  # 2 = Hard
        win = subprocess.run(["xdotool", "search", "--sync", "--onlyvisible", "--pid", str(pid)],
                             check=True, capture_output=True, text=True, env=xenv).stdout.split()[-1]
        ww, wh = a.size.split("x")
        for cmd in (["windowsize", win, ww, wh], ["windowmove", win, "60", "60"]):
            subprocess.run(["xdotool", *cmd], check=True, env=xenv)
        # raising needs a window manager; a bare Xvfb has none and needs no raising
        subprocess.run(["xdotool", "windowactivate", win], env=xenv, stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        info = subprocess.run(["xwininfo", "-id", win], check=True, capture_output=True, text=True,
                              env=xenv).stdout
        geo = {k.strip(): v.strip() for k, v in (ln.split(":", 1) for ln in info.splitlines() if ":" in ln)}
        x, y = int(geo["Absolute upper-left X"]), int(geo["Absolute upper-left Y"])
        gw, gh = int(geo["Width"]), int(geo["Height"])
        gw, gh = gw - gw % 2, gh - gh % 2
        raw = tempfile.mktemp(suffix=".mkv")
        rec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "x11grab", "-draw_mouse", "0",
                                "-framerate", "25", "-video_size", f"{gw}x{gh}", "-i", f"{a.display}+{x},{y}",
                                "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", raw],
                               stdin=subprocess.PIPE, env=xenv)
        time.sleep(2.0)
        b = br.board
        net_lines = 0
        while not b.done:
            assert b.player == net_seat, "bridge returned while the AI is to move"
            root = mcts.search([b], a.sims)[0]
            if root.terminal:  # winner already fixed: the search has nothing to say
                ln = int(np.argmax(ev([b])[0][0]))
            else:
                ln = choose_move(root, 0.0, rng)
            time.sleep(a.move_delay)
            st = br.step(ln)
            net_lines += 1
            b = br.board
        time.sleep(3.0)
        rec.communicate(b"q")
        score = st["scores"]
    print(f"net (player {net_seat}) {score[net_seat]} : Hard {score[1 - net_seat]}, {net_lines} net lines")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", raw, "-vf", f"scale={a.width}:-2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", a.out],
                   check=True)
    os.unlink(raw)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
