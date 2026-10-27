#!/usr/bin/env python3
"""Self-check after install.sh: is everything the app needs in place?

    .venv/bin/python scripts/selfcheck.py

Checks, as the app uses them: the KSquares binary with the RL bridge
(build/src/ksquares); a private Xvfb; KSquares started on it over the
bridge, a whole 5x5 game against its Hard AI, its window found with xdotool;
torch and whether it sees a CUDA GPU; pygame with a font that has Cyrillic
letters; ffmpeg (recording). Prints a short report; exit status 1 if
anything the app cannot do without failed (no GPU is only a warning).
"""

from __future__ import annotations

import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from launcher import core  # noqa: E402

rows = []   # (status OK / WARN / FAIL, what, detail)


def check(what, fn, required=True):
    t0 = time.time()
    try:
        status, detail = fn()
    except Exception as e:   # the report says what broke, the next checks still run
        status, detail = ("FAIL" if required else "WARN"), f"{type(e).__name__}: {e}"
    rows.append((status, what, f"{detail} ({time.time() - t0:.1f} s)" if status == "OK" else detail))


def game_binary():
    if not os.access(core.GAME, os.X_OK):
        return "FAIL", f"{os.path.relpath(core.GAME, core.ROOT)} is missing: the build failed"
    return "OK", os.path.relpath(core.GAME, core.ROOT)


def xvfb():
    x = core.PrivateX()
    x.close()
    return "OK", f"started on {x.name}"


def ksquares_game():
    from env.bridge import KSquaresBridge

    x = core.PrivateX()
    bridge = None
    try:
        env = {"DISPLAY": x.name, "QT_QPA_PLATFORM": "xcb", "KSQUARES_RL_AI_DELAY_MS": "0"}
        bridge = KSquaresBridge(core.GAME, env=env)
        bridge.reset(5, 5, ai=(False, True), difficulty=2, seed=1)
        rng = random.Random(1)
        board = bridge.board
        while not board.done:
            bridge.step(rng.choice(board.legal_moves()))
        box = core.find_window(x.name, bridge._res["proc"].pid, (900, 900))
        return "OK", f"a 5x5 game against Hard over the bridge, {board.score[0]}:{board.score[1]}; window {box}"
    finally:
        if bridge is not None:
            bridge.close()
        x.close()


def torch_gpu():
    import torch

    smi = shutil.which("nvidia-smi")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return "OK", f"torch {torch.__version__}, GPU {p.name}, {p.total_memory / 2**30:.1f} GB"
    if smi:
        return "WARN", (f"torch {torch.__version__} does not see the GPU although nvidia-smi is there: "
                        "the driver may be too old for this torch build (see nvidia-smi)")
    return "WARN", f"torch {torch.__version__}, no GPU: the app plays fine, training is ~10x slower"


def pygame_font():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    pygame.font.init()
    path = pygame.font.match_font("dejavusans")
    if not path:
        return "FAIL", "no DejaVu Sans font (fonts-dejavu-core): Russian and Ukrainian text would not show"
    pygame.font.Font(path, 20).render("Точки и квадраты", True, (255, 255, 255))
    return "OK", f"pygame {pygame.version.ver}, {os.path.basename(path)}"


def ffmpeg():
    if not shutil.which("ffmpeg"):
        return "WARN", "no ffmpeg: --record does not work, the rest does"
    return "OK", shutil.which("ffmpeg")


def main():
    check("KSquares built", game_binary)
    check("Xvfb", xvfb)
    if rows[0][0] == "OK":
        check("KSquares on Xvfb over the bridge", ksquares_game)
    check("torch / GPU", torch_gpu)
    check("pygame and fonts", pygame_font)
    check("ffmpeg", ffmpeg, required=False)
    print("\nSelf-check:")
    for status, what, detail in rows:
        print(f"  [{status:4}] {what}: {detail}")
    failed = [r for r in rows if r[0] == "FAIL"]
    print("\n" + ("all good" if not failed else f"{len(failed)} check(s) failed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
