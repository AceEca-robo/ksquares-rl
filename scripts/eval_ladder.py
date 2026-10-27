"""Round robin of agents in the simulator; prints markdown tables.

Every pair plays ``games`` games, the first mover alternating. Games run in
a process pool. Per-move times are measured per agent (note: the strong
agent searches against a wall-clock limit, so its depth depends on the
machine load and results are not bit-reproducible).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import time
from multiprocessing import Pool

from _common import ROOT, load_config

from agents.registry import make_agent
from sim.board import Board

_CFG = None


def _init(cfg):
    global _CFG
    _CFG = cfg


def play_game(task):
    """task = (a, b, game index, seed_a, seed_b). Returns per-agent results."""
    a, b, g, seed_a, seed_b = task
    w, h = _CFG["board"]
    params = _CFG.get("params") or {}
    agents = [make_agent(a, seed_a, **params.get(a, {})), make_agent(b, seed_b, **params.get(b, {}))]
    first = g % 2  # agents[first] moves first
    board = Board(w, h)
    seat = [first, 1 - first]  # seat[player] = agent index
    times = [0.0, 0.0]
    moves = [0, 0]
    tmax = [0.0, 0.0]
    while not board.done:
        k = seat[board.player]
        t0 = time.perf_counter()
        ln = agents[k].act(board)
        dt = time.perf_counter() - t0
        times[k] += dt
        moves[k] += 1
        tmax[k] = max(tmax[k], dt)
        board.step(ln)
    score = [board.score[seat.index(0)], board.score[seat.index(1)]]
    return a, b, first, score, times, moves, tmax


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/eval_ladder.yaml"))
    cfg = load_config(ap.parse_args().config)
    names = cfg["agents"]
    pairs = [tuple(p) for p in cfg.get("pairs") or itertools.combinations(names, 2)]
    rng = random.Random(cfg["seed"])
    tasks = [(a, b, g, rng.randrange(2**32), rng.randrange(2**32))
             for a, b in pairs for g in range(cfg["games"])]
    procs = cfg.get("processes") or max(1, (os.cpu_count() or 2) - 2)
    res = {}
    speed = {}  # agent -> [time, moves, max]
    t0 = time.time()
    with Pool(procs, initializer=_init, initargs=(cfg,)) as pool:
        for a, b, first, score, times, moves, tmax in pool.imap_unordered(play_game, tasks, chunksize=1):
            r = res.setdefault((a, b), {"w": 0, "l": 0, "t": 0, "margin": 0, "first_w": 0, "n": 0})
            r["n"] += 1
            r["margin"] += score[0] - score[1]
            if score[0] == score[1]:
                r["t"] += 1
            else:
                win = 0 if score[0] > score[1] else 1
                r["w" if win == 0 else "l"] += 1
                r["first_w"] += win == first
            for k, name in enumerate((a, b)):
                sp = speed.setdefault(name, [0.0, 0, 0.0])
                sp[0] += times[k]
                sp[1] += moves[k]
                sp[2] = max(sp[2], tmax[k])
    wall = time.time() - t0
    w, h = cfg["board"]
    if cfg.get("json"):  # wins per ordered pair, the Elo anchors of scripts/az_eval.py
        (ROOT / cfg["json"]).write_text(json.dumps({
            "board": [w, h], "games": cfg["games"], "seed": cfg["seed"], "params": cfg.get("params") or {},
            "pairs": [[a, b, res[(a, b)]["w"], res[(a, b)]["l"], res[(a, b)]["t"]] for a, b in pairs]}, indent=1))
    print(f"Board {w}x{h}, {cfg['games']} games per pair, seed {cfg['seed']}, {procs} processes, "
          f"wall {wall / 60:.1f} min.\n")
    print("| A vs B | games | A wins | B wins | ties | A win % (±95%) | mean margin A−B | first-mover wins % |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for a, b in pairs:
        r = res[(a, b)]
        n = r["n"]
        p = r["w"] / n
        ci = 1.96 * math.sqrt(p * (1 - p) / n) * 100
        decided = n - r["t"]
        fm = 100 * r["first_w"] / decided if decided else float("nan")
        print(f"| {a} vs {b} | {n} | {r['w']} | {r['l']} | {r['t']} | {100 * p:.1f} ± {ci:.1f} | "
              f"{r['margin'] / n:+.2f} | {fm:.1f} |")
    print("\nWin % of the row agent against the column agent (ties count half):\n")
    print("| | " + " | ".join(names) + " |")
    print("|---|" + "---:|" * len(names))
    for x in names:
        cells = []
        for y in names:
            if (x, y) in res:
                r = res[(x, y)]
                cells.append(f"{100 * (r['w'] + r['t'] / 2) / r['n']:.1f}")
            elif (y, x) in res:
                r = res[(y, x)]
                cells.append(f"{100 * (r['l'] + r['t'] / 2) / r['n']:.1f}")
            else:
                cells.append("—")
        print(f"| **{x}** | " + " | ".join(cells) + " |")
    print("\n| agent | moves timed | mean ms/move | max ms/move |")
    print("|---|---:|---:|---:|")
    for name in names:
        t, m, mx = speed[name]
        print(f"| {name} | {m} | {1000 * t / m:.2f} | {1000 * mx:.1f} |")


if __name__ == "__main__":
    main()
