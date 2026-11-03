"""Where do Dot-2's routs against Hard go wrong? (part 3 (c) check)

    python scripts/blowouts.py play [--seeds 11 200]   # dot2c gen 16 vs the real Hard, as the app plays it
    python scripts/blowouts.py judge                   # the middle-game judge from the end backwards
    python scripts/blowouts.py report                  # docs/blowouts.md

play: the app's "vs KSquares" game (launcher/session.py): the network first,
Hard (level 2), the game's srand and the MCTS rng both = the seed, 50 sims, on
the CPU (as the 2026-10-29 run of seeds 11-200: 17 losses of 190). Moves are
saved, one game per line, in configs/blowouts.yaml ``out``.

judge: for every lost game, the positions where Dot-2 is to move, from the
last one backwards; each gets ``mid_judge.outcome`` within ``time_limit`` s
(fresh table). The walk stops at the first position that is not lost for
Dot-2 (the mistake is the move played there: its label is checked with
``classify``) or at the first one out of time (the wall).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

from _common import ROOT, load_config

CFG = ROOT / "configs/blowouts.yaml"


def _play(args):
    seeds, cfg = args
    import torch

    torch.set_num_threads(cfg["threads"])
    from env.bridge import KSquaresBridge
    from launcher import brain

    out = []
    with KSquaresBridge() as br:
        for seed in seeds:
            bn = brain.Brain(str(ROOT / cfg["net"]["config"]), str(ROOT / cfg["net"]["checkpoint"]), "dot2c",
                             "cpu", seed=seed)
            moves = [m[0] for m in br.reset(10, 7, ai=[False, True], difficulty=2, seed=seed)["moves"]]
            b = br.board
            while not b.done:
                moves += [m[0] for m in br.step(bn.think(b, cfg["sims"])["move"])["moves"]]
            r = {"seed": seed, "net": int(b.score[0]), "hard": int(b.score[1]), "moves": moves}
            out.append(r)
            print(f"seed {seed}: {r['net']}:{r['hard']}", flush=True)
    return out


def cmd_play(cfg, lo, hi):
    seeds = list(range(lo, hi + 1))
    k = cfg["processes"]
    parts = [(seeds[i::k], cfg) for i in range(k)]
    path = ROOT / cfg["out"] / "games.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with get_context("spawn").Pool(k) as pool, open(path, "w") as fh:
        for res in pool.imap_unordered(_play, parts):
            for r in res:
                fh.write(json.dumps(r) + "\n")


def _judge(args):
    game, cfg = args
    sys.setrecursionlimit(10000)
    from agents.baselines import safe_moves
    from sim import mid_judge as mj
    from sim.board import Board

    j = mj.Judge(10, 7, tt_bits=cfg["tt_bits"])
    moves = game["moves"]
    b = Board(10, 7)
    pos = []  # (index of the move, safe lines) where Dot-2 (seat 0) moves
    for k, ln in enumerate(moves):
        if b.player == 0:
            pos.append((k, len(safe_moves(b))))
        b.step(ln)
    r = {"seed": game["seed"], "net": game["net"], "hard": game["hard"], "walk": []}
    for k, s in reversed(pos):
        b = Board(10, 7)
        for ln in moves[:k]:
            b.step(ln)
        j.clear()
        t0 = time.perf_counter()
        try:
            o = mj.outcome(j, b, cfg["time_limit"])
        except mj.Timeout:
            r["walk"].append({"k": k, "s": s, "outcome": None})
            r.update(status="wall", k=k, s=s)
            return r
        r["walk"].append({"k": k, "s": s, "outcome": o, "t": round(time.perf_counter() - t0, 2)})
        if o >= 0:   # not lost here: the move played here lost it
            j.clear()
            try:
                lab = mj.classify(j, b, cfg["time_limit"] * 4)
                r["move_label"] = lab.get(moves[k])
                r["best"] = [m for m, v in lab.items() if v == max(lab.values())]
            except mj.Timeout:
                r["move_label"] = None
            r.update(status="found", k=k, s=s, line=moves[k], outcome_before=o)
            return r
    r.update(status="lost from the start")
    return r


def cmd_judge(cfg):
    games = [json.loads(x) for x in (ROOT / cfg["out"] / "games.jsonl").read_text().splitlines()]
    lost = sorted((g for g in games if g["net"] < g["hard"]), key=lambda g: g["seed"])
    print(f"{len(lost)} lost of {len(games)}", flush=True)
    with get_context("spawn").Pool(cfg["processes"]) as pool, \
            open(ROOT / cfg["out"] / "judge.jsonl", "w") as fh:
        for r in pool.imap_unordered(_judge, [(g, cfg) for g in lost]):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"seed {r['seed']}: {r['status']} at move {r.get('k')} (s {r.get('s')})", flush=True)


def cmd_report(cfg):
    games = [json.loads(x) for x in (ROOT / cfg["out"] / "games.jsonl").read_text().splitlines()]
    rs = sorted((json.loads(x) for x in (ROOT / cfg["out"] / "judge.jsonl").read_text().splitlines()),
                key=lambda r: r["seed"])
    lost = sum(g["net"] < g["hard"] for g in games)
    sign = {1: "win", 0: "draw", -1: "loss", None: "?"}
    L = ["# Dot-2's routs against Hard: where the judge finds the mistake", "",
         f"Generated by `scripts/blowouts.py report` ({time.strftime('%Y-%m-%d %H:%M')}), configs/blowouts.yaml.",
         f"Games: dot2c gen 16 (MCTS-{cfg['sims']}, CPU, first) vs the real KSquares Hard over the bridge, "
         f"seeds {min(g['seed'] for g in games)}-{max(g['seed'] for g in games)} as the app plays them: "
         f"{lost} lost of {len(games)}.", "",
         f"Judge: sim/mid_judge.py, {cfg['time_limit']} s per position, positions where Dot-2 is to move from "
         "the end backwards. *Found*: the first position (going back) that is not lost for Dot-2; the line "
         "played there is the mistake (its own label from `classify` in brackets). *Wall*: the judge ran out "
         "of time on a position that was still lost, so the mistake is earlier than it can see.", "",
         "| seed | score (Dot-2 : Hard) | lines drawn | safe lines | judge | before the move | the move | "
         "positions judged |",
         "|---:|---:|---:|---:|---|---|---|---:|"]
    for r in rs:
        if r["status"] == "found":
            mv = f"line {r['line']} ({sign.get(r.get('move_label'), '?')})"
            L.append(f"| {r['seed']} | {r['net']}:{r['hard']} | {r['k']} | {r['s']} | found | "
                     f"{sign[r['outcome_before']]} | {mv} | {len(r['walk'])} |")
        else:
            L.append(f"| {r['seed']} | {r['net']}:{r['hard']} | {r.get('k', '—')} | {r.get('s', '—')} | "
                     f"{r['status']} | — | — | {len(r['walk'])} |")
    L.append("")
    concl = ROOT / "docs/blowouts_conclusions.md"
    if concl.exists():
        L += [concl.read_text().rstrip(), ""]
    (ROOT / "docs/blowouts.md").write_text("\n".join(L))
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["play", "judge", "report"])
    ap.add_argument("--config", default=str(CFG))
    ap.add_argument("--seeds", type=int, nargs=2, default=[11, 200])
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.cmd == "play":
        cmd_play(cfg, *a.seeds)
    elif a.cmd == "judge":
        cmd_judge(cfg)
    else:
        cmd_report(cfg)


if __name__ == "__main__":
    main()
