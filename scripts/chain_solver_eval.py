"""The formula judge (sim/chain_solver.py) against the exact judge.

    python scripts/chain_solver_eval.py agree    # late judge positions (5x5, 10x7): agreement
    python scripts/chain_solver_eval.py mid      # 10x7 middle-game positions: does it answer at all
    python scripts/chain_solver_eval.py timing   # whole games: from how many safe lines on each answers
    python scripts/chain_solver_eval.py report   # docs/chain_solver.md

configs/chain_solver.yaml. "Answers" = the position is settled (no box to
take, the safe phase cannot change length, the board after the safe lines
is cut into arms, junctions and loops) and the recursion stays within the
budget. The exact judge answers when agents/strong.py exact_move_values
solves the position within the time limit.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from multiprocessing import get_context

import numpy as np

from _common import ROOT, load_config

import transfer

_G = {}


def _init(cfg):
    sys.setrecursionlimit(10000)
    _G["cfg"] = cfg


def sign(x):
    return (x > 0) - (x < 0)


def formula(board, budget):
    """{'why': reason or None, 'v', 'move', 'kind'}."""
    from sim import chain_solver as cs

    r = cs.best_move(board, budget)
    if r is None:
        return {"why": cs.why_not(board) if not cs.capturable(board) else "capture: not settled after"}
    return {"why": None, "v": r[1], "move": r[0], "kind": r[2]}


def compare(board, f, vals):
    """Formula answer vs exact move values (net future boxes per line)."""
    best = max(vals.values())
    me = board.score[board.player] - board.score[board.player ^ 1]
    res = {ln: sign(me + v) for ln, v in vals.items()}
    rbest = max(res.values())
    return {"value": f["v"] == best, "sign": sign(me + f["v"]) == rbest,
            "move": res[f["move"]] == rbest, "move_value": vals[f["move"]] == best}


# --- agreement on the judge sets -----------------------------------------------------

def _agree_one(task):
    from agents.baselines import safe_moves
    from sim import chain_solver as cs
    from sim.board import Board

    bname, p = task
    b = Board(*transfer.wh_of(bname))
    for ln in p["moves"]:
        b.step(ln)
    f = formula(b, _G["cfg"]["budget"])
    r = {"safe": len(safe_moves(b)), "why": f["why"]}
    if f["why"] is None:
        vals = {int(k): v for k, v in p["values"].items()}
        r.update(compare(b, f, vals), kind=f["kind"])
        g = cs.ply_move(b, _G["cfg"]["budget"])
        r["ply"] = None if g is None else compare(b, {"v": g[1], "move": g[0]}, vals)
    return r


def cmd_agree(cfg):
    out = ROOT / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for bname, path in cfg["agree"]["sets"].items():
        ps = []
        for line in (ROOT / path).read_text().splitlines():
            ps += json.loads(line)["found"]
        uniq = {}
        for p in ps:
            uniq.setdefault((tuple(sorted(p["moves"])), tuple(p["score"]), p["player"]), p)
        ps = list(uniq.values())
        random.Random(cfg["seed"]).shuffle(ps)
        ps = ps[:cfg["agree"]["n"]]
        t0 = time.time()
        with get_context("spawn").Pool(cfg["processes"], initializer=_init, initargs=(cfg,)) as pool:
            res[bname] = pool.map(_agree_one, [(bname, p) for p in ps], chunksize=8)
        print(f"{bname}: {len(ps)} positions, {time.time() - t0:.0f} s, answered "
              f"{sum(r['why'] is None for r in res[bname])}", flush=True)
    transfer.save(out / "agree.json", res)


# --- middle game coverage ----------------------------------------------------------

def _mid_one(task):
    from agents.baselines import safe_moves
    from parity_arch import board_from_state

    b = board_from_state(10, 7, *task)
    f = formula(b, _G["cfg"]["budget"])
    return {"safe": len(safe_moves(b)), "why": f["why"], "v": f.get("v")}


def cmd_mid(cfg):
    out = ROOT / cfg["out"]
    md = np.load(ROOT / cfg["agree"]["mid"])
    tasks = [(md["lines"][i], md["sides"][i], md["owner"][i], int(md["diff"][i]), bool(md["first"][i]))
             for i in range(len(md["lines"]))]
    with get_context("spawn").Pool(cfg["processes"], initializer=_init, initargs=(cfg,)) as pool:
        res = pool.map(_mid_one, tasks, chunksize=16)
    print(f"mid: {len(res)} positions, answered {sum(r['why'] is None for r in res)}", flush=True)
    transfer.save(out / "mid.json", res)


# --- whole games ---------------------------------------------------------------------

def _game(task):
    from agents.az_mcts import result_for_mover
    from agents.baselines import safe_moves
    from agents.registry import make_agent
    from agents.strong import Engine, exact_move_values
    from sim import chain_solver as cs
    from sim.board import Board

    g, wh, a, b_, seed, first, bc = task
    cfg = _G["cfg"]
    agents = [make_agent(a, seed), make_agent(b_, seed + 1)]
    seat = [first, 1 - first]
    board = Board(*wh)
    eng = Engine(*wh)
    rows = []
    while result_for_mover(board) is None:
        s = len(safe_moves(board))
        cap = bool(cs.capturable(board))
        f = formula(board, cfg["budget"])
        row = {"n": board.n_drawn, "safe": s, "cap": cap, "f": f["why"] is None, "why": f["why"]}
        if s <= bc["exact_max_safe"]:
            if len(eng.tt) > 1_500_000:
                eng.tt.clear()
            vals = exact_move_values(eng, board, None, bc["time_limit"])
            row["e"] = vals is not None
            if vals is not None and f["why"] is None:
                row.update(compare(board, f, vals))
        else:
            row["e"] = False
        rows.append(row)
        board.step(agents[seat[board.player]].act(board))
    return {"game": g, "pair": [a, b_], "rows": rows}


def cmd_timing(cfg, sizes):
    out = ROOT / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    tc = cfg["timing"]
    for bname in sizes or tc["boards"]:
        bc = tc["boards"][bname]
        path = out / f"timing_{bname}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(x)["game"] for x in path.read_text().splitlines() if x.strip()}
        rng = random.Random(f"{cfg['seed']}/{bname}")
        seeds = [rng.randrange(2**31) for _ in range(bc["games"])]
        tasks = [(g, transfer.wh_of(bname), *tc["pairs"][g % len(tc["pairs"])], seeds[g], g % 2, bc)
                 for g in range(bc["games"]) if g not in done]
        t0 = time.time()
        with get_context("spawn").Pool(cfg["processes"], initializer=_init, initargs=(cfg,)) as pool, \
                open(path, "a") as fh:
            for i, r in enumerate(pool.imap_unordered(_game, tasks)):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                if (i + 1) % 20 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] {bname}: {len(done) + i + 1} games, "
                          f"{time.time() - t0:.0f} s", flush=True)


# --- report --------------------------------------------------------------------------

def pct(k, n):
    return f"{100 * k / n:.1f}" if n else "—"


def cmd_report(cfg):
    from parity_probe import wilson

    out = ROOT / cfg["out"]

    def ci(k, n):
        if not n:
            return "—"
        lo, hi = wilson(k, n)
        return f"{100 * k / n:.1f} [{lo:.0f}-{hi:.0f}]"

    L = ["# The formula judge: chain counting instead of search", "",
         f"Generated by `scripts/chain_solver_eval.py report` ({time.strftime('%Y-%m-%d %H:%M')}), "
         "configs/chain_solver.yaml. Judge: sim/chain_solver.py (Berlekamp's recursion over arms, junctions "
         "and loops with the safe lines as tempo moves; see its docstring). It answers when the position is "
         "*settled*: no box to take, every box has drawn sides + safe lines on it <= 2 (the safe phase has a "
         "fixed length), and the board after the safe lines cuts into arms (runs of boxes with two drawn sides) "
         "between junctions (boxes with at most one drawn side) or the ground, and loops; within a budget of "
         f"{cfg['budget']} recursion states. Exact judge: agents/strong.py exact_move_values.", ""]
    agree = json.loads((out / "agree.json").read_text()) if (out / "agree.json").exists() else {}
    if agree:
        L += ["## Agreement on late positions where the exact judge knows the answer", "",
              "Part 3 (a) judge sets (runs/parity_hand/data_*.jsonl): no box to take, 1-6 (5x5) / 1-2 (10x7) "
              "safe lines, and the move decides the result (only some moves win / draw), so the sets are "
              "the hard cases of the long chain rule. Value: the formula's net boxes = the exact one. "
              "Result: same win / draw / loss for the mover. Move: the formula's move keeps the best result "
              "(the probe's criterion); in brackets: keeps the exact value.", "",
              "| board | n | answers % | value % | result % | move % (exact value %) | ply: answers % | ply: value % | ply: move % | no answer: why |",
              "|---|---:|---|---|---|---|---:|---:|---:|---|"]
        for b, rs in agree.items():
            a = [r for r in rs if r["why"] is None]
            why = Counter(r["why"] for r in rs if r["why"] is not None)
            L.append(f"| {b} | {len(rs)} | {ci(len(a), len(rs))} | {ci(sum(r['value'] for r in a), len(a))} | "
                     f"{ci(sum(r['sign'] for r in a), len(a))} | {ci(sum(r['move'] for r in a), len(a))} "
                     f"({pct(sum(r['move_value'] for r in a), len(a))}) | "
                     f"{pct(len(pl := [r['ply'] for r in a if r.get('ply')]), len(rs))} | "
                     f"{pct(sum(x['value'] for x in pl), len(pl))} | {pct(sum(x['move'] for x in pl), len(pl))} | "
                     + ", ".join(f"{k} {v}" for k, v in why.most_common()) + " |")
        L += ["", "By safe lines left (answered positions):", "",
              "| board | safe lines | n | answers % | value % | result % | move % |", "|---|---:|---:|---:|---:|---:|---:|"]
        for b, rs in agree.items():
            for s in sorted({r["safe"] for r in rs}):
                rr = [r for r in rs if r["safe"] == s]
                a = [r for r in rr if r["why"] is None]
                L.append(f"| {b} | {s} | {len(rr)} | {pct(len(a), len(rr))} | {pct(sum(r['value'] for r in a), len(a))} | "
                         f"{pct(sum(r['sign'] for r in a), len(a))} | {pct(sum(r['move'] for r in a), len(a))} |")
        L.append("")
    if (out / "mid.json").exists():
        rs = json.loads((out / "mid.json").read_text())
        L += ["## The middle game of 10x7 (Dot-2 self-play positions)", "",
              f"{cfg['agree']['mid']}: the part 3 (b) middle-game positions (20-40 safe lines, dot2c gen 16 "
              "self-play). No exact answer exists here; only whether the formula answers.", "",
              "| safe lines | n | answers % | no answer: why |", "|---|---:|---:|---|"]
        for lo, hi in ((20, 24), (25, 29), (30, 34), (35, 40)):
            rr = [r for r in rs if lo <= r["safe"] <= hi]
            why = Counter(r["why"] for r in rr if r["why"] is not None)
            L.append(f"| {lo}-{hi} | {len(rr)} | {pct(sum(r['why'] is None for r in rr), len(rr))} | "
                     + ", ".join(f"{k} {v}" for k, v in why.most_common()) + " |")
        L.append("")
    tc = cfg["timing"]
    for bname, bc in tc["boards"].items():
        path = out / f"timing_{bname}.jsonl"
        if not path.exists():
            continue
        games = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        rows = [r for g in games for r in g["rows"] if not r["cap"]]
        L += [f"## Whole games, {bname}: when does each judge answer", "",
              f"{len(games)} games of the ladder agents ({', '.join('-'.join(p) for p in tc['pairs'])}), every "
              f"position without a box to take. Exact: solved within {bc['time_limit']} s (tried with up to "
              f"{bc['exact_max_safe']} safe lines; with more it counts as no answer). Agreement where both answer.", "",
              "| safe lines | positions | formula answers % | exact answers % | both | value % | result % | move % |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for lo, hi in tc["buckets"]:
            rr = [r for r in rows if lo <= r["safe"] <= hi]
            both = [r for r in rr if r["f"] and r["e"]]
            lab = f"{lo}" if lo == hi else (f"{lo}+" if hi >= 999 else f"{lo}-{hi}")
            L.append(f"| {lab} | {len(rr)} | {pct(sum(r['f'] for r in rr), len(rr))} | "
                     f"{pct(sum(r['e'] for r in rr), len(rr))} | {len(both)} | "
                     f"{pct(sum(r['value'] for r in both), len(both))} | {pct(sum(r['sign'] for r in both), len(both))} | "
                     f"{pct(sum(r['move'] for r in both), len(both))} |")
        caps = [r for g in games for r in g["rows"] if r["cap"] and r.get("e") and r["f"]]
        if caps:
            L += ["", f"Positions with boxes to take (the double-cross decision), both answer: {len(caps)}, "
                  f"value {pct(sum(r['value'] for r in caps), len(caps))}%, "
                  f"move keeps the result {pct(sum(r['move'] for r in caps), len(caps))}%."]

        def first(g, key):
            """Most safe lines at which the judge answers, and from which on
            it answers at every later position (no box to take)."""
            rs = [r for r in g["rows"] if not r["cap"]]
            any_ = max((r["safe"] for r in rs if r[key]), default=None)
            steady = None
            for r in reversed(rs):
                if not r[key]:
                    break
                steady = r["safe"]
            return any_, steady

        fa = [first(g, "f") for g in games]
        ea = [first(g, "e") for g in games]

        def med(xs):
            xs = [x for x in xs if x is not None]
            return f"{np.median(xs):.0f}" if xs else "—"

        L += ["", "Per game: the most safe lines left at which the judge answers at all, and from which on it "
              "answers at every position to the end (medians over games; games where it never answers left out).", "",
              "| judge | first answer at (safe lines) | answers steadily from | games with an answer at 20+ safe lines |",
              "|---|---:|---:|---:|",
              f"| formula | {med([a for a, _ in fa])} | {med([s for _, s in fa])} | "
              f"{sum((a or 0) >= 20 for a, _ in fa)} / {len(games)} |",
              f"| exact | {med([a for a, _ in ea])} | {med([s for _, s in ea])} | "
              f"{sum((a or 0) >= 20 for a, _ in ea)} / {len(games)} |", ""]
        why = Counter(r["why"] for r in rows if not r["f"] and r["safe"] >= 20)
        if why:
            L += ["Why no formula answer at 20+ safe lines: " + ", ".join(f"{k} {v}" for k, v in why.most_common())
                  + ".", ""]
    concl = ROOT / "docs/chain_solver_conclusions.md"
    if concl.exists():
        L += [concl.read_text().rstrip(), ""]
    (ROOT / "docs/chain_solver.md").write_text("\n".join(L))
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["agree", "mid", "timing", "report"])
    ap.add_argument("--config", default=str(ROOT / "configs/chain_solver.yaml"))
    ap.add_argument("--sizes", nargs="+")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.cmd == "timing":
        cmd_timing(cfg, a.sizes)
    else:
        {"agree": cmd_agree, "mid": cmd_mid, "report": cmd_report}[a.cmd](cfg)


if __name__ == "__main__":
    main()
