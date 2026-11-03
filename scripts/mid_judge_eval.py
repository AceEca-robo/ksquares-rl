"""Part 3 (c): the middle-game judge (sim/mid_judge.py) on 10x7.

    python scripts/mid_judge_eval.py games          # Dot-2 self-play games (move lists)
    python scripts/mid_judge_eval.py check          # sign agreement on the late judge set
    python scripts/mid_judge_eval.py reach --tier 10 # solved within 10 s / 60 s / 600 s by safe lines left
    python scripts/mid_judge_eval.py label          # judge labels for positions of the reached range
    python scripts/mid_judge_eval.py nets           # hand, intuition, MCTS-200 / 2000 on labelled positions
    python scripts/mid_judge_eval.py others         # formula and exact judges on the chart positions
    python scripts/mid_judge_eval.py report         # docs/mid_judge.md, docs/mid_judge.png

configs/mid_judge.yaml. A position is *solved* when every legal line has its
result sign for the mover (win / draw / loss, ``mid_judge.classify``). Stages
run one at a time; ``processes`` judge workers at most.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from _common import ROOT, load_config

import transfer

_G = {}


def out_dir(cfg) -> Path:
    p = ROOT / cfg["out"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def board_of(moves, wh=(10, 7)):
    from sim.board import Board

    b = Board(*wh)
    for ln in moves:
        b.step(ln)
    return b


# --- games ---------------------------------------------------------------------------

def cmd_games(cfg, n=None, name="games"):
    import parity_hand
    from agents.az_mcts import choose_move, result_for_mover
    from agents.az_play import make_mcts
    from sim.board import Board

    gc = cfg["games"]
    n = n or gc["n"]
    ev, rc = parity_hand.evaluator({"device": cfg["device"]}, (cfg["net"]["config"], ROOT / cfg["net"]["checkpoint"]))
    rng = np.random.default_rng(cfg["seed"] + {"games": 0, "games2": 1}.get(name, sum(map(ord, name))))
    path = out_dir(cfg) / f"{name}.jsonl"
    t0 = time.time()
    with open(path, "w") as fh:
        for start in range(0, n, gc["concurrent"]):
            boards = [Board(10, 7) for _ in range(min(gc["concurrent"], n - start))]
            moves = [[] for _ in boards]
            live = list(range(len(boards)))
            mcts = make_mcts(ev, rc["mcts"], rng, leaves_per_tree=1)
            while live:
                roots = mcts.search([boards[i] for i in live], gc["sims"])
                for i, r in zip(live, roots):
                    t = 1.0 if len(moves[i]) < gc["temp_moves"] else 0.0
                    ln = choose_move(r, t, rng)
                    boards[i].step(ln)
                    moves[i].append(int(ln))
                live = [i for i in live if result_for_mover(boards[i]) is None
                        and boards[i].n_drawn < boards[i].geo.n_lines]
            for m in moves:
                fh.write(json.dumps(m) + "\n")
            print(f"[{time.strftime('%H:%M:%S')}] {start + len(boards)} games, {time.time() - t0:.0f} s", flush=True)


def load_games(cfg, name="games"):
    return [json.loads(x) for x in (out_dir(cfg) / f"{name}.jsonl").read_text().splitlines() if x.strip()]


def positions_by_safe(games, s_values):
    """{s: [(game, move prefix)]}: per game the first position without a box
    to take with exactly s safe lines."""
    from agents.baselines import safe_moves
    from sim import chain_solver as cs
    from sim.board import Board

    out = {s: [] for s in s_values}
    for g, moves in enumerate(games):
        b = Board(10, 7)
        got = set()
        for k in range(len(moves) + 1):
            if not cs.capturable(b):
                s = len(safe_moves(b))
                if s in out and s not in got:
                    got.add(s)
                    out[s].append((g, moves[:k]))
            if k < len(moves):
                b.step(moves[k])
    return out


def positions_in_range(games, lo, hi):
    """[(game, move prefix, s)]: every position without a box to take with
    lo..hi safe lines."""
    from agents.baselines import safe_moves
    from sim import chain_solver as cs
    from sim.board import Board

    out = []
    for g, moves in enumerate(games):
        b = Board(10, 7)
        for k in range(len(moves) + 1):
            if not cs.capturable(b):
                s = len(safe_moves(b))
                if lo <= s <= hi:
                    out.append((g, moves[:k], s))
            if k < len(moves):
                b.step(moves[k])
    return out


# --- judge workers -------------------------------------------------------------------

def _init(tt_bits):
    sys.setrecursionlimit(10000)
    from sim import mid_judge as mj

    _G["j"] = mj.Judge(10, 7, tt_bits=tt_bits)


def _solve(task):
    """(key, prefix, limit) -> times and labels; None where out of time."""
    from sim import mid_judge as mj

    key, prefix, limit = task
    j = _G["j"]
    j.clear()
    b = board_of(prefix)
    t0 = time.perf_counter()
    r = {"key": key, "moves": prefix}
    try:
        r["outcome"] = mj.outcome(j, b, limit)
        r["t_outcome"] = time.perf_counter() - t0
    except mj.Timeout:
        r.update(outcome=None, t_outcome=None, labels=None, t_solve=None)
        return r
    try:
        r["labels"] = mj.classify(j, b, limit - r["t_outcome"])
        r["t_solve"] = time.perf_counter() - t0
    except mj.Timeout:
        r.update(labels=None, t_solve=None)
    return r


def run_pool(cfg, tasks, tt_bits, path, log_every=20):
    t0 = time.time()
    res = []
    with get_context("spawn").Pool(cfg["processes"], initializer=_init, initargs=(tt_bits,)) as pool, \
            open(path, "a") as fh:
        for i, r in enumerate(pool.imap_unordered(_solve, tasks)):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            res.append(r)
            if (i + 1) % log_every == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(tasks)}, {time.time() - t0:.0f} s", flush=True)
    return res


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


# --- check ---------------------------------------------------------------------------

def _check_one(p):
    from sim import mid_judge as mj

    b = board_of(p["moves"])
    me = b.score[b.player] - b.score[b.player ^ 1]
    want = {int(k): (me + v > 0) - (me + v < 0) for k, v in p["values"].items()}
    _G["j"].clear()
    t0 = time.perf_counter()
    got = mj.classify(_G["j"], b)
    return {"ok": got == want, "best_ok": max(got.values()) == max(want.values()),
            "t": time.perf_counter() - t0}


def cmd_check(cfg):
    ps = [p for x in (ROOT / cfg["check"]["set"]).read_text().splitlines() for p in json.loads(x)["found"]]
    uniq = {}
    for p in ps:
        uniq.setdefault((tuple(sorted(p["moves"])), tuple(p["score"]), p["player"]), p)
    ps = list(uniq.values())
    random.Random(cfg["seed"]).shuffle(ps)
    ps = ps[:cfg["check"]["n"]]
    with get_context("spawn").Pool(cfg["processes"], initializer=_init, initargs=(22,)) as pool:
        res = pool.map(_check_one, ps, chunksize=8)
    r = {"n": len(res), "all_moves": sum(x["ok"] for x in res), "outcome": sum(x["best_ok"] for x in res),
         "median_s": float(np.median([x["t"] for x in res])), "max_s": float(max(x["t"] for x in res))}
    transfer.save(out_dir(cfg) / "check.json", r)
    print(r, flush=True)


# --- reach ---------------------------------------------------------------------------

def cmd_reach(cfg, tier):
    rc = cfg["reach"]
    tc = rc["tiers"][str(tier)] if str(tier) in rc["tiers"] else rc["tiers"][tier]
    pos = positions_by_safe(load_games(cfg), rc["safe"])
    path = out_dir(cfg) / f"reach_{tier}.jsonl"
    done = {tuple(r["key"]) for r in read_jsonl(path)}
    # buckets: up to the first one where the previous tier solved nothing
    prev = {t: read_jsonl(out_dir(cfg) / f"reach_{t}.jsonl") for t in rc["tiers"] if int(t) < int(tier)}
    buckets = list(rc["safe"])
    if prev:
        pt = max(prev, key=int)
        keep = []
        for s in buckets:
            keep.append(s)
            rs = [r for r in prev[pt] if r["key"][0] == s]
            if rs and not any(r["labels"] is not None for r in rs):
                break
        buckets = keep
    tasks = [((s, g), pre, float(tier)) for s in buckets for g, pre in pos[s][:tc["n"]] if (s, g) not in done]
    print(f"tier {tier} s: {buckets}, {len(tasks)} positions", flush=True)
    run_pool(cfg, tasks, tc["tt_bits"], path, log_every=4)


# --- labels and the nets -------------------------------------------------------------

def cmd_label(cfg, name):
    lc = cfg["label"][name]
    s_vals = list(range(lc["safe"][0], lc["safe"][1] + 1))
    if lc.get("all_in_range"):  # every position without a box to take in the range, from several game sets
        tasks = []
        for gi, gname in enumerate(lc["games"]):
            for g, pre, s in positions_in_range(load_games(cfg, gname), *lc["safe"]):
                tasks.append(((s, gi * 10**5 + g, len(pre)), pre, float(lc["time_limit"])))
    else:
        pos = positions_by_safe(load_games(cfg, lc["games"]), s_vals)
        per = lc.get("per_s")
        tasks = [((s, g), pre, float(lc["time_limit"])) for s in s_vals for g, pre in (pos[s][:per] if per else pos[s])]
    random.Random(cfg["seed"]).shuffle(tasks)
    tasks = tasks[:lc.get("n", len(tasks))]
    path = out_dir(cfg) / f"labels_{name}.jsonl"
    done = {tuple(r["key"]) for r in read_jsonl(path)}
    tasks = [t for t in tasks if t[0] not in done]
    print(f"label {name}: {len(tasks)} positions, s {s_vals[0]}-{s_vals[-1]}", flush=True)
    run_pool(cfg, tasks, lc["tt_bits"], path, log_every=100)


def labelled(cfg, name):
    return [r for r in read_jsonl(out_dir(cfg) / f"labels_{name}.jsonl") if r["labels"] is not None]


def cmd_nets(cfg, name):
    """Hand (policy argmax), intuition (move with the best child V), MCTS at
    the configured simulations: their moves on the labelled positions."""
    import parity_hand
    from agents.az_mcts import choose_move, result_for_mover
    from agents.az_play import make_mcts

    rs = labelled(cfg, name)
    ev, rc = parity_hand.evaluator({"device": cfg["device"]}, (cfg["net"]["config"], ROOT / cfg["net"]["checkpoint"]))
    boards = [board_of(r["moves"]) for r in rs]
    out = {"hand": [], "intuition": []}
    for i in range(0, len(boards), 256):
        bs = boards[i:i + 256]
        pri, _ = ev(bs)
        out["hand"] += [int(np.argmax(p)) for p in pri]
        for b in bs:
            kids = [(ln, b.clone()) for ln in range(b.geo.n_lines) if not b.lines[ln]]
            for ln, c in kids:
                c.step(ln)
            v = ev([c for _, c in kids])[1]
            best, bv = None, -9.0
            for (ln, c), x in zip(kids, v):
                ex = result_for_mover(c)
                x = ex if ex is not None else float(x)
                x = x if c.player == b.player else -x
                if x > bv:
                    best, bv = ln, x
            out["intuition"].append(best)
    rng = np.random.default_rng(cfg["seed"])
    for sims in cfg["nets"]["sims"]:
        mv = []
        for i in range(0, len(boards), cfg["nets"]["trees"]):
            roots = make_mcts(ev, rc["mcts"], rng, leaves_per_tree=1).search(boards[i:i + cfg["nets"]["trees"]], sims)
            mv += [choose_move(r, 0.0, rng) for r in roots]
            print(f"[{time.strftime('%H:%M:%S')}] MCTS-{sims}: {len(mv)}/{len(boards)}", flush=True)
        out[f"mcts{sims}"] = mv
    keys = [list(r["key"]) for r in rs]
    transfer.save(out_dir(cfg) / f"nets_{name}.json", {"keys": keys, "moves": out})


def _others_one(task):
    from agents.strong import Engine, exact_move_values
    from sim import chain_solver as cs
    from sim import mid_judge as mj

    key, prefix, ec = task
    b = board_of(prefix)
    g = cs.ply_move(b, ec["formula_budget"])
    r = {"key": key, "formula": None if g is None else g[0]}
    if "eng" not in _G:
        _G["eng"] = Engine(10, 7)
    eng = _G["eng"]
    if len(eng.tt) > 1_500_000:
        eng.tt.clear()
    vals = exact_move_values(eng, b, None, ec["exact_limit"])
    r["exact"] = None if vals is None else max(vals, key=vals.get)
    # the middle-game judge as a player: the move it finds within its limit
    j = _G["j"]
    j.clear()
    try:
        lab = mj.classify(j, b, ec["mid_limit"])
        r["mid"] = max(lab, key=lab.get)
    except mj.Timeout:
        r["mid"] = None
    return r


def cmd_others(cfg, name):
    ec = cfg["others"]
    rs = labelled(cfg, name)
    tasks = [(r["key"], r["moves"], ec) for r in rs]
    with get_context("spawn").Pool(cfg["processes"], initializer=_init, initargs=(22,)) as pool:
        res = []
        t0 = time.time()
        for i, r in enumerate(pool.imap_unordered(_others_one, tasks, chunksize=4)):
            res.append(r)
            if (i + 1) % 200 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] others: {i + 1}/{len(tasks)}, {time.time() - t0:.0f} s", flush=True)
    transfer.save(out_dir(cfg) / f"others_{name}.json", res)


# --- report --------------------------------------------------------------------------

def correct(r, ln):
    lab = {int(k): v for k, v in r["labels"].items()}
    return ln is not None and lab.get(int(ln), -9) == max(lab.values())


def decisive(r):
    return len(set(r["labels"].values())) > 1


def cmd_report(cfg):
    from parity_probe import wilson

    out = out_dir(cfg)
    rc = cfg["reach"]
    L = ["# Part 3 (c): a judge for the middle game of 10x7", "",
         f"Generated by `scripts/mid_judge_eval.py report` ({time.strftime('%Y-%m-%d %H:%M')}), "
         "configs/mid_judge.yaml. Judge: sim/mid_judge.py + agents/csrc/mid_judge.c — exact search on the real "
         "board (captures and double-dealing as agents/strong.py; safe lines plus one representative per chain / "
         "loop), Berlekamp's recursion as the leaf of the loony endgame without junctions (the formula judge at "
         "s = 0, where it is exact), a transposition table of value bounds keyed by the canonical position over "
         "the 4 board symmetries, sacrifices tried before safe lines, null-window searches for the result "
         "sign only. *Solved* = every legal line has its result sign (win / draw / loss for the mover).", ""]
    ck = out / "check.json"
    if ck.exists():
        c = json.loads(ck.read_text())
        L += ["## Check against the exact judge (late 10x7 positions)", "",
              f"{c['n']} positions of the part 3 (a) judge set ({cfg['check']['set']}, 1-2 safe lines, the move "
              f"decides the result): every move's sign equal in {c['all_moves']}/{c['n']}, the position's "
              f"result in {c['outcome']}/{c['n']}. Time per position (all moves) median {c['median_s']:.2f} s, "
              f"max {c['max_s']:.1f} s.", ""]
    tiers = sorted(rc["tiers"], key=int)
    reach = {t: read_jsonl(out / f"reach_{t}.jsonl") for t in tiers}
    if any(reach.values()):
        L += ["## How far it reaches", "",
              "Positions from Dot-2 self-play on 10x7 (MCTS-200, temperature 1 for the first 10 lines; "
              "per game the first position with exactly s safe lines and no box to take). Per time limit its own "
              "run with a fresh table. Solved % (the position's result found % in brackets); median / max time of "
              "the solved ones. A limit is tried up to the first s where the shorter one solved nothing.", "",
              "| safe lines | " + " | ".join(f"{t} s: n | solved % | median / max s" for t in tiers) + " |",
              "|---:|" + "---:|---:|---:|" * len(tiers)]
        for s in rc["safe"]:
            cells = []
            for t in tiers:
                rs = [r for r in reach[t] if r["key"][0] == s]
                if not rs:
                    cells += ["—", "—", "—"]
                    continue
                sol = [r for r in rs if r["labels"] is not None]
                oc = sum(r["outcome"] is not None for r in rs)
                ts = [r["t_solve"] for r in sol]
                cells += [str(len(rs)), f"{100 * len(sol) / len(rs):.0f} ({100 * oc / len(rs):.0f})",
                          f"{np.median(ts):.1f} / {max(ts):.1f}" if ts else "—"]
            L.append(f"| {s} | " + " | ".join(cells) + " |")
        L.append("")
    for name in cfg["label"]:
        rs = read_jsonl(out / f"labels_{name}.jsonl")
        if not rs:
            continue
        lc = cfg["label"][name]
        sol = [r for r in rs if r["labels"] is not None]
        nets = out / f"nets_{name}.json"
        oth = out / f"others_{name}.json"
        L += [f"## Labelled set '{name}': safe lines {lc['safe'][0]}-{lc['safe'][1]}", "",
              f"{len(rs)} positions ({lc['games']}.jsonl games), limit {lc['time_limit']} s: solved "
              f"{len(sol)} ({100 * len(sol) / len(rs):.1f}%), decisive (not every move has the same result) "
              f"{sum(decisive(r) for r in sol)}.", ""]
        if nets.exists():
            nd = json.loads(nets.read_text())
            od = {tuple(r["key"]): r for r in json.loads(oth.read_text())} if oth.exists() else {}
            by = {tuple(r["key"]): r for r in sol}
            methods = list(nd["moves"])
            if od:
                methods += ["formula", "mid", "exact"]
            L += ["Correct move = keeps the best result. Formula: sim/chain_solver.py ply_move; mid: this judge "
                  f"within {cfg['others']['mid_limit']} s; exact: agents/strong.py exact_move_values within "
                  f"{cfg['others']['exact_limit']} s (no answer = wrong).", "",
                  "| positions | n | " + " | ".join(methods) + " |", "|---|---:|" + "---:|" * len(methods)]
            rows = {m: {} for m in methods}
            for i, k in enumerate(map(tuple, nd["keys"])):
                for m in nd["moves"]:
                    rows[m][k] = nd["moves"][m][i]
                if k in od:
                    for m in ("formula", "mid", "exact"):
                        rows[m][k] = od[k][m]
            for lab, keep in (("all solved", lambda r: True), ("decisive", decisive)):
                ks = [k for k in by if keep(by[k]) and k in rows["hand"]]
                cells = []
                for m in methods:
                    kk = [k for k in ks if k in rows[m]]
                    c = sum(correct(by[k], rows[m][k]) for k in kk)
                    cells.append(f"{100 * c / len(kk):.1f}" if kk else "—")
                L.append(f"| {lab} | {len(ks)} | " + " | ".join(cells) + " |")
            L.append("")
            chart(cfg, name, by, rows, methods)
            L += [f"![who knows what](mid_judge.png)", "",
                  "Decisive positions by safe lines left (% correct moves):", "",
                  "| safe lines | n | " + " | ".join(methods) + " |", "|---:|---:|" + "---:|" * len(methods)]
            for s in sorted({k[0] for k in by}, reverse=True):
                ks = [k for k in by if k[0] == s and decisive(by[k]) and k in rows["hand"]]
                cells = []
                for m in methods:
                    kk = [k for k in ks if k in rows[m]]
                    cells.append(f"{100 * sum(correct(by[k], rows[m][k]) for k in kk) / len(kk):.0f}" if kk else "—")
                L.append(f"| {s} | {len(ks)} | " + " | ".join(cells) + " |")
            L.append("")
    concl = ROOT / "docs/mid_judge_conclusions.md"
    if concl.exists():
        L += [concl.read_text().rstrip(), ""]
    (ROOT / "docs/mid_judge.md").write_text("\n".join(L))
    print("\n".join(L))


def chart(cfg, name, by, rows, methods):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {"hand": "hand (policy argmax)", "intuition": "intuition (best child V)", "mcts200": "MCTS-200",
             "mcts2000": "MCTS-2000", "formula": "formula judge (+1 ply)",
             "mid": f"middle-game judge ({cfg['others']['mid_limit']} s)",
             "exact": f"exact judge, Python ({cfg['others']['exact_limit']} s)"}
    colors = {"hand": "#4C78A8", "intuition": "#9ECAE9", "mcts200": "#F58518", "mcts2000": "#E45756",
              "formula": "#54A24B", "mid": "#000000", "exact": "#B279A2"}
    s_all = sorted({k[0] for k in by})
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for m in methods:
        xs, ys = [], []
        for s in s_all:
            ks = [k for k in by if k[0] == s and decisive(by[k]) and k in rows[m]]
            if len(ks) >= 10:
                xs.append(s)
                ys.append(100 * sum(correct(by[k], rows[m][k]) for k in ks) / len(ks))
        ax.plot(xs, ys, marker="o", ms=3, lw=2 if m == "mid" else 1.5, label=names.get(m, m), color=colors.get(m))
    ax.set_xlim(max(s_all) + 0.5, -0.5)
    ax.set_ylim(0, 102)
    ax.set_xlabel("safe lines left (game goes left to right)")
    ax.set_ylabel("% correct moves (decisive positions)")
    ax.set_title("10x7, Dot-2 self-play positions: who knows the right move")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "docs/mid_judge.png", dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["games", "check", "reach", "label", "nets", "others", "report"])
    ap.add_argument("--config", default=str(ROOT / "configs/mid_judge.yaml"))
    ap.add_argument("--tier", default="10")
    ap.add_argument("--set", default="mid")
    ap.add_argument("--n", type=int)
    ap.add_argument("--name", default="games")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.cmd == "games":
        cmd_games(cfg, a.n, a.name)
    elif a.cmd == "reach":
        cmd_reach(cfg, a.tier)
    elif a.cmd in ("label", "nets", "others"):
        {"label": cmd_label, "nets": cmd_nets, "others": cmd_others}[a.cmd](cfg, a.set)
    else:
        {"check": cmd_check, "report": cmd_report}[a.cmd](cfg)


if __name__ == "__main__":
    main()
