"""Middle-game mistakes from routs only: does the hand generalise them?

    python scripts/rout_mistakes.py collect   # rounds: play vs Hard, judge the new routs, until judge.want mistakes
    python scripts/rout_mistakes.py check     # 30% held out: train the head on the rest, held-out accuracy before / after
    python scripts/rout_mistakes.py routs     # only if check passes: routs on the 6 seed sets (1200 games per net)
    python scripts/rout_mistakes.py report    # docs/rout_mistakes.md

collect: dot2c gen 16 (MCTS ``sims``) vs the simulator's Hard on 10x7, net
first alternating, in rounds of ``batch`` games. Only routs (a loss by
``margin``+ boxes) are judged: the net's positions from the end backwards,
``mid_judge.outcome`` per position. Passes: the first with the shortest
limit; a later pass reruns the walls of the last one from the wall
backwards with its longer limit (the same walk a single long pass makes,
cheaper). The walk stops at the first position that is not lost (*found*,
the line played there is the mistake, ``classify``'s best lines the target),
at a wall, or when the safe-line count goes above ``safe`` (*above*: no
mistake in range is possible further back). A mistake counts if its safe
lines are within ``safe``.

check: the head-only fine-tune of self_correct.py (policy head; body, value,
score frozen) on 70% of the mistakes; the held-out 30%'s accuracy (raw
policy argmax is a judge-correct line) before and after, with a paired sign
test. The hand generalises if a variant's held-out accuracy rises with
p < 0.05 (one-sided).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from math import comb
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from _common import ROOT, load_config

import self_correct as sc

CFG = ROOT / "configs/rout_mistakes.yaml"


def margin(g):
    s, me = g["score"], g["net_seat"]
    return s[me] - s[1 - me]


def routs_of(cfg, games):
    return [(i, g) for i, g in enumerate(games) if margin(g) <= -cfg["collect"]["margin"]]


# --- judge -----------------------------------------------------------------------------

def _jinit(tt_bits):
    sys.setrecursionlimit(10000)
    from sim import mid_judge as mj

    sc._G["j"] = mj.Judge(10, 7, tt_bits=tt_bits)


def _judge(task):
    from agents.baselines import safe_moves
    from sim import mid_judge as mj

    gid, g, limit, start, hi = task
    j = sc._G["j"]
    seat, moves = g["net_seat"], g["moves"]
    b = sc.board_of([])
    pos = []
    for k, ln in enumerate(moves):
        if b.player == seat:
            pos.append((k, len(safe_moves(b))))
        b.step(ln)
    r = {"gid": gid, "score": g["score"], "seat": seat, "limit": limit, "walk": []}
    for k, s in reversed([p for p in pos if p[0] <= start]):
        if s > hi:
            r.update(status="above", k=k, s=s)
            return r
        b = sc.board_of(moves[:k])
        j.clear()
        t0 = time.perf_counter()
        try:
            o = mj.outcome(j, b, limit)
        except mj.Timeout:
            r["walk"].append({"k": k, "s": s, "outcome": None})
            r.update(status="wall", k=k, s=s)
            return r
        r["walk"].append({"k": k, "s": s, "outcome": o, "t": round(time.perf_counter() - t0, 2)})
        if o >= 0:
            j.clear()
            try:
                lab = {int(m): v for m, v in mj.classify(j, b, 4 * limit).items()}
            except mj.Timeout:
                r.update(status="found-unlabelled", k=k, s=s, line=moves[k], outcome_before=o)
                return r
            best = max(lab.values())
            r.update(status="found", k=k, s=s, line=moves[k], outcome_before=o, played_label=lab.get(moves[k]),
                     correct=sorted(m for m, v in lab.items() if v == best), n_legal=len(lab))
            return r
    r.update(status="lost from the start")
    return r


def pass_path(cfg, i):
    return sc.out_dir(cfg) / f"judge_{cfg['judge']['passes'][i]['time_limit']}.jsonl"


def judged(cfg):
    """gid -> the walk's end after all passes (later passes replace the walls)."""
    rs = {}
    for i in range(len(cfg["judge"]["passes"])):
        for x in sc.read_jsonl(pass_path(cfg, i)):
            if x["gid"] in rs:
                x = {**x, "walk": rs[x["gid"]]["walk"][:-1] + x["walk"]}
            rs[x["gid"]] = x
    return rs


def in_range(cfg, r):
    lo, hi = cfg["judge"]["safe"]
    return r["status"] == "found" and lo <= r["s"] <= hi


def run_passes(cfg, games):
    jc = cfg["judge"]
    hi = jc["safe"][1]
    todo = {i: g for i, g in routs_of(cfg, games)}
    for p, pc in enumerate(jc["passes"]):
        path = pass_path(cfg, p)
        done = {r["gid"] for r in sc.read_jsonl(path)}
        if p == 0:
            tasks = [(i, g, pc["time_limit"], 10**9, hi) for i, g in todo.items() if i not in done]
        else:
            last = judged(cfg)
            tasks = [(i, todo[i], pc["time_limit"], r["k"], hi) for i, r in last.items()
                     if r["status"] == "wall" and r["limit"] < pc["time_limit"] and i not in done]
        if not tasks:
            continue
        print(f"[{time.strftime('%H:%M:%S')}] pass {pc['time_limit']} s: {len(tasks)} games", flush=True)
        t0 = time.time()
        with get_context("spawn").Pool(cfg["processes"], initializer=_jinit, initargs=(pc["tt_bits"],)) as pool, \
                open(path, "a") as fh:
            for n, r in enumerate(pool.imap_unordered(_judge, tasks)):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                if (n + 1) % 10 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}]   {n + 1}/{len(tasks)} ({time.time() - t0:.0f} s)", flush=True)


def cmd_collect(cfg):
    import torch

    import parity_hand
    from az_eval import _oinit, _opp_move

    cc = cfg["collect"]
    path = sc.out_dir(cfg) / "games.jsonl"
    while True:
        games = sc.read_jsonl(path)
        run_passes(cfg, games)
        found = [r for r in judged(cfg).values() if in_range(cfg, r)]
        print(f"[{time.strftime('%H:%M:%S')}] {len(games)} games, {len(routs_of(cfg, games))} routs, "
              f"{len(found)} mistakes in range", flush=True)
        if len(found) >= cfg["judge"]["want"] or len(games) >= cc["max_games"]:
            return
        ev, rc = parity_hand.evaluator({"device": cfg["device"]}, (cfg["net"]["config"], ROOT / cfg["net"]["checkpoint"]))
        torch.manual_seed(cc["seed"] + len(games))
        rng = np.random.default_rng([cc["seed"], len(games)])
        with get_context("spawn").Pool(cfg["processes"], initializer=_oinit,
                                       initargs=({}, cc["seed"] + len(games))) as pool:
            new = sc.play_batch(ev, rc, cc["opp"], cc["sims"], cc["batch"], pool, rng, 0, _opp_move)
        with open(path, "a") as fh:
            for g in new:
                fh.write(json.dumps(g) + "\n")
        del ev


# --- check -----------------------------------------------------------------------------

def mistakes(cfg):
    games = sc.read_jsonl(sc.out_dir(cfg) / "games.jsonl")
    rs = sorted((r for r in judged(cfg).values() if in_range(cfg, r)), key=lambda r: r["gid"])
    ps = [{"moves": games[r["gid"]]["moves"][:r["k"]], "correct": r["correct"], "played": r["line"],
           "gid": r["gid"], "s": r["s"]} for r in rs]
    random.Random(f"{cfg['seed']}/split").shuffle(ps)
    k = int(round(len(ps) * cfg["train"]["val_share"]))
    return ps[k:], ps[:k]


def correct_vec(net, positions):
    import torch
    import torch.nn.functional as F

    import parity_hand as ph

    codec, x, mask = ph.parity_arrays("10x7", positions)
    xt = torch.from_numpy(x.reshape(len(x), -1, codec.rows, codec.cols))
    with torch.no_grad():
        logits, _, _ = net.eval()(xt)
    legal = xt[:, 1].flatten(1)[:, net.line_cells(codec.rows, codec.cols)] > 0
    am = F.log_softmax(logits.masked_fill(~legal, -1e9), 1).argmax(1).numpy()
    return mask[np.arange(len(am)), am].astype(bool)


def sign_p(up, down):
    """One-sided sign test: P(at least ``up`` of up+down rises by chance)."""
    n = up + down
    return sum(comb(n, i) for i in range(up, n + 1)) / 2 ** n if n else 1.0


def cmd_check(cfg):
    import torch

    import parity_hand as ph

    sc.mistakes = mistakes   # the trainer reads its mistakes through this
    tr, va = mistakes(cfg)
    names = list(cfg["train"]["variants"])
    sc.cmd_train(cfg, [n for n in names if not (sc.out_dir(cfg) / f"{n}.pt").exists()])
    base, _ = ph.load_base({"net": cfg["net"]})
    b_tr, b_va = correct_vec(base, tr), correct_vec(base, va)
    res = {"n_train": len(tr), "n_val": len(va), "base": {"train": float(b_tr.mean()), "val": float(b_va.mean())}}
    for n in names:
        net, _ = ph.load_base({"net": cfg["net"]})
        net.load_state_dict(torch.load(sc.out_dir(cfg) / f"{n}.pt", map_location="cpu"))
        t, v = correct_vec(net, tr), correct_vec(net, va)
        up, down = int((v & ~b_va).sum()), int((~v & b_va).sum())
        res[n] = {"train": float(t.mean()), "val": float(v.mean()), "up": up, "down": down, "p": sign_p(up, down)}
    res["generalises"] = any(res[n]["val"] > res["base"]["val"] and res[n]["p"] < 0.05 for n in names)
    (sc.out_dir(cfg) / "check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    return res


# --- routs (only if check passes) ------------------------------------------------------

def cmd_routs(cfg):
    import parity_hand as ph

    res = json.loads((sc.out_dir(cfg) / "check.json").read_text())
    if not res["generalises"]:
        print("check failed: the hand does not generalise; no games", flush=True)
        return
    names = list(cfg["nets"])
    ph.cmd_routs(cfg, names)
    sc.cmd_noise(cfg)
    ph.cmd_bridge(cfg, names)


# --- report ----------------------------------------------------------------------------

def cmd_report(cfg):
    out = sc.out_dir(cfg)
    games = sc.read_jsonl(out / "games.jsonl")
    js = judged(cfg)
    jc, cc = cfg["judge"], cfg["collect"]
    lo, hi = jc["safe"]
    st = {}
    for r in js.values():
        key = ("found in range" if in_range(cfg, r) else "found below range" if r["status"] == "found"
               else r["status"])
        st[key] = st.get(key, 0) + 1
    L = ["# Middle-game mistakes from routs: does the hand generalise?", "",
         f"Generated by `scripts/rout_mistakes.py report` ({time.strftime('%Y-%m-%d %H:%M')}), "
         "configs/rout_mistakes.yaml. Net: Dot-2 (dot2c gen 16).", "",
         "## 1. Mistakes", "",
         f"{len(games)} games vs the simulator's Hard on 10x7 (MCTS {cc['sims']}, net first alternating), "
         f"{len(routs_of(cfg, games))} routs (lost by {cc['margin']}+). Judge from the end backwards, passes "
         + ", ".join(f"{p['time_limit']} s" for p in jc["passes"])
         + f" (each later pass from the last one's wall). A mistake counts at {lo}-{hi} safe lines; the walk "
         f"stops above {hi} (*above*).", "",
         "| outcome | routs |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in sorted(st.items())] + [""]
    fr = [r for r in js.values() if in_range(cfg, r)]
    if fr:
        edges = [(lo, 10), (11, 15), (16, 20), (21, hi)]
        L += ["Mistakes in range by safe lines: " + ", ".join(
            f"{a}-{b}: {sum(a <= r['s'] <= b for r in fr)}" for a, b in edges) + ".", ""]
    cp = out / "check.json"
    if cp.exists():
        c = json.loads(cp.read_text())
        tc = cfg["train"]
        L += ["## 2. Generalisation check", "",
              f"{c['n_train']} mistakes train, {c['n_val']} held out ({100 * tc['val_share']:.0f}%). Policy head "
              f"only, {tc['steps']} steps, lr {tc['lr']}, batch {tc['batch']} with the variant's share of mistakes, "
              "the rest dot2c replay positions. Accuracy = raw policy argmax is a judge-correct line. Up / down: "
              "held-out positions that became right / wrong; p: one-sided sign test.", "",
              "| net | train acc % | held-out acc % | up | down | p |", "|---|---:|---:|---:|---:|---:|",
              f"| base | {100 * c['base']['train']:.1f} | {100 * c['base']['val']:.1f} | | | |"]
        for n in tc["variants"]:
            x = c[n]
            L.append(f"| {n} | {100 * x['train']:.1f} | {100 * x['val']:.1f} | {x['up']} | {x['down']} | {x['p']:.3f} |")
        L += ["", f"Generalises (a variant up with p < 0.05): **{'yes' if c['generalises'] else 'no'}**.", ""]
    ev = out / "eval.json"
    nz = out / "noise.json"
    if ev.exists() and nz.exists():
        res, nd = json.loads(ev.read_text()), json.loads(nz.read_text())
        m = cfg["eval"]["routs"]["margin"]
        seeds = [cfg["seed"], *cfg["noise"]["seeds"]]
        L += ["## 3. Routs and bridge", "",
              f"Routs: 200 games vs the simulator's Hard (MCTS 50) per seed set, rout = lost by {m}+. Bridge: 20 "
              "games vs the real KSquares Hard.", "",
              "| net | " + " | ".join(f"seed {s}" for s in seeds) + " | total routs | per 200 | losses | bridge W-L-T |",
              "|---|" + "---:|" * (len(seeds) + 4)]
        for n in cfg["nets"]:
            rr = [res.get(f"routs/{n}")] + [nd.get(f"{s}/{n}") for s in cfg["noise"]["seeds"]]
            if any(r is None for r in rr):
                continue
            ro = [sum(x <= -m for x in r["margins"]) for r in rr]
            br = res.get(f"bridge/{n}")
            L.append(f"| {n} | " + " | ".join(map(str, ro)) + f" | {sum(ro)} | {sum(ro) / len(ro):.1f} | "
                     f"{sum(r['wlt'][1] for r in rr)} | "
                     + (f"{br['wins']}-{br['losses']}-{br['ties']}" if br else "—") + " |")
        L.append("")
    concl = ROOT / "docs/rout_mistakes_conclusions.md"
    if concl.exists():
        L += [concl.read_text().rstrip(), ""]
    (ROOT / "docs/rout_mistakes.md").write_text("\n".join(L))
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["collect", "check", "routs", "report"])
    ap.add_argument("--config", default=str(CFG))
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"collect": cmd_collect, "check": cmd_check, "routs": cmd_routs, "report": cmd_report}[a.cmd](cfg)


if __name__ == "__main__":
    main()
