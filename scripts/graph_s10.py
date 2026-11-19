"""Part 4, series 10 of docs/plan_part4.md: matches only. Was Dot gen 54 the best Dot, or did the ladder err there
too? (configs/graph_s10.yaml)

    python scripts/graph_s10.py run       # items 1-4 and the round robin (resumable: runs/graph_s10/results.json)
    python scripts/graph_s10.py report    # docs/graph_s10.md

Items 1-2: late / best-raw Dot against Dot gen 54; item 3: g5 gen 114 against the strongest Dot of items 1-2;
item 4: g5 gen 114 against g5 gen 49. Modes: MCTS 200 both, without thinking both (policy, sampled in the first
temp_moves lines, then argmax), equal time per move (only where the two nets differ in cost: g5 58 sims against
Dot 200, series 9's calibration; between nets of one architecture equal time = MCTS 200 both). First mover
alternating; nets play the first temp_moves lines at temperature 1. Round robin: 6 players (nets with MCTS 200,
Hard and Strong as on the ladder), 100 games per pair, descriptive.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from multiprocessing import get_context

import numpy as np

from _common import ROOT, load_config

AGENTS = ("hard", "strong")
LABEL = {"dot54": "Dot gen 54", "dot_last": "Dot gen 84 (last)", "dot_raw": "Dot gen 52 (best raw)",
         "g5_114": "g5 gen 114", "g5_49": "g5 gen 49", "hard": "Hard", "strong": "Strong"}
MODE = {"mcts": "MCTS 200 both", "raw": "without thinking both", "equal_time": "equal time per move"}


def config(path="configs/graph_s10.yaml"):
    c = load_config(ROOT / path)
    base = load_config(ROOT / c["base"])
    return {**base, **c, "eval": base["eval"]}


def wilson(w, n, z=1.96):
    """95 % Wilson interval of a share w / n."""
    if n == 0:
        return 0.0, 1.0
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def score(r):
    """Share of the first player from [wins a, wins b, ties, ...]."""
    return (r[0] + r[2] / 2) / (r[0] + r[1] + r[2])


def strongest(res):
    """The strongest Dot by items 1-2 (fixed in the plan): a candidate whose MCTS-200 score against gen 54 has its
    95 % interval above 50 %; of two, the higher score; none -> dot54."""
    best, bs = "dot54", None
    for item, cand in (("1", "dot_last"), ("2", "dot_raw")):
        r = res.get(f"{item}:mcts")
        if r is None:
            return None
        lo, _ = wilson(r[0] + r[2] / 2, sum(r[:3]))
        if lo > 0.5 and (bs is None or score(r) > bs):
            best, bs = cand, score(r)
    return best


def reading(res):
    """The protocol's reading (series 10)."""
    s = strongest(res)
    if s is None:
        return None
    r = res["1:mcts"]
    late = wilson(r[0] + r[2] / 2, sum(r[:3]))[0] > 0.5
    if late:
        return ("the late Dot beats gen 54: the graph vs convolution comparison is item 3; part 1's \"Dot did not "
                "grow after round 54\" is a measurement error")
    if s != "dot54":
        return ("the late Dot does not beat gen 54, the best-raw Dot does: the comparison is item 3 (against it); "
                "part 1's note stands")
    return "no Dot beats gen 54: the comparison stays as in series 9"


def play_pair(players, n, cfg, seed, pool, concurrent=200):
    """players[k]: {"ev", "sims"} (a net; sims 0 = policy) or {"agent": name}. Game g: players[g % 2] moves
    first. Returns [wins 0, wins 1, ties, margin 0]."""
    from agents.az_mcts import choose_move, result_for_mover
    from agents.az_play import make_mcts
    from sim.board import Board

    import az_eval

    w, h = cfg["board"]
    tm = cfg["temp_moves"]
    rng = np.random.default_rng(seed)
    srch = [make_mcts(p["ev"], cfg["mcts"], rng) if "ev" in p else None for p in players]
    out = [0, 0, 0, 0]
    started, live = 0, []
    while live or started < n:
        while len(live) < concurrent and started < n:
            live.append([Board(w, h), started % 2, 0])
            started += 1
        pending = {}
        for k in (0, 1):
            grp = [g for g in live if (g[0].player == 0) == (g[1] == k) and result_for_mover(g[0]) is None]
            if grp and "agent" in players[k]:
                pending[k] = (grp, pool.map_async(az_eval._opp_move, [(players[k]["agent"], g[0]) for g in grp],
                                                  chunksize=1))
        for k in (0, 1):
            p = players[k]
            if "agent" in p:
                continue
            grp = [g for g in live if (g[0].player == 0) == (g[1] == k) and result_for_mover(g[0]) is None]
            if not grp:
                continue
            if p["sims"] == 0:
                pri, _ = p["ev"]([g[0] for g in grp])
                for g, q in zip(grp, pri):
                    q = np.asarray(q, dtype=np.float64)
                    g[0].step(int(rng.choice(len(q), p=q / q.sum())) if g[2] < tm else int(np.argmax(q)))
                    g[2] += 1
            else:
                for g, r in zip(grp, srch[k].search([g[0] for g in grp], p["sims"])):
                    g[0].step(choose_move(r, 1.0 if g[2] < tm else 0.0, rng))
                    g[2] += 1
        for k, (grp, ar) in pending.items():
            for g, m in zip(grp, ar.get()):
                g[0].step(m)
                g[2] += 1
        nxt = []
        for g in live:
            b, first, _ = g
            if result_for_mover(b) is None:
                nxt.append(g)
                continue
            s0 = b.score[0] if first == 0 else b.score[1]
            s1 = sum(b.score) - s0
            out[3] += s0 - s1
            out[0 if s0 > s1 else 1 if s1 > s0 else 2] += 1
        live = nxt
    return out


def cmd_run(cfg):
    import torch

    from agents.az_net import Evaluator, make_net

    torch.set_num_threads(2)
    run = ROOT / cfg["run_dir"]
    run.mkdir(parents=True, exist_ok=True)
    path = run / "results.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    evs = {}

    def ev(name):
        if name not in evs:
            c, ck = cfg["nets"][name]
            net = make_net(load_config(ROOT / c))
            net.load_state_dict(torch.load(ROOT / ck, map_location="cpu"))
            evs[name] = Evaluator(net, cfg["device"])
        return evs[name]

    def player(name, mode, other):
        if name in AGENTS:
            return {"agent": name}
        sims = {"mcts": cfg["sims"], "raw": 0}.get(mode)
        if mode == "equal_time":   # only g5 against Dot: g5 gets the calibrated sims
            sims = cfg["equal_time_sims"] if name.startswith("g5") else cfg["sims"]
        return {"ev": ev(name), "sims": sims}

    def save():
        path.with_suffix(".tmp").write_text(json.dumps(res, indent=1))
        os.replace(path.with_suffix(".tmp"), path)

    ctx = get_context("spawn")
    with ctx.Pool(cfg["processes"], initializer=__import__("az_eval")._oinit,
                  initargs=(cfg["eval"].get("params") or {}, cfg["seed"] + 1000)) as pool:
        for item, (a, b, modes) in cfg["matches"].items():
            if b == "strongest":
                b = strongest(res)
                res["strongest"] = b
            for i, mode in enumerate(modes):
                key = f"{item}:{mode}"
                if key in res:
                    continue
                t0 = time.time()
                r = play_pair([player(a, mode, b), player(b, mode, a)], cfg["games"], cfg,
                              [cfg["seed"], int(item), i], pool)
                res[key] = r
                res[key + ":who"] = [a, b]
                print(f"[{time.strftime('%H:%M:%S')}] item {item} {LABEL[a]} vs {LABEL[b]}, {MODE[mode]}: {r} "
                      f"({time.time() - t0:.0f} s)", flush=True)
                save()
        rr = cfg["round_robin"]
        ps = rr["players"]
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                key = f"rr:{ps[i]}:{ps[j]}"
                if key in res:
                    continue
                t0 = time.time()
                r = play_pair([player(ps[i], "mcts", ps[j]), player(ps[j], "mcts", ps[i])], rr["games"], cfg,
                              [cfg["seed"], 100 + i, j], pool)
                res[key] = r
                print(f"[{time.strftime('%H:%M:%S')}] round robin {LABEL[ps[i]]} vs {LABEL[ps[j]]}: {r} "
                      f"({time.time() - t0:.0f} s)", flush=True)
                save()
    cmd_report(cfg)


def cmd_report(cfg):
    res = json.loads((ROOT / cfg["run_dir"] / "results.json").read_text())
    L = ["# Part 4, series 10: was Dot gen 54 the best Dot?", "",
         f"Generated by `scripts/graph_s10.py report` ({time.strftime('%Y-%m-%d %H:%M')}). Protocol: docs/plan_part4.md "
         "(series 10), configs/graph_s10.yaml. Matches only, no training. "
         f"{cfg['games']} games per match, first mover alternating, the first {cfg['temp_moves']} lines of the nets at "
         "temperature 1, then the best move. Score % of the first-named player with a 95 % Wilson interval. "
         f"Equal time: g5 {cfg['equal_time_sims']} sims against Dot 200 (series 9 calibration); between two nets of "
         "one architecture equal time is MCTS 200 both and is not played again.", "",
         "## Items 1-4", "",
         "| item | player | opponent | mode | wins | losses | ties | score % | 95 % |", "|---|---|---|---|---:|---:|---:|---:|---|"]
    for item, (_, _, modes) in cfg["matches"].items():
        for mode in modes:
            r = res.get(f"{item}:{mode}")
            if r is None:
                continue
            a, b = res[f"{item}:{mode}:who"]
            lo, hi = wilson(r[0] + r[2] / 2, sum(r[:3]))
            L.append(f"| {item} | {LABEL[a]} | {LABEL[b]} | {MODE[mode]} | {r[0]} | {r[1]} | {r[2]} | "
                     f"{100 * score(r):.1f} | {100 * lo:.1f}-{100 * hi:.1f} |")
    s, rd = strongest(res), reading(res)
    L += ["", f"Strongest Dot by items 1-2 (MCTS 200, interval above 50 %, else gen 54): "
          f"{LABEL[s] if s else '—'}.", "", f"**Reading (fixed before the run): {rd or '—'}.**", ""]
    rr = cfg["round_robin"]
    ps = rr["players"]
    if any(k.startswith("rr:") for k in res):
        L += ["## Round robin (descriptive)", "",
              f"MCTS 200 for the nets, Hard and Strong as on the ladder; {rr['games']} games per pair, first mover "
              "alternating. Cell = score % of the row player against the column player.", "",
              "| | " + " | ".join(LABEL[p] for p in ps) + " | mean |", "|---|" + "---:|" * (len(ps) + 1)]
        for p in ps:
            row, vals = [], []
            for q in ps:
                if p == q:
                    row.append("—")
                    continue
                r = res.get(f"rr:{p}:{q}")
                v = None if r is None else score(r)
                if v is None and f"rr:{q}:{p}" in res:
                    v = 1 - score(res[f"rr:{q}:{p}"])
                row.append("…" if v is None else f"{100 * v:.0f}")
                if v is not None:
                    vals.append(v)
            mean = f"{100 * np.mean(vals):.0f}" if vals else "…"
            L.append(f"| {LABEL[p]} | " + " | ".join(row) + f" | {mean} |")
        L.append("")
    concl = ROOT / "docs/graph_s10_notes.md"
    if concl.exists():
        L += [concl.read_text().rstrip(), ""]
    (ROOT / "docs/graph_s10.md").write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--config", default="configs/graph_s10.yaml")
    a = ap.parse_args()
    cfg = config(a.config)
    {"run": cmd_run, "report": cmd_report}[a.cmd](cfg)


if __name__ == "__main__":
    main()
