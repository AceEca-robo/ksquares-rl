"""Large parity probe: the long chain rule on 1000 fresh positions.

    python scripts/parity_probe.py positions   # generate the position set
    python scripts/parity_probe.py evaluate    # raw / MCTS per generation, baselines
    python scripts/parity_probe.py report      # table with 95% intervals, paired tests

Positions come from the generator of scripts/strategy_probe.py (same agent
pairs and exact solving, parity category only), with a fresh seed and the
150 parity positions of the strategy probe excluded, so the result is an
independent replication of its parity column. Results in
runs/parity_probe/eval.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from multiprocessing import get_context

import numpy as np

from _common import ROOT, load_config

import strategy_probe as sp

OUT = ROOT / "runs/parity_probe"


def key(p):
    return tuple(sorted(p["moves"])), tuple(p["score"]), p["player"]


def merged(cfg):
    base = load_config(ROOT / cfg["base"])
    c = dict(base, board=cfg["board"], processes=cfg["processes"], device=cfg["device"])
    c["positions"] = dict(base["positions"], **cfg["positions"])
    return c


def cmd_positions(cfg):
    c = merged(cfg)
    pc = c["positions"]
    old = json.loads((ROOT / load_config(ROOT / cfg["base"])["positions"]["out"]).read_text())
    seen = {key(p) for p in old["positions"]}
    rng = random.Random(cfg["seed"])
    tasks = []
    for g in range(pc["games"]):
        a, b = pc["pairs"][g % len(pc["pairs"])]
        tasks.append((a, b, rng.randrange(2**31), g % 2))
    uniq = {}
    t0 = time.time()
    with get_context("spawn").Pool(c["processes"], initializer=sp._gen_init, initargs=(c,)) as pool:
        for i, (found, _) in enumerate(pool.imap_unordered(sp._gen_game, tasks, chunksize=2)):
            for p in found:
                if p["category"] == "parity" and key(p) not in seen:
                    uniq.setdefault(key(p), p)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(tasks)} games, {time.time() - t0:.0f} s, parity {len(uniq)}", flush=True)
    ps = list(uniq.values())
    rng.shuffle(ps)
    if len(ps) < pc["n"]:
        print(f"only {len(ps)} distinct parity positions (wanted {pc['n']})")
    out = ps[:pc["n"]]
    for p in out:
        p["values"] = {str(k): v for k, v in p["values"].items()}
    path = ROOT / pc["out"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"board": cfg["board"], "positions": out}))
    print(f"{len(uniq)} distinct new parity positions; saved {len(out)} to {path}")


def cmd_evaluate(cfg):
    import torch

    from agents.az_mcts import choose_move
    from agents.az_net import Evaluator, make_net
    from agents.az_play import make_mcts

    c = merged(cfg)
    positions, boards = sp.load_positions(c)
    correct = [set(p["correct"]) for p in positions]
    path = OUT / "eval.json"
    res = json.loads(path.read_text()) if path.exists() else {}

    def score(moves):
        return [int(m in s) for m, s in zip(moves, correct)]

    wh = cfg["board"]
    with get_context("spawn").Pool(cfg["processes"]) as pool:
        for name in cfg["baselines"]:
            if name in res:
                continue
            params = c["agents"]["strong"] if name == "strong" else {}
            moves = pool.map(sp._agent_move, [(name, params, cfg["seed"] + i, p["moves"], wh)
                                              for i, p in enumerate(positions)], chunksize=4)
            res[name] = score(moves)
            print(f"{name}: {sum(res[name])}/{len(moves)}", flush=True)
            path.write_text(json.dumps(res))
    rc = load_config(ROOT / cfg["gens"]["run_config"])
    for gen in cfg["gens"]["list"]:
        k = f"{rc['run_dir'].split('/')[-1]}/{gen}"
        if k in res:
            continue
        net = make_net(rc)
        net.load_state_dict(torch.load(ROOT / rc["run_dir"] / f"gen_{gen:04d}.pt", map_location="cpu"))
        ev = Evaluator(net, cfg["device"])
        pri, _ = ev(boards)
        raw = [int(np.argmax(p)) for p in pri]
        rng = np.random.default_rng(cfg["seed"])
        mcts = make_mcts(ev, rc["mcts"], rng)
        mc = [choose_move(root, 0.0, rng) for root in mcts.search(boards, c["agents"]["sims"])]
        res[k] = {"raw": score(raw), "mcts": score(mc)}
        print(f"{k}: raw {sum(res[k]['raw'])}, mcts {sum(res[k]['mcts'])} / {len(boards)}", flush=True)
        path.write_text(json.dumps(res))
    res["_chance"] = [len(p["correct"]) / len(p["values"]) for p in positions]
    path.write_text(json.dumps(res))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (m - h), 100 * (m + h)


def mcnemar(a, b):
    """Exact two-sided McNemar p-value on paired 0/1 results."""
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    tail = sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2**n
    return n01, n10, min(1.0, 2 * tail)


def cmd_report(cfg):
    res = json.loads((OUT / "eval.json").read_text())
    n = len(res["_chance"])
    L = [f"{n} parity positions; uniform random legal move: {100 * np.mean(res['_chance']):.1f}%", "",
         "| player | raw % [95%] | MCTS % [95%] |", "|---|---|---|"]
    for name in cfg["baselines"]:
        k = sum(res[name])
        lo, hi = wilson(k, n)
        L.append(f"| {name} | {100 * k / n:.1f} [{lo:.1f}, {hi:.1f}] | |")
    run = load_config(ROOT / cfg["gens"]["run_config"])["run_dir"].split("/")[-1]
    keys = [f"{run}/{g}" for g in cfg["gens"]["list"]]
    for k in keys:
        cells = []
        for mode in ("raw", "mcts"):
            s = sum(res[k][mode])
            lo, hi = wilson(s, n)
            cells.append(f"{100 * s / n:.1f} [{lo:.1f}, {hi:.1f}]")
        L.append(f"| {k} | {cells[0]} | {cells[1]} |")
    L += ["", "Paired (same positions), exact McNemar: first wrong & second right / first right & second wrong, p", ""]
    for mode in ("raw", "mcts"):
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                n01, n10, p = mcnemar(res[a][mode], res[b][mode])
                L.append(f"- {mode} {a} -> {b}: +{n01} / -{n10}, p = {p:.3g}")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["positions", "evaluate", "report"])
    ap.add_argument("--config", default=str(ROOT / "configs/parity_probe.yaml"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"positions": cmd_positions, "evaluate": cmd_evaluate, "report": cmd_report}[a.cmd](cfg)


if __name__ == "__main__":
    main()
