"""Large parity probe: the long chain rule on 1000 fresh positions.

    python scripts/parity_probe.py positions   # generate the position set
    python scripts/parity_probe.py evaluate    # raw / MCTS per generation, baselines
    python scripts/parity_probe.py report      # table with 95% intervals, paired tests
    python scripts/parity_probe.py curves      # every accepted generation: policy, V, MCTS
    python scripts/parity_probe.py plot        # the curves -> table + docs/parity_curves.png

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


def run_gens(cfg):
    """(run, gen, checkpoint, run config) for every accepted generation
    (gen_NNNN.pt) of the runs in curves.runs, gen 0 (the initial net) left out."""
    out = []
    for rc_path in cfg["curves"]["runs"]:
        rc = load_config(ROOT / rc_path)
        run = rc["run_dir"].split("/")[-1]
        for ck in sorted((ROOT / rc["run_dir"]).glob("gen_*.pt")):
            g = int(ck.stem.split("_")[1])
            if g > 0:
                out.append((run, g, ck, rc))
    return out


def gen_curves(ev, rc, boards, positions, sims, seed):
    """One generation on the parity set. policy: argmax correct; v_child:
    sign of V after every legal move right (mover's view; balanced over
    winning / losing moves, as in docs/probes/summary.md); v_greedy: the
    move with the best child V correct; mcts: the MCTS move correct."""
    from agents.az_mcts import choose_move, result_for_mover
    from agents.az_play import make_mcts

    correct = [set(p["correct"]) for p in positions]
    pri, _ = ev(boards)
    kids = [[(ln, b.clone()) for ln in range(b.geo.n_lines) if not b.lines[ln]] for b in boards]
    for ks in kids:
        for ln, c in ks:
            c.step(ln)
    _, cval = ev([c for ks in kids for _, c in ks])
    it = iter(cval)
    win_ok, lose_ok, greedy = [], [], []
    for b, ks, cor in zip(boards, kids, correct):
        best, best_v = None, -2.0
        for ln, c in ks:
            v = next(it)
            ex = result_for_mover(c)
            v = ex if ex is not None else float(v)
            v = v if c.player == b.player else -v
            # every parity position is critical with no ties: correct = wins, the rest lose
            if ln in cor:
                win_ok.append(int(v > 0))
            else:
                lose_ok.append(int(v < 0))
            if v > best_v:
                best, best_v = ln, v
        greedy.append(int(best in cor))
    rng = np.random.default_rng(seed)
    roots = make_mcts(ev, rc["mcts"], rng).search(boards, sims)
    mc = [choose_move(r, 0.0, rng) for r in roots]
    return {"policy": [int(int(np.argmax(p)) in c) for p, c in zip(pri, correct)],
            "mcts": [int(m in c) for m, c in zip(mc, correct)],
            "v_greedy": greedy,
            "v_child": 50 * (np.mean(win_ok) + np.mean(lose_ok)),
            "v_child_win": 100 * np.mean(win_ok), "v_child_lose": 100 * np.mean(lose_ok)}


def cmd_curves(cfg):
    import torch

    from agents.az_net import Evaluator, make_net

    c = merged(cfg)
    positions, boards = sp.load_positions(c)
    path = OUT / "curves.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for run, g, ck, rc in run_gens(cfg):
        k = f"{run}/{g}"
        if k in res:
            continue
        t0 = time.time()
        net = make_net(rc)
        net.load_state_dict(torch.load(ck, map_location="cpu"))
        r = gen_curves(Evaluator(net, cfg["device"]), rc, boards, positions, c["agents"]["sims"], cfg["seed"])
        res[k] = r
        print(f"{k}: policy {sum(r['policy'])}, V child {r['v_child']:.1f}%, V greedy {sum(r['v_greedy'])}, "
              f"mcts {sum(r['mcts'])} / {len(boards)}  ({time.time() - t0:.0f} s)", flush=True)
        path.write_text(json.dumps(res))


def cmd_plot(cfg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads((OUT / "curves.json").read_text())
    base = json.loads((OUT / "eval.json").read_text())
    keys = [f"{run}/{g}" for run, g, _, _ in run_gens(cfg) if f"{run}/{g}" in res]
    n = len(res[keys[0]]["policy"])
    rows = []
    L = ["| run | gen | x | policy % | V child % (win / lose) | V greedy % | MCTS % |", "|---|---:|---:|---:|---:|---:|---:|"]
    for x, k in enumerate(keys, 1):
        r = res[k]
        run, g = k.split("/")
        row = (x, 100 * sum(r["policy"]) / n, r["v_child"], 100 * sum(r["v_greedy"]) / n, 100 * sum(r["mcts"]) / n)
        rows.append(row)
        L.append(f"| {run} | {g} | {x} | {row[1]:.1f} | {row[2]:.1f} ({r['v_child_win']:.0f} / "
                 f"{r['v_child_lose']:.0f}) | {row[3]:.1f} | {row[4]:.1f} |")
    print("\n".join(L))
    xs, pol, vc, vg, mc = zip(*rows)
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=120)
    ax.plot(xs, pol, "o-", ms=3, label="policy (no search)")
    ax.plot(xs, vc, "o-", ms=3, label="value: sign after each move")
    ax.plot(xs, vg, "o-", ms=3, alpha=0.6, label="value: best move by one-ply V")
    ax.plot(xs, mc, "o-", ms=3, label=f"MCTS {cfg['curves']['sims_label']}")
    for name, style in (("strong", ":"), ("hard", "--")):
        ax.axhline(100 * sum(base[name]) / n, ls=style, c="grey", lw=1)
        ax.text(len(xs) + 0.5, 100 * sum(base[name]) / n, name, va="center", fontsize=9, color="grey")
    chance = 100 * np.mean(base["_chance"])
    ax.axhline(chance, ls=":", c="lightgrey", lw=1)
    ax.text(len(xs) + 0.5, chance, "random move", va="center", fontsize=9, color="grey")
    split = sum(1 for k in keys if k.startswith("az5/"))
    ax.axvline(split + 0.5, c="lightgrey", lw=1)
    ax.set_xticks(xs, [k.split("/")[1] for k in keys], fontsize=7)
    ax.set_xlabel(f"accepted generation (az5 1-{keys[split - 1].split('/')[1]} | az5b)")
    ax.set_ylabel(f"% right on {n} parity positions")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / cfg["curves"]["plot"])
    print(f"saved {cfg['curves']['plot']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["positions", "evaluate", "report", "curves", "plot"])
    ap.add_argument("--config", default=str(ROOT / "configs/parity_probe.yaml"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"positions": cmd_positions, "evaluate": cmd_evaluate, "report": cmd_report,
     "curves": cmd_curves, "plot": cmd_plot}[a.cmd](cfg)


if __name__ == "__main__":
    main()
