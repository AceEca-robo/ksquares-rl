"""Evaluate accepted AlphaZero generations against the 5x5 ladder.

For every accepted generation without ``run_dir/eval/gen_XXXX.json``: the
net alone (argmax of the policy over legal lines, no search) and the net
with MCTS (``eval.sims`` simulations, no noise, temperature 0) play
``eval.games`` games against each opponent, the first mover alternating.
Net moves are batched over all games on the GPU; opponent moves run in a
CPU process pool (one agent instance per worker and opponent; strong keeps
its transposition table across games, which is keyed by the position only).
Games stop as soon as the winner is fixed (more than half of the boxes).

Then rebuilds ``docs/<run>_ladder.md`` (table + Elo) and
``docs/<run>_ladder.png``, <run> = the last part of ``run_dir``. Elo: Bradley-Terry fit, random = 0. The anchor
agents are fitted once from the 200-games-per-pair ladder in
docs/results.md (LADDER below); each net is then fitted alone against the
fixed anchors. Every pair gets one virtual game split half/half so that
100% scores stay finite.

``--report`` only rebuilds the table and plot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from multiprocessing import get_context

import numpy as np

from _common import ROOT, load_config

# wins of the first agent over the second, 200 games per pair, no ties
# (docs/results.md, "Ladder on 5x5", seed 777)
LADDER = {
    ("greedy", "random"): 200, ("easy", "random"): 200, ("medium", "random"): 200,
    ("hard", "random"): 200, ("strong", "random"): 200,
    ("greedy", "easy"): 105, ("medium", "greedy"): 184, ("hard", "greedy"): 186, ("strong", "greedy"): 198,
    ("medium", "easy"): 192, ("hard", "easy"): 185, ("strong", "easy"): 194,
    ("hard", "medium"): 193, ("strong", "medium"): 188, ("strong", "hard"): 187,
}
LADDER_GAMES = 200
ANCHORS = ("random", "greedy", "easy", "medium", "hard", "strong")

_A = {}


def _oinit(params, seed):
    from agents.registry import make_agent
    pid = os.getpid()
    _A["agents"] = {}
    _A["make"] = lambda name: make_agent(name, seed + pid, **params.get(name, {}))


def _opp_move(task):
    name, board = task
    ag = _A["agents"].get(name)
    if ag is None:
        ag = _A["agents"][name] = _A["make"](name)
    return ag.act(board)


def play(evaluator, mode, opponents, cfg, pool, rng):
    """All games of one mode at once; returns {opp: [net wins, losses, ties]}."""
    from agents.az_mcts import choose_move, result_for_mover
    from agents.az_play import make_mcts
    from sim.board import Board

    w, h = cfg["board"]
    ec = cfg["eval"]
    mcts = make_mcts(evaluator, cfg["mcts"], rng)
    live = [[Board(w, h), opp, g % 2 == 0] for opp in opponents for g in range(ec["games"])]
    res = {o: [0, 0, 0] for o in opponents}
    while live:
        net_turn = [g for g in live if (g[0].player == 0) == g[2]]
        opp_turn = [g for g in live if (g[0].player == 0) != g[2]]
        async_res = pool.map_async(_opp_move, [(g[1], g[0]) for g in opp_turn], chunksize=1)
        if net_turn:
            boards = [g[0] for g in net_turn]
            if mode == "raw":
                pri, _ = evaluator(boards)
                moves = [int(np.argmax(p)) for p in pri]
            else:
                moves = [choose_move(r, 0.0, rng) for r in mcts.search(boards, ec["sims"])]
            for g, m in zip(net_turn, moves):
                g[0].step(m)
        for g, m in zip(opp_turn, async_res.get()):
            g[0].step(m)
        nxt = []
        for g in live:
            board, opp, net_first = g
            if result_for_mover(board) is None:
                nxt.append(g)
                continue
            net = board.score[0 if net_first else 1]
            other = sum(board.score) - net
            res[opp][0 if net > other else 1 if other > net else 2] += 1
        live = nxt
    return res


# --- Elo --------------------------------------------------------------------------

def _p(d):
    return 1.0 / (1.0 + 10 ** (-d / 400))


def fit_anchors():
    """Bradley-Terry by MM iterations on the ladder; Elo with random = 0."""
    idx = {a: i for i, a in enumerate(ANCHORS)}
    n = len(ANCHORS)
    W = np.full((n, n), 0.5)  # W[i, j] = wins of i over j, incl. the virtual half game
    np.fill_diagonal(W, 0)
    for (a, b), wins in LADDER.items():
        W[idx[a], idx[b]] += wins
        W[idx[b], idx[a]] += LADDER_GAMES - wins
    N = W + W.T
    g = np.ones(n)
    for _ in range(20000):
        g_new = W.sum(1) / np.array([sum(N[i, j] / (g[i] + g[j]) for j in range(n) if j != i) for i in range(n)])
        g_new /= g_new[0]
        if np.max(np.abs(np.log(g_new / g))) < 1e-12:
            break
        g = g_new
    return {a: 400 * math.log10(g[idx[a]]) for a in ANCHORS}


def fit_one(results, anchors):
    """Elo of one player from {opp: [w, l, t]} against fixed anchor ratings."""
    data = [(anchors[o], w + t / 2 + 0.5, l + t / 2 + 0.5) for o, (w, l, t) in results.items()]
    lo, hi = -2000.0, 6000.0
    for _ in range(100):  # d logL / dr is decreasing in r: bisection
        r = (lo + hi) / 2
        grad = sum(w * (1 - _p(r - ro)) - l * _p(r - ro) for ro, w, l in data)
        lo, hi = (r, hi) if grad > 0 else (lo, r)
    return (lo + hi) / 2


# --- report -----------------------------------------------------------------------

def report(cfg):
    run = ROOT / cfg["run_dir"]
    state = json.loads((run / "state.json").read_text())
    evals = []
    for h in state["history"]:
        p = run / "eval" / f"gen_{h['gen']:04d}.json"
        if h["accepted"] and p.exists():
            evals.append(json.loads(p.read_text()))
    anchors = fit_anchors()
    opps = cfg["eval"]["opponents"]
    name = os.path.basename(cfg["run_dir"].rstrip("/"))
    lines = [f"# AlphaZero 5x5 ({name}) — ladder per accepted generation", "",
             f"Generated by `scripts/az_eval.py` from `{cfg['run_dir']}` ({time.strftime('%Y-%m-%d %H:%M')}). "
             f"Win % of the net, {cfg['eval']['games']} games per cell, first mover alternating; "
             f"raw = policy argmax without search, mcts = {cfg['eval']['sims']} simulations. "
             "Elo: Bradley-Terry vs the fixed ladder anchors, random = 0. "
             f"strong here: {cfg['eval'].get('params', {}).get('strong', {})}.", "",
             "Anchor Elo (from the 200-game ladder in docs/results.md): "
             + ", ".join(f"{a} {anchors[a]:.0f}" for a in ANCHORS) + ".", "",
             "| gen | gating % | " + " | ".join(f"raw vs {o}" for o in opps) + " | raw Elo | "
             + " | ".join(f"mcts vs {o}" for o in opps) + " | mcts Elo |",
             "|---:|---:|" + "---:|" * (2 * len(opps) + 2)]
    hist = {h["gen"]: h for h in state["history"]}
    curve = []
    for e in evals:
        row = [str(e["gen"]), f"{100 * hist[e['gen']]['score']:.0f}"]
        elo = {}
        for mode in ("raw", "mcts"):
            r = e[mode]
            row += [f"{100 * (r[o][0] + r[o][2] / 2) / sum(r[o]):.0f}" for o in opps]
            elo[mode] = fit_one(r, anchors)
            row.append(f"{elo[mode]:.0f}")
        lines.append("| " + " | ".join(row) + " |")
        curve.append((e["gen"], e, elo))
    (ROOT / f"docs/{name}_ladder.md").write_text("\n".join(lines) + "\n")
    if curve:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        gens = [c[0] for c in curve]
        colors = dict(zip(opps, ("#8c8c8c", "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3")))
        for ax, mode in zip(axes[:2], ("raw", "mcts")):
            for o in opps:
                ax.plot(gens, [100 * (c[1][mode][o][0] + c[1][mode][o][2] / 2) / sum(c[1][mode][o]) for c in curve],
                        marker="o", color=colors.get(o), label=o)
            ax.axhline(50, color="grey", lw=0.8, ls="--")
            ax.set_ylim(-3, 103)
            ax.set_xlabel("generation")
            ax.set_ylabel("net win %")
            ax.set_title(f"{mode}: net {'without search' if mode == 'raw' else str(cfg['eval']['sims']) + ' sims'}")
            ax.legend(loc="lower right", fontsize=8)
        ax = axes[2]
        for mode, ls in (("raw", "--"), ("mcts", "-")):
            ax.plot(gens, [c[2][mode] for c in curve], marker="o", ls=ls, color="black", label=f"net {mode}")
        for o in opps:
            ax.axhline(anchors[o], color=colors.get(o), lw=0.8, ls=":")
            ax.text(gens[0], anchors[o], o, color=colors.get(o), fontsize=8, va="bottom")
        ax.set_xlabel("generation")
        ax.set_ylabel("Elo (random = 0)")
        ax.set_title("Elo on the ladder")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(ROOT / f"docs/{name}_ladder.png", dpi=110)
    return curve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/az5.yaml"))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.report:
        report(cfg)
        return
    import torch

    from agents.az_net import Evaluator, make_net

    torch.set_num_threads(2)
    run = ROOT / cfg["run_dir"]
    (run / "eval").mkdir(exist_ok=True)
    ec = cfg["eval"]
    ctx = get_context("spawn")
    with ctx.Pool(ec["processes"], initializer=_oinit, initargs=(ec.get("params") or {}, cfg["seed"])) as pool:
        while True:
            state = json.loads((run / "state.json").read_text())
            todo = [h["gen"] for h in state["history"]
                    if h["accepted"] and not (run / "eval" / f"gen_{h['gen']:04d}.json").exists()]
            if not todo:
                break
            gen = todo[0]
            net = make_net(cfg)
            net.load_state_dict(torch.load(run / f"gen_{gen:04d}.pt", map_location="cpu"))
            ev = Evaluator(net, cfg["device"])
            rng = np.random.default_rng([cfg["seed"], gen, 7])
            out = {"gen": gen}
            for mode in ("raw", "mcts"):
                t0 = time.time()
                out[mode] = play(ev, mode, ec["opponents"], cfg, pool, rng)
                out[f"{mode}_seconds"] = time.time() - t0
                print(f"[{time.strftime('%H:%M:%S')}] gen {gen} {mode}: "
                      + ", ".join(f"{o} {r[0]}-{r[1]}" for o, r in out[mode].items())
                      + f" ({out[f'{mode}_seconds']:.0f} s)", flush=True)
            p = run / "eval" / f"gen_{gen:04d}.json"
            p.with_suffix(".tmp").write_text(json.dumps(out, indent=1))
            os.replace(p.with_suffix(".tmp"), p)
            report(cfg)
            # policy / value / PV snapshots of the fixed probe set (docs/probes)
            subprocess.run([sys.executable, str(ROOT / "scripts/az_probe_maps.py"), "--config", a.config])


if __name__ == "__main__":
    main()
