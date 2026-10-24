"""Policy / value / MCTS snapshots of a fixed probe set, per accepted generation.

For an animation of how the net discovers the long chain rule: a fixed set
of positions from the strategy probe (``maps`` in configs/strategy_probe.yaml)
is evaluated by every accepted generation of a run, and saved to
``{out}/{run}/gen_NN.json`` and ``.png``.

select   Picks the set once into ``{out}/positions.json`` (never changed
         afterwards): ``n_parity`` parity and ``n_dd`` double-dealing
         positions, only those where at most ``max_correct_share`` of the
         legal moves are correct (no near-free ones), spread evenly over
         the probe file in its order.
run      (default) Every accepted generation of ``--config`` without a
         snapshot. Per position: the policy over the legal lines, the value
         head (and score head) for the side to move, MCTS (``sims``, no
         noise, temperature 0, as in the probe) visits and Q per move, the
         chosen move and the principal variation (most visited child, down
         to ``pv_len`` lines). PNG: one small board per position, undrawn
         lines coloured by the policy (fixed scale over all generations),
         correct moves ringed, the MCTS move starred, the PV numbered.
         Called by scripts/az_eval.py after each evaluated generation.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from _common import ROOT, load_config

from sim.board import Board


def select(pc):
    mc = pc["maps"]
    d = json.loads((ROOT / pc["positions"]["out"]).read_text())
    picked = []
    for cat, n in (("parity", mc["n_parity"]), ("dd", mc["n_dd"])):
        pool = [i for i, p in enumerate(d["positions"]) if p["category"] == cat
                and len(p["correct"]) <= mc["max_correct_share"] * len(p["values"])]
        picked += [pool[int(k)] for k in np.linspace(0, len(pool) - 1, n)]
    out = ROOT / mc["out"] / "positions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    sel = [dict(probe_index=i, **d["positions"][i]) for i in picked]
    out.write_text(json.dumps({"board": d["board"], "source": pc["positions"]["out"], "positions": sel}, indent=1))
    print(f"{len(sel)} positions -> {out}")


def principal_variation(root, n):
    pv, node = [], root
    while node is not None and node.expanded and node.N.sum() > 0 and len(pv) < n:
        k = int(np.argmax(node.N))
        pv.append({"line": int(node.moves[k]), "player": int(node.player)})
        node = node.children[k]
    return pv


def snapshot(rc, pc, gen):
    import torch

    from agents.az_mcts import choose_move
    from agents.az_net import Evaluator, make_net
    from agents.az_play import make_mcts

    mc = pc["maps"]
    sel = json.loads((ROOT / mc["out"] / "positions.json").read_text())
    boards = []
    for p in sel["positions"]:
        b = Board(*sel["board"])
        for ln in p["moves"]:
            b.step(ln)
        boards.append(b)
    net = make_net(rc)
    net.load_state_dict(torch.load(ROOT / rc["run_dir"] / f"gen_{gen:04d}.pt", map_location="cpu"))
    ev = Evaluator(net, pc["device"])
    pri, val = ev(boards)
    score = [None] * len(boards)
    if net.s_fc is not None:
        with torch.no_grad():
            x = torch.from_numpy(ev.codec.encode_boards(boards)).to(ev.device)
            score = (net(x)[2].float().cpu().numpy() * boards[0].geo.n_boxes).tolist()
    rng = np.random.default_rng(pc["seed"])
    roots = make_mcts(ev, rc["mcts"], rng).search(boards, mc["sims"])
    out = []
    for p, b, pr, v, s, root in zip(sel["positions"], boards, pri, val, score, roots):
        legal = [ln for ln in range(b.geo.n_lines) if not b.lines[ln]]
        raw = int(np.argmax(pr))
        mv = choose_move(root, 0.0, rng)
        n = root.N.sum()
        out.append({
            "probe_index": p["probe_index"], "category": p["category"], "player": p["player"],
            "score": p["score"],
            "owner_rel": [0 if o < 0 else 1 if o == b.player else -1 for o in b.owner], "correct": p["correct"], "values": p["values"],
            "policy": {ln: round(float(pr[ln]), 4) for ln in legal},
            "value": round(float(v), 4), "score_pred": None if s is None else round(s, 2),
            "mcts": {"sims": mc["sims"],
                     "visits": {int(m): int(c) for m, c in zip(root.moves, root.N)},
                     "q": {int(m): round(float(w / c), 4) for m, w, c in zip(root.moves, root.W, root.N) if c},
                     "root_q": round(float(root.W.sum() / n), 4) if n else None,
                     "move": mv, "pv": principal_variation(root, mc["pv_len"])},
            "raw_move": raw, "raw_correct": raw in p["correct"], "mcts_correct": mv in p["correct"],
        })
    return {"run": rc["run_dir"].split("/")[-1], "gen": gen, "board": sel["board"], "positions": out}


def plot(snap, path, pc):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    from agents.az_net import Codec

    w, h = snap["board"]
    codec = Codec(w, h)
    cmap = plt.get_cmap("Blues")
    norm = PowerNorm(0.5, 0, 1)  # fixed over generations, sqrt so small priors show
    pos = snap["positions"]
    ncol = 5
    nrow = -(-len(pos) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.65 * nrow + 0.5))
    for ax, p in zip(axes.flat, pos):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-0.6, 2 * w + 0.6)
        ax.set_ylim(2 * h + 0.6, -0.6)
        p_lines = {int(k): v for k, v in p["policy"].items()}
        for cell, o in zip(codec.box_cell, p["owner_rel"]):
            r, c = divmod(int(cell), codec.cols)
            if o:
                ax.add_patch(plt.Rectangle((c - 1, r - 1), 2, 2, color="#cfe3c9" if o > 0 else "#f0cccc", lw=0))
        for ln, (r, c) in enumerate(codec.line_rc):
            xy = ([c - 1, c + 1], [r, r]) if r % 2 == 0 else ([c, c], [r - 1, r + 1])
            if ln in p_lines:
                ax.plot(*xy, color=cmap(norm(p_lines[ln])), lw=3.2, solid_capstyle="butt", zorder=2)
            else:
                ax.plot(*xy, color="#333333", lw=1.2, zorder=3)
        corr = set(p["correct"])
        for ln in corr:
            r, c = codec.line_rc[ln]
            ax.plot(c, r, "o", ms=9, mfc="none", mec="#1a7f37", mew=1.4, zorder=4)
        r, c = codec.line_rc[p["mcts"]["move"]]
        ax.plot(c, r, "*", ms=9, color="#d4a017" if p["mcts_correct"] else "#c62828", mec="black", mew=0.4,
                zorder=5)
        for k, step in enumerate(p["mcts"]["pv"][1:5], start=2):
            r, c = codec.line_rc[step["line"]]
            ax.text(c, r, str(k), fontsize=6, ha="center", va="center", zorder=6,
                    color="white", bbox=dict(boxstyle="circle,pad=0.12", fc="#555555", lw=0))
        ax.plot(*np.meshgrid(np.arange(0, 2 * w + 1, 2), np.arange(0, 2 * h + 1, 2)), "k.", ms=3, zorder=4)
        mark = lambda ok: "+" if ok else "-"
        ax.set_title(f"#{p['probe_index']} {p['category']}  V {p['value']:+.2f}\n"
                     f"raw {mark(p['raw_correct'])}  mcts {mark(p['mcts_correct'])}  "
                     f"{p['score'][p['player']]}:{p['score'][1 - p['player']]}", fontsize=7)
    for ax in axes.flat[len(pos):]:
        ax.axis("off")
    n_raw = sum(p["raw_correct"] for p in pos)
    n_mc = sum(p["mcts_correct"] for p in pos)
    fig.suptitle(f"{snap['run']} gen {snap['gen']}: correct raw {n_raw}/{len(pos)}, MCTS {n_mc}/{len(pos)}   "
                 "(lines: policy; ring: correct; star: MCTS move; numbers: PV)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="run", choices=["select", "run"])
    ap.add_argument("--config", default=str(ROOT / "configs/az5b.yaml"), help="run config")
    ap.add_argument("--probe-config", default=str(ROOT / "configs/strategy_probe.yaml"))
    a = ap.parse_args()
    pc = load_config(a.probe_config)
    if a.cmd == "select":
        select(pc)
        return
    rc = load_config(a.config)
    run = ROOT / rc["run_dir"]
    out = ROOT / pc["maps"]["out"] / run.name
    out.mkdir(parents=True, exist_ok=True)
    st = json.loads((run / "state.json").read_text())
    for g in [h["gen"] for h in st["history"] if h["accepted"]]:
        if (out / f"gen_{g:02d}.json").exists() or not (run / f"gen_{g:04d}.pt").exists():
            continue
        snap = snapshot(rc, pc, g)
        plot(snap, out / f"gen_{g:02d}.png", pc)
        tmp = out / f"gen_{g:02d}.tmp"
        tmp.write_text(json.dumps(snap, indent=1))
        os.replace(tmp, out / f"gen_{g:02d}.json")
        pos = snap["positions"]
        print(f"{run.name} gen {g}: raw {sum(p['raw_correct'] for p in pos)}, "
              f"mcts {sum(p['mcts_correct'] for p in pos)} / {len(pos)}", flush=True)


if __name__ == "__main__":
    main()
