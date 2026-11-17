"""Part 4, series 7 of docs/plan_part4.md: does the jump carry over to the middle game. Evaluation only.

    python scripts/graph_s7.py overlap   # mid / chart positions against the nets' training and selection positions
    python scripts/graph_s7.py eval      # moves of every net and of Dot-2 without thinking on mid and chart
    python scripts/graph_s7.py report    # docs/graph_s7.md, docs/who_knows_s7.png

configs/graph_s7.yaml. Sets: the labels of part 3 (c) (runs/mid_judge/labels_{mid,chart}.jsonl), not relabelled.
A move is correct when it keeps the best result (scripts/mid_judge_eval.py correct).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np

from _common import ROOT, load_config

import graph_s3
import parity_hand
from graph_hand import correct_rate, judge_sets, probe_sets
from graph_s2 import splits
from graph_s3p_checks import canon

GROUPS = {"gc_jump": "Gc with a jump", "gc_nojump": "Gc without a jump", "ch": "C+h (series 3')"}


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def mj_out(cfg):
    return ROOT / load_config(ROOT / cfg["mid_judge"])["out"]


def best_moves(r):
    lab = {int(k): v for k, v in r["labels"].items()}
    m = max(lab.values())
    return sorted(k for k, v in lab.items() if v == m)


def decisive(r):
    return len(set(r["labels"].values())) > 1


def load_sets(cfg):
    """{set: [record]}: solved positions in the set's safe-line range, file order."""
    out = {}
    for name, sc in cfg["sets"].items():
        lo, hi = sc["safe"]
        rs = [r for r in read_jsonl(mj_out(cfg) / f"labels_{name}.jsonl") if r["labels"] is not None]
        out[name] = [dict(r, correct=best_moves(r)) for r in rs if lo <= r["key"][0] <= hi]
    return out


def net_list(cfg):
    """[(group, name, config, arm, weights path)]."""
    L = []
    for g, gc in cfg["groups"].items():
        sub = load_config(ROOT / gc["config"])
        if "runs" in gc:
            for w, t in gc["runs"]:
                L.append((g, f"Gc w{w}/{t}", sub, gc["arm"], ROOT / sub["out"] / f"w{w}" / f"s2_{gc['arm']}_s{t}.pt"))
        else:
            for s in gc["seeds"]:
                L.append((g, f"C+h s{s}", sub, gc["arm"], ROOT / sub["out"] / f"s2_{gc['arm']}_s{s}.pt"))
    return L


def limit_gpu(cfg):
    import torch

    if cfg["device"].startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(cfg["limits"]["gpu_memory_fraction"])


# --- overlap -------------------------------------------------------------------------

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def cmd_overlap(cfg):
    from agents.az_net import Codec

    planes = 17
    codec = Codec(10, 7, planes=planes)
    sets = load_sets(cfg)
    ours = {k: parity_hand.parity_arrays("10x7", v, planes)[1] for k, v in sets.items()}
    theirs, notes = {}, []
    for cp in cfg["overlap"]["configs"]:
        c = load_config(ROOT / cp)
        tag = c["out"].split("/")[-1]
        z = np.load(ROOT / c["out"] / "data.npz")
        notes.append(f"{c['out']}/data.npz sha256 {sha(ROOT / c['out'] / 'data.npz')}")
        s0 = splits(c, {k: z[k] for k in z.files})["10x7"]
        _, sp = judge_sets(c)
        tr, va = sp["10x7"]
        theirs[f"{tag}: stage 0 train"] = s0[0][:, :planes]
        theirs[f"{tag}: stage 0 validation"] = s0[2][:, :planes]
        theirs[f"{tag}: judge train"] = parity_hand.parity_arrays("10x7", tr, planes)[1]
        theirs[f"{tag}: judge validation"] = parity_hand.parity_arrays("10x7", va, planes)[1]
        theirs[f"{tag}: judge probe"] = probe_sets(c, planes)["10x7"][1]
    res = {"notes": notes, "tables": {}}
    for lo, title in ((True, "drawn lines only"), (False, "full encoding (17 planes)")):
        ko = {k: canon(v.reshape(len(v), planes, -1), codec, lo) for k, v in ours.items()}
        kt = {k: set(canon(np.asarray(v).reshape(len(v), planes, -1), codec, lo)) for k, v in theirs.items()}
        tab = {}
        for a, keys in ko.items():
            tab[a] = {"distinct": len(set(keys)),
                      "shared": {b: len(set(keys) & s) for b, s in kt.items()},
                      "shared_positions": [i for i, k in enumerate(keys) if any(k in s for s in kt.values())]}
        res["tables"][title] = {"theirs_distinct": {b: len(s) for b, s in kt.items()}, "rows": tab}
    out = ROOT / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "overlap.json").write_text(json.dumps(res, indent=1))
    for title, t in res["tables"].items():
        print(title, {a: (r["distinct"], sum(r["shared"].values())) for a, r in t["rows"].items()})


# --- eval ----------------------------------------------------------------------------

def cmd_eval(cfg):
    import torch

    limit_gpu(cfg)
    dev = cfg["device"]
    sets = load_sets(cfg)
    res = {"keys": {k: [r["key"] for r in v] for k, v in sets.items()}, "nets": {}, "dot2": {}}
    for g, name, sub, arm, path in net_list(cfg):
        net, zero, _ = graph_s3.build_net(sub, arm)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.to(dev).eval()
        res["nets"][name] = {"group": g, "weights": str(path.relative_to(ROOT)), "sha256": sha(path)}
        for k, v in sets.items():
            codec, x, mask = parity_hand.parity_arrays("10x7", v, net.planes)
            res["nets"][name][k] = correct_rate(net, codec, graph_s3.zeroed(x, zero), mask, dev).astype(int).tolist()
        print(f"[{time.strftime('%H:%M:%S')}] {name}: " + ", ".join(
            f"{k} {100 * np.mean(res['nets'][name][k]):.1f}" for k in sets), flush=True)
        del net
        torch.cuda.empty_cache()
    d = cfg["dot2"]
    ev, _ = parity_hand.evaluator({"device": dev}, (d["config"], ROOT / d["checkpoint"]))
    from sim.board import Board

    for k, v in sets.items():
        moves = []
        for i in range(0, len(v), 256):
            bs = []
            for r in v[i:i + 256]:
                b = Board(10, 7)
                for ln in r["moves"]:
                    b.step(ln)
                bs.append(b)
            moves += [int(np.argmax(p)) for p in ev(bs)[0]]
        stored = json.loads((mj_out(cfg) / f"nets_{k}.json").read_text())
        sm = {tuple(kk): i for i, kk in enumerate(stored["keys"])}
        old = {m: [stored["moves"][m][sm[tuple(r["key"])]] if tuple(r["key"]) in sm else None for r in v]
               for m in stored["moves"]}
        res["dot2"][k] = {"hand": moves, "hand_correct": [int(m in r["correct"]) for m, r in zip(moves, v)],
                          "stored_hand_equal": sum(a == b for a, b in zip(moves, old["hand"])),
                          "stored": {m: [int(x is not None and x in r["correct"]) for x, r in zip(old[m], v)]
                                     for m in old}}
        print(f"Dot-2 {k}: {100 * np.mean(res['dot2'][k]['hand_correct']):.1f}, equal to the stored hand "
              f"{res['dot2'][k]['stored_hand_equal']}/{len(v)}", flush=True)
    (ROOT / cfg["out"] / "eval.json").write_text(json.dumps(res))


# --- report --------------------------------------------------------------------------

def interval(a, b, n, seed, level):
    """a: (nets, positions) 0/1, b: (positions,) 0/1 on the same positions. Positions with replacement (the same
    draw for both), nets with replacement. Returns (observed mean(a) - mean(b), lower, upper) in points,
    two-sided percentile interval."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for i in range(n):
        pos = rng.integers(0, a.shape[1], a.shape[1])
        k = rng.integers(0, len(a), len(a))
        d[i] = a[k][:, pos].mean() - b[pos].mean()
    q = (1 - level) / 2
    return 100 * (a.mean() - b.mean()), 100 * float(np.quantile(d, q)), 100 * float(np.quantile(d, 1 - q))


def reading(d, lo, hi):
    if lo > 0:
        return "the jump carries over to the middle game"
    if hi < 0:
        return "no transfer; Gc with a jump is below Dot-2 without thinking"
    return "no transfer to the middle game"


def pct(x):
    return f"{100 * float(np.mean(x)):.1f}" if len(x) else "—"


def cmd_report(cfg):
    sets = load_sets(cfg)
    ev = json.loads((ROOT / cfg["out"] / "eval.json").read_text())
    ov = json.loads((ROOT / cfg["out"] / "overlap.json").read_text())
    for k, v in sets.items():
        assert [r["key"] for r in v] == ev["keys"][k], "sets changed since eval"
    dec = {k: np.array([decisive(r) for r in v]) for k, v in sets.items()}
    safe = {k: np.array([r["key"][0] for r in v]) for k, v in sets.items()}
    nets = ev["nets"]
    grp = {g: [n for n in nets if nets[n]["group"] == g] for g in GROUPS}
    A = {k: {g: np.array([nets[n][k] for n in grp[g]]) for g in GROUPS} for k in sets}
    D2 = {k: np.array(ev["dot2"][k]["hand_correct"]) for k in sets}
    ST = {k: {m: np.array(x) for m, x in ev["dot2"][k]["stored"].items()} for k in sets}
    bc = cfg["bootstrap"]
    m = dec["mid"]
    rows = {g: interval(A["mid"][g][:, m], D2["mid"][m], bc["resamples"], bc["seed"], bc["level"]) for g in GROUPS}
    d, lo, hi = rows["gc_jump"]
    nm, nc = len(sets["mid"]), len(sets["chart"])
    L = ["# Part 4, series 7: does the jump carry over to the middle game (evaluation only)", "",
         f"Generated by `scripts/graph_s7.py report` ({time.strftime('%Y-%m-%d %H:%M')}). Protocol: docs/plan_part4.md "
         "(series 7), configs/graph_s7.yaml. No training. Nets: best point weights. Gc and C+h move: policy argmax "
         "over legal lines; Dot-2 without thinking: its policy argmax (dot2c gen 16), recomputed here. A move is "
         "correct when it keeps the best result of the judge's labels (part 3 (c), runs/mid_judge, not relabelled).",
         "",
         f"Sets: **mid** = {nm} solved positions at 10-15 safe lines, {int(m.sum())} decisive (primary); **chart** = "
         f"{nc} solved positions at 0-18 safe lines, {int(dec['chart'].sum())} decisive (descriptive).", "",
         "## Primary and reading", "",
         f"D = mean of the 9 Gc nets with a jump - Dot-2 without thinking, decisive mid positions: "
         f"**{d:+.1f} points**, 95% interval [{lo:+.1f}, {hi:+.1f}] (bootstrap over positions and nets, "
         f"{bc['resamples']} resamples, seed {bc['seed']}).", "",
         f"Row of the protocol: **{reading(d, lo, hi)}**.", "",
         "| nets | n | mean on decisive mid % | Dot-2 without thinking % | difference | 95% interval |",
         "|---|---:|---:|---:|---:|---:|"]
    for g, lab in GROUPS.items():
        dd, l2, h2 = rows[g]
        L.append(f"| {lab}{' (primary)' if g == 'gc_jump' else ''} | {len(grp[g])} | {pct(A['mid'][g][:, m])} | "
                 f"{pct(D2['mid'][m])} | {dd:+.1f} | [{l2:+.1f}, {h2:+.1f}] |")
    L += ["", f"Reference, Dot-2 on the same {int(m.sum())} decisive positions from the stored moves of "
          f"runs/mid_judge/nets_mid.json (not rerun): intuition {pct(ST['mid']['intuition'][m])}%, MCTS-200 "
          f"{pct(ST['mid']['mcts200'][m])}%, MCTS-2000 {pct(ST['mid']['mcts2000'][m])}%. The protocol quotes the "
          f"user's {cfg['reference']['mcts200']} / {cfg['reference']['intuition']}%; docs/mid_judge.md has 95.6 / 91.0 "
          "on these positions.", "",
          f"Recomputed Dot-2 moves equal the stored hand moves of part 3: mid "
          f"{ev['dot2']['mid']['stored_hand_equal']}/{nm}, chart {ev['dot2']['chart']['stored_hand_equal']}/{nc}.", ""]
    # overlap
    L += ["## Overlap with the nets' training and selection positions", "",
          "mid and chart positions against the judge's 10x7 train / validation / probe and the stage 0 10x7 train / "
          "validation of series 6 (Gc) and series 3' (C+h); key: minimum over the 4 grid symmetries. Cells: shared "
          "distinct positions. 5x5 positions cannot coincide with 10x7 ones.", ""]
    for title, t in ov["tables"].items():
        cols = list(t["theirs_distinct"])
        L += [f"Key: {title}.", "", "| | " + " | ".join(f"{c} ({t['theirs_distinct'][c]})" for c in cols) + " |",
              "|---|" + "---:|" * len(cols)]
        for a, r in t["rows"].items():
            L.append(f"| {a} ({r['distinct']} distinct) | " + " | ".join(str(r["shared"][c]) for c in cols) + " |")
        L.append("")
    shared = {a: set(r["shared_positions"]) for a, r in ov["tables"]["drawn lines only"]["rows"].items()}
    L += [f"Data files: {'; '.join(ov['notes'])}.", ""]
    if shared["mid"]:
        keep = np.array([i not in shared["mid"] for i in range(nm)]) & m
        d3, l3, h3 = interval(A["mid"]["gc_jump"][:, keep], D2["mid"][keep], bc["resamples"], bc["seed"], bc["level"])
        L += [f"Shared mid positions: {len(shared['mid'])}. Without them ({int(keep.sum())} decisive): D {d3:+.1f}, "
              f"[{l3:+.1f}, {h3:+.1f}] -> {reading(d3, l3, h3)}.", ""]
    else:
        L += ["No mid position is shared with any training or selection position (by drawn lines, so not by the full "
              "encoding either)." if not any(shared.values()) else
              f"No mid position is shared; chart positions shared (drawn lines): {len(shared['chart'])}.", ""]
    L += ["Dot-2 is not in this check: the mid and chart positions come from its own self-play games (MCTS-200, "
          "part 3 (c)), so they are from its own distribution, which the Gc and C+h nets never trained on.", ""]
    # nets table
    L += ["## Table 1. Nets", "",
          "| group | net | mid all % | mid decisive % | chart decisive % (0-18) | weights sha256 |",
          "|---|---|---:|---:|---:|---|"]
    for n, r in nets.items():
        L.append(f"| {GROUPS[r['group']]} | {n} | {pct(r['mid'])} | {pct(np.array(r['mid'])[m])} | "
                 f"{pct(np.array(r['chart'])[dec['chart']])} | {r['sha256']} |")
    L.append(f"| Dot-2 | without thinking | {pct(D2['mid'])} | {pct(D2['mid'][m])} | {pct(D2['chart'][dec['chart']])} | |")
    L.append("")
    # by safe lines
    for k, title in (("mid", "Table 2. mid, decisive positions by safe lines left"),
                     ("chart", "Table 3. chart, decisive positions by safe lines left")):
        L += [f"## {title}", "", "% correct moves; group = mean over its nets. Dot-2 intuition / MCTS-200 from the "
              "stored moves of part 3.", "",
              "| safe lines | n | Gc with a jump | Gc without a jump | C+h | Dot-2 without thinking | intuition | MCTS-200 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for s in sorted(set(safe[k]), reverse=True):
            sel = dec[k] & (safe[k] == s)
            L.append(f"| {s} | {int(sel.sum())} | " + " | ".join(pct(A[k][g][:, sel]) for g in GROUPS) +
                     f" | {pct(D2[k][sel])} | {pct(ST[k]['intuition'][sel])} | {pct(ST[k]['mcts200'][sel])} |")
        L.append("")
    chart(cfg, sets, dec, safe, A, D2, ST)
    L += ["## Who knows what", "", f"![who knows what, with Gc]({cfg['chart_png'].split('/')[-1]})", "",
          "The lines of docs/who_knows.png (part 3, stored moves; the judge's solved share from all chart positions) "
          f"plus Gc with a jump (mean of the 9 nets). A point where a bucket has >= {cfg['sets']['chart']['min_bucket']} "
          "decisive positions.", ""]
    (ROOT / cfg["report"]).write_text("\n".join(L))
    print("\n".join(L))


def chart(cfg, sets, dec, safe, A, D2, ST):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k, mb = "chart", cfg["sets"]["chart"]["min_bucket"]
    oth = {tuple(r["key"]): r for r in json.loads((mj_out(cfg) / "others_chart.json").read_text())}
    formula = np.array([int(tuple(r["key"]) in oth and oth[tuple(r["key"])]["formula"] in r["correct"])
                        for r in sets[k]])
    series = [(D2[k], "Dot-2 without thinking (policy argmax)", "#2a78d6", "-"),
              (ST[k]["intuition"], "intuition (sign of child V)", "#1baf7a", "-"),
              (ST[k]["mcts200"], "MCTS-200", "#eda100", "-"),
              (ST[k]["mcts2000"], "MCTS-2000", "#eb6834", "-"),
              (formula, "formula judge (+1 ply)", "#6250d6", "-"),
              (A[k]["gc_jump"].mean(0), "Gc with a jump (mean of 9 nets)", "#d62a6e", "-")]
    s_all = list(range(0, 21))
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for y, lab, col, ls in series:
        xs, ys = [], []
        for s in s_all:
            sel = dec[k] & (safe[k] == s)
            if sel.sum() >= mb:
                xs.append(s)
                ys.append(100 * float(np.mean(y[sel])))
        ax.plot(xs, ys, marker="o", ms=4, lw=3 if "Gc" in lab else 2, ls=ls, label=lab, color=col)
    allr = read_jsonl(mj_out(cfg) / "labels_chart.jsonl")
    xs, ys = [], []
    for s in s_all:
        a = [r for r in allr if r["key"][0] == s]
        if a:
            xs.append(s)
            ys.append(100 * sum(r["labels"] is not None for r in a) / len(a))
    tl = load_config(ROOT / cfg["mid_judge"])["label"]["chart"]["time_limit"]
    ax.plot(xs, ys, ls="--", marker="s", ms=4, lw=2, color="#444444", label=f"middle-game judge: % positions solved ({tl} s)")
    ax.set_xlim(-0.5, 20.5)
    ax.set_xticks(s_all)
    ax.set_ylim(0, 102)
    ax.set_xlabel("safe lines left")
    ax.set_ylabel("% correct moves (decisive positions) / % solved")
    ax.set_title("10x7, Dot-2 self-play positions: who knows the right move (with the graph net)")
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / cfg["chart_png"], dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["overlap", "eval", "report"])
    ap.add_argument("--config", default="configs/graph_s7.yaml")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    {"overlap": cmd_overlap, "eval": cmd_eval, "report": cmd_report}[a.cmd](cfg)


if __name__ == "__main__":
    main()
