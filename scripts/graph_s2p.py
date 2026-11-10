"""Part 4, series 2' of docs/plan_part4.md: what gets in the way of transfer.

    python scripts/graph_s2p.py run           # g (2b reproduced), b, a, cC, cCh; finished runs skipped
    python scripts/graph_s2p.py commit-preds  # waits for the 10 first-loss predictions of b, commits them
    python scripts/graph_s2p.py analyse       # (g): sign inversion on 5x5 validation -> runs/graph_s2p/g.json
    python scripts/graph_s2p.py report        # docs/graph_s2p.md

configs/graph_s2p.yaml. Training is the run loop of scripts/graph_s2.py.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time

import numpy as np

from _common import ROOT, load_config

import graph_s2
from graph_s2 import SIZES, fisher, fmt, load, splits


def cmd_run(cfg):
    graph_s2.cmd_run(cfg)


def cmd_commit_preds(cfg):
    out, seeds = ROOT / cfg["out"], cfg["parts"]["b"]["seeds"]
    files = [out / f"b_Gc_s{s}.pred.json" for s in seeds]
    while not all(f.exists() for f in files):
        time.sleep(60)
    preds = [json.loads(f.read_text()) for f in files]
    done = [s for s in seeds if (out / f"b_Gc_s{s}.json").exists()]
    doc = {"threshold": cfg["parts"]["b"]["predict"], "runs_finished_when_written": done, "predictions": preds}
    path = ROOT / "docs/graph_s2p_predictions.json"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    msg = ("Series 2' (b): first-loss predictions, written before the outcomes\n\n"
           f"Runs finished at this point: {done or 'none'}.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    subprocess.run(["git", "add", str(path)], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-m", msg, "--", str(path)], cwd=ROOT, check=True)
    print("predictions committed", done, flush=True)


# ---------------------------------------------------------------- (g)

def features(b):
    from sim.chains import analyse

    a = analyse(b)
    ch, lo = a["chains"], a["loops"]
    lng = [c for c in ch if c >= 3]
    return {"player": b.player, "long": len(lng), "long_len": sum(lng), "loops": len(lo), "chains": len(ch),
            "chains2": sum(c >= 2 for c in ch), "short": sum(c < 3 for c in ch)}


def bin_long(n):
    return "3+" if n >= 3 else str(n)


def bin_len(n):
    return "0" if n == 0 else ("3-5" if n <= 5 else ("6-8" if n <= 8 else ("9-11" if n <= 11 else "12+")))


def bin_loops(n):
    return "2+" if n >= 2 else str(n)


BINS = {"long": (bin_long, ["0", "1", "2", "3+"]), "long_len": (bin_len, ["0", "3-5", "6-8", "9-11", "12+"]),
        "loops": (bin_loops, ["0", "1", "2+"])}
# candidate rules: +1 iff (count % 2) == player, as the true rule is with count = long
RULES = ["long", "chains2", "chains", "short", "loops", "long+loops", "chains+loops"]


def rule_count(f, r):
    return sum(f[k] for k in r.split("+"))


def cmd_analyse(cfg):
    import torch

    from agents.az_net import Codec, make_net
    from graph_hand import settled_boards

    ac = cfg["analysis"]
    s2 = load_config(ROOT / "configs/graph_s2.yaml")
    old = {r["seed"]: r for r in load(s2, "2b", "Gc", s2["parts"]["2b"]["seeds"])}
    seeds = [s for s in cfg["parts"]["g"]["seeds"]]
    out = ROOT / cfg["out"]
    rng = random.Random(cfg["data_seed"])
    nt, nv = cfg["n_train"], cfg["n_val"]
    boards = [b for b, _ in settled_boards(5, 5, nt + nv, rng)][nt:]
    z = np.load(out / "data.npz")
    x, y = splits(cfg, {k: z[k] for k in z.files})["5x5"][2:]
    assert np.array_equal(Codec(5, 5, planes=17).encode_boards(boards[:200]), x[:200]), "5x5 val boards not reproduced"
    feats = [features(b) for b in boards]
    target = np.array([1.0 if (f["long"] % 2) == f["player"] else -1.0 for f in feats])
    assert np.array_equal(target, y), "target != long parity rule"
    res = {"n": len(boards), "seeds": {}, "features": feats}
    for s in seeds:
        r = json.loads((out / f"g_Gc_s{s}.json").read_text())
        same = all(a["val"]["5x5"]["acc"] == b["val"]["5x5"]["acc"] and a["val"]["10x7"]["acc"] == b["val"]["10x7"]["acc"]
                   and a.get("loss") == b.get("loss") for a, b in zip(r["log"], old[s]["log"])) and len(r["log"]) == len(old[s]["log"])
        gc = load_config(ROOT / cfg["arms"]["Gc"]["config"])
        gc["net"] = {**gc["net"], "v_out": cfg["recipe"]["v_out"], "readout_norm": False, "readout_init": None}
        net = make_net(gc)
        net.load_state_dict(torch.load(out / f"g_Gc_s{s}.pt", map_location="cpu"))
        net.eval()
        pre = []
        with torch.no_grad():
            for i in range(0, len(x), 500):
                net(torch.from_numpy(x[i:i + 500, :net.planes]))
                pre.append(net.v_pre.numpy())
        pre = np.concatenate(pre)
        res["seeds"][s] = {"reproduced": same, "last_5x5_series2": old[s]["last_5x5"], "pred": (pre > 0).astype(int).tolist(),
                           "acc": 100 * float(np.mean((pre > 0) == (y > 0)))}
        print(f"s{s}: reproduced {same}, 5x5 acc {res['seeds'][s]['acc']:.1f} (series 2: {old[s]['last_5x5']})", flush=True)
    res["target"] = (y > 0).astype(int).tolist()
    (out / "g.json").write_text(json.dumps(res))


def g_tables(cfg):
    out = ROOT / cfg["out"]
    p = out / "g.json"
    if not p.exists():
        return ["(g) not analysed yet.", ""]
    g = json.loads(p.read_text())
    ac, feats, t = cfg["analysis"], g["features"], np.array(g["target"])
    seeds = [int(s) for s in g["seeds"]]
    low = [s for s in seeds if g["seeds"][str(s)]["last_5x5_series2"] < ac["low_5x5"]]
    show = low + [ac["reference_seed"]]
    L = [f"Seeds with last 5x5 < {ac['low_5x5']}% in 2b: {low}; reference s{ac['reference_seed']}. "
         "Error % on the 2000 5x5 validation positions, by class (n = positions in the class).", "",
         "| run | reproduced bit-exact | 5x5 acc |", "|---|---|---:|"]
    for s in show:
        e = g["seeds"][str(s)]
        L.append(f"| s{s} | {e['reproduced']} | {e['acc']:.1f} |")
    L.append("")
    for key, (fn, labels) in BINS.items():
        cls = np.array([fn(f[key]) for f in feats])
        L += [f"Error % by {key}:", "", "| run | " + " | ".join(f"{l} (n={int((cls == l).sum())})" for l in labels) + " |",
              "|---|" + "---:|" * len(labels)]
        for s in show:
            pr = np.array(g["seeds"][str(s)]["pred"])
            L.append(f"| s{s} | " + " | ".join(f"{100 * np.mean(pr[cls == l] != t[cls == l]):.1f}" if (cls == l).any() else "-"
                                              for l in labels) + " |")
        L.append("")
    L += ["Agreement % of the net's sign with candidate rules: +1 iff (count % 2) == player; "
          "'long' is the true rule. 'chains2' counts chains of 2+, 'short' chains of 1-2. "
          "Also the player alone (+1 iff player 0). Below 50 = the inverted rule.", "",
          "| run | " + " | ".join(RULES) + " | player |", "|---|" + "---:|" * (len(RULES) + 1)]
    for s in show:
        pr = np.array(g["seeds"][str(s)]["pred"])
        cells = []
        for r in RULES:
            cand = np.array([int(rule_count(f, r) % 2 == f["player"]) for f in feats])
            cells.append(f"{100 * np.mean(cand == pr):.1f}")
        cand = np.array([int(f["player"] == 0) for f in feats])
        cells.append(f"{100 * np.mean(cand == pr):.1f}")
        L.append(f"| s{s} | " + " | ".join(cells) + " |")
    L.append("")
    return L


# ---------------------------------------------------------------- report

def run_rows(cfg, rs, mix):
    """Per run: switch, 10x7 click, hold 10x7 after the click, hold 5x5 after the mix, both at the end."""
    out = []
    for r in rs:
        log, h = r["log"], cfg["hold_acc"]
        sw = r["switch"]
        hold5 = None if sw is None else all(x["val"]["5x5"]["acc"] >= h for x in log if x["step"] >= sw + mix)
        last = log[-1]["val"]
        both = last["5x5"]["acc"] >= h and last["10x7"]["acc"] >= h
        out.append({**r, "hold5": hold5, "both": both})
    return out


def cmd_report(cfg):
    pc = cfg["parts"]
    L = ["# Part 4, series 2': what gets in the way of transfer", "",
         f"Protocol: docs/plan_part4.md (series 2'), configs/graph_s2p.yaml. Recipe of series 2, {cfg['steps']} steps, "
         f"validation every {cfg['log_every']}. Click: first validation >= {cfg['pass_acc']}%; hold 10x7: every validation "
         f"after the click >= {cfg['hold_acc']}%; hold 5x5: every validation from the end of the mix (from the switch "
         f"when there is no mix) >= {cfg['hold_acc']}%; both: the last validation >= {cfg['hold_acc']}% on both sizes.", "",
         "## Table 1. (g) What the 2b nets learned instead of parity on 5x5", ""] + g_tables(cfg)
    counts = {}
    titles = {"a": "Table 2. (a) Soft switch: mix of sizes for {m} steps after the event, then 10x7 only",
              "b": "Table 3. (b) Plain switch (as 2b), seeds 11-20, first-loss prediction",
              "cC": "Table 4a. (c) C, fixed switch at step {sa}", "cCh": "Table 4b. (c) C+h with plane 15 zeroed, event switch"}
    for part in ["a", "b", "cC", "cCh"]:
        p = pc[part]
        arm = p["arms"][0]
        rs = run_rows(cfg, load(cfg, part, arm, p["seeds"]), p.get("mix_steps", 0))
        n = len(p["seeds"])
        c = {k: sum(bool(r[k]) for r in rs) for k in ["click10", "hold10", "hold5", "both"]}
        c["click10"] = sum(r["click10"] is not None for r in rs)
        c["n"], c["done"] = n, len(rs)
        counts[part] = c
        L += [f"## {titles[part].format(m=p.get('mix_steps', 0), sa=p.get('switch_at'))}", "",
              f"{arm}, {c['done']}/{n} runs. 10x7 click {c['click10']}/{n}, hold 10x7 {c['hold10']}/{n}, "
              f"hold 5x5 {c['hold5']}/{n}, both at the end {c['both']}/{n}.", "",
              "| seed | 5x5 click | switch | 10x7 click | click - switch | hold 10x7 | hold 5x5 | both | best 10x7 | last 10x7 | last 5x5 |",
              "|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|"]
        for r in rs:
            dl = None if r["click10"] is None or r["switch"] is None else r["click10"] - r["switch"]
            L.append(f"| {r['seed']} | {fmt(r['click5'])} | {fmt(r['switch'])} | {fmt(r['click10'])} | {fmt(dl)} | "
                     f"{fmt(r['hold10'])} | {fmt(r['hold5'])} | {r['both']} | {r['best_10x7']:.1f} | {r['last_10x7']:.1f} | "
                     f"{r['last_5x5']:.1f} |")
        L.append("")
        if part == "b":
            L += b_prediction(cfg, rs)
    L += decide(cfg, counts)
    (ROOT / "docs/graph_s2p.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def b_prediction(cfg, rs):
    pp = ROOT / "docs/graph_s2p_predictions.json"
    if not pp.exists():
        return ["Predictions not committed yet.", ""]
    preds = {p["name"]: p for p in json.loads(pp.read_text())["predictions"]}
    fast = cfg["prediction_window"]
    L = [f"Prediction (threshold {cfg['parts']['b']['predict']}, committed before the outcomes): 'learns' if the mean "
         f"train loss of the first {cfg['log_every']} steps after the switch is below it. Outcome: 10x7 click within "
         f"{fast} steps of the switch.", "",
         "| seed | first loss | prediction | 10x7 click - switch | outcome |", "|---:|---:|---|---:|---|"]
    tab = {(a, b): 0 for a in (0, 1) for b in (0, 1)}
    for r in rs:
        p = preds.get(f"b_Gc_s{r['seed']}")
        if p is None:
            continue
        dl = None if r["click10"] is None else r["click10"] - r["switch"]
        ok = dl is not None and dl <= fast
        tab[int(p["prediction"] == "learns"), int(ok)] += 1
        L.append(f"| {r['seed']} | {p['first_loss']:.3f} | {p['prediction']} | {fmt(dl)} | {'learns' if ok else 'not'} |")
    a, n = tab[1, 1], tab[1, 1] + tab[1, 0]
    b, m = tab[0, 1], tab[0, 1] + tab[0, 0]
    L.append("")
    if n and m:
        p1 = fisher(a, n, b, m)[0]
        L.append(f"Predicted 'learns': {a}/{n} learned; predicted 'collapses': {b}/{m} learned. Fisher one-sided p = {p1:.3g}; "
                 f"wrong predictions: {tab[1, 0] + tab[0, 1]}. Separation confirmed (p < 0.05): {p1 < 0.05}.")
    else:
        L.append(f"All predictions in one class ({n} 'learns', {m} 'collapses'): the separation is not testable.")
    L.append("")
    cfg["_sep"] = bool(n and m and fisher(a, n, b, m)[0] < 0.05)
    return L


def decide(cfg, c):
    a, cc, ch = c["a"], c["cC"], c["cCh"]
    L = ["## Verdict by the protocol", ""]
    if a["click10"] >= 8:
        v = "series 3 (curriculum: soft switch)"
    elif a["click10"] >= 4:
        v = "series 2'' (aimed at the loss spike at the switch)" if cfg.get("_sep") else "stop: partial transfer, cause not found"
    else:
        v = "stop: the soft switch does not help; 2b's 7/10 is the transfer we have"
    conv = [k for k, x in (("C fixed", cc), ("C+h without plane 15", ch)) if x["click10"] >= 3]
    L.append(f"(a) 10x7 clicks {a['click10']}/{a['n']}; (b) separation {'confirmed' if cfg.get('_sep') else 'not confirmed'}: {v}.")
    if conv:
        L.append(f"Convolution arms with >= 3/5 clicks: {', '.join(conv)}: series 3 for comparison, over a stop row.")
        v = "series 3" if v.startswith("stop") else v
    L += ["", f"Next: {v}.", ""]
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "commit-preds", "analyse", "report"])
    ap.add_argument("--config", default="configs/graph_s2p.yaml")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    (ROOT / cfg["out"]).mkdir(parents=True, exist_ok=True)
    {"run": cmd_run, "commit-preds": cmd_commit_preds, "analyse": cmd_analyse, "report": cmd_report}[a.cmd](cfg)


if __name__ == "__main__":
    main()
