"""Part 4, series 2'' of docs/plan_part4.md: forgetting of 5x5 with a kept 5x5 share.

    python scripts/graph_s2pp.py run       # k10, k25 (Gc, school (a), 5x5 share kept after the mix)
    python scripts/graph_s2pp.py report    # docs/graph_s2pp.md, compared with series 2' (a) on the same seeds

configs/graph_s2pp.yaml. Training is the run loop of scripts/graph_s2.py (option keep_5x5).
"""

from __future__ import annotations

import argparse
import math
import shutil

from _common import ROOT, load_config

import graph_s2
from graph_s2 import fmt, load
from graph_s2p import run_rows

KEYS = ["click10", "hold10", "hold5", "both"]


def sign_test(a, b):
    """Paired 0/1 per seed: (seeds only a has, seeds only b has, one-sided exact p that a > b)."""
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n10 + n01
    p = sum(math.comb(n, i) for i in range(n10, n + 1)) / 2 ** n if n else 1.0
    return n10, n01, p


def cmd_run(cfg):
    out = ROOT / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "data.npz").exists():
        shutil.copy(ROOT / "runs/graph_s2/data.npz", out / "data.npz")
    graph_s2.cmd_run(cfg)


def cmd_report(cfg):
    ref = cfg["reference"]
    rcfg = {**cfg, "out": ref["out"]}
    L = ["# Part 4, series 2'': forgetting of 5x5 with a kept 5x5 share", "",
         "Protocol: docs/plan_part4.md (series 2''), configs/graph_s2pp.yaml. Gc, school (a) of series 2'; after the "
         "1000-step mix a share of 5x5 batches is kept to the end. Click: first validation >= "
         f"{cfg['pass_acc']}%; hold 10x7: every validation after the click >= {cfg['hold_acc']}%; hold 5x5: every "
         f"validation from the end of the mix >= {cfg['hold_acc']}%; both: the last validation >= {cfg['hold_acc']}% on both sizes. "
         "Reference: series 2' (a), share 0, the same seeds.", ""]
    seeds = cfg["parts"]["k10"]["seeds"]
    base = {r["seed"]: r for r in run_rows(rcfg, load(rcfg, ref["part"], "Gc", seeds), 1000)}
    for part, pc in cfg["parts"].items():
        rs = run_rows(cfg, load(cfg, part, "Gc", pc["seeds"]), pc["mix_steps"])
        n = len(pc["seeds"])
        c = {k: sum(bool(r[k]) for r in rs) for k in KEYS}
        L += [f"## {part}: share {pc['keep_5x5']:.2f}, {len(rs)}/{n} runs", "",
              f"10x7 click {c['click10']}/{n}, hold 10x7 {c['hold10']}/{n}, hold 5x5 {c['hold5']}/{n}, both at the end {c['both']}/{n}.", "",
              "| seed | switch | 10x7 click | hold 10x7 | hold 5x5 | both | min 5x5 after mix | last 10x7 | last 5x5 | (a): hold 5x5 / both / last 5x5 |",
              "|---:|---:|---:|---|---|---|---:|---:|---:|---|"]
        for r in rs:
            m5 = min((x["val"]["5x5"]["acc"] for x in r["log"] if r["switch"] is not None and x["step"] >= r["switch"] + pc["mix_steps"]),
                     default=None)
            b = base.get(r["seed"])
            L.append(f"| {r['seed']} | {fmt(r['switch'])} | {fmt(r['click10'])} | {fmt(r['hold10'])} | {fmt(r['hold5'])} | {r['both']} | "
                     f"{fmt(m5 if m5 is None else round(m5, 1))} | {r['last_10x7']:.1f} | {r['last_5x5']:.1f} | "
                     + (f"{b['hold5']} / {b['both']} / {b['last_5x5']:.1f} |" if b else "- |"))
        if len(rs) == n:
            L += ["", "Paired with (a) on the same seeds (seeds only this variant has / only (a) has / one-sided exact p):", "",
                  "| metric | this | (a) | only this / only (a) | p |", "|---|---:|---:|---|---:|"]
            for k in KEYS:
                a = [bool(r[k]) for r in rs]
                bb = [bool(base[r["seed"]][k]) for r in rs]
                n10, n01, p = sign_test(a, bb)
                L.append(f"| {k} | {sum(a)} | {sum(bb)} | {n10} / {n01} | {p:.3g} |")
        L.append("")
    (ROOT / "docs/graph_s2pp.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--config", default="configs/graph_s2pp.yaml")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    {"run": cmd_run, "report": cmd_report}[a.cmd](cfg)


if __name__ == "__main__":
    main()
