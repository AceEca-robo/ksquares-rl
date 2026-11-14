"""Part 4, series 5 of docs/plan_part4.md: how often does Gc make the move-choice jump of series 3' Gc s1.

    python scripts/graph_s5.py check                                 # before the series: new code = series 3' Gc s1, bit for bit
    python scripts/graph_s3.py run   --config configs/graph_s5.yaml  # stage 1 + stage 2 of series 3', Gc seeds 6-25
    python scripts/graph_s3.py probe --config configs/graph_s5.yaml  # once, after all training
    python scripts/graph_s5.py report                                # docs/graph_s5.md

Training is scripts/graph_s3.py unchanged; this file only checks and reports.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np

from _common import ROOT, load_config

import graph_s3
from parity_probe import wilson

BINS = [-np.inf, -10, -3, -1, 0, 1, 3, 10, np.inf]


def cmd_check(cfg, out, steps):
    """Stage 2 of series 3' Gc s1 rerun with this code and config (snapshots on) to a checkpoint; its
    log must equal the stored series 3' log exactly, and the kept weights must give the logged V-sign."""
    import torch

    src = ROOT / "runs/graph_s3p"
    cfg = {**cfg, "out": str(out), "_stop_at": steps}
    graph_s3.prepare(cfg)
    for ext in (".json", ".pt"):
        shutil.copy(src / f"a_Gc_s1{ext}", ROOT / cfg["out"] / f"a_Gc_s1{ext}")
    t = time.time()
    assert graph_s3.run_stage2(cfg, "Gc", 1) is None
    sec = time.time() - t
    ck = torch.load(ROOT / cfg["out"] / "s2_Gc_s1.ckpt", map_location="cpu", weights_only=False)
    ref = json.loads((src / "s2_Gc_s1.json").read_text())["log"]
    strip = lambda x: {k: v for k, v in x.items() if k != "seconds"}  # noqa: E731
    new = ck["loop"]["log"]
    assert len(new) == steps // cfg["stage2"]["log_every"] + 1
    for a, b in zip(new, ref):
        assert strip(a) == strip(b), (a, b)
    cfg["stage2"] = {**cfg["stage2"], "snapshots": {**cfg["stage2"]["snapshots"],
                                                    "steps": [s for s in cfg["stage2"]["snapshots"]["steps"] if s <= steps]}}
    graph_s3.snapshot_hists(cfg, "Gc", 1)
    vh = json.loads((ROOT / cfg["out"] / "s2_Gc_s1_vhist.json").read_text())
    by = {x["step"]: x for x in new}
    for s, v in vh.items():
        if int(s) in by:
            assert all(v[k]["acc"] == by[int(s)]["stage0"][k] for k in ("5x5", "10x7")), (s, v["10x7"]["acc"])
    print(f"check passed: {len(new)} log entries to step {steps} equal series 3' Gc s1 exactly; "
          f"V-sign from the kept weights equals the log at steps {[int(s) for s in vh if int(s) in by]}; "
          f"{sec:.0f} s for {steps} steps alone")


def fmt_hist(pre):
    return " ".join(str(int(c)) for c in np.histogram(pre, BINS)[0])


def rise(log, sc):
    """Largest rise of the 10x7 judge validation between two consecutive validations after the switch."""
    after = [i for i, x in enumerate(log) if x.get("policy_step", 0) > sc["switch_at"]]
    i = max(after, key=lambda i: log[i]["val"]["10x7"] - log[i - 1]["val"]["10x7"])
    return log[i]["step"], log[i - 1]["val"]["10x7"], log[i]["val"]["10x7"]


def verdict(n_jump, n_done, n_all, cr):
    if n_done < n_all:
        return f"series not finished ({n_done} of {n_all} runs): no verdict"
    if n_jump >= cr["reproducible_min"]:
        return f">= {cr['reproducible_min']} of {n_all}: the jump is reproducible"
    if n_jump <= cr["chance_max"]:
        return f"0-{cr['chance_max']} of {n_all}: series 3' Gc s1 was chance; part 4 closed"
    return f"{cr['chance_max'] + 1}-{cr['reproducible_min'] - 1} of {n_all}: described; the decision is the user's"


def cmd_report(cfg):
    out, sc, cr = ROOT / cfg["out"], cfg["stage2"], cfg["criterion"]
    seeds = cfg["parts"]["a"]["seeds"]
    probe = json.loads((out / "probe.json").read_text()) if (out / "probe.json").exists() else {}
    thr, gate = cr["jump_val_10x7"], sc["gate_stage0_10x7"]
    L = ["# Part 4, series 5: how often does Gc make the move-choice jump", "",
         "Protocol: docs/plan_part4.md (series 5), configs/graph_s5.yaml. Gc only, seeds "
         f"{seeds[0]}-{seeds[-1]}; stage 1 and stage 2 are series 3' unchanged (40000 stage 2 steps, every 4th a "
         "stage 0 step, switch at policy step 5500 = step 7333, best point by 10x7 judge validation after the switch). "
         f"Slots: {cfg['jobs']} runs at once, GPU memory fraction {cfg['limits']['gpu_memory_fraction']}, nice -n 10, "
         f"no work below {cfg['limits']['min_available_gb']} GB free RAM (user, 2026-11-14). Steps below are total stage 2 steps.", "",
         "## Table 1. Runs", "",
         f"| seed | s1 5x5 click | s1 10x7 click | s1 last 10x7 | best step | val 5x5 | best val 10x7 | >= {thr:.0f}% | "
         f"V-sign 10x7 at start / best / end | sign kept (>= {gate:.0f}% at best) | largest rise after the switch | "
         "probe 5x5 | probe 10x7 [95%] |",
         "|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|"]
    done, rows = [], {}
    for s in seeds:
        p1, p2 = out / f"a_Gc_s{s}.json", out / f"s2_Gc_s{s}.json"
        if not p2.exists():
            ck = out / f"s2_Gc_s{s}.ckpt"
            state = "stage 2 running" if ck.exists() else "stage 1 done" if p1.exists() else \
                "stage 1 running" if (out / f"a_Gc_s{s}.ckpt").exists() else "not started"
            L.append(f"| {s} | {state} ||||||||||||")
            continue
        r1, r2 = json.loads(p1.read_text()), json.loads(p2.read_text())
        log = r2["log"]
        bl = next(x for x in log if x["step"] == r2["best_step"])
        jump, kept = r2["best_val_10x7"] >= thr, bl["stage0"]["10x7"] >= gate
        rows[s] = (jump, kept)
        done.append(s)
        rs, ra, rb = rise(log, sc)
        pr = probe.get(f"Gc_s{s}")
        pc = "- | -"
        if pr:
            lo, hi = wilson(sum(pr["10x7"]), len(pr["10x7"]))
            pc = f"{100 * np.mean(pr['5x5']):.1f} | {100 * np.mean(pr['10x7']):.1f} [{lo:.0f}-{hi:.0f}]"
        L.append(f"| {s} | {graph_s3.fmt(r1['click5'])} | {graph_s3.fmt(r1['click10'])} | {r1['last_10x7']:.1f} | {r2['best_step']} | "
                 f"{bl['val']['5x5']:.1f} | {r2['best_val_10x7']:.1f} | {'yes' if jump else 'no'} | "
                 f"{log[0]['stage0']['10x7']:.1f} / {bl['stage0']['10x7']:.1f} / {log[-1]['stage0']['10x7']:.1f} | "
                 f"{'yes' if kept else 'no'} | {rb - ra:+.1f} at {rs} ({ra:.1f} -> {rb:.1f}) | {pc} |")
    nj = sum(j for j, _ in rows.values())
    L += ["", "## Primary metric and verdict", "",
          f"Runs with best 10x7 validation >= {thr:.0f}% after the switch: **{nj} of {len(done)}** finished "
          f"({len(seeds)} planned)" + (f"; seeds {' '.join(str(s) for s in done if rows[s][0])}." if nj else "."),
          "", "| | V sign kept | V sign lost |", "|---|---:|---:|",
          f"| best val 10x7 >= {thr:.0f}% | {sum(j and k for j, k in rows.values())} | {sum(j and not k for j, k in rows.values())} |",
          f"| best val 10x7 < {thr:.0f}% | {sum(k and not j for j, k in rows.values())} | {sum(not j and not k for j, k in rows.values())} |",
          "", f"Row of the protocol: {verdict(nj, len(done), len(seeds), cr)}.", ""]
    L += ["## Table 2. V on the stage 0 10x7 validation at steps 250, 500, 1000 of stage 2", "",
          "From the weights kept at those steps (inference after the run; 2000 positions, 1013 negative / 987 positive). "
          "Bins of the V pre-activation: <-10, -10..-3, -3..-1, -1..0, 0..1, 1..3, 3..10, >10. Stage 0 loss: mean over the "
          "stage 0 steps of the 500 steps before (ln 2 = 0.693 is a V that says nothing).", "",
          "| seed | V-sign 250 / 500 / 1000 | stage 0 loss 0-500 / 500-1000 | V hist 250 | V hist 500 | V hist 1000 |",
          "|---:|---|---|---|---|---|"]
    for s in done:
        vp = out / f"s2_Gc_s{s}_vhist.json"
        if not vp.exists():
            continue
        vh = json.loads(vp.read_text())
        lg = {x["step"]: x for x in json.loads((out / f"s2_Gc_s{s}.json").read_text())["log"]}
        st = sc["snapshots"]["steps"]
        L.append(f"| {s} | " + " / ".join(f"{vh[str(k)]['10x7']['acc']:.1f}" for k in st)
                 + f" | {lg[500]['loss_stage0']:.3f} / {lg[1000]['loss_stage0']:.3f} | "
                 + " | ".join(fmt_hist(np.array(vh[str(k)]["10x7"]["pre"])) for k in st) + " |")
    n_snap = len(list((out / "snap").glob("*.pt"))) if (out / "snap").exists() else 0
    lo, hi = sc["snapshots"]["val_window"]
    L += ["", "## Kept for later analysis (runs/graph_s5, not in git)", "",
          f"- snap/s2_Gc_s<seed>_step<step>.pt: weights at steps {', '.join(map(str, sc['snapshots']['steps']))} and at every "
          f"validation step {lo}-{hi} ({len(graph_s3.snapshot_steps(sc))} per run; {n_snap} files now).",
          "- s2_Gc_s<seed>_vhist.json: V pre-activations of all stage 0 validation positions (5x5 and 10x7) at steps "
          f"{', '.join(map(str, sc['snapshots']['steps']))}.",
          "- s2_Gc_s<seed>.pt (best point), s2_Gc_s<seed>_last.pt, a_Gc_s<seed>.pt (end of stage 1), logs in the .json files.", ""]
    ref = ROOT / "runs/graph_s3p"
    if all((ref / f"s2_Gc_s{s}.json").exists() for s in range(1, 6)):
        L += ["## For reference: Gc seeds 1-5 of series 3' (not counted here)", "",
              "| seed | best step | best val 10x7 | V-sign 10x7 at best |", "|---:|---:|---:|---:|"]
        for s in range(1, 6):
            r = json.loads((ref / f"s2_Gc_s{s}.json").read_text())
            bl = next(x for x in r["log"] if x["step"] == r["best_step"])
            L.append(f"| {s} | {r['best_step']} | {r['best_val_10x7']:.1f} | {bl['stage0']['10x7']:.1f} |")
        L.append("")
    (ROOT / cfg["report"]).write_text("\n".join(L))
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "report"])
    ap.add_argument("--config", default="configs/graph_s5.yaml")
    ap.add_argument("--out", help="check: scratch directory")
    ap.add_argument("--steps", type=int, default=1500, help="check: stage 2 steps (a checkpoint step)")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    if a.cmd == "check":
        cmd_check(cfg, a.out, a.steps)
    else:
        cmd_report(cfg)


if __name__ == "__main__":
    main()
