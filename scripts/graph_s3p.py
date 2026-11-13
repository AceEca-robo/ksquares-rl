"""Part 4, series 3' of docs/plan_part4.md: report with the stage 0 gate (scripts/graph_s3.py does run / probe).

    python scripts/graph_s3.py run   --config configs/graph_s3p.yaml
    python scripts/graph_s3.py probe --config configs/graph_s3p.yaml
    python scripts/graph_s3p.py report      # docs/graph_s3p.md: runs table, gate, tests, verdict, descriptive

The "Data checks" section (scripts/graph_s3p_checks.py) is kept. The descriptive section reads logs and
saved weights only (inference on the stage 0 10x7 validation; no training).
"""

from __future__ import annotations

import json

import numpy as np

from _common import ROOT, load_config

import graph_s3
from graph_s3 import bootstrap
from parity_probe import mcnemar

NAMES = {"C": "C", "Ch": "C+h", "Ch14": "C+h14", "Ga": "Ga", "Gc": "Gc"}


def runs(cfg):
    out = ROOT / cfg["out"]
    r = {}
    for arm in cfg["parts"]["a"]["arms"]:
        for s in cfg["parts"]["a"]["seeds"]:
            r[arm, s] = json.loads((out / f"s2_{arm}_s{s}.json").read_text())
    return r


def at_best(r):
    return next(x for x in r["log"] if x["step"] == r["best_step"])


def gate_section(cfg, R, probe):
    sc, tc = cfg["stage2"], cfg["test"]
    arms, seeds = cfg["parts"]["a"]["arms"], cfg["parts"]["a"]["seeds"]
    g, need = sc["gate_stage0_10x7"], sc["gate_min_runs"]
    passing = {a: [s for s in seeds if at_best(R[a, s])["stage0"]["10x7"] >= g] for a in arms}
    acc = {a: [probe[f"{a}_s{s}"]["10x7"] for s in passing[a]] for a in arms}
    allacc = {a: [probe[f"{a}_s{s}"]["10x7"] for s in seeds] for a in arms}
    L = [f"## Gate (stage 0 V-sign on 10x7 >= {g}% at the best point; a pair needs >= {need} of 5 passing runs in both arms)", "",
         "| arm | passing | seeds passing | probe 10x7 mean, passing runs | probe 10x7 mean, all runs |", "|---|---:|---|---:|---:|"]
    for a in arms:
        m = np.mean([100 * np.mean(x) for x in acc[a]]) if acc[a] else None
        L.append(f"| {NAMES[a]} | {len(passing[a])}/5 | {' '.join(map(str, passing[a])) or '-'} | "
                 f"{'-' if m is None else f'{m:.1f}'} | {np.mean([100 * np.mean(x) for x in allacc[a]]):.1f} |")
    L += ["", f"## Tests on the 10x7 probe (bootstrap {tc['resamples']} resamples over passing seeds and positions; "
          f"one-sided {int(100 * tc['level'])}% lower bound)", "",
          "| pair | role | runs | difference, points | lower bound | McNemar seed pairs: first better / second better / n.s. |",
          "|---|---|---|---:|---:|---|"]
    res = {}
    rows = [(tc["primary"], "primary"), (tc["second"], "second (pre-registered)")] + [(p, "descriptive") for p in tc["descriptive"]]
    for (x, y), role in rows:
        nm = f"{NAMES[x]} - {NAMES[y]}"
        if len(passing[x]) < need or len(passing[y]) < need:
            res[x, y] = None
            L.append(f"| {nm} | {role} | {len(passing[x])} / {len(passing[y])} | not comparable | - | - |")
            continue
        d, lo = bootstrap(acc[x], acc[y], tc["resamples"], tc["boot_seed"], tc["level"])
        res[x, y] = (d, lo)
        w = [mcnemar(acc[y][j], acc[x][i]) for i in range(len(acc[x])) for j in range(len(acc[y]))]
        fb = sum(1 for n01, n10, p in w if p < 0.05 and n01 > n10)
        sb = sum(1 for n01, n10, p in w if p < 0.05 and n10 > n01)
        L.append(f"| {nm} | {role} | {len(passing[x])} / {len(passing[y])} | {d:+.1f} | {lo:+.1f} | {fb} / {sb} / {len(w) - fb - sb} |")
    p, s2 = res[tuple(tc["primary"])], res[tuple(tc["second"])]
    gc = np.mean([100 * np.mean(x) for x in acc["Gc"]]) if acc["Gc"] else float("nan")
    if p is None:
        row = "primary pair not comparable (Gc or C+h < 3 runs past the gate): the share does not keep the chain count during stage 2. Next: stop; report"
    elif p[1] > 0:
        row = ("Gc - C+h lower bound > 0 and Gc probe > 80%: graph adds something beyond chain planes, late. Next: series 4"
               if gc > tc["strong_probe"] else
               "Gc - C+h lower bound > 0, probe <= 80%: a small gain, still weak. Next: series 4, reported as weak")
    elif s2 is not None and s2[1] > 0:
        row = "Gc <= C+h, Gc - C+h14 bound > 0: graph beats the chain count alone, not with the target plane. Next: stop part 4; reported"
    else:
        row = ("Gc <= C+h and Gc <= C+h14: graph gives nothing beyond chain planes late, even keeping the count. Next: stop part 4"
               + ("" if s2 is not None else " (second pair not comparable; the row is read from the primary)"))
    L += ["", "## Verdict by the decision table (docs/plan_part4.md, series 3')", "",
          f"Gc passing runs {len(passing['Gc'])}/5, C+h {len(passing['Ch'])}/5, C+h14 {len(passing['Ch14'])}/5. "
          + (f"Primary Gc - C+h: {p[0]:+.1f} points, lower bound {p[1]:+.1f}; Gc probe 10x7 mean (passing runs) {gc:.1f}%. "
             if p else "")
          + (f"Second Gc - C+h14: {s2[0]:+.1f}, lower bound {s2[1]:+.1f}. " if s2 else "Second Gc - C+h14: not comparable. "),
          "", f"Row: {row}.", ""]
    return L


def v_hist(cfg, arm, state, x, dev):
    import torch

    net, zero, _ = graph_s3.build_net(cfg, arm)
    net.load_state_dict(state)
    net.to(dev).eval()
    x = graph_s3.zeroed(x[:, :net.planes], zero)
    pre = []
    with torch.no_grad():
        for i in range(0, len(x), 500):
            net(torch.from_numpy(x[i:i + 500]).to(dev))
            pre.append(net.v_pre.cpu().numpy())
    return np.concatenate(pre)


def fmt_hist(pre):
    e = [-np.inf, -10, -3, -1, 0, 1, 3, 10, np.inf]
    return " ".join(str(int(c)) for c in np.histogram(pre, e)[0])


def descriptive(cfg, R):
    import torch

    from graph_s2 import splits

    out = ROOT / cfg["out"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.1)
    z = np.load(out / "data.npz")
    _, _, xv, yv = splits(cfg, {k: z[k] for k in z.files})["10x7"]
    L = ["## Descriptive (outside the verdict; logs and saved weights only, no new runs)", "",
         "### (1) Gc s1 (the jump) next to Gc s2 (sign kept, no jump)", "",
         "Per validation (every 500 steps of 40000; policy step = step x 3/4): judge validation 10x7, stage 0 V-sign "
         "on 10x7, mean policy loss and mean stage 0 loss over the 500 steps before.", "",
         "| step | policy step | s1 val 10x7 | s1 V-sign | s1 policy loss | s1 stage 0 loss | s2 val 10x7 | s2 V-sign | s2 policy loss | s2 stage 0 loss |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    a, b = R["Gc", 1]["log"], R["Gc", 2]["log"]
    for x, y in zip(a, b):
        if x["step"] % 1000 and not (abs(x["val"]["10x7"] - (a[a.index(x) - 1]["val"]["10x7"] if a.index(x) else 0)) >= 5):
            continue
        f = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
        L.append(f"| {x['step']} | {x.get('policy_step', 0)} | {x['val']['10x7']:.1f} | {x['stage0']['10x7']:.1f} | "
                 f"{f(x.get('loss'))} | {f(x.get('loss_stage0'))} | {y['val']['10x7']:.1f} | {y['stage0']['10x7']:.1f} | "
                 f"{f(y.get('loss'))} | {f(y.get('loss_stage0'))} |")
    jumps = [(a[i]["step"], a[i]["val"]["10x7"] - a[i - 1]["val"]["10x7"]) for i in range(2, len(a))]
    js, jd = max(jumps, key=lambda t: t[1])
    i = next(k for k, x in enumerate(a) if x["step"] == js)
    L += ["", f"Largest rise between two validations of s1: {jd:+.1f} points at step {js} (policy step {a[i]['policy_step']}), "
          f"{a[i - 1]['val']['10x7']:.1f} -> {a[i]['val']['10x7']:.1f}; policy loss {a[i - 1]['loss']:.3f} -> {a[i]['loss']:.3f}, "
          f"stage 0 loss {a[i - 1]['loss_stage0']:.3f} -> {a[i]['loss_stage0']:.3f}, V-sign {a[i - 1]['stage0']['10x7']:.1f} -> "
          f"{a[i]['stage0']['10x7']:.1f}. Rows above: every 1000 steps plus every validation with a change >= 5 points.", ""]
    L += ["### (2) Losing the V sign in the first 500 steps of stage 2: Gc s3, s4 against the others", "",
          "Stage 2 logs have no V histogram (only V-sign accuracy and the mean stage 0 loss per 500 steps), and the "
          "step 500 weights are not kept (checkpoints are deleted when a run ends). Histograms below are recomputed by "
          "inference on the 2000 stage 0 10x7 validation positions from the weights that exist: end of stage 1 "
          "(= stage 2 step 0), the best point and the end of stage 2. Bins of the V pre-activation: "
          "<-10, -10..-3, -3..-1, -1..0, 0..1, 1..3, 3..10, >10.", "",
          "| run | V-sign 0 / 500 / 1000 / 2000 | stage 0 loss 0-500 / 500-1000 / 1000-2000 | V hist step 0 | V hist best | V hist end |",
          "|---|---|---|---|---|---|"]
    for arm in ("Gc", "Ch", "Ch14"):
        for s in cfg["parts"]["a"]["seeds"]:
            r = R[arm, s]
            lg = {x["step"]: x for x in r["log"]}
            h0 = v_hist(cfg, arm, torch.load(out / f"a_{arm}_s{s}.pt", map_location="cpu"), xv, dev)
            hb = v_hist(cfg, arm, torch.load(out / f"s2_{arm}_s{s}.pt", map_location="cpu"), xv, dev)
            he = v_hist(cfg, arm, torch.load(out / f"s2_{arm}_s{s}_last.pt", map_location="cpu"), xv, dev)
            l2 = np.mean([lg[k]["loss_stage0"] for k in (1500, 2000)])
            L.append(f"| {NAMES[arm]} s{s} | " + " / ".join(f"{lg[k]['stage0']['10x7']:.1f}" for k in (0, 500, 1000, 2000))
                     + f" | {lg[500]['loss_stage0']:.3f} / {lg[1000]['loss_stage0']:.3f} / {l2:.3f} | {fmt_hist(h0)} | "
                     f"{fmt_hist(hb)} | {fmt_hist(he)} |")
    L += ["", "Reading aid: ln 2 = 0.693 is the stage 0 loss of a V that says nothing; a loss near 1 with V-sign ~99% means "
          "few, very confident wrong answers; positions are 1013 negative / 987 positive.", "" + [
          "Observations (descriptive, not tested):",
          "- (1) Both Gc s1 and s2 sit at 56-58% until the switch to 10x7 judge positions (policy step 5500 = step 7333); "
          "both rise right after it. s1 goes 61.6 -> 81.8 between steps 7500 and 8000 and keeps climbing to ~89; s2 stops at "
          "62-66. The policy loss shows no matching event (0.567 -> 0.549 in s1, 0.517 -> 0.668 in s2); both nets keep the V "
          "sign there (99.9 / 99.6 at step 8000). Nothing in the logged quantities separates s1 from s2 at the jump.",
          "- (2) At step 0 every Gc, C+h and C+h14 net has a saturated V on 10x7 (all 2000 positions |pre| > 10, right sign). "
          "In the first 500 steps the stage 0 loss of every Gc is 0.49-1.19 (C+h: 0.002), i.e. the policy steps push Gc's V "
          "to confident wrong answers on part of the positions. s3 and s4 then sit at 0.694-0.699 = ln 2 from step 500 on, and "
          "their V histograms (best, end) are piled in -1..1: V became near-constant and never recovered; s1, s2, s5 had "
          "a lower or falling loss and stayed saturated with the right sign. Which run falls cannot be read from step 0.",
          "- C+h and C+h14 behave differently: their stage 0 training loss stays low (C+h <= 0.003, C+h14 0.02-0.08) while "
          "C+h14's validation V-sign is 25-60% in the first 2000 steps. Low loss in training mode with wrong eval sign points to "
          "the BatchNorm of the CNN (batch statistics on the stage 0 batch in training, running statistics mixed with judge "
          "batches in eval). This is a hypothesis; the graph nets are a different case (their training loss itself is at ln 2).",
          "- The step 500 weights are not kept, so the histogram at the moment of the loss is not available without a rerun.", ""]
    return L


def cmd_report(cfg):
    graph_s3.cmd_report(cfg)                 # runs table (and keeps the Data checks section)
    rep = ROOT / cfg["report"]
    head, _, checks = rep.read_text().partition("\n## Data checks")
    head = head.replace("# Part 4, series 3: parity probe on real positions", "# Part 4, series 3': stage 0 kept as an auxiliary share of stage 2")
    head = head.replace("Protocol: docs/plan_part4.md (series 3), configs/graph_s3.yaml.",
                        "Protocol: docs/plan_part4.md (series 3'), configs/graph_s3p.yaml. Stage 2 has 40000 steps, every 4th a "
                        "stage 0 step (10x7, V-sign loss); the switch and the best point count policy steps; 3 runs at once. "
                        "Steps below are total stage 2 steps.")
    R = runs(cfg)
    probe = json.loads((ROOT / cfg["out"] / "probe.json").read_text())
    body = head.rstrip() + "\n\n" + "\n".join(gate_section(cfg, R, probe) + descriptive(cfg, R))
    rep.write_text(body + ("\n## Data checks" + checks if checks else ""))
    print(body)


if __name__ == "__main__":
    cmd_report(load_config(ROOT / "configs/graph_s3p.yaml"))
