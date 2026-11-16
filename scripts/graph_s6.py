"""Part 4, series 6 of docs/plan_part4.md: what decides the move-choice jump, the stage 1 weights or the stage 2 seed.

    python scripts/graph_s6.py check --out <scratch dir>   # before the series: control to step 1500 = series 5 s2_Gc_s8
    python scripts/graph_s6.py run                         # control + 4 stage 1 weights x 5 stage 2 seeds
    python scripts/graph_s6.py probe                       # once, after all training
    python scripts/graph_s6.py report                      # docs/graph_s6.md

Stage 2 is graph_s3.run_stage2 unchanged. It takes the weights a_<arm>_s<seed>.pt of its directory and seeds
everything with <seed>, so each stage 1 seed W gets a directory runs/graph_s6/w<W> where the series 5 weights
a_Gc_s<W>.pt are copied under the name of every stage 2 seed; the control (stage 2 seed = stage 1 seed) runs in
runs/graph_s6/control under the series 5 names. STOP in a run directory is a link to runs/graph_s6/STOP, so the
series stop command is the usual one.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time

import numpy as np

from _common import ROOT, load_config

import graph_s3
from graph_hand import stop_requested
from graph_s5 import fmt_hist, rise, strip_seconds
from parity_probe import wilson

ARM = "Gc"
SLOTS = ("2 runs at once, GPU memory fraction 0.4 per run, nice -n 10 from the start (2026-11-16 10:17) to the stop at "
         "checkpoints at 21:19 (14 runs finished, w9 / 104 and w6 / 104 at step 13500); from the restart 6 runs at once, "
         "no GPU memory limit, no nice (user, 2026-11-16 21:15; protocol and seeds unchanged). The table marks the runs "
         "that ran wholly or partly at 6 slots.")


def groups(cfg):
    """[(group, stage 1 seed)] in the order of the config."""
    return [(g, w) for g, ws in cfg["series6"]["weights"].items() for w in ws]


def sub_config(cfg, sub, seeds):
    """The config of one run directory: the series config with its own out and stage 2 seeds; no stage 1."""
    return {**cfg, "out": f"{cfg['out']}/{sub}", "reuse_stage1": None, "parts": {"a": {"arms": [ARM], "seeds": list(seeds)}}}


def job_list(cfg):
    """[(sub config, stage 1 seed, stage 2 seed)]: the control first, then stage 2 seed by stage 2 seed over
    the four weights, so that a stopped series has the groups filled evenly."""
    s6 = cfg["series6"]
    c = s6["control"]
    jobs = [(sub_config(cfg, "control", [c["stage2_seed"]]), c["weights"], c["stage2_seed"])]
    for t in s6["stage2_seeds"]:
        jobs += [(sub_config(cfg, f"w{w}", s6["stage2_seeds"]), w, t) for _, w in groups(cfg)]
    return jobs


def prepare(cfg):
    """Run directories: data.npz (one copy, linked), the STOP link, the series 5 stage 1 weights under the
    stage 2 seed names. Returns the sha256 of each stage 1 weight file used."""
    import hashlib

    out, src = ROOT / cfg["out"], ROOT / cfg["series6"]["from"]
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "data.npz").exists():
        shutil.copy(src / "data.npz", out / "data.npz")
    sha = {}
    for sub, w, t in job_list(cfg):
        d = ROOT / sub["out"]
        d.mkdir(exist_ok=True)
        for link, target in (("data.npz", "../data.npz"), ("STOP", "../STOP")):
            if not (d / link).is_symlink():
                os.symlink(target, d / link)
        dst = d / f"a_{ARM}_s{t}.pt"
        if not dst.exists():
            shutil.copy(src / f"a_{ARM}_s{w}.pt", dst)
        sha[w] = hashlib.sha256((src / f"a_{ARM}_s{w}.pt").read_bytes()).hexdigest()
        assert hashlib.sha256(dst.read_bytes()).hexdigest() == sha[w], dst
    (out / "stage1_sha256.json").write_text(json.dumps(sha))
    return sha


def run_job(args):
    sub, w, t = args
    name = graph_s3.run_stage2(sub, ARM, t)
    if name:
        graph_s3.snapshot_hists(sub, ARM, t)
    return f"{sub['out'].rsplit('/', 1)[1]}/{name}" if name else None


def cmd_run(cfg):
    prepare(cfg)
    with mp.get_context("spawn").Pool(cfg["jobs"], maxtasksperchild=1) as pool:
        for name in pool.imap_unordered(run_job, job_list(cfg)):
            if name:
                print("done", name, flush=True)
    if stop_requested(ROOT / cfg["out"]):
        print("series stopped at checkpoints", flush=True)


def cmd_probe(cfg):
    """Probes of the best stage 2 nets, once, after all training (graph_s3.cmd_probe per run directory)."""
    done = set()
    for sub, _, _ in job_list(cfg):
        if sub["out"] not in done:
            graph_s3.cmd_probe(sub)
            done.add(sub["out"])


def compare_stage2(ref_dir, new_dir, name, snap_steps):
    """A finished stage 2 run against a reference run of the same name: log entry by entry (wall time aside),
    best point, best / last / kept weights tensor by tensor, V pre-activations of the kept early weights."""
    import torch

    la, lb = (json.loads((d / f"{name}.json").read_text()) for d in (ref_dir, new_dir))
    first = next((x["step"] for x, y in zip(la["log"], lb["log"]) if strip_seconds(x) != strip_seconds(y)), None)

    def same(pa, pb):
        a, b = (torch.load(p, map_location="cpu") for p in (pa, pb))
        return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)

    files = [f"{name}.pt", f"{name}_last.pt"] + [f"snap/{name}_step{s:05d}.pt" for s in sorted(snap_steps)]
    differ = [f for f in files if not same(ref_dir / f, new_dir / f)]
    vh = [json.loads((d / f"{name}_vhist.json").read_text()) for d in (ref_dir, new_dir)]
    return {"name": name, "ref": str(ref_dir.relative_to(ROOT)), "new": str(new_dir.relative_to(ROOT)),
            "log_entries": [len(la["log"]), len(lb["log"])], "first_log_diff_step": first,
            "log_equal": len(la["log"]) == len(lb["log"]) and first is None,
            "best_equal": (la["best_step"], la["best_val_10x7"]) == (lb["best_step"], lb["best_val_10x7"]),
            "best": [[la["best_step"], la["best_val_10x7"]], [lb["best_step"], lb["best_val_10x7"]]],
            "weight_files": len(files), "weight_files_differ": differ, "vhist_equal": vh[0] == vh[1],
            "equal": len(la["log"]) == len(lb["log"]) and first is None and not differ and vh[0] == vh[1]}


def cmd_check(cfg, out, steps):
    """Before the series, in a scratch directory: (1) the control run to a checkpoint gives the stored series 5
    log of s2_Gc_s8 exactly and the same kept weights; (2) the same stage 1 weights with a new stage 2 seed
    start from the same validation (step 0) and differ from the control afterwards."""
    import torch

    cfg = {**cfg, "out": str(out), "_stop_at": steps}
    prepare(cfg)
    jobs = job_list(cfg)
    ctl, w, t = jobs[0]
    new, _, t_new = next(j for j in jobs[1:] if j[1] == w)
    src = ROOT / cfg["series6"]["from"]
    name = f"s2_{ARM}_s{t}"
    ref = json.loads((src / f"{name}.json").read_text())["log"]
    logs = []
    for sub, seed in ((ctl, t), (new, t_new)):
        t0 = time.time()
        assert graph_s3.run_stage2(sub, ARM, seed) is None
        sec = time.time() - t0
        ck = torch.load(ROOT / sub["out"] / f"s2_{ARM}_s{seed}.ckpt", map_location="cpu", weights_only=False)
        logs.append([strip_seconds(x) for x in ck["loop"]["log"]])
        if sub is ctl:
            sec_ctl = sec
    n = steps // cfg["stage2"]["log_every"] + 1
    assert len(logs[0]) == n and logs[0] == [strip_seconds(x) for x in ref[:n]], "control differs from series 5"
    kept = [s for s in sorted(graph_s3.snapshot_steps(cfg["stage2"])) if s <= steps]
    for s in kept:
        a, b = (torch.load(d / "snap" / f"{name}_step{s:05d}.pt", map_location="cpu") for d in (src, ROOT / ctl["out"]))
        assert all(torch.equal(a[k], b[k]) for k in a), s
    assert logs[1][0] == logs[0][0], "step 0 validation must not depend on the stage 2 seed"
    assert all(x != y for x, y in zip(logs[0][1:], logs[1][1:])), "a new stage 2 seed must change the run"
    print(f"check passed: control w{w} / stage 2 seed {t}: {n} log entries to step {steps} equal series 5 {name} exactly, "
          f"kept weights at steps {kept} equal bit for bit ({sec_ctl:.0f} s alone); stage 2 seed {t_new} from the same "
          f"weights: step 0 equal, every later entry differs")


def reading(a, b, na, nb, s6):
    """Row of the protocol from the jump counts of the two groups (a: weights that jumped in series 5)."""
    n = len(s6["stage2_seeds"]) * len(s6["weights"]["jump"])
    if na < n or nb < n:
        return f"series not finished ({na} and {nb} of {n} runs per group): no reading"
    if a >= s6["decide_min"] and b <= s6["decide_max"]:
        return (f">= {s6['decide_min']} of {n} against <= {s6['decide_max']} of {n}: the stage 1 weights decide; "
                "next: look for what distinguishes these weights")
    if abs(a - b) < s6["close_below"]:
        return (f"the counts are close (difference {abs(a - b)}, below {s6['close_below']} of {n}): the course of stage 2 "
                "decides; next: change the stage 2 procedure")
    return "neither row: described; the decision is the user's"


def cmd_report(cfg):
    out, sc, s6 = ROOT / cfg["out"], cfg["stage2"], cfg["series6"]
    src = ROOT / s6["from"]
    thr, gate = cfg["criterion"]["jump_val_10x7"], sc["gate_stage0_10x7"]
    seeds = s6["stage2_seeds"]
    gs = groups(cfg)
    L = ["# Part 4, series 6: what decides the jump, the stage 1 weights or the stage 2 seed", "",
         f"Protocol: docs/plan_part4.md (series 6), configs/graph_s6.yaml. Stage 2 only, exactly the stage 2 of series 5 "
         f"(40000 steps, every 4th a stage 0 step, switch at policy step {sc['switch_at']} = step 7333, best point by 10x7 "
         f"judge validation after the switch). Stage 1 weights: the finished series 5 weights of seeds "
         f"{', '.join(str(w) for _, w in gs)} ({s6['from']}/a_Gc_s<seed>.pt, copied). Stage 2 seeds "
         f"{seeds[0]}-{seeds[-1]}: the seed of torch, numpy and python in stage 2 (batch order and everything else random "
         f"there). Slots: {SLOTS} Steps below are total stage 2 steps.", "",
         "## Table 1. Runs", "",
         f"| stage 1 weights | series 5 run of these weights | stage 2 seed | slots | best step | val 5x5 | best val 10x7 | >= {thr:.0f}% | "
         f"V-sign 10x7 at start / best / end | sign kept (>= {gate:.0f}% at best) | largest rise after the switch | "
         "probe 5x5 | probe 10x7 [95%] |", "|---:|---|---:|---|---:|---:|---:|---|---|---|---|---:|---:|"]
    two = (out / "done_at_2_slots.txt").read_text().split() if (out / "done_at_2_slots.txt").exists() else None
    res, orig = {}, {}
    for g, w in gs:
        o = json.loads((src / f"s2_{ARM}_s{w}.json").read_text())
        orig[w] = o
        d = out / f"w{w}"
        probe = json.loads((d / "probe.json").read_text()) if (d / "probe.json").exists() else {}
        for t in seeds:
            slots = "2" if two is None or f"w{w}:{t}" in two else "2, then 6 from step 13500" if (w, t) in ((9, 104), (6, 104)) else "6"
            head = f"| s{w} | {'jump' if g == 'jump' else 'no jump'} ({o['best_val_10x7']:.1f}) | {t} | {slots} | "
            p = d / f"s2_{ARM}_s{t}.json"
            if not p.exists():
                ck = d / f"s2_{ARM}_s{t}.ckpt"
                st = "not started"
                if ck.exists():
                    import torch
                    st = f"at checkpoint {torch.load(ck, map_location='cpu', weights_only=False)['step']}"
                L.append(head + st + " |||||||||")
                continue
            r = json.loads(p.read_text())
            log = r["log"]
            bl = next(x for x in log if x["step"] == r["best_step"])
            jump, kept = r["best_val_10x7"] >= thr, bl["stage0"]["10x7"] >= gate
            res[w, t] = (jump, kept, r["best_val_10x7"])
            rs, ra, rb = rise(log, sc)
            pr, pc = probe.get(f"{ARM}_s{t}"), "- | -"
            if pr:
                lo, hi = wilson(sum(pr["10x7"]), len(pr["10x7"]))
                pc = f"{100 * np.mean(pr['5x5']):.1f} | {100 * np.mean(pr['10x7']):.1f} [{lo:.0f}-{hi:.0f}]"
            L.append(head + f"{r['best_step']} | {bl['val']['5x5']:.1f} | {r['best_val_10x7']:.1f} | {'yes' if jump else 'no'} | "
                     f"{log[0]['stage0']['10x7']:.1f} / {bl['stage0']['10x7']:.1f} / {log[-1]['stage0']['10x7']:.1f} | "
                     f"{'yes' if kept else 'no'} | {rb - ra:+.1f} at {rs} ({ra:.1f} -> {rb:.1f}) | {pc} |")
    L += ["", "## Table 2. Jumps by stage 1 weights", "",
          f"A jump: best 10x7 judge validation after the switch >= {thr:.0f}%. The series 5 runs (stage 2 seed = stage 1 "
          "seed) are shown for reference and are not counted.", "",
          f"| stage 1 weights | series 5 run (not counted) | runs finished | jumps | stage 2 seeds with a jump | best val 10x7 by stage 2 seed {seeds[0]}-{seeds[-1]} | V sign kept at best |",
          "|---:|---|---:|---:|---|---|---:|"]
    cnt = {}
    for g, w in gs:
        fin = [t for t in seeds if (w, t) in res]
        j = [t for t in fin if res[w, t][0]]
        cnt.setdefault(g, [0, 0])
        cnt[g][0] += len(j)
        cnt[g][1] += len(fin)
        L.append(f"| s{w} | {'jump' if g == 'jump' else 'no jump'}, {orig[w]['best_val_10x7']:.1f} at {orig[w]['best_step']} | {len(fin)} | "
                 f"{len(j)} | {' '.join(map(str, j)) or '-'} | {' / '.join(f'{res[w, t][2]:.1f}' if (w, t) in res else '-' for t in seeds)} | "
                 f"{sum(res[w, t][1] for t in fin)} |")
    (a, na), (b, nb) = cnt["jump"], cnt["no_jump"]
    wj, wn = (" and ".join(f"s{w}" for w in s6["weights"][g]) for g in ("jump", "no_jump"))
    L += ["", "## Primary metric and reading", "",
          f"- weights {wj} (jump in series 5): **{a} of {na}** runs jump;",
          f"- weights {wn} (no jump in series 5): **{b} of {nb}** runs jump.", "",
          f"Row of the protocol: {reading(a, b, na, nb, s6)}.", ""]
    c = s6["control"]
    name = f"s2_{ARM}_s{c['stage2_seed']}"
    L += ["## Reproducibility control (not counted)", "",
          f"Stage 2 from the weights s{c['weights']} with the stage 2 seed {c['stage2_seed']} of series 5, rerun here "
          f"({cfg['out']}/control) and compared with {s6['from']}/{name}.", ""]
    if (out / "control" / f"{name}_vhist.json").exists():
        r = compare_stage2(src, out / "control", name, graph_s3.snapshot_steps(sc))
        (out / "control" / "compare.json").write_text(json.dumps(r))
        L += [f"- log ({r['log_entries'][1]} entries: losses, judge validation of both sizes, stage 0 V-sign; wall time aside): "
              + ("equal entry by entry;" if r["log_equal"] else f"**differs**, first at step {r['first_log_diff_step']};"),
              f"- best point: series 5 {r['best'][0][1]:.1f} at step {r['best'][0][0]}, here {r['best'][1][1]:.1f} at step {r['best'][1][0]};",
              f"- weights (best point, last step and the {r['weight_files'] - 2} kept steps): "
              + ("equal bit for bit;" if not r["weight_files_differ"] else f"**differ** in {len(r['weight_files_differ'])} of {r['weight_files']} files: {' '.join(r['weight_files_differ'])};"),
              f"- V pre-activations from the kept weights of steps {', '.join(map(str, sc['snapshots']['steps']))}: "
              + ("equal." if r["vhist_equal"] else "**differ**."), "",
              "Control " + ("passed: the rerun equals the series 5 run bit for bit." if r["equal"]
                            else "**failed**: the rerun is not the series 5 run; the series is not comparable with series 5 as it stands.")]
        if not r["equal"]:                   # protocol: said first, and no reading before the user has seen it
            L.insert(2, "**The reproducibility control failed (see its section): no reading is given.**\n")
            i = next(i for i, x in enumerate(L) if x.startswith("Row of the protocol:"))
            L[i] = "Row of the protocol: not given, the control failed."
    else:
        L += ["Not finished."]
    sha = json.loads((out / "stage1_sha256.json").read_text()) if (out / "stage1_sha256.json").exists() else {}
    L += ["", "## Table 3. V on the stage 0 10x7 validation at steps 250, 500, 1000 of stage 2", "",
          "From the weights kept at those steps, as Table 2 of docs/graph_s5.md (same bins: <-10, -10..-3, -3..-1, -1..0, "
          "0..1, 1..3, 3..10, >10).", "",
          "| stage 1 weights | stage 2 seed | V-sign 250 / 500 / 1000 | stage 0 loss 0-500 / 500-1000 | V hist 250 | V hist 500 | V hist 1000 |",
          "|---:|---:|---|---|---|---|---|"]
    st = sc["snapshots"]["steps"]
    for _, w in gs:
        for t in seeds:
            vp = out / f"w{w}" / f"s2_{ARM}_s{t}_vhist.json"
            if (w, t) not in res or not vp.exists():
                continue
            vh = json.loads(vp.read_text())
            lg = {x["step"]: x for x in json.loads((out / f"w{w}" / f"s2_{ARM}_s{t}.json").read_text())["log"]}
            L.append(f"| s{w} | {t} | " + " / ".join(f"{vh[str(k)]['10x7']['acc']:.1f}" for k in st)
                     + f" | {lg[500]['loss_stage0']:.3f} / {lg[1000]['loss_stage0']:.3f} | "
                     + " | ".join(fmt_hist(np.array(vh[str(k)]["10x7"]["pre"])) for k in st) + " |")
    L += ["", "## Kept (runs/graph_s6, not in git)", "",
          "- w<stage 1 seed>/s2_Gc_s<stage 2 seed>.json (log), .pt (best point), _last.pt, _vhist.json, snap/ (weights at "
          "the steps of series 5: 250, 500, 1000 and every validation 7000-14000), probe.json; control/ the same under "
          "the series 5 names.",
          "- stage 1 weights used, sha256 of the series 5 files: "
          + "; ".join(f"s{w} {h[:16]}" for w, h in sorted(sha.items(), key=lambda x: int(x[0]))) + ".", ""]
    (ROOT / cfg["report"]).write_text("\n".join(L))
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "run", "probe", "report"])
    ap.add_argument("--config", default="configs/graph_s6.yaml")
    ap.add_argument("--out", help="check: scratch directory (relative to the project root or absolute)")
    ap.add_argument("--steps", type=int, default=1500, help="check: stage 2 steps (a checkpoint step)")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    if a.cmd == "check":
        assert a.out, "check needs --out (a scratch directory)"
        cmd_check(cfg, a.out, a.steps)
    else:
        {"run": cmd_run, "probe": cmd_probe, "report": cmd_report}[a.cmd](cfg)


if __name__ == "__main__":
    main()
