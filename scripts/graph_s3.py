"""Part 4, series 3 of docs/plan_part4.md: parity probe on real positions, two stages per run.

    python scripts/graph_s3.py run       # stage 1 (school (a) on stage 0) then stage 2 (judge positions)
    python scripts/graph_s3.py probe     # 10x7 / 5x5 probes of the best stage 2 nets, once, after training
    python scripts/graph_s3.py report    # docs/graph_s3.md

configs/graph_s3.yaml. Stage 1 is graph_s2.run_one unchanged; finished stages are skipped, stage 2
checkpoints every 500 steps and resumes bit-exact; touch out/STOP to stop the series at checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import shutil
import time

import numpy as np

from _common import ROOT, load_config

import graph_s2
import parity_hand
from graph_hand import CKPT_EVERY, Limits, correct_rate, judge_sets, load_checkpoint, probe_sets, save_checkpoint, \
    set_deterministic, stop_requested
from graph_s2 import splits
from parity_probe import mcnemar, wilson


def arm_config(cfg, arm):
    ac = cfg["arms"][arm]
    zero = ac.get("zero_planes", [])
    ac = load_config(ROOT / ac["config"]) if "config" in ac else dict(ac)
    r = cfg["recipe"]
    ac["net"] = {**ac["net"], "v_out": r["v_out"]}
    if ac["net"].get("arch") == "graph":
        ac["net"].update(readout_norm=r["readout_norm"], readout_init=r["readout_init"])
    return ac, zero


def build_net(cfg, arm):
    from agents.az_net import make_net

    ac, zero = arm_config(cfg, arm)
    return make_net(ac), zero, ac["net"].get("arch") == "graph"


def prepare(cfg):
    out = ROOT / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "data.npz").exists():
        shutil.copy(ROOT / "runs/graph_s2/data.npz", out / "data.npz")
    ru = cfg.get("reuse_stage1")
    if ru:
        src_cfg = load_config(ROOT / "configs/graph_s2p.yaml")
        keys = ["recipe", "steps", "log_every", "pass_acc", "n_train", "n_val", "data_seed"]
        assert all(src_cfg[k] == cfg[k] for k in keys), "series 2' stage 1 config differs"
        assert src_cfg["parts"]["a"]["mix_steps"] == cfg["parts"]["a"]["mix_steps"]
        for s in cfg["parts"]["a"]["seeds"]:
            for ext in (".json", ".pt"):
                src, dst = ROOT / ru["from"] / f"a_{ru['arm']}_s{s}{ext}", out / f"a_{ru['arm']}_s{s}{ext}"
                if not dst.exists():
                    shutil.copy(src, dst)


def zeroed(x, zero):
    if zero:
        x = x.copy()
        x[:, zero] = 0
    return x


def run_stage2(cfg, arm, seed):
    import torch
    import torch.nn.functional as F

    out = ROOT / cfg["out"]
    name = f"s2_{arm}_s{seed}"
    path = out / f"{name}.json"
    if path.exists():
        return name
    if stop_requested(out):
        return None
    sc, dev = cfg["stage2"], cfg["device"]
    lim = Limits(cfg["limits"])
    set_deterministic(seed)
    net, zero, graph = build_net(cfg, arm)
    net.load_state_dict(torch.load(out / f"a_{arm}_s{seed}.pt", map_location="cpu"))
    net.to(dev)
    planes = net.planes
    _, sp = judge_sets(cfg)
    par = {}
    for b, (tr, va) in sp.items():
        ct, xt_, mt = parity_hand.parity_arrays(b, tr, planes)
        cv, xv, mv = parity_hand.parity_arrays(b, va, planes)
        par[b] = ((ct, zeroed(xt_, zero), mt), (cv, zeroed(xv, zero), mv))
    z = np.load(out / "data.npz")
    s0 = splits(cfg, {k: z[k] for k in z.files})
    s0val = {k: (zeroed(s0[k][2][:, :planes], zero), s0[k][3]) for k in ("5x5", "10x7")}
    opt = torch.optim.Adam(net.parameters(), lr=sc["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, sc["steps"])
    np_rng = np.random.default_rng(seed)
    # RL series 3': a share of the stage 2 steps are stage 0 steps (10x7 only, V-sign loss) to the end;
    # switch_at then counts policy steps. Off (share 0) in series 3.
    share = sc.get("stage0_share", 0.0)
    if share:
        from agents.az_net import Codec

        s0x = zeroed(s0["10x7"][0][:, :planes], zero)
        s0y = (s0["10x7"][1] > 0).astype(np.float32)
        c0 = Codec(*graph_s2.SIZES["10x7"], planes=planes)

    def is_s0(step):
        return bool(share) and math.floor(step * share) > math.floor((step - 1) * share)

    def policy_steps(step):
        return step - math.floor(step * share) if share else step

    def validate():
        net.eval()
        val = {k: 100 * float(np.mean(correct_rate(net, *par[k][1], dev))) for k in par}
        s0acc = {}
        with torch.no_grad():
            for k, (x, y) in s0val.items():
                pre = []
                for i in range(0, len(x), 500):
                    net(torch.from_numpy(x[i:i + 500]).to(dev))
                    pre.append(net.v_pre.cpu().numpy())
                s0acc[k] = 100 * float(np.mean((np.concatenate(pre) > 0) == (y > 0)))
        net.train()
        return val, s0acc

    def policy_step(b):
        codec, X, M = par[b][0]
        idx = np_rng.integers(0, len(X), sc["batch"])
        x, m = parity_hand.augment(np_rng, codec, X[idx], M[idx], on=not graph)
        xt = torch.from_numpy(x.reshape(-1, planes, codec.rows, codec.cols)).to(dev)
        logits = net(xt)[0]
        legal = xt[:, 1].flatten(1)[:, net.line_cells(codec.rows, codec.cols)] > 0
        lp = F.log_softmax(logits.masked_fill(~legal, -1e9), 1)
        loss = -torch.logsumexp(lp.masked_fill(~torch.from_numpy(m).to(dev), -1e9), 1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), sc["clip"])
        opt.step()
        sched.step()
        lim.tick()
        return loss.item()

    ckpt = out / f"{name}.ckpt"
    start, st = load_checkpoint(ckpt, net, opt, sched, np_rng)
    if st:
        log, acc_loss, best = st["log"], st["acc_loss"], st["best"]
        acc_s0 = st.get("acc_s0", 0.0)
        t0 = time.time() - st["seconds"]
        print(f"{name}: resumed at step {start}", flush=True)
    else:
        v, s = validate()
        log, acc_loss, best, t0 = [{"step": 0, "val": v, "stage0": s}], 0.0, None, time.time()
        acc_s0 = 0.0
    net.train()
    for step in range(start + 1, sc["steps"] + 1):
        b = "5x5" if policy_steps(step) <= sc["switch_at"] else "10x7"
        if is_s0(step):
            idx = np_rng.integers(0, len(s0x), sc["batch"])
            xb = s0x[idx]
            if not graph:
                xb = graph_s2.augment(np_rng, c0, xb.reshape(len(xb), planes, -1))[0].reshape(-1, planes, c0.rows, c0.cols)
            v0 = net(torch.from_numpy(np.ascontiguousarray(xb)).to(dev))[1]
            loss = F.binary_cross_entropy_with_logits(v0, torch.from_numpy(s0y[idx]).to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), sc["clip"])
            opt.step()
            sched.step()
            acc_s0 += loss.item()
            lim.tick()
            b = "stage0"
        else:
            acc_loss += policy_step(b)
        if step % sc["log_every"] == 0:
            v, s = validate()
            npol = max(1, sc["log_every"] - (math.floor(step * share) - math.floor((step - sc["log_every"]) * share)))
            log.append({"step": step, "policy_step": policy_steps(step), "train_on": b, "loss": acc_loss / npol,
                        "loss_stage0": acc_s0 / max(1, sc["log_every"] - npol) if share else None,
                        "val": v, "stage0": s, "seconds": time.time() - t0})
            acc_loss = acc_s0 = 0.0
            ok = policy_steps(step) > sc["switch_at"] or not sc["select_after_switch"]
            if ok and (best is None or v["10x7"] > best["val"]):
                best = {"val": v["10x7"], "step": step, "state": {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}}
            if step % (10 * sc["log_every"]) == 0:
                print(f"{name}: step {step} val 5x5 {v['5x5']:.1f} 10x7 {v['10x7']:.1f} "
                      f"stage0 10x7 {s['10x7']:.1f} ({log[-1]['seconds']:.0f} s)", flush=True)
        if step % CKPT_EVERY == 0 and step < sc["steps"]:
            save_checkpoint(ckpt, step, net, opt, sched, np_rng,
                            {"log": log, "acc_loss": acc_loss, "acc_s0": acc_s0, "best": best, "seconds": time.time() - t0})
            if stop_requested(out) or step == cfg.get("_stop_at"):
                print(f"{name}: stopped at checkpoint {step}", flush=True)
                return None
    return finish(cfg, name, path, ckpt, net, best, log, arm, seed, sc)


def finish(cfg, name, path, ckpt, net, best, log, arm, seed, sc):
    import torch

    out = ROOT / cfg["out"]
    torch.save(best["state"], out / f"{name}.pt")
    torch.save(net.state_dict(), out / f"{name}_last.pt")
    res = {"arm": arm, "seed": seed, "stage2": sc, "log": log, "best_step": best["step"], "best_val_10x7": best["val"]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.rename(path)
    ckpt.unlink(missing_ok=True)
    print(f"{name}: best 10x7 val {best['val']:.1f} at {best['step']}", flush=True)
    return name


def run_job(args):
    cfg, arm, seed = args
    if graph_s2.run_one((cfg, "a", arm, seed)) is None:
        return None
    return run_stage2(cfg, arm, seed)


def jobs_of(cfg):
    pa = cfg["parts"]["a"]
    return [(cfg, arm, s) for s in pa["seeds"] for arm in pa["arms"]]


def cmd_run(cfg):
    prepare(cfg)
    with mp.get_context("spawn").Pool(cfg["jobs"], maxtasksperchild=1) as pool:
        for name in pool.imap_unordered(run_job, jobs_of(cfg)):
            if name:
                print("done", name, flush=True)
    if stop_requested(ROOT / cfg["out"]):
        print("series stopped at checkpoints", flush=True)


def cmd_probe(cfg):
    import torch

    out, dev = ROOT / cfg["out"], cfg["device"]
    res = {}
    for _, arm, s in jobs_of(cfg):
        net, zero, _ = build_net(cfg, arm)
        net.load_state_dict(torch.load(out / f"s2_{arm}_s{s}.pt", map_location="cpu"))
        net.to(dev).eval()
        for b, (codec, x, mask) in probe_sets(cfg, net.planes).items():
            res.setdefault(f"{arm}_s{s}", {})[b] = correct_rate(net, codec, zeroed(x, zero), mask, dev).astype(int).tolist()
    (out / "probe.json").write_text(json.dumps(res))


def bootstrap(a, b, n, seed, level):
    """a, b: (seeds, positions) 0/1 arrays of two arms on the same positions. Seeds are drawn with
    replacement within each arm, positions with replacement (the same for both arms).
    Returns (observed difference of mean accuracy, one-sided lower bound at level), in points."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for i in range(n):
        pos = rng.integers(0, a.shape[1], a.shape[1])
        sa, sb = rng.integers(0, len(a), len(a)), rng.integers(0, len(b), len(b))
        d[i] = a[sa][:, pos].mean() - b[sb][:, pos].mean()
    return 100 * (a.mean() - b.mean()), 100 * float(np.quantile(d, 1 - level))


def fmt(x, f="{}"):
    return "-" if x is None else f.format(x)


def cmd_report(cfg):
    out, tc, seeds = ROOT / cfg["out"], cfg["test"], cfg["parts"]["a"]["seeds"]
    arms = cfg["parts"]["a"]["arms"]
    probe = json.loads((out / "probe.json").read_text()) if (out / "probe.json").exists() else {}
    L = ["# Part 4, series 3: parity probe on real positions", "",
         "Protocol: docs/plan_part4.md (series 3), configs/graph_s3.yaml. Stage 1: school (a) of series 2' on the "
         "stage 0 target (20000 steps). Stage 2: the judge's positions from the stage 1 weights, 30000 steps, 5x5 to "
         f"{cfg['stage2']['switch_at']}, then 10x7; best point by 10x7 validation after the switch. Stage 0 columns: "
         "V-sign accuracy on the stage 0 10x7 validation (diagnostic only).", "",
         "## Table 1. Runs", "",
         "| arm | seed | s1 5x5 click | s1 10x7 click | s1 last 10x7 | best step | val 5x5 | val 10x7 | probe 5x5 | "
         "probe 10x7 [95%] | stage0 10x7 at start / best / end |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    acc = {}
    for arm in arms:
        for s in seeds:
            p1, p2 = out / f"a_{arm}_s{s}.json", out / f"s2_{arm}_s{s}.json"
            if not p2.exists():
                L.append(f"| {arm} | {s} | " + ("stage 1 done" if p1.exists() else "running") + " |||||||||")
                continue
            r1, r2 = json.loads(p1.read_text()), json.loads(p2.read_text())
            bl = next(x for x in r2["log"] if x["step"] == r2["best_step"])
            pr = probe.get(f"{arm}_s{s}")
            pc = "-"
            if pr:
                k, n = sum(pr["10x7"]), len(pr["10x7"])
                lo, hi = wilson(k, n)
                pc = f"{100 * k / n:.1f} [{lo:.0f}-{hi:.0f}]"
                acc.setdefault(arm, []).append(pr["10x7"])
            L.append(f"| {arm} | {s} | {fmt(r1['click5'])} | {fmt(r1['click10'])} | {r1['last_10x7']:.1f} | {r2['best_step']} | "
                     f"{bl['val']['5x5']:.1f} | {bl['val']['10x7']:.1f} | "
                     f"{100 * np.mean(pr['5x5']):.1f} | {pc} | " if pr else
                     f"| {arm} | {s} | {fmt(r1['click5'])} | {fmt(r1['click10'])} | {r1['last_10x7']:.1f} | {r2['best_step']} | "
                     f"{bl['val']['5x5']:.1f} | {bl['val']['10x7']:.1f} | - | - | ")
            L[-1] += f"{r2['log'][0]['stage0']['10x7']:.1f} / {bl['stage0']['10x7']:.1f} / {r2['log'][-1]['stage0']['10x7']:.1f} |"
    L.append("")
    if all(len(acc.get(a, [])) == len(seeds) for a in arms):
        L += ["## Table 2. Arm means, 10x7 probe", "", "| arm | mean % | seeds |", "|---|---:|---|"]
        for a in arms:
            m = [100 * np.mean(x) for x in acc[a]]
            L.append(f"| {a} | {np.mean(m):.1f} | {' '.join(f'{v:.1f}' for v in m)} |")
        L += ["", f"## Table 3. Bootstrap over seeds and positions ({tc['resamples']} resamples), 10x7 probe", "",
              f"| pair | role | difference, points | one-sided {int(100 * tc['level'])}% lower bound | McNemar 25 seed pairs: "
              "first better p<0.05 / second better p<0.05 / n.s. |", "|---|---|---:|---:|---|"]
        rows = [(tc["primary"], "primary"), (tc["second"], "second (pre-registered)")] + [(p, "descriptive") for p in tc["descriptive"]]
        bounds = {}
        for (x, y), role in rows:
            d, lo = bootstrap(acc[x], acc[y], tc["resamples"], tc["boot_seed"], tc["level"])
            bounds[x, y] = (d, lo)
            w = [mcnemar(acc[y][j], acc[x][i]) for i in range(len(seeds)) for j in range(len(seeds))]
            fb = sum(1 for n01, n10, p in w if p < 0.05 and n01 > n10)
            sb = sum(1 for n01, n10, p in w if p < 0.05 and n10 > n01)
            L.append(f"| {x} - {y} | {role} | {d:+.1f} | {lo:+.1f} | {fb} / {sb} / {len(w) - fb - sb} |")
        d, lo = bounds[tuple(tc["primary"])]
        gc = np.mean([100 * np.mean(x) for x in acc["Gc"]])
        dc, loc = bootstrap(acc["Gc"], acc["C"], tc["resamples"], tc["boot_seed"], tc["level"])
        if lo > 0:
            row = "graph adds something beyond chain planes, late: series 4" if gc > tc["strong_probe"] else \
                "a small gain, still weak: series 4, reported as weak"
        else:
            row = "graph gives nothing beyond chain planes late: stop part 4"
        L += ["", "## Verdict by the protocol", "",
              f"Primary Gc - C+h: {d:+.1f} points, lower bound {lo:+.1f}; Gc probe 10x7 mean {gc:.1f}%. Row: {row}.",
              f"Gc - C: {dc:+.1f}, lower bound {loc:+.1f} (C did not learn the stage 0 rule in stage 1; read as in the plan)."]
        d2, lo2 = bounds[tuple(tc["second"])]
        L.append(f"Second test Gc - C+h14: {d2:+.1f}, lower bound {lo2:+.1f}: "
                 + ("positive." if lo2 > 0 else "not positive."))
    (ROOT / cfg.get("report", "docs/graph_s3.md")).write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "probe", "report"])
    ap.add_argument("--config", default="configs/graph_s3.yaml")
    ap.add_argument("--out")
    ap.add_argument("--steps", type=int, help="smoke test: stage 1 and stage 2 steps")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    if a.out:
        cfg["out"] = a.out
    if a.steps:
        cfg["steps"] = cfg["stage2"]["steps"] = a.steps
        cfg["stage2"]["switch_at"] = a.steps // 2
        cfg["reuse_stage1"] = None
    if cfg.get("stage1_from") and not a.steps:
        copy_stage1(cfg)
    {"run": cmd_run, "probe": cmd_probe, "report": cmd_report}[a.cmd](cfg)


def copy_stage1(cfg):
    """RL series 3': finished stage 1 runs of an earlier series (same stage 1 config) are copied, not rerun."""
    src_dir, out = ROOT / cfg["stage1_from"]["out"], ROOT / cfg["out"]
    src = load_config(ROOT / cfg["stage1_from"]["config"])
    keys = ["recipe", "steps", "log_every", "pass_acc", "n_train", "n_val", "data_seed", "parts", "arms", "seed"]
    assert all(src[k] == cfg[k] for k in keys), "stage 1 config differs from " + cfg["stage1_from"]["config"]
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_dir.glob("a_*.json")):
        for ext in (".json", ".pt"):
            d = out / f.with_suffix(ext).name
            if not d.exists():
                shutil.copy(f.with_suffix(ext), d)


if __name__ == "__main__":
    main()
