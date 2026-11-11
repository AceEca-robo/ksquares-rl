"""Part 4, series 2 of docs/plan_part4.md: transfer of the parity skill to 10x7.

    python scripts/graph_s2.py run       # 2a, 2b, 2c; finished runs are skipped
    python scripts/graph_s2.py report    # docs/graph_s2.md

configs/graph_s2.yaml. Every run is deterministic. 2d is read from the logs of 2b.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import time

import numpy as np

from _common import ROOT, load_config

from graph_hand import CKPT_EVERY, Limits, load_checkpoint, save_checkpoint, settled_boards, set_deterministic, stop_requested
from parity_hand import augment

SIZES = {"5x5": (5, 5), "10x7": (10, 7)}
PLANES = 17                     # stored; a 12-plane net takes the first 12


def build_data(cfg):
    """5x5 train + val, 10x7 val (as series 1), then 10x7 train; out/data.npz."""
    from agents.az_net import Codec

    path = ROOT / cfg["out"] / "data.npz"
    if path.exists():
        return
    rng = random.Random(cfg["data_seed"])
    nt, nv = cfg["n_train"], cfg["n_val"]
    parts = {"5x5": [settled_boards(5, 5, nt + nv, rng)], "10x7": [settled_boards(10, 7, nv, rng)]}
    parts["10x7"].append(settled_boards(10, 7, nt, rng))
    d = {}
    for k, ps in parts.items():
        c = Codec(*SIZES[k], planes=PLANES)
        d[f"x{k}"] = np.concatenate([c.encode_boards([b for b, _ in p[i:i + 2000]]).astype(np.float32)
                                     for p in ps for i in range(0, len(p), 2000)])
        d[f"y{k}"] = np.array([t for p in ps for _, t in p], dtype=np.float32)
    s1 = np.load(ROOT / "runs/graph_s1/data.npz")
    assert np.array_equal(s1["x5x5"], d["x5x5"][:, :12]) and np.array_equal(s1["x10x7"], d["x10x7"][:nv, :12]), \
        "series 1 data not reproduced"
    np.savez_compressed(path, **d)


def splits(cfg, data):
    """{size: (train x, train y, val x, val y)}; 10x7 stores val first."""
    nt, nv = cfg["n_train"], cfg["n_val"]
    a = data["x5x5"], data["y5x5"]
    b = data["x10x7"], data["y10x7"]
    return {"5x5": (a[0][:nt], a[1][:nt], a[0][nt:], a[1][nt:]),
            "10x7": (b[0][nv:], b[1][nv:], b[0][:nv], b[1][:nv])}


def run_one(args):
    cfg, part, arm, seed = args
    import torch
    import torch.nn.functional as F

    from agents.az_net import Codec, make_net

    name = f"{part}_{arm}_s{seed}"
    path = ROOT / cfg["out"] / f"{name}.json"
    if path.exists():
        return name
    if stop_requested(ROOT / cfg["out"]):
        return None                          # series stopping: do not start new runs
    lim = Limits(cfg["limits"])
    dev, r, pc, steps = cfg["device"], cfg["recipe"], cfg["parts"][part], cfg["steps"]
    z = np.load(ROOT / cfg["out"] / "data.npz")
    sp = splits(cfg, {k: z[k] for k in z.files})
    set_deterministic(seed)
    ac = cfg["arms"][arm]
    ac = load_config(ROOT / ac["config"]) if "config" in ac else dict(ac)
    ac["net"] = {**ac["net"], "v_out": r["v_out"]}
    if ac["net"].get("arch") == "graph":
        ac["net"].update(readout_norm=r["readout_norm"], readout_init=r["readout_init"])
    net = make_net(ac).to(dev)
    planes, graph = net.planes, ac["net"].get("arch") == "graph"
    opt = torch.optim.Adam(net.parameters(), lr=r["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    np_rng = np.random.default_rng(seed)
    codecs = {k: Codec(*SIZES[k], planes=planes) for k in SIZES}
    edges_pre, edges_out = np.linspace(-10, 10, 21), np.linspace(-1, 1, 21)

    def validate():
        net.eval()
        res = {}
        with torch.no_grad():
            for k, (_, _, x, y) in sp.items():
                pre = []
                for i in range(0, len(x), 500):
                    xv = x[i:i + 500, :planes]
                    if zero:
                        xv = xv.copy()
                        xv[:, zero] = 0
                    net(torch.from_numpy(xv).to(dev))
                    pre.append(net.v_pre.cpu().numpy())
                pre = np.concatenate(pre)
                bo = np.tanh(pre / 2)
                res[k] = {"acc": 100 * float(np.mean((pre > 0) == (y > 0))),
                          "sat": float(np.mean(np.abs(bo) > cfg["sat"])),
                          "hist_pre": np.histogram(np.clip(pre, -10, 10), edges_pre)[0].tolist(),
                          "hist_out": np.histogram(bo, edges_out)[0].tolist()}
        net.train()
        return res

    # RL series 2' options (all off in series 2): soft switch, fixed switch step, zeroed input planes,
    # first-loss prediction, final weights
    mix, fixed, zero = pc.get("mix_steps", 0), pc.get("switch_at"), ac.get("zero_planes", [])

    def train_size(step):
        if switch is None or step > switch + mix:
            return size
        return "5x5" if step % 2 else "10x7"

    ckpt = ROOT / cfg["out"] / f"{name}.ckpt"
    start, st = load_checkpoint(ckpt, net, opt, sched, np_rng)
    if st:                                   # resume: loop state as saved at the checkpoint
        size, switch, acc_loss, log = st["size"], st["switch"], st["acc_loss"], st["log"]
        t0 = time.time() - st["seconds"]
        print(f"{name}: resumed at step {start}", flush=True)
    else:
        size, switch, acc_loss, t0 = "5x5", None, 0.0, time.time()
        log = [{"step": 0, "train_on": size, "val": validate()}]
    net.train()
    for step in range(start + 1, steps + 1):
        ts = train_size(step)
        X, Y, _, _ = sp[ts]
        idx = np_rng.integers(0, len(X), r["batch"])
        xb = X[idx, :planes]
        if zero:
            xb = xb.copy()
            xb[:, zero] = 0
        if not graph:
            c = codecs[ts]
            xb = augment(np_rng, c, xb.reshape(len(xb), planes, -1))[0].reshape(-1, planes, c.rows, c.cols)
        v = net(torch.from_numpy(xb).to(dev))[1]
        loss = F.binary_cross_entropy_with_logits(v, torch.from_numpy((Y[idx] > 0).astype(np.float32)).to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        total = float(torch.nn.utils.clip_grad_norm_(net.parameters(), r["clip"]))
        opt.step()
        sched.step()
        acc_loss += loss.item()
        lim.tick()
        if step % cfg["log_every"] == 0:
            g = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in net.v_fc.parameters() if p.grad is not None))
            val = validate()
            log.append({"step": step, "train_on": "mix" if switch is not None and step <= switch + mix and mix else size, "loss": acc_loss / cfg["log_every"], "readout_grad": g,
                        "grad_total": total, "lr": sched.get_last_lr()[0], "val": val, "seconds": time.time() - t0})
            acc_loss = 0.0
            if pc.get("predict") and switch is not None and step == switch + cfg["log_every"]:
                fl = log[-1]["loss"]
                pred = {"name": name, "switch": switch, "first_loss": fl, "threshold": pc["predict"],
                        "prediction": "learns" if fl < pc["predict"] else "collapses", "written": time.time(),
                        "val_10x7_now": val["10x7"]["acc"]}
                (ROOT / cfg["out"] / f"{name}.pred.json").write_text(json.dumps(pred))
            if switch is None and (step == fixed if fixed else
                                   pc["curriculum"] and val["5x5"]["acc"] >= cfg["pass_acc"]):
                switch, size = step, "10x7"
            if step % (10 * cfg["log_every"]) == 0:
                print(f"{name}: step {step} on {log[-1]['train_on']} 5x5 {val['5x5']['acc']:.1f} "
                      f"10x7 {val['10x7']['acc']:.1f} ({log[-1]['seconds']:.0f} s)", flush=True)
        if step % CKPT_EVERY == 0 and step < steps:
            save_checkpoint(ckpt, step, net, opt, sched, np_rng,
                            {"size": size, "switch": switch, "acc_loss": acc_loss, "log": log,
                             "seconds": time.time() - t0})
            if stop_requested(ROOT / cfg["out"]) or step == cfg.get("_stop_at"):
                print(f"{name}: stopped at checkpoint {step}", flush=True)
                return None
    res = {"part": part, "arm": arm, "seed": seed, "recipe": r, "steps": steps, "switch": switch, "log": log,
           **summary(cfg, log)}
    if cfg.get("save_weights"):
        torch.save(net.state_dict(), path.with_suffix(".pt"))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.rename(path)
    ckpt.unlink(missing_ok=True)
    print(f"{name}: switch {switch} click10 {res['click10']} hold10 {res['hold10']}", flush=True)
    return name


def summary(cfg, log):
    def first(k):
        return next((r["step"] for r in log if r["val"][k]["acc"] >= cfg["pass_acc"]), None)

    c5, c10 = first("5x5"), first("10x7")
    hold = None if c10 is None else all(r["val"]["10x7"]["acc"] >= cfg["hold_acc"] for r in log if r["step"] >= c10)
    return {"click5": c5, "click10": c10, "hold10": hold, "best_10x7": max(r["val"]["10x7"]["acc"] for r in log),
            "last_5x5": log[-1]["val"]["5x5"]["acc"], "last_10x7": log[-1]["val"]["10x7"]["acc"]}


def fisher(a, n, b, m):
    """2x2 Fisher, a of n vs b of m: (p that the first share is this high or higher, two-sided p)."""
    k, N = a + b, n + m
    pmf = {x: math.comb(n, x) * math.comb(m, k - x) / math.comb(N, k) for x in range(max(0, k - m), min(n, k) + 1)}
    return sum(p for x, p in pmf.items() if x >= a), sum(p for p in pmf.values() if p <= pmf[a] * (1 + 1e-9))


def diagnose(cfg, r):
    """Class of a 2b run without a 10x7 click, by the rule of the config."""
    if r["switch"] is None:
        return "no 10x7 phase", {}
    d = cfg["diagnosis"]
    rows = [x for x in r["log"][1:] if x["step"] > 0.75 * r["steps"] and x["train_on"] == "10x7"]
    m = {"loss": float(np.median([x["loss"] for x in rows])), "grad": float(np.median([x["readout_grad"] for x in rows])),
         "sat": float(np.median([x["val"]["10x7"]["sat"] for x in rows]))}
    if m["loss"] < d["loss_fit"]:
        return "fits train without the rule", m
    return ("frozen" if m["grad"] < d["grad_frozen"] else "does not find the rule"), m


def load(cfg, part, arm, seeds):
    out = ROOT / cfg["out"]
    return [json.loads((out / f"{part}_{arm}_s{s}.json").read_text()) for s in seeds if (out / f"{part}_{arm}_s{s}.json").exists()]


def cmd_run(cfg):
    build_data(cfg)
    jobs = [(cfg, part, arm, s) for part, pc in cfg["parts"].items() for s in pc["seeds"] for arm in pc["arms"]]
    with mp.get_context("spawn").Pool(cfg["jobs"], maxtasksperchild=1) as pool:
        for name in pool.imap_unordered(run_one, jobs):
            if name:
                print("done", name, flush=True)
    if stop_requested(ROOT / cfg["out"]):
        print("series stopped at checkpoints", flush=True)


def fmt(x):
    return "-" if x is None else str(x)


def cmd_report(cfg):
    pc = cfg["parts"]
    L = ["# Part 4, series 2: transfer to 10x7", "",
         f"Protocol: docs/plan_part4.md, configs/graph_s2.yaml. Recipe B + V-sign, {cfg['steps']} steps, deterministic, "
         f"validation every {cfg['log_every']} steps. Click: first validation >= {cfg['pass_acc']}%; "
         f"hold: every later 10x7 validation >= {cfg['hold_acc']}%.", ""]
    res, counts = {}, {}
    for part in pc:
        for arm in pc[part]["arms"]:
            rs = load(cfg, part, arm, pc[part]["seeds"])
            res[part, arm] = rs
            n = len(pc[part]["seeds"])
            k = sum(r["click10"] is not None for r in rs)
            h = sum(bool(r["hold10"]) for r in rs)
            counts[part, arm] = (k, h, n)
            L += [f"## {part} {arm}: 10x7 click {k}/{n}, click and hold {h}/{n}", "",
                  "| seed | 5x5 click | switch | 10x7 click | click - switch | hold | best 10x7 | last 10x7 | last 5x5 |",
                  "|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
            for r in rs:
                dl = None if r["click10"] is None or r["switch"] is None else r["click10"] - r["switch"]
                L.append(f"| {r['seed']} | {fmt(r['click5'])} | {fmt(r['switch'])} | {fmt(r['click10'])} | {fmt(dl)} | "
                         f"{fmt(r['hold10'])} | {r['best_10x7']:.1f} | {r['last_10x7']:.1f} | {r['last_5x5']:.1f} |")
            L.append("")
    a, b = counts["2a", "Gc"], counts["2b", "Gc"]
    pone, ptwo = fisher(b[0], b[2], a[0], a[2])
    L += ["## Fisher and criterion", "", f"2b {b[0]}/{b[2]} vs 2a {a[0]}/{a[2]}: one-sided p = {pone:.3g}, two-sided p = {ptwo:.3g}.", ""]
    c = counts["2c", "C"]
    pc2 = fisher(c[0], c[2], b[0], b[2])[1]
    same = c[0] > 0 and c[0] / c[2] >= b[0] / b[2] - 0.2
    row = lambda k: "series 3" if k >= 8 else ("series 2'" if k >= 4 else "stop: the graph does not transfer")
    L += [f"C {c[0]}/{c[2]} vs 2b, two-sided p = {pc2:.3g}: \"the convolutions click the same\" = {same}. "
          f"C+h {counts['2c', 'Ch'][0]}/{counts['2c', 'Ch'][2]} is not a test (its plane 15 is the target).", "",
          f"By clicks: 2b row = {row(b[0])}. By click + hold: {row(b[1])}." + (" The two rows differ." if row(b[0]) != row(b[1]) else ""),
          f"Next: {'series 3 (the school, not the graph)' if same else row(b[0])}.", ""]
    L += ["## 2d: 2b runs without a 10x7 click", "",
          "Medians over the last quarter of the logs. Histogram: raw V on 10x7 validation, bins of 1 from -10 to 10.", "",
          "| seed | class | train loss | readout grad | sat | V at the switch | V at the end |", "|---:|---|---:|---:|---:|---|---|"]
    for r in res["2b", "Gc"]:
        if r["click10"] is not None:
            continue
        cl, m = diagnose(cfg, r)
        at = next((x for x in r["log"] if x["step"] == r["switch"]), None)
        L.append(f"| {r['seed']} | {cl} | {m.get('loss', float('nan')):.3f} | {m.get('grad', float('nan')):.2e} | "
                 f"{m.get('sat', float('nan')):.2f} | {' '.join(map(str, at['val']['10x7']['hist_pre'])) if at else '-'} | "
                 f"{' '.join(map(str, r['log'][-1]['val']['10x7']['hist_pre']))} |")
    L.append("")
    (ROOT / "docs/graph_s2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--config", default="configs/graph_s2.yaml")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    (ROOT / cfg["out"]).mkdir(parents=True, exist_ok=True)
    {"run": cmd_run, "report": cmd_report}[a.cmd](cfg)


if __name__ == "__main__":
    main()
