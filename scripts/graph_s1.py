"""Part 4, series 1 of docs/plan_part4.md: reliable training of Gc on stage 0, 5x5 only.

    python scripts/graph_s1.py time --n 2000   # timing run of B (not counted), steps per 10 min
    python scripts/graph_s1.py run             # 10 variants x 5 seeds, then R1 on seeds 6-10
    python scripts/graph_s1.py report          # docs/graph_s1.md

configs/graph_s1.yaml. Finished runs are skipped. Every run is deterministic
(graph_hand.set_deterministic). The 10x7 validation is logged, never used.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np

from _common import ROOT, load_config

from graph_hand import Limits, settled_boards, set_deterministic


def boards_data(cfg):
    """5x5 train + val and 10x7 val planes, cached in out/data.npz."""
    path = ROOT / cfg["out"] / "data.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    from agents.az_net import Codec

    rng = random.Random(cfg["data_seed"])
    d = {}
    for (w, h), n in (((5, 5), cfg["n_train"] + cfg["n_val"]), ((10, 7), cfg["n_val"])):
        items = settled_boards(w, h, n, rng)
        c = Codec(w, h, planes=12)
        x = np.concatenate([c.encode_boards([b for b, _ in items[i:i + 2000]]) for i in range(0, n, 2000)])
        d[f"x{w}x{h}"] = x.astype(np.float32)
        d[f"y{w}x{h}"] = np.array([t for _, t in items], dtype=np.float32)
    np.savez_compressed(path, **d)
    return d


def recipe(cfg, name):
    r = dict(cfg["base"])
    if name == "R1":
        r.update(r1_recipe(cfg)[1])
    else:
        r.update(cfg["variants"][name])
    return r


def bounded(v, r):
    import torch

    if r["v_out"] == "tanh":
        return v
    return torch.tanh(v / 2) if r["loss"] == "sign" else v.clamp(-1, 1)   # sign: 2 sigmoid - 1


def run_one(cfg, lim, name, seed, steps, data):
    import torch
    import torch.nn.functional as F

    from agents.az_net import make_net

    out = ROOT / cfg["out"]
    path = out / f"{name}_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    r = recipe(cfg, name)
    dev = cfg["device"]
    set_deterministic(seed)
    ac = load_config(ROOT / cfg["arm"])
    ac["net"] = {**ac["net"], "v_out": r["v_out"], "readout_norm": r["readout_norm"],
                 "readout_init": r["readout_init"]}
    net = make_net(ac).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=r["lr"])
    wu = int(r["warmup"] * steps)

    def lr_at(k):                                          # k = steps done
        if k < wu:
            return (k + 1) / wu
        if not r["cosine"]:
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * (k - wu) / max(1, steps - wu)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    X, Y = data["x5x5"], data["y5x5"]
    n = cfg["n_train"]
    np_rng = np.random.default_rng(seed)
    rows, cols = X.shape[2], X.shape[3]
    vals = {"5x5": (X[n:], Y[n:]), "10x7": (data["x10x7"], data["y10x7"])}
    edges_pre, edges_out = np.linspace(-10, 10, 21), np.linspace(-1, 1, 21)

    def validate():
        net.eval()
        res = {}
        with torch.no_grad():
            for k, (x, y) in vals.items():
                pre, bo = [], []
                for i in range(0, len(x), 500):
                    v = net(torch.from_numpy(x[i:i + 500]).to(dev))[1]
                    pre.append(net.v_pre.cpu().numpy())
                    bo.append(bounded(v, r).cpu().numpy())
                pre, bo = np.concatenate(pre), np.concatenate(bo)
                res[k] = {"acc": 100 * float(np.mean((pre > 0) == (y > 0))),
                          "sat": float(np.mean(np.abs(bo) > cfg["sat"])),
                          "hist_pre": np.histogram(np.clip(pre, -10, 10), edges_pre)[0].tolist(),
                          "hist_out": np.histogram(bo, edges_out)[0].tolist()}
        net.train()
        return res

    log, acc_loss, t0 = [{"step": 0, "val": validate()}], 0.0, time.time()
    net.train()
    for step in range(1, steps + 1):
        idx = np_rng.integers(0, n, r["batch"])
        xt = torch.from_numpy(X[idx]).to(dev)
        y = torch.from_numpy(Y[idx]).to(dev)
        v = net(xt)[1]
        if r["loss"] == "mse":
            loss = F.mse_loss(v, y)
        elif r["loss"] == "clip":
            loss = F.mse_loss(v.clamp(-1, 1), y)
        else:
            loss = F.binary_cross_entropy_with_logits(v, (y > 0).float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if r["clip"]:
            torch.nn.utils.clip_grad_norm_(net.parameters(), r["clip"])
        opt.step()
        sched.step()
        acc_loss += loss.item()
        lim.tick()
        if step % cfg["log_every"] == 0:
            g = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in net.v_fc.parameters() if p.grad is not None))
            log.append({"step": step, "loss": acc_loss / cfg["log_every"], "readout_grad": g,
                        "lr": sched.get_last_lr()[0], "val": validate(), "seconds": time.time() - t0})
            acc_loss = 0.0
            if step % (10 * cfg["log_every"]) == 0:
                print(f"{name}_s{seed}: step {step} 5x5 {log[-1]['val']['5x5']['acc']:.1f} "
                      f"10x7 {log[-1]['val']['10x7']['acc']:.1f} sat {log[-1]['val']['5x5']['sat']:.2f} "
                      f"({log[-1]['seconds']:.0f} s)", flush=True)
    res = {"variant": name, "seed": seed, "recipe": r, "steps": steps, "log": log, **summary(cfg, log)}
    path.write_text(json.dumps(res))
    print(f"{name}_s{seed}: click {res['click']} hold {res['hold']}", flush=True)
    del net, opt
    torch.cuda.empty_cache()
    return res


def summary(cfg, log):
    click = next((r["step"] for r in log if r["val"]["5x5"]["acc"] >= cfg["pass_acc"]), None)
    hold = None if click is None else all(r["val"]["5x5"]["acc"] >= cfg["hold_acc"] for r in log if r["step"] >= click)
    return {"click": click, "hold": hold, "best_5x5": max(r["val"]["5x5"]["acc"] for r in log),
            "last_5x5": log[-1]["val"]["5x5"]["acc"], "last_10x7": log[-1]["val"]["10x7"]["acc"]}


def load_runs(cfg, name, seeds):
    out = ROOT / cfg["out"]
    return [json.loads((out / f"{name}_s{s}.json").read_text()) for s in seeds if (out / f"{name}_s{s}.json").exists()]


def share(runs):
    clicks = [r["click"] for r in runs if r["click"] is not None]
    return len(clicks), (float(np.median(clicks)) if clicks else math.inf)


def r1_recipe(cfg):
    """(chosen variants, recipe changes) by the rule of configs/graph_s1.yaml."""
    seeds = cfg["seeds"]
    b = share(load_runs(cfg, "B", seeds))[0]
    chosen, changes = [], {}
    for grp in cfg["r1"]["groups"]:
        sc = [(share(load_runs(cfg, v, seeds)), i, v) for i, v in enumerate(grp)]
        (k, med), _, v = min(sc, key=lambda t: (-t[0][0], t[0][1], t[1]))
        if k > b:
            chosen.append(v)
            changes.update(cfg["variants"][v])
    return chosen, changes


def cmd_time(cfg, lim, n):
    data = boards_data(cfg)
    t = time.time()
    tmp = dict(cfg, out=cfg["out"] + "/timing")
    (ROOT / tmp["out"]).mkdir(parents=True, exist_ok=True)
    res = run_one(tmp, lim, "B", 1, n, data)
    sec = res["log"][-1]["seconds"]
    print(f"B: {n} steps in {sec:.0f} s ({time.time() - t:.0f} s wall) -> 600 s = {600 * n / sec:.0f} steps", flush=True)


def cmd_run(cfg, lim):
    assert cfg["steps"], "steps not fixed: run `time` first and set steps in the config"
    data = boards_data(cfg)
    for seed in cfg["seeds"]:
        for name in cfg["variants"]:
            run_one(cfg, lim, name, seed, cfg["steps"], data)
    chosen, changes = r1_recipe(cfg)
    print(f"R1 = B + {chosen or 'nothing'}: {changes}", flush=True)
    for seed in cfg["r1"]["seeds"]:
        run_one(cfg, lim, "R1", seed, cfg["steps"], data)


def cmd_report(cfg):
    seeds = cfg["seeds"]
    L = [f"# Part 4, series 1: reliable training of Gc (stage 0, 5x5 only)", "",
         f"Protocol: configs/graph_s1.yaml, docs/plan_part4.md. {cfg['steps']} steps per run, deterministic, "
         f"data seed {cfg['data_seed']}. Cell: click step (first 5x5 validation >= {cfg['pass_acc']}%), "
         f"'-' no click; * the run does not hold >= {cfg['hold_acc']}% after it.", "",
         "| variant | " + " | ".join(f"s{s}" for s in seeds) + " | clicks | median step | best 5x5 (mean) | last 10x7 (mean) |",
         "|---|" + "---:|" * (len(seeds) + 4)]

    def row(name, ss):
        runs = {r["seed"]: r for r in load_runs(cfg, name, ss)}
        cells = [("-" if runs[s]["click"] is None else f"{runs[s]['click']}{'' if runs[s]['hold'] else '*'}")
                 if s in runs else "n/a" for s in ss]
        k, med = share(list(runs.values()))
        rs = list(runs.values())
        return (f"| {name} | " + " | ".join(cells) + f" | {k}/{len(ss)} | {'-' if med == math.inf else int(med)} | "
                f"{np.mean([r['best_5x5'] for r in rs]):.1f} | {np.mean([r['last_10x7'] for r in rs]):.1f} |"), rs

    allruns = {}
    for name in cfg["variants"]:
        line, allruns[name] = row(name, seeds)
        L.append(line)
    chosen, changes = r1_recipe(cfg)
    L += ["", f"R1 = B + {', '.join(chosen) if chosen else 'nothing'} ({changes}), seeds {cfg['r1']['seeds']}:", "",
          "| variant | " + " | ".join(f"s{s}" for s in cfg["r1"]["seeds"]) + " | clicks | median step | best 5x5 (mean) | last 10x7 (mean) |",
          "|---|" + "---:|" * (len(cfg["r1"]["seeds"]) + 4)]
    line, allruns["R1"] = row("R1", cfg["r1"]["seeds"])
    L.append(line)
    L += ["", "Freeze diagnosis, runs that click: share of 5x5 validation with |V| > "
          f"{cfg['sat']} (bounded output) and readout gradient norm, at the click and at the end; "
          "5x5 validation change from the click to the end.", "",
          "| run | sat at click | sat at end | grad at click | grad at end | 5x5 click -> end |", "|---|---:|---:|---:|---:|---:|"]
    for name, rs in allruns.items():
        for r in rs:
            if r["click"] is None:
                continue
            c = next(x for x in r["log"] if x["step"] == r["click"])
            e = r["log"][-1]
            L.append(f"| {name}_s{r['seed']} | {c['val']['5x5']['sat']:.2f} | {e['val']['5x5']['sat']:.2f} | "
                     f"{c.get('readout_grad', 0):.2e} | {e.get('readout_grad', 0):.2e} | "
                     f"{c['val']['5x5']['acc']:.1f} -> {e['val']['5x5']['acc']:.1f} |")
    (ROOT / "docs" / "graph_s1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["time", "run", "report"])
    ap.add_argument("--config", default="configs/graph_s1.yaml")
    ap.add_argument("--n", type=int, default=2000, help="time: steps of the timing run")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    (ROOT / cfg["out"]).mkdir(parents=True, exist_ok=True)
    if a.cmd == "report":
        return cmd_report(cfg)
    lim = Limits(cfg["limits"])
    {"time": lambda: cmd_time(cfg, lim, a.n), "run": lambda: cmd_run(cfg, lim)}[a.cmd]()


if __name__ == "__main__":
    main()
