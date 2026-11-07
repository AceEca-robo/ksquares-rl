"""Part 4 of docs/plan_part4.md: stage 0 and experiment (i).

    python scripts/graph_hand.py stage0   # can the net read dots + long chains mod 2?
    python scripts/graph_hand.py train    # arms from scratch on the judge's late positions
    python scripts/graph_hand.py probe    # raw policy on the probe sets, once, after training
    python scripts/graph_hand.py report   # docs/graph_hand.md

configs/graph_hand.yaml. Finished runs are skipped, so a stopped run resumes
at the next arm and seed. The loss of (i) is part 3's: -log(mass on the
correct moves), no self-play data. CNN arms use symmetry augmentation; the
graph nets are equivariant and need none.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np

from _common import ROOT, load_config

import parity_hand
from parity_probe import mcnemar, wilson


class Limits:
    """Holds the trainer to the configured GPU memory and free memory."""

    def __init__(self, c: dict):
        import torch

        torch.set_num_threads(c["cpu_threads"])
        torch.cuda.set_per_process_memory_fraction(c["gpu_memory_fraction"])
        self.min_kb = c["min_available_gb"] * 1024 * 1024
        self.n = 0

    def tick(self) -> None:
        self.n += 1
        if self.n % 100 == 0:
            while self.available_kb() < self.min_kb:
                print("waiting for free memory", flush=True)
                time.sleep(30)

    @staticmethod
    def available_kb() -> int:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1])
        raise RuntimeError("no MemAvailable")


def make_arm(cfg, arm, seed):
    import torch

    from agents.az_net import make_net

    torch.manual_seed(seed)
    ac = cfg["arms"][arm]
    return make_net(load_config(ROOT / ac["config"]) if "config" in ac else ac)


class Optim:
    """Adam with the configured gradient clipping and cosine lr decay."""

    def __init__(self, net, lr, steps, oc):
        import torch

        self.net, self.clip = net, oc.get("clip")
        self.opt = torch.optim.Adam(net.parameters(), lr=lr)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, steps) if oc.get("cosine") else None

    def step(self, loss) -> None:
        import torch

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.clip:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
        self.opt.step()
        if self.sched is not None:
            self.sched.step()


def best_state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def is_graph(net) -> bool:
    return net.__class__.__name__ == "GraphNet"


def settled_boards(w, h, n, rng):
    """(board, target) after a random safe-move playout; +1: the long chain
    rule favours the mover."""
    from agents.baselines import safe_moves
    from sim.board import Board
    from sim.chains import analyse

    out = []
    for _ in range(n):
        b = Board(w, h)
        while True:
            m = safe_moves(b)
            if not m:
                break
            b.step(rng.choice(m))
        out.append((b, 1.0 if analyse(b)["favoured"] == b.player else -1.0))
    return out


def cmd_stage0(cfg, lim, tag):
    import torch
    import torch.nn.functional as F

    from agents.az_net import Codec

    sc = cfg["stage0"]
    dev = cfg["device"]
    out = ROOT / cfg["out"]
    rng = random.Random(cfg["seed"])
    rc = sc["runs"][tag]
    res_path = out / f"stage0_{tag}.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    if "log" in res:
        return stage0_verdict(sc, rc, res)
    t0 = time.time()
    data = {}
    for wh in ((5, 5), (10, 7)):
        items = settled_boards(*wh, sc["n_train"] + sc["n_val"], rng)
        c = Codec(*wh, planes=12)
        x = np.concatenate([c.encode_boards([b for b, _ in items[i:i + 2000]])
                            for i in range(0, len(items), 2000)])
        data[wh] = (c, x.reshape(len(x), 12, -1), np.array([t for _, t in items], dtype=np.float32))
        print(f"{wh}: mover favoured {np.mean(data[wh][2] > 0):.3f} ({time.time() - t0:.0f} s)", flush=True)
    res["base_rate"] = {f"{k[0]}x{k[1]}": float(np.mean(v[2] > 0)) for k, v in data.items()}
    n = sc["n_train"]
    for arm in [rc["arm"]]:
        net = make_arm(cfg, arm, cfg["seed"])
        if rc.get("init"):                           # fine-tune a finished stage 0 net
            import torch as _t
            net.load_state_dict(_t.load(out / rc["init"], map_location="cpu"))
        net.to(dev)
        res["params"] = sum(p.numel() for p in net.parameters())
        steps = rc.get("steps", sc["steps"])
        opt = Optim(net, sc["lr"] * rc.get("lr_scale", 1.0), steps, cfg.get("optim", {}))
        best = (-1.0, None, None)
        np_rng = np.random.default_rng(cfg["seed"])
        sizes = [wh for wh in data if f"{wh[0]}x{wh[1]}" in rc["train_sizes"]]
        switched = None

        def validate():
            net.eval()
            r = {}
            with torch.no_grad():
                for wh, (c, x, y) in data.items():
                    v = []
                    for i in range(n, len(x), 512):
                        xt = torch.from_numpy(x[i:i + 512].reshape(-1, 12, c.rows, c.cols)).to(dev)
                        v.append(net(xt)[1].cpu().numpy())
                    r[f"{wh[0]}x{wh[1]}"] = 100 * float(np.mean((np.concatenate(v) > 0) == (y[n:] > 0)))
            net.train()
            return r

        log = []
        net.train()
        log.append({"step": 0, "loss": None, "val": validate()})
        print(f"stage0 {tag}: {log[-1]}", flush=True)
        for step in range(1, steps + 1):
            c, X, Y = data[sizes[step % len(sizes)]]
            idx = np_rng.integers(0, n, sc["batch"])
            xb = X[idx]
            if not is_graph(net):
                xb, = parity_hand.augment(np_rng, c, xb)
            xt = torch.from_numpy(xb.reshape(-1, 12, c.rows, c.cols)).to(dev)
            loss = F.mse_loss(net(xt)[1], torch.from_numpy(Y[idx]).to(dev))
            opt.step(loss)
            loss = loss.item()
            lim.tick()
            if step % sc["log_every"] == 0:
                log.append({"step": step, "loss": loss, "val": validate()})
                print(f"stage0 {tag}: {log[-1]}", flush=True)
                m = min(log[-1]["val"].values())
                if m > best[0]:
                    best = (m, step, best_state(net))
                if rc.get("curriculum") and switched is None and log[-1]["val"]["5x5"] >= sc["pass_acc"]:
                    switched = step                  # 5x5 has clicked: 10x7 only from here
                    sizes = [(10, 7)]
                    print(f"stage0 {tag}: curriculum switch to 10x7 at step {step}", flush=True)
        res.update(run=rc, log=log, switched=switched, best_step=best[1])
        torch.save(best[2], out / f"stage0_{tag}_best.pt")
        torch.save(best_state(net), out / f"stage0_{tag}_final.pt")
        res_path.write_text(json.dumps(res, indent=1))
        del net, opt
        torch.cuda.empty_cache()
    return stage0_verdict(sc, rc, res)


def first_at(log, size, acc):
    return next((r["step"] for r in log if r["val"][size] >= acc), None)


def stage0_verdict(sc, rc, res):
    """Pass: the best point (min over sizes) >= pass_acc on both sizes, and
    every later log >= hold_acc on both sizes to the end."""
    log = res["log"]
    v = log[-1]["val"]
    i = next((k for k, r in enumerate(log) if min(r["val"].values()) >= sc["pass_acc"]), None)
    ok = i is not None and all(min(r["val"].values()) >= sc["hold_acc"] for r in log[i:])
    print("stage 0:", "PASS" if ok else "FAIL", "last", v, "| best step", res.get("best_step"),
          "| both >= pass first at", None if i is None else log[i]["step"], "| 95% first reached at",
          {b: first_at(res["log"], b, sc["pass_acc"]) for b in v}, flush=True)
    if not ok:
        raise SystemExit(3)


def run_name(arm, seed):
    return f"{arm}_s{seed}"


def judge_sets(cfg):
    pc = load_config(ROOT / cfg["parity_config"])
    return pc, {b: parity_hand.split_data(pc, b) for b in pc["data"]["boards"]}


def correct_rate(net, codec, x, mask, dev, bs=512):
    import torch

    am = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xt = torch.from_numpy(x[i:i + bs].reshape(-1, codec.planes, codec.rows, codec.cols)).to(dev)
            lg = net(xt)[0]
            legal = xt[:, 1].flatten(1)[:, net.line_cells(codec.rows, codec.cols)] > 0
            am.append(lg.masked_fill(~legal, -1e9).argmax(1).cpu().numpy())
    am = np.concatenate(am)
    return mask[np.arange(len(am)), am]


def cmd_train(cfg, lim):
    import torch
    import torch.nn.functional as F

    tc = cfg["train"]
    dev = cfg["device"]
    out = ROOT / cfg["out"]
    _, splits = judge_sets(cfg)
    sizes = list(splits)
    for b, (tr, va) in splits.items():
        print(f"{b}: {len(tr)} train, {len(va)} validation", flush=True)
    for seed in tc["seeds"]:
        for arm in cfg["train"]["arms"]:
            name = run_name(arm, seed)
            if (out / f"{name}.pt").exists():
                continue
            net = make_arm(cfg, arm, cfg["seed"] + seed).to(dev)
            planes = net.planes
            par = {b: (parity_hand.parity_arrays(b, tr, planes), parity_hand.parity_arrays(b, va, planes))
                   for b, (tr, va) in splits.items()}
            n_par = sum(p.numel() for p in net.parameters())
            print(f"{name}: {n_par} parameters", flush=True)
            opt = Optim(net, tc["lr"], tc["steps"], cfg.get("optim", {}))
            best = (-1.0, None, None)
            rng = np.random.default_rng([cfg["seed"], seed, len(arm), ord(arm[-1])])
            graph = is_graph(net)
            log, acc, t0 = [], 0.0, time.time()
            net.train()
            for step in range(1, tc["steps"] + 1):
                cur = tc.get("curriculum")           # {"5x5": n}: 5x5 only for n steps, then 10x7 only
                b = sizes[step % len(sizes)] if not cur else ("5x5" if step <= cur["5x5"] else "10x7")
                codec, X, M = par[b][0]
                idx = rng.integers(0, len(X), tc["batch"])
                x, m = parity_hand.augment(rng, codec, X[idx], M[idx], on=not graph)
                xt = torch.from_numpy(x.reshape(-1, planes, codec.rows, codec.cols)).to(dev)
                logits = net(xt)[0]
                legal = xt[:, 1].flatten(1)[:, net.line_cells(codec.rows, codec.cols)] > 0
                lp = F.log_softmax(logits.masked_fill(~legal, -1e9), 1)
                loss = -torch.logsumexp(lp.masked_fill(~torch.from_numpy(m).to(dev), -1e9), 1).mean()
                opt.step(loss)
                acc += loss.item()
                lim.tick()
                if step % tc["log_every"] == 0:
                    net.eval()
                    val = {k: 100 * float(np.mean(correct_rate(net, *par[k][1], dev))) for k in sizes}
                    net.train()
                    log.append({"step": step, "loss": acc / tc["log_every"], "val": val,
                                "seconds": time.time() - t0})
                    acc = 0.0
                    print(f"{name}: {log[-1]}", flush=True)
                    mv = val[tc["select_on"]] if tc.get("select_on") else float(np.mean(list(val.values())))
                    if mv > best[0]:
                        best = (mv, step, best_state(net))
            net.cpu()
            (out / f"{name}_train.json").write_text(json.dumps({"params": n_par, "log": log, "best_step": best[1]},
                                                               indent=1))
            torch.save(best[2] if cfg.get("optim", {}).get("keep_best") else net.state_dict(), out / f"{name}.pt")
            del net, opt, par
            torch.cuda.empty_cache()


def probe_sets(cfg, planes):
    pc = load_config(ROOT / cfg["parity_config"])
    r = {}
    for b, bc in pc["data"]["boards"].items():
        ps = json.loads((ROOT / bc["heldout"]).read_text())["positions"]
        r[b] = parity_hand.parity_arrays(b, ps, planes)
    return r


def cmd_probe(cfg):
    import torch

    dev = cfg["device"]
    out = ROOT / cfg["out"]
    res = {}
    for seed in cfg["train"]["seeds"]:
        for arm in cfg["train"]["arms"]:
            name = run_name(arm, seed)
            net = make_arm(cfg, arm, 0)
            net.load_state_dict(torch.load(out / f"{name}.pt", map_location="cpu"))
            net.to(dev).eval()
            for b, (codec, x, mask) in probe_sets(cfg, net.planes).items():
                res.setdefault(name, {})[b] = correct_rate(net, codec, x, mask, dev).astype(int).tolist()
    (out / "probe.json").write_text(json.dumps(res))


def cmd_report(cfg):
    out = ROOT / cfg["out"]
    probe = json.loads((out / "probe.json").read_text())
    seeds = cfg["train"]["seeds"]
    sizes = ["5x5", "10x7"]
    keep = cfg.get("optim", {}).get("keep_best")
    lines = [f"<!-- report {cfg['out']} -->", f"## (i) From scratch, raw policy ({cfg['out']}, {cfg['train']['steps']} steps)", "",
             f"Training: {cfg.get('optim', {})}. Validation and probe, % correct, "
             + ("at the best-validation step" if keep else "at the last step")
             + "; probes 1000 (5x5) and 304 (10x7).", "",
              "| arm | seed | params | val 5x5 | val 10x7 | probe 5x5 | probe 10x7 [95%] |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    best = {}
    for arm in cfg["train"]["arms"]:
        scored = []
        for seed in seeds:
            name = run_name(arm, seed)
            t = json.loads((out / f"{name}_train.json").read_text())
            v = next(r for r in t["log"] if r["step"] == t["best_step"])["val"] if keep else t["log"][-1]["val"]
            p = probe[name]
            k, n = sum(p["10x7"]), len(p["10x7"])
            lo, hi = wilson(k, n)
            lines.append(f"| {arm} | {seed} | {t['params']} | {v['5x5']:.1f} | {v['10x7']:.1f} | "
                         f"{100 * np.mean(p['5x5']):.1f} | {100 * k / n:.1f} [{lo:.0f}-{hi:.0f}] |")
            scored.append((v["5x5"] + v["10x7"], name))
        best[arm] = max(scored)[1]
    lines += ["", "Best-validation seed of each arm, McNemar on the 10x7 probe (positions only the first / only the second has right):", "",
              "| pair | first only | second only | p |", "|---|---:|---:|---:|"]
    for a, b in [("Gc", "C"), ("Gc", "Ch"), ("Ga", "C"), ("Gc", "Ga")]:
        n01, n10, p = mcnemar(probe[best[b]]["10x7"], probe[best[a]]["10x7"])
        lines.append(f"| {best[a]} vs {best[b]} | {n01} | {n10} | {p:.3g} |")
    lines.append(f"<!-- end report {cfg['out']} -->")
    doc = ROOT / "docs/graph_hand.md"
    text = doc.read_text() if doc.exists() else ""
    a, b = lines[0], lines[-1]
    if a in text:
        text = text[:text.index(a)] + text[text.index(b) + len(b) + 1:]
    doc.write_text(text.rstrip("\n") + "\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage0", "train", "probe", "report"])
    ap.add_argument("--config", default="configs/graph_hand.yaml")
    ap.add_argument("--steps", type=int, help="train: overrides train.steps")
    ap.add_argument("--run", default="v1", help="stage0: a key of stage0.runs")
    ap.add_argument("--out", help="overrides out")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    if a.out:
        cfg["out"] = a.out
    if a.steps:
        cfg["train"]["steps"] = a.steps
    (ROOT / cfg["out"]).mkdir(parents=True, exist_ok=True)
    if a.cmd == "report":
        return cmd_report(cfg)
    lim = Limits(cfg["limits"])
    {"stage0": lambda: cmd_stage0(cfg, lim, a.run), "train": lambda: cmd_train(cfg, lim),
     "probe": lambda: cmd_probe(cfg)}[a.cmd]()


if __name__ == "__main__":
    main()
