"""Part 4, series 12 of docs/plan_part4.md: the same judge answers, graph net against convolution.

    python scripts/graph_s12.py check     # before the series: the trainer reproduces mid_hand r30 (Dot-2, mid only)
    python scripts/graph_s12.py train     # every net x variant (frozen, lr01); finished ones are skipped
    python scripts/graph_s12.py overlap   # held-out positions against every training / selection position
    python scripts/graph_s12.py eval      # correct moves without thinking, before and after, per net
    python scripts/graph_s12.py report    # docs/graph_s12.md

configs/graph_s12.yaml. The trainer is self_correct.cmd_train generalised to any net: the same batches (same seed,
same RNG calls) for every net; the head learns at lr, the rest at the variant's rest_lr (0 = frozen).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np

from _common import ROOT, load_config

GROUPS = {"dotg": "Dot-G", "dot2": "Dot-2", "ch": "C+h"}
VARIANTS = {"frozen": "(a) head only", "lr01": "(b) all, rest at 0.1x lr"}


def out_dir(cfg):
    p = ROOT / cfg["out"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def limit_gpu(cfg):
    import torch

    if cfg["device"].startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(cfg["limits"]["gpu_memory_fraction"])


# --- data ----------------------------------------------------------------------------

def data(cfg):
    """{mid_train, mid_held, mis_train, mis_held}: positions {moves, correct, gid, ...} in the existing splits."""
    import mid_hand
    import rout_mistakes as rm

    mc = load_config(ROOT / cfg["data"]["mid_hand"])
    rc = load_config(ROOT / cfg["data"]["rout_mistakes"])
    mt, mh = mid_hand.mistakes(mc)
    rt, rh = rm.mistakes(rc)
    for p in rt + rh:
        p["decisive"] = True          # a mistake: the played line loses what the best keeps
    return {"mid_train": mt, "mid_held": mh, "mis_train": rt, "mis_held": rh}


def net_list(cfg):
    """[(group, name, kind, loader args)]; kind 'dot2' (AZNet2) or 'graph'."""
    L = []
    for g, gc in cfg["nets"].items():
        if "checkpoint" in gc:
            L.append((g, "Dot-2", "dot2", gc))
            continue
        sub = load_config(ROOT / gc["config"])
        kind = "graph" if gc["arm"].startswith("G") else "dot2"
        if "runs" in gc:
            for w, t in gc["runs"]:
                L.append((g, f"Dot-G w{w}/{t}", kind,
                          {"sub": sub, "arm": gc["arm"], "weights": ROOT / sub["out"] / f"w{w}" / f"s2_{gc['arm']}_s{t}.pt"}))
        else:
            for s in gc["seeds"]:
                L.append((g, f"C+h s{s}", kind,
                          {"sub": sub, "arm": gc["arm"], "weights": ROOT / sub["out"] / f"s2_{gc['arm']}_s{s}.pt"}))
    return L


def weights_path(spec):
    return ROOT / spec["checkpoint"] if "checkpoint" in spec else spec["weights"]


def load_net(spec):
    """The net as it was before series 12 (CPU)."""
    import torch

    import graph_s3
    import parity_hand as ph

    if "checkpoint" in spec:
        return ph.load_base({"net": spec})[0]
    net, zero, _ = graph_s3.build_net(spec["sub"], spec["arm"])
    assert not zero
    net.load_state_dict(torch.load(spec["weights"], map_location="cpu"))
    return net


def file_name(name, variant):
    return name.replace("/", "_").replace(" ", "_").replace("+", "p") + f"__{variant}"


# --- train ---------------------------------------------------------------------------

def head_names(cfg, kind):
    return set(cfg["heads"][kind])


def load_buffers(tc):
    from agents.az_net import Codec, N_PLANES_2
    from az_train import Buffer

    import transfer

    bufs = {}
    for bname, path in tc["selfplay_buffers"].items():
        c = Codec(*transfer.wh_of(bname), planes=N_PLANES_2)
        bufs[bname] = Buffer(1, c.n_lines, c.n_boxes)
        d = np.load(ROOT / path)
        bufs[bname].a = {k: d[k] for k in bufs[bname].a}
        bufs[bname].n = int(d["n"])
    return bufs


def train_one(cfg, net, kind, pool, bufs, variant, log_prefix=""):
    """Fine-tune ``net`` in place on the judge pool + replay; returns the log. Same RNG calls as
    self_correct.cmd_train (seed, share), so every net sees the same batches."""
    import torch
    import torch.nn.functional as F

    import parity_hand as ph
    import transfer
    from agents.az_net import Codec

    tc, dev = cfg["train"], cfg["device"]
    share = tc["share"]
    nm = int(round(share * tc["batch"]))
    rest_lr = tc["variants"][variant]["rest_lr"]
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng([cfg["seed"], int(100 * share)])
    heads = head_names(cfg, kind)
    codec, X, M = ph.parity_arrays("10x7", pool, net.planes)
    bw = np.array([tc["selfplay_weights"][b] for b in bufs], dtype=float)
    bw = bw / bw.sum() if len(bw) else bw
    assert nm < tc["batch"] or not bufs         # series 12b: judge positions only, no replay
    codecs = {b: Codec(*transfer.wh_of(b), planes=net.planes) for b in bufs}
    net.to(dev).eval()                       # BatchNorm statistics as trained
    hp = [p for n, p in net.named_parameters() if n.split(".")[0] in heads]
    rp = [p for n, p in net.named_parameters() if n.split(".")[0] not in heads]
    assert hp and rp
    groups = [{"params": hp, "lr": tc["lr"]}]
    if rest_lr > 0:
        groups.append({"params": rp, "lr": rest_lr})
    else:
        for p in rp:
            p.requires_grad_(False)
    opt = torch.optim.Adam(groups, lr=tc["lr"])

    def logp_of(x, c):
        xt = torch.from_numpy(x.reshape(len(x), -1, c.rows, c.cols)).to(dev)
        logits = net(xt)[0]
        legal = xt[:, 1].flatten(1)[:, net.line_cells(c.rows, c.cols)] > 0
        return F.log_softmax(logits.masked_fill(~legal, -1e9), 1)

    def acc():
        with torch.no_grad():
            am = np.concatenate([logp_of(X[i:i + 512], codec).argmax(1).cpu().numpy() for i in range(0, len(X), 512)])
        return 100 * float(np.mean(M[np.arange(len(am)), am]))

    log = [{"step": 0, "train_acc": acc()}]
    run = {"judge": 0.0, "policy": 0.0}
    t0 = time.time()
    for step in range(1, tc["steps"] + 1):
        loss = 0.0
        idx = rng.integers(0, len(X), nm)
        x, m = ph.augment(rng, codec, X[idx], M[idx], on=tc["symmetries"])
        lp = logp_of(x, codec)
        lm = -torch.logsumexp(lp.masked_fill(~torch.from_numpy(m).to(dev), -1e9), 1).mean()
        loss = loss + share * lm
        run["judge"] += lm.item()
        if nm == tc["batch"]:                    # series 12b: no replay share
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % tc["log_every"] == 0:
                rec = {"step": step, **{k: v / tc["log_every"] for k, v in run.items()}, "train_acc": acc(),
                       "seconds": round(time.time() - t0)}
                log.append(rec)
                run = dict.fromkeys(run, 0.0)
                print(f"[{time.strftime('%H:%M:%S')}] {log_prefix}{rec}", flush=True)
            continue
        j = list(bufs)[int(rng.choice(len(bufs), p=bw))]
        s = bufs[j].sample(rng, tc["batch"] - nm)
        c = codecs[j]
        xs = c.encode(s["lines"], s["sides"], s["owner"], s["diff"].astype(np.float32), s["first"])
        xs, pi = ph.augment(rng, c, xs.reshape(len(xs), c.planes, -1), s["pi"].astype(np.float32), on=tc["symmetries"])
        lpol = -(torch.from_numpy(pi).to(dev) * logp_of(xs, c)).sum(1).mean()
        loss = loss + (1 - share) * lpol
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run["policy"] += lpol.item()
        if step % tc["log_every"] == 0:
            rec = {"step": step, **{k: v / tc["log_every"] for k, v in run.items()}, "train_acc": acc(),
                   "seconds": round(time.time() - t0)}
            log.append(rec)
            run = dict.fromkeys(run, 0.0)
            print(f"[{time.strftime('%H:%M:%S')}] {log_prefix}{rec}", flush=True)
    net.cpu()
    return log


def frozen_check(net, net0, kind, cfg, positions):
    """Every non-head tensor (weights and buffers) as before; V and score bit for bit on ``positions`` (CPU fp32)."""
    import torch

    import parity_hand as ph

    heads = head_names(cfg, kind)
    sd, sd0 = net.state_dict(), net0.state_dict()
    same = all(torch.equal(sd[k], sd0[k]) for k in sd0 if k.split(".")[0] not in heads)
    codec, x, _ = ph.parity_arrays("10x7", positions, net.planes)
    xt = torch.from_numpy(x.reshape(len(x), -1, codec.rows, codec.cols))
    net.eval()
    net0.eval()
    with torch.no_grad():
        _, v, s = net(xt)
        _, v0, s0 = net0(xt)
    return {"weights_unchanged": bool(same), "v_bitwise_equal": bool(torch.equal(v, v0) and torch.equal(s, s0)),
            "positions": len(x)}


def cmd_train(cfg):
    import torch

    limit_gpu(cfg)
    D = data(cfg)
    pool = D["mid_train"] + D["mis_train"]
    print(f"judge pool: {len(D['mid_train'])} mid + {len(D['mis_train'])} mistakes", flush=True)
    bufs = load_buffers(cfg["train"])
    out = out_dir(cfg)
    for g, name, kind, spec in net_list(cfg):
        for variant in cfg["train"]["variants"]:
            fn = file_name(name, variant)
            if (out / f"{fn}.json").exists():
                continue
            net = load_net(spec)
            log = train_one(cfg, net, kind, pool, bufs, variant, f"{name} {variant}: ")
            res = {"net": name, "group": g, "variant": variant, "weights_before": str(weights_path(spec).relative_to(ROOT)),
                   "sha_before": sha(weights_path(spec)), "log": log}
            if cfg["train"]["variants"][variant]["rest_lr"] == 0:
                res["frozen_check"] = frozen_check(net, load_net(spec), kind, cfg, D["mid_held"] + D["mis_held"])
                print(f"{name} {variant}: frozen check {res['frozen_check']}", flush=True)
            torch.save(net.state_dict(), out / f"{fn}.pt")
            tmp = out / f"{fn}.tmp"
            tmp.write_text(json.dumps(res, indent=1))
            tmp.rename(out / f"{fn}.json")


def cmd_check(cfg):
    """The trainer with Dot-2, frozen, mid train only = mid_hand r30 (same held-out moves; weights difference)."""
    import torch

    import parity_hand as ph

    limit_gpu(cfg)
    D = data(cfg)
    spec = cfg["nets"]["dot2"]
    net = load_net(spec)
    train_one(cfg, net, "dot2", D["mid_train"], load_buffers(cfg["train"]), "frozen", "check: ")
    ref = load_net(spec)
    ref.load_state_dict(torch.load(ROOT / "runs/mid_hand/r30.pt", map_location="cpu"))
    codec, x, mask = ph.parity_arrays("10x7", D["mid_held"], net.planes)
    xt = torch.from_numpy(x.reshape(len(x), -1, codec.rows, codec.cols))
    legal = xt[:, 1].flatten(1)[:, net.line_cells(codec.rows, codec.cols)] > 0
    with torch.no_grad():
        a = net.eval()(xt)[0].masked_fill(~legal, -1e9).argmax(1)
        b = ref.eval()(xt)[0].masked_fill(~legal, -1e9).argmax(1)
    sd, sr = net.state_dict(), ref.state_dict()
    res = {"held_out_moves_equal": int((a == b).sum()), "held_out": len(x),
           "weights_bitwise_equal": all(torch.equal(sd[k], sr[k]) for k in sr),
           "max_abs_weight_diff": max(float((sd[k].float() - sr[k].float()).abs().max()) for k in sr),
           "acc_ours": float(mask[np.arange(len(a)), a.numpy()].mean()),
           "acc_r30": float(mask[np.arange(len(b)), b.numpy()].mean())}
    (out_dir(cfg) / "check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


# --- overlap -------------------------------------------------------------------------

def lines_of(positions, w=10, h=7):
    from sim.board import Board

    L = []
    for p in positions:
        b = Board(w, h)
        for ln in p["moves"]:
            b.step(ln)
        L.append(np.frombuffer(bytes(b.lines), dtype=np.uint8).copy())
    return np.array(L).reshape(len(positions), -1)


def canon_lines(lines, codec):
    """= graph_s3p_checks.canon(x, codec, lines_only=True) from the drawn-lines array (N, n_lines)."""
    keys = []
    for _, lperm in codec.syms:
        y = np.ascontiguousarray(lines[:, lperm] == 0)
        keys.append([hashlib.sha1(r.tobytes()).hexdigest() for r in y])
    return [min(k) for k in zip(*keys)]


def cmd_overlap(cfg):
    from agents.az_net import Codec

    import graph_s2
    from graph_hand import judge_sets, probe_sets
    from graph_s3p_checks import canon

    codec = Codec(10, 7, planes=17)
    D = data(cfg)
    held = {"mid held out": D["mid_held"], "mid held out, decisive": [p for p in D["mid_held"] if p["decisive"]],
            "mistakes held out": D["mis_held"]}
    ko = {k: canon_lines(lines_of(v), codec) for k, v in held.items()}
    theirs = {"fine-tune: mid train": canon_lines(lines_of(D["mid_train"]), codec),
              "fine-tune: mistakes train": canon_lines(lines_of(D["mis_train"]), codec)}
    notes = []
    for bname, path in cfg["train"]["selfplay_buffers"].items():
        if bname != "10x7":
            continue                  # other sizes cannot coincide with 10x7 positions
        d = np.load(ROOT / path)
        n = int(d["n"])
        theirs[f"fine-tune: dot2c replay {bname}"] = canon_lines(d["lines"][:n], codec)
        notes.append(f"{path} sha256 {sha(ROOT / path)} ({n} positions)")
    for cp in ("configs/graph_s6.yaml", "configs/graph_s3p.yaml"):
        c = load_config(ROOT / cp)
        tag = c["out"].split("/")[-1]
        z = np.load(ROOT / c["out"] / "data.npz")
        s0 = graph_s2.splits(c, {k: z[k] for k in z.files})["10x7"]
        _, sp = judge_sets(c)
        tr, va = sp["10x7"]
        for lab, x in ((f"{tag}: stage 0 train", s0[0]), (f"{tag}: stage 0 validation", s0[2])):
            x = np.asarray(x)[:, :17]
            theirs[lab] = canon(x.reshape(len(x), 17, -1), codec, True)
        theirs[f"{tag}: judge train"] = canon_lines(lines_of(tr), codec)
        theirs[f"{tag}: judge validation"] = canon_lines(lines_of(va), codec)
        pr = np.asarray(probe_sets(c, 17)["10x7"][1])
        theirs[f"{tag}: judge probe"] = canon(pr.reshape(len(pr), 17, -1), codec, True)
        notes.append(f"{c['out']}/data.npz sha256 {sha(ROOT / c['out'] / 'data.npz')}")
    ts = {k: set(v) for k, v in theirs.items()}
    rows = {}
    for a, keys in ko.items():
        rows[a] = {"n": len(keys), "distinct": len(set(keys)), "shared": {b: len(set(keys) & s) for b, s in ts.items()},
                   "shared_positions": [i for i, k in enumerate(keys) if any(k in s for s in ts.values())]}
    gid = {"mid": (sorted({p["gid"] for p in D["mid_train"]} & {p["gid"] for p in D["mid_held"]})),
           "mistakes": sorted({p["gid"] for p in D["mis_train"]} & {p["gid"] for p in D["mis_held"]})}
    res = {"notes": notes, "theirs_distinct": {b: len(s) for b, s in ts.items()}, "rows": rows, "shared_game_ids": gid,
           "sizes": {k: len(v) for k, v in D.items()},
           "games": {"mid": {"train": len({p['gid'] for p in D['mid_train']}), "held": len({p['gid'] for p in D['mid_held']})},
                     "mistakes": {"train": len({p['gid'] for p in D['mis_train']}),
                                  "held": len({p['gid'] for p in D['mis_held']})}}}
    (out_dir(cfg) / "overlap.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({a: (r["distinct"], sum(r["shared"].values())) for a, r in rows.items()}), json.dumps(gid), flush=True)


# --- eval ----------------------------------------------------------------------------

def correct_of(net, positions, dev):
    import parity_hand as ph
    from graph_hand import correct_rate

    codec, x, mask = ph.parity_arrays("10x7", positions, net.planes)
    net.to(dev).eval()
    r = correct_rate(net, codec, x, mask, dev).astype(int).tolist()
    net.cpu()
    return r


def cmd_eval(cfg):
    import torch

    dev = cfg["eval_device"]     # CPU fp32, as part 3's accuracy (rout_mistakes.correct_vec): near ties of the
    D = data(cfg)                # logits can make GPU and CPU argmax differ (one Dot-2 position, 10.6402 vs 10.6399)
    out = out_dir(cfg)
    res = {"eval_device": dev, "keys": {k: [p["moves"] for p in v] for k, v in D.items()}, "nets": {}}
    for g, name, kind, spec in net_list(cfg):
        r = {"group": g, "kind": kind, "weights": str(weights_path(spec).relative_to(ROOT)), "sha256": sha(weights_path(spec))}
        net = load_net(spec)
        r["before"] = {k: correct_of(net, v, dev) for k, v in D.items()}
        for variant in cfg["train"]["variants"]:
            fn = file_name(name, variant)
            net = load_net(spec)
            net.load_state_dict(torch.load(out / f"{fn}.pt", map_location="cpu"))
            r[variant] = {k: correct_of(net, v, dev) for k, v in D.items()}
            r[f"{variant}_sha256"] = sha(out / f"{fn}.pt")
        res["nets"][name] = r
        print(f"[{time.strftime('%H:%M:%S')}] {name}: " + ", ".join(
            f"{s} {100 * np.mean([x for x, p in zip(r[s]['mid_held'], D['mid_held']) if p['decisive']]):.1f}"
            for s in ["before", *cfg["train"]["variants"]]), flush=True)
    # reference: Dot-2 MCTS 200, stored moves of part 3 (c)
    st = json.loads((ROOT / cfg["data"]["stored_moves"]).read_text())
    idx = {json.dumps(m): i for i, m in enumerate(st["keys"])}
    import mid_hand

    mc = load_config(ROOT / cfg["data"]["mid_hand"])
    keyed = {json.dumps(p["moves"]): p for p in mid_hand.positions(mc)}
    labels = {json.dumps(r["moves"]): tuple(r["key"]) for r in __import__("self_correct").read_jsonl(ROOT / mc["labels"])}
    res["stored"] = {}
    for m in ("hand", "mcts200"):
        res["stored"][m] = []
        for p in D["mid_held"]:
            k = labels[json.dumps(p["moves"])]
            i = idx.get(json.dumps(list(k)))
            res["stored"][m].append(None if i is None else int(st["moves"][m][i] in keyed[json.dumps(p["moves"])]["correct"]))
    (out / "eval.json").write_text(json.dumps(res))


# --- statistics ----------------------------------------------------------------------

def interval(a, b, n, seed, level):
    """a: (nets_a, positions), b: (nets_b, positions), 0/1 on the same positions. Positions with replacement (the same
    draw for both), each side's nets with replacement. (mean a - mean b, lower, upper) in points, two-sided."""
    a, b = np.atleast_2d(np.asarray(a, float)), np.atleast_2d(np.asarray(b, float))
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for i in range(n):
        pos = rng.integers(0, a.shape[1], a.shape[1])
        ka = rng.integers(0, len(a), len(a))
        kb = rng.integers(0, len(b), len(b))
        d[i] = a[ka][:, pos].mean() - b[kb][:, pos].mean()
    q = (1 - level) / 2
    return 100 * (a.mean() - b.mean()), 100 * float(np.quantile(d, q)), 100 * float(np.quantile(d, 1 - q))


def gain(after, before, n, seed, level):
    """Paired: after, before (nets, positions); the same nets and positions resampled for both."""
    a, b = np.atleast_2d(np.asarray(after, float)), np.atleast_2d(np.asarray(before, float))
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for i in range(n):
        pos = rng.integers(0, a.shape[1], a.shape[1])
        k = rng.integers(0, len(a), len(a))
        d[i] = a[k][:, pos].mean() - b[k][:, pos].mean()
    q = (1 - level) / 2
    return 100 * (a.mean() - b.mean()), 100 * float(np.quantile(d, q)), 100 * float(np.quantile(d, 1 - q))


def reading(lo, hi):
    if lo > 0:
        return "the graph net learns the middle game better"
    if hi < 0:
        return "the graph net learns the middle game worse"
    return "no difference seen"


def pct(x):
    x = [v for v in np.asarray(x, float).ravel() if v == v]
    return f"{100 * float(np.mean(x)):.1f}" if x else "—"


# --- report --------------------------------------------------------------------------

def cmd_report(cfg):
    out = out_dir(cfg)
    D = data(cfg)
    ev = json.loads((out / "eval.json").read_text())
    ov = json.loads((out / "overlap.json").read_text())
    for k, v in D.items():
        assert [p["moves"] for p in v] == ev["keys"][k], "data changed since eval"
    dec = np.array([p["decisive"] for p in D["mid_held"]])
    nets = ev["nets"]
    grp = {g: [n for n in nets if nets[n]["group"] == g] for g in GROUPS}
    variants = list(cfg["train"]["variants"])
    states = ["before", *variants]

    def A(g, s, k="mid_held", mask=None):
        a = np.array([nets[n][s][k] for n in grp[g]])
        return a[:, mask] if mask is not None else a

    bc = cfg["bootstrap"]
    bs = (bc["resamples"], bc["seed"], bc["level"])
    shared = set(ov["rows"]["mid held out"]["shared_positions"])
    keep = dec & np.array([i not in shared for i in range(len(dec))])
    m = keep
    tc = cfg["train"]
    L = ["# Part 4, series 12: the same judge answers, graph net against convolution", "",
         f"Generated by `scripts/graph_s12.py report` ({time.strftime('%Y-%m-%d %H:%M')}). Protocol: docs/plan_part4.md "
         "(series 12, user 2026-11-20), configs/graph_s12.yaml. Fine-tune on the judge's middle-game answers, the same "
         "for every net (hyperparameters of Dot-2's mid_hand r30, not tuned for Dot-G): "
         f"{tc['steps']} steps, Adam lr {tc['lr']}, batch {tc['batch']}, {100 * tc['share']:.0f} % judge positions "
         f"({ov['sizes']['mid_train']} mid train + {ov['sizes']['mis_train']} mistakes train, pooled), the rest dot2c "
         "replay positions with their MCTS policy; the same seed and batches for every net; the weights after the last "
         "step are evaluated (no selection). Variants: **(a)** only the move-choice head learns (body, V, score frozen); "
         "**(b)** everything learns, body / V / score at 0.1x lr, no V loss.", "",
         "Metric: correct moves without thinking (policy argmax over legal lines keeps the judge's best result) on the "
         f"**{int(m.sum())} held-out decisive mid positions** (of {len(dec)} held-out mid positions, split by game).", ""]
    if shared:
        L += [f"{len(shared)} held-out mid positions are shared with a training position (see Overlap); "
              f"they are left out of every number below ({int(dec.sum())} decisive before that).", ""]
    # primary
    L += ["## Primary and reading", "",
          "D = mean of the 3 Dot-G after - Dot-2 after, the same positions; bootstrap over positions and nets "
          f"({bc['resamples']} resamples, seed {bc['seed']}, two-sided {100 * bc['level']:.0f} % percentile interval).", "",
          "| variant | Dot-G after % | Dot-2 after % | D, points | 95 % interval | reading |", "|---|---:|---:|---:|---:|---|"]
    prim = {}
    for v in variants:
        d, lo, hi = interval(A("dotg", v, mask=m), A("dot2", v, mask=m), *bs)
        prim[v] = (d, lo, hi)
        L.append(f"| {VARIANTS[v]} | {pct(A('dotg', v, mask=m))} | {pct(A('dot2', v, mask=m))} | {d:+.1f} | "
                 f"[{lo:+.1f}, {hi:+.1f}] | **{reading(lo, hi)}** |")
    rs = {reading(*prim[v][1:]) for v in variants}
    L += ["", "Both variants give the same row." if len(rs) == 1 else
          "The variants give different rows; both are reported, no joint verdict (protocol).", ""]
    # gains
    L += ["## Gain of each net against itself before", "",
          "after - before, paired (the same nets and positions resampled for both), the same interval.", "",
          "| nets | n | before % | " + " | ".join(f"after {VARIANTS[v]} % | gain {v} | 95 % interval" for v in variants)
          + " |", "|---|---:|---:|" + "---:|---:|---:|" * len(variants)]
    for g, lab in GROUPS.items():
        row = f"| {lab} | {len(grp[g])} | {pct(A(g, 'before', mask=m))} |"
        for v in variants:
            d, lo, hi = gain(A(g, v, mask=m), A(g, "before", mask=m), *bs)
            row += f" {pct(A(g, v, mask=m))} | {d:+.1f} | [{lo:+.1f}, {hi:+.1f}] |"
        L.append(row)
    st = ev["stored"]
    mc = np.array([x if x is not None else np.nan for x in st["mcts200"]], float)
    hd = np.array([x if x is not None else np.nan for x in st["hand"]], float)
    hand_eq = int(np.nansum(hd[m] == np.array(nets["Dot-2"]["before"]["mid_held"])[m]))
    L += ["", f"Reference on the same positions: Dot-2 **with thinking (MCTS 200)** {pct(mc[m])} % (the stored moves of "
          f"runs/mid_judge/nets_mid.json, part 3 (c), not rerun; stored moves found for {int(np.sum(~np.isnan(mc[m])))} of "
          f"{int(m.sum())}). Dot-2 before, recomputed here, agrees with the stored raw moves' correctness on {hand_eq} of "
          f"{int(m.sum())}.", ""]
    # descriptive comparisons
    L += ["## Descriptive comparisons", "", "| comparison | variant | difference, points | 95 % interval |", "|---|---|---:|---:|"]
    for a, b in (("ch", "dot2"), ("dotg", "ch")):
        for v in ["before", *variants]:
            d, lo, hi = interval(A(a, v, mask=m), A(b, v, mask=m), *bs)
            L.append(f"| {GROUPS[a]} - {GROUPS[b]} | {VARIANTS.get(v, 'before')} | {d:+.1f} | [{lo:+.1f}, {hi:+.1f}] |")
    d, lo, hi = interval(A("dotg", "before", mask=m), A("dot2", "before", mask=m), *bs)
    L += [f"| Dot-G - Dot-2 | before | {d:+.1f} | [{lo:+.1f}, {hi:+.1f}] |", ""]
    # nets table
    L += ["## Table 1. Nets", "",
          "% correct moves without thinking. mid held out: decisive (the metric) / all; mistakes: the 24 held-out / 56 "
          "train; mid train: all 1387.", "",
          "| net | state | mid held out, decisive | mid held out, all | mistakes held out | mistakes train | mid train | "
          "weights sha256 |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for n, r in nets.items():
        for s in states:
            L.append(f"| {n} | {VARIANTS.get(s, s)} | {pct(np.array(r[s]['mid_held'])[m])} | {pct(r[s]['mid_held'])} | "
                     f"{pct(r[s]['mis_held'])} | {pct(r[s]['mis_train'])} | {pct(r[s]['mid_train'])} | "
                     f"{r['sha256'] if s == 'before' else r[f'{s}_sha256']} |")
    L.append("")
    # training checks
    L += ["## Training checks", ""]
    fc = []
    for g, name, kind, spec in net_list(cfg):
        for v in variants:
            j = json.loads((out / f"{file_name(name, v)}.json").read_text())
            lg = j["log"]
            if "frozen_check" in j:
                f = j["frozen_check"]
                fc.append(f"{name}: weights unchanged {f['weights_unchanged']}, V / score bitwise {f['v_bitwise_equal']} "
                          f"({f['positions']} positions)")
            L.append(f"- {name} {v}: judge train accuracy {lg[0]['train_acc']:.1f} -> {lg[-1]['train_acc']:.1f} %, "
                     f"judge loss {lg[1]['judge']:.3f} -> {lg[-1]['judge']:.3f}, replay policy loss {lg[1]['policy']:.3f} "
                     f"-> {lg[-1]['policy']:.3f}, {lg[-1]['seconds']} s")
    L += ["", "Frozen check of variant (a), after training: " + "; ".join(fc) + ".", ""]
    ck = out / "check.json"
    if ck.exists():
        c = json.loads(ck.read_text())
        L += [f"Code check before the run: the trainer with Dot-2, (a), mid train only, against runs/mid_hand/r30.pt: "
              f"held-out moves equal {c['held_out_moves_equal']} / {c['held_out']}, weights bit for bit "
              f"{c['weights_bitwise_equal']} (max abs difference {c['max_abs_weight_diff']:.2e}), held-out accuracy "
              f"{100 * c['acc_ours']:.1f} vs {100 * c['acc_r30']:.1f} %.", ""]
    heads = {}
    for g, name, kind, spec in net_list(cfg):
        if kind not in heads:
            net = load_net(spec)
            hn = head_names(cfg, kind)
            heads[kind] = (sum(p.numel() for n, p in net.named_parameters() if n.split(".")[0] in hn),
                           sum(p.numel() for p in net.parameters()))
    L += ["Move-choice head, trainable in (a): " + "; ".join(
        f"{'Gc (Dot-G)' if k == 'graph' else 'AZNet2 (Dot-2, C+h)'} {h} of {t} parameters" for k, (h, t) in heads.items())
          + ". The plan's \"heads within 10 %\" does not hold; the nets are fine-tuned as they are (protocol).", ""]
    # overlap
    L += ["## Overlap", "",
          "Held-out positions against every training / selection position; key = drawn lines, minimum over the grid "
          "symmetries. Cells: shared distinct positions. Other board sizes cannot coincide with 10x7 positions.", "",
          "| | " + " | ".join(f"{c} ({ov['theirs_distinct'][c]})" for c in ov["theirs_distinct"]) + " |",
          "|---|" + "---:|" * len(ov["theirs_distinct"])]
    for a, r in ov["rows"].items():
        L.append(f"| {a} ({r['distinct']} distinct) | " + " | ".join(str(r["shared"][c]) for c in ov["theirs_distinct"]) + " |")
    g = ov["games"]
    L += ["", f"Games: mid {g['mid']['train']} train / {g['mid']['held']} held out, shared game ids "
          f"{len(ov['shared_game_ids']['mid'])}; mistakes {g['mistakes']['train']} / {g['mistakes']['held']} rout games, "
          f"shared {len(ov['shared_game_ids']['mistakes'])}. Mid games are Dot-2 self-play games (part 3 (c)), the "
          "mistakes' games Dot-2 vs the simulator's Hard (runs/rout_mistakes): separate game sets.", "",
          f"Data files: {'; '.join(ov['notes'])}.", ""]
    L += ["## Caveat", "",
          "The positions come from Dot-2's own games (mid: its self-play, MCTS 200; the mistakes: its games against "
          "Hard), and 70 % of every batch is Dot-2's own replay positions with its MCTS targets: for Dot-2 an anchor, for "
          "Dot-G and C+h new data. Both favour Dot-2.", ""]
    concl = ROOT / "docs/graph_s12_notes.md"
    if concl.exists():
        L += [concl.read_text().rstrip(), ""]
    sb = cfg.get("series_b")
    if sb and (ROOT / load_config(ROOT / sb)["out"] / "eval.json").exists():
        L += section_b(load_config(ROOT / sb), prim, m)
    (ROOT / cfg["report"]).write_text("\n".join(L))
    print("\n".join(L))


def scene6_phrase(prim12, prim12b):
    """The one sentence for scene 6 from both series (rule of series 12b, fixed before its run).
    prim*: {variant: (D, lo, hi)}."""
    better = [v for v, (_, lo, _) in prim12b.items() if lo > 0]
    if not better:
        return "те же ответы судьи графовой сети помогли не больше, чем свёрточной", better
    return "на одних ответах судьи графовая сеть учится лучше, а в смеси с чужими партиями хуже", better


def section_b(cb, prim12, m):
    """Series 12b section of docs/graph_s12.md; m: the primary position mask of series 12 (the same positions)."""
    out = ROOT / cb["out"]
    D = data(cb)
    ev = json.loads((out / "eval.json").read_text())
    for k, v in D.items():
        assert [p["moves"] for p in v] == ev["keys"][k], "data changed since eval"
    nets = ev["nets"]
    grp = {g: [n for n in nets if nets[n]["group"] == g] for g in GROUPS}
    variants = list(cb["train"]["variants"])
    bc = cb["bootstrap"]
    bs = (bc["resamples"], bc["seed"], bc["level"])
    tc = cb["train"]

    def A(g, s, k="mid_held"):
        a = np.array([nets[n][s][k] for n in grp[g]])
        return a[:, m] if k == "mid_held" else a

    def T(g, s):                                   # the pooled training judge answers
        return np.array([nets[n][s]["mid_train"] + nets[n][s]["mis_train"] for n in grp[g]])

    L = ["## Series 12b: judge answers only (no Dot-2 self-play share)", "",
         "Protocol: docs/plan_part4.md (series 12b, user 2026-11-20), configs/graph_s12b.yaml, committed before the run. "
         f"As series 12, but every batch is {tc['batch']} judge positions ({len(D['mid_train'])} mid train + "
         f"{len(D['mis_train'])} mistakes train, pooled), no replay share; {tc['steps']} steps = the judge positions a "
         f"net saw in series 12 ({tc['steps'] * tc['batch']} of 4000 x 115). The same nets, splits, {int(m.sum())} "
         "held-out decisive positions, variants, seed; weights after the last step; evaluation on the CPU in fp32.", "",
         "### Primary and reading", "",
         "| variant | Dot-G after % | Dot-2 after % | D, points | 95 % interval | reading |", "|---|---:|---:|---:|---:|---|"]
    prim = {}
    for v in variants:
        d, lo, hi = interval(A("dotg", v), A("dot2", v), *bs)
        prim[v] = (d, lo, hi)
        L.append(f"| {VARIANTS[v]} | {pct(A('dotg', v))} | {pct(A('dot2', v))} | {d:+.1f} | [{lo:+.1f}, {hi:+.1f}] | "
                 f"**{reading(lo, hi)}** |")
    rs = {reading(*prim[v][1:]) for v in variants}
    L += ["", "Both variants give the same row." if len(rs) == 1 else
          "The variants give different rows; both are reported, no joint verdict (protocol).", "",
          "### Gain against itself before, and the training answers", "",
          "Gain: after - before on the held-out positions, paired, the same interval. Train: accuracy on the "
          f"{len(D['mid_train']) + len(D['mis_train'])} training judge answers (pooled) before -> after.", "",
          "| nets | n | before % | " + " | ".join(f"after {v} % | gain {v} | 95 % interval | train {v} %" for v in variants)
          + " |", "|---|---:|---:|" + "---:|---:|---:|---:|" * len(variants)]
    for g, lab in GROUPS.items():
        row = f"| {lab} | {len(grp[g])} | {pct(A(g, 'before'))} |"
        for v in variants:
            d, lo, hi = gain(A(g, v), A(g, "before"), *bs)
            row += f" {pct(A(g, v))} | {d:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {pct(T(g, 'before'))} -> {pct(T(g, v))} |"
        L.append(row)
    L += ["", "### Table 1b. Nets", "",
          "| net | state | mid held out, decisive | mid held out, all | mistakes held out | mistakes train | mid train | "
          "weights sha256 |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for n, r in nets.items():
        for s in variants:
            L.append(f"| {n} | {VARIANTS[s]} | {pct(np.array(r[s]['mid_held'])[m])} | {pct(r[s]['mid_held'])} | "
                     f"{pct(r[s]['mis_held'])} | {pct(r[s]['mis_train'])} | {pct(r[s]['mid_train'])} | {r[f'{s}_sha256']} |")
    L += ["", "Before = the series 12 rows (recomputed here, equal: "
          + str(all(nets[n]["before"] == json.loads((ROOT / "runs/graph_s12/eval.json").read_text())["nets"][n]["before"]
                    for n in nets)) + ").", "", "### Training checks", ""]
    fc = []
    for g, name, kind, spec in net_list(cb):
        for v in variants:
            j = json.loads((out / f"{file_name(name, v)}.json").read_text())
            lg = j["log"]
            if "frozen_check" in j:
                f = j["frozen_check"]
                fc.append(f"{name}: {f['weights_unchanged']} / {f['v_bitwise_equal']}")
            L.append(f"- {name} {v}: judge train accuracy {lg[0]['train_acc']:.1f} -> {lg[-1]['train_acc']:.1f} %, "
                     f"judge loss {lg[1]['judge']:.3f} -> {lg[-1]['judge']:.3f}, {lg[-1]['seconds']} s")
    L += ["", "Frozen check of (a) (non-head weights unchanged / V and score bitwise on the held-out positions): "
          + "; ".join(fc) + ".", ""]
    phrase, better = scene6_phrase(prim12, prim)
    L += ["## Both series: the sentence for scene 6", "",
          "Rule fixed before series 12b: no variant of 12b better -> the first sentence; a variant of 12b better -> the "
          "second, with its numbers.", "",
          "| series | variant | D, points | 95 % interval | reading |", "|---|---|---:|---:|---|"]
    for ser, pr in (("12 (with replay)", prim12), ("12b (judge only)", prim)):
        for v, (d, lo, hi) in pr.items():
            L.append(f"| {ser} | {VARIANTS[v]} | {d:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {reading(lo, hi)} |")
    L += ["", f"Sentence: **«{phrase}»**" + (" — 12b " + ", ".join(
        f"{VARIANTS[v]} D {prim[v][0]:+.1f} [{prim[v][1]:+.1f}, {prim[v][2]:+.1f}]" for v in better)
        + f"; series 12 (b) D {prim12['lr01'][0]:+.1f} [{prim12['lr01'][1]:+.1f}, {prim12['lr01'][2]:+.1f}]" if better else "")
          + ".", ""]
    nb = ROOT / "docs/graph_s12b_notes.md"
    if nb.exists():
        L += [nb.read_text().rstrip(), ""]
    return L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "train", "overlap", "eval", "report"])
    ap.add_argument("--config", default="configs/graph_s12.yaml")   # series 12b: configs/graph_s12b.yaml (train, eval)
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    {"check": cmd_check, "train": cmd_train, "overlap": cmd_overlap, "eval": cmd_eval, "report": cmd_report}[a.cmd](cfg)


if __name__ == "__main__":
    main()
