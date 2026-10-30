"""Part 3, experiment (a): the policy head learns parity from the judge.

    python scripts/parity_hand.py data      # parity positions with the judge's answers, per size
    python scripts/parity_hand.py stats     # their distribution -> docs/parity_hand_data.md
    python scripts/parity_hand.py train     # fine-tune the arms (train.arms) -> runs/parity_hand/{arm}.pt
    python scripts/parity_hand.py probe     # held-out parity probe: base Dot-2 and every arm
    python scripts/parity_hand.py elo       # raw policy ladder games on 5x5 and 10x7
    python scripts/parity_hand.py routs     # MCTS 50 vs the simulator's Hard on 10x7, margins
    python scripts/parity_hand.py bridge    # 20 games vs the real Hard (scripts/az_vs_hard.py)
    python scripts/parity_hand.py report    # docs/parity_hand.md

configs/parity_hand.yaml, plan in docs/plan_part3.md. Positions come from
the generator of scripts/transfer.py (games of ladder agents; positions
without a capture and with 1..max_safe safe lines go to the judge with a
time limit; only solved positions where the move decides the result are
kept). The probe sets (5x5 docs/parity_probe_positions.json, 10x7
docs/transfer_parity_10x7.json) are held out: positions in them are
skipped. Every finished game is appended to runs/parity_hand/data_{size}.jsonl
(a copy of the sets used is in data/parity_hand_{size}.jsonl.gz),
so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from multiprocessing import get_context

import numpy as np

from _common import ROOT, load_config

import transfer


def key(p):
    return tuple(sorted(p["moves"])), tuple(p["score"]), p["player"]


def heldout_keys(path):
    return {key(p) for p in json.loads((ROOT / path).read_text())["positions"]}


def read_data(cfg, bname):
    """Games done and distinct positions (held-out ones dropped) of one size."""
    path = ROOT / cfg["data"]["out"] / f"data_{bname}.jsonl"
    held = heldout_keys(cfg["data"]["boards"][bname]["heldout"])
    done, uniq, tried, solved = set(), {}, 0, 0
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                g = json.loads(line)
            except json.JSONDecodeError:  # the last line while the generator writes it
                continue
            done.add(g["game"])
            tried += g["tried"]
            solved += g["solved"]
            for p in g["found"]:
                if key(p) not in held:
                    uniq.setdefault(key(p), p)
    return path, done, uniq, tried, solved


def cmd_data(cfg):
    dc = cfg["data"]
    for bname, bc in dc["boards"].items():
        path, done, uniq, tried, solved = read_data(cfg, bname)
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(uniq) >= dc["n"]:
            print(f"{bname}: {len(uniq)} positions already", flush=True)
            continue
        wh = transfer.wh_of(bname)
        rng = random.Random(f"{cfg['seed']}/{bname}")
        seeds = [rng.randrange(2**31) for _ in range(bc["games"])]
        gpc = dict(bc, pairs=dc["pairs"])
        tasks = [(g, (wh, *dc["pairs"][g % len(dc["pairs"])], seeds[g], g % 2, gpc))
                 for g in range(bc["games"]) if g not in done]
        held = heldout_keys(bc["heldout"])
        t0 = time.time()
        with get_context("spawn").Pool(cfg["processes"], initializer=transfer._ginit, initargs=(cfg,)) as pool, \
                open(path, "a") as f:
            for i, (g, (found, t, s)) in enumerate(pool.imap_unordered(_one, tasks, chunksize=1)):
                f.write(json.dumps({"game": g, "tried": t, "solved": s, "found": found}) + "\n")
                f.flush()
                tried += t
                solved += s
                for p in found:
                    if key(p) not in held:
                        uniq.setdefault(key(p), p)
                if (i + 1) % 50 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] {bname}: {len(done) + i + 1} games, "
                          f"{time.time() - t0:.0f} s, positions {len(uniq)}, judge solved {solved}/{tried}",
                          flush=True)
                if len(uniq) >= dc["n"]:
                    pool.terminate()
                    break
        print(f"{bname}: {len(uniq)} distinct positions, judge solved {solved}/{tried}", flush=True)


def _one(task):
    g, t = task
    return g, transfer._gen_game(t)


# --- the distribution -------------------------------------------------------------

def describe(bname, positions):
    """Chains / loops formed at each position (sim/chains.py), safe lines, etc."""
    from agents.baselines import safe_moves
    from sim.board import Board
    from sim.chains import analyse

    rows = []
    for p in positions:
        b = Board(*transfer.wh_of(bname))
        for ln in p["moves"]:
            b.step(ln)
        a = analyse(b)
        rows.append({"long": a["long"], "chains": a["chains"], "loops": a["loops"], "safe": len(safe_moves(b)),
                     "undrawn": p["undrawn"], "n_correct": len(p["correct"]), "n_moves": len(p["values"]),
                     "source": p["source"]})
    return rows


def cmd_stats(cfg):
    L = ["# Parity training sets (part 3, experiment a)", "",
         f"Generated by `scripts/parity_hand.py data` / `stats` ({time.strftime('%Y-%m-%d %H:%M')}), "
         "configs/parity_hand.yaml. Positions without a capture, with 1..max_safe safe lines, solved exactly "
         "by the judge within its time limit, where the move decides the result (some moves win, others "
         "lose). Chains / loops are those formed on the board at the position (sim/chains.py: runs of boxes "
         "with two undrawn sides); long = 3+ boxes. The held-out probe sets are excluded.", ""]
    sets = {"train": {}, "held-out": {}}
    for bname, bc in cfg["data"]["boards"].items():
        _, done, uniq, tried, solved = read_data(cfg, bname)
        sets["train"][bname] = (list(uniq.values()), f"{len(done)} games, judge solved {solved}/{tried}")
        sets["held-out"][bname] = (json.loads((ROOT / bc["heldout"]).read_text())["positions"], bc["heldout"])
    L += ["| set | board | positions | source | safe lines | long chains | loops | correct moves / legal | "
          "undrawn lines |", "|---|---|---:|---|---|---|---|---|---|"]
    hists = {}
    for sname, by in sets.items():
        for bname, (ps, src) in by.items():
            if not ps:
                continue
            rows = describe(bname, ps)
            hists[(sname, bname)] = rows

            def dist(xs, top=6):
                c = Counter(min(x, top) for x in xs)
                return ", ".join(f"{k if k < top else f'{top}+'}: {100 * c[k] / len(xs):.0f}%" for k in sorted(c))

            L.append(f"| {sname} | {bname} | {len(ps)} | {src} | {dist([r['safe'] for r in rows])} | "
                     f"{dist([r['long'] for r in rows])} | {dist([len(r['loops']) for r in rows], 3)} | "
                     f"{np.mean([r['n_correct'] for r in rows]):.1f} / {np.mean([r['n_moves'] for r in rows]):.1f} | "
                     f"{np.median([r['undrawn'] for r in rows]):.0f} (median) |")
    L += ["", "Chain lengths (all chains formed, share of chains):", "",
          "| set | board | 1 | 2 | 3 | 4 | 5 | 6-8 | 9+ |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (sname, bname), rows in hists.items():
        ls = [c for r in rows for c in r["chains"]]
        bins = [sum(c == k for c in ls) for k in (1, 2, 3, 4, 5)] + [sum(6 <= c <= 8 for c in ls),
                                                                     sum(c >= 9 for c in ls)]
        L.append(f"| {sname} | {bname} | " + " | ".join(f"{100 * b / max(1, len(ls)):.0f}%" for b in bins) + " |")
    L += ["", "Source games (pairs of agents):", ""]
    for (sname, bname), rows in hists.items():
        c = Counter(r["source"] for r in rows)
        L.append(f"- {sname} {bname}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    out = ROOT / "docs/parity_hand_data.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


# --- fine-tuning ------------------------------------------------------------------

HEAD = ("p_conv", "g_conv", "p_g", "p_out")   # AZNet2's policy head


def load_base(cfg):
    import torch

    from agents.az_net import make_net

    rc = load_config(ROOT / cfg["net"]["config"])
    net = make_net(rc)
    net.load_state_dict(torch.load(ROOT / cfg["net"]["checkpoint"], map_location="cpu"))
    return net, rc


def parity_arrays(bname, positions):
    """Encoded positions (N, planes, cells) and correct-move masks (N, n_lines)."""
    from agents.az_net import Codec, N_PLANES_2
    from sim.board import Board

    codec = Codec(*transfer.wh_of(bname), planes=N_PLANES_2)
    boards = []
    for p in positions:
        b = Board(codec.w, codec.h)
        for ln in p["moves"]:
            b.step(ln)
        boards.append(b)
    x = codec.encode_boards(boards).reshape(len(boards), codec.planes, -1)
    mask = np.zeros((len(boards), codec.n_lines), dtype=bool)
    for i, p in enumerate(positions):
        mask[i, p["correct"]] = True
    return codec, x, mask


def split_data(cfg, bname):
    _, _, uniq, _, _ = read_data(cfg, bname)
    ps = sorted(uniq.values(), key=lambda p: json.dumps(key(p)))
    random.Random(f"{cfg['seed']}/split/{bname}").shuffle(ps)
    k = int(len(ps) * cfg["train"]["val_share"])
    return ps[k:], ps[:k]


def augment(rng, codec, x, *lines_arrays, on=True):
    """Random grid symmetry per sample: x (B, planes, cells), arrays (B, n_lines)."""
    syms = codec.syms if on else codec.syms[:1]
    t = rng.integers(0, len(syms), len(x))
    x = x.copy()
    out = [a.copy() for a in lines_arrays]
    for k, (cperm, lperm) in enumerate(syms):
        m = t == k
        if k and m.any():
            x[m] = x[m][:, :, cperm]
            for a in out:
                a[m] = a[m][:, lperm]
    return (x, *out)


def cmd_train(cfg, arms):
    import torch
    import torch.nn.functional as F

    from agents.az_net import Codec, N_PLANES_2
    from az_train import Buffer

    tc = cfg["train"]
    dev = cfg["device"]
    out = ROOT / cfg["data"]["out"]
    sizes = list(cfg["data"]["boards"])
    par = {}
    for bname in sizes:
        tr, va = split_data(cfg, bname)
        par[bname] = (parity_arrays(bname, tr), parity_arrays(bname, va))
        print(f"{bname}: {len(tr)} train, {len(va)} validation positions", flush=True)
    bufs, codecs = {}, {}
    for bname, path in tc["selfplay_buffers"].items():
        codecs[bname] = Codec(*transfer.wh_of(bname), planes=N_PLANES_2)
        bufs[bname] = Buffer(1, codecs[bname].n_lines, codecs[bname].n_boxes)
        d = np.load(ROOT / path)
        bufs[bname].a = {k: d[k] for k in bufs[bname].a}
        bufs[bname].n = int(d["n"])
    bw = np.array([tc["selfplay_weights"][b] for b in bufs], dtype=float)
    bw /= bw.sum()

    for arm in arms:
        ac = tc["arms"][arm]
        torch.manual_seed(cfg["seed"])
        rng = np.random.default_rng([cfg["seed"], len(arm)])
        net, _ = load_base(cfg)
        net.to(dev).eval()  # BatchNorm statistics stay as trained
        head = [p for n, p in net.named_parameters() if n.split(".")[0] in HEAD]
        rest = [p for n, p in net.named_parameters() if n.split(".")[0] not in HEAD]
        if ac["train"] == "head":
            for p in rest:
                p.requires_grad_(False)
            groups = [{"params": head, "lr": tc["lr"]}]
        else:
            groups = [{"params": head, "lr": tc["lr"]}, {"params": rest, "lr": tc["lr"] * ac["body_lr_scale"]}]
        opt = torch.optim.Adam(groups)

        def logp_of(x, codec):
            xt = torch.from_numpy(x.reshape(len(x), -1, codec.rows, codec.cols)).to(dev)
            logits, v, sc = net(xt)
            legal = xt[:, 1].flatten(1)[:, net.line_cells(codec.rows, codec.cols)] > 0
            return F.log_softmax(logits.masked_fill(~legal, -1e9), 1), v, sc

        def validate():
            r = {}
            with torch.no_grad():
                for bname, (_, (codec, x, mask)) in par.items():
                    lp, _, _ = logp_of(x, codec)
                    am = lp.argmax(1).cpu().numpy()
                    r[bname] = 100 * float(np.mean(mask[np.arange(len(am)), am]))
            return r

        log = [{"step": 0, "val": validate()}]
        print(f"{arm}: step 0 validation {log[-1]['val']}", flush=True)
        acc = {"parity": 0.0, "policy": 0.0, "value": 0.0}
        t0 = time.time()
        for step in range(1, tc["steps"] + 1):
            bname = sizes[step % len(sizes)]
            (codec, X, M), _ = par[bname]
            idx = rng.integers(0, len(X), tc["parity_batch"])
            x, m = augment(rng, codec, X[idx], M[idx], on=tc["symmetries"])
            lp, _, _ = logp_of(x, codec)
            mt = torch.from_numpy(m).to(dev)
            lpar = -torch.logsumexp(lp.masked_fill(~mt, -1e9), 1).mean()
            j = list(bufs)[int(rng.choice(len(bufs), p=bw))]
            s = bufs[j].sample(rng, tc["selfplay_batch"])
            c = codecs[j]
            xs = c.encode(s["lines"], s["sides"], s["owner"], s["diff"].astype(np.float32), s["first"])
            xs, pi = augment(rng, c, xs.reshape(len(xs), c.planes, -1), s["pi"].astype(np.float32),
                             on=tc["symmetries"])
            lps, v, sc = logp_of(xs, c)
            lpol = -(torch.from_numpy(pi).to(dev) * lps).sum(1).mean()
            loss = tc["parity_weight"] * lpar + lpol
            if ac.get("value_loss"):
                lv = F.mse_loss(v, torch.from_numpy(s["z"].astype(np.float32)).to(dev))
                loss = loss + lv
                if sc is not None:
                    mg = torch.from_numpy(s["margin"].astype(np.float32) / c.n_boxes).to(dev)
                    loss = loss + rc_score_weight(cfg) * F.mse_loss(sc, mg)
                acc["value"] += lv.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            acc["parity"] += lpar.item()
            acc["policy"] += lpol.item()
            if step % tc["log_every"] == 0:
                rec = {"step": step, **{k: v / tc["log_every"] for k, v in acc.items()}, "val": validate(),
                       "seconds": time.time() - t0}
                log.append(rec)
                acc = dict.fromkeys(acc, 0.0)
                print(f"{arm}: step {step} parity loss {rec['parity']:.3f} self-play policy {rec['policy']:.3f} "
                      f"value {rec['value']:.3f} validation {rec['val']} ({rec['seconds']:.0f} s)", flush=True)
        net.cpu()
        torch.save(net.state_dict(), out / f"{arm}.pt")
        (out / f"{arm}_train.json").write_text(json.dumps(log, indent=1))


def rc_score_weight(cfg):
    return load_config(ROOT / cfg["net"]["config"])["train"].get("score_weight", 0.0)


# --- evaluation -------------------------------------------------------------------

def nets_of(cfg, names):
    """name -> checkpoint path: 'base' is Dot-2 as it is, others the arms."""
    out = ROOT / cfg["data"]["out"]
    return {n: ROOT / cfg["net"]["checkpoint"] if n == "base" else out / f"{n}.pt" for n in names}


def evaluator(cfg, path):
    import torch

    from agents.az_net import Evaluator, make_net

    rc = load_config(ROOT / cfg["net"]["config"])
    net = make_net(rc)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return Evaluator(net, cfg["device"]), rc


def results(cfg):
    path = ROOT / cfg["data"]["out"] / "eval.json"
    return path, (json.loads(path.read_text()) if path.exists() else {})


def cmd_probe(cfg, names):
    from parity_probe import gen_curves

    path, res = results(cfg)
    for name, ck in nets_of(cfg, names).items():
        ev, rc = evaluator(cfg, ck)
        for bname, bc in cfg["data"]["boards"].items():
            k = f"probe/{name}/{bname}"
            if k in res:
                continue
            d, boards = transfer.load_boards(bc["heldout"])
            r = gen_curves(ev, rc, boards, d["positions"], cfg["eval"]["probe_sims"], cfg["seed"])
            res[k] = {"n": len(boards), "policy": 100 * float(np.mean(r["policy"])),
                      "mcts": 100 * float(np.mean(r["mcts"])), "v_child": float(r["v_child"]),
                      "v_child_win": float(r["v_child_win"]), "v_child_lose": float(r["v_child_lose"])}
            transfer.save(path, res)
            print(f"{k}: {res[k]}", flush=True)


def cmd_elo(cfg, names):
    import torch

    from az_eval import _oinit, play

    torch.set_num_threads(2)
    path, res = results(cfg)
    ec = cfg["eval"]["elo"]
    for bname, bc in ec["boards"].items():
        with get_context("spawn").Pool(cfg["processes"], initializer=_oinit,
                                       initargs=(bc["params"], cfg["seed"])) as pool:
            for name, ck in nets_of(cfg, names).items():
                ev, rc = evaluator(cfg, ck)
                ecfg = {"board": list(transfer.wh_of(bname)), "mcts": rc["mcts"], "eval": {"games": ec["games"]}}
                for o in ec["opponents"]:
                    k = f"elo/{name}/{bname}/{o}"
                    if k in res:
                        continue
                    t0 = time.time()
                    rng = np.random.default_rng([cfg["seed"], len(o), 7])
                    res[k] = play(ev, "raw", [o], ecfg, pool, rng)[o]
                    transfer.save(path, res)
                    print(f"[{time.strftime('%H:%M:%S')}] {k}: {res[k]} ({time.time() - t0:.0f} s)", flush=True)


def cmd_routs(cfg, names):
    from az_eval import _oinit, play

    path, res = results(cfg)
    rc_ = cfg["eval"]["routs"]
    with get_context("spawn").Pool(cfg["processes"], initializer=_oinit,
                                   initargs=({}, cfg["seed"])) as pool:
        for name, ck in nets_of(cfg, names).items():
            k = f"routs/{name}"
            if k in res:
                continue
            ev, rc = evaluator(cfg, ck)
            ecfg = {"board": [10, 7], "mcts": rc["mcts"], "eval": {"games": rc_["games"], "sims": rc_["sims"]}}
            t0 = time.time()
            margins = {}
            r = play(ev, "mcts", ["hard"], ecfg, pool, np.random.default_rng([cfg["seed"], 50]), margins)["hard"]
            res[k] = {"wlt": r, "margins": margins["hard"]}
            transfer.save(path, res)
            print(f"[{time.strftime('%H:%M:%S')}] {k}: {r}, routs "
                  f"{sum(m <= -rc_['margin'] for m in margins['hard'])} ({time.time() - t0:.0f} s)", flush=True)


def cmd_bridge(cfg, names):
    import subprocess
    import sys

    path, res = results(cfg)
    out = ROOT / cfg["data"]["out"]
    for name, ck in nets_of(cfg, names).items():
        k = f"bridge/{name}"
        if k in res:
            continue
        js = out / f"bridge_{name}.json"
        subprocess.run([sys.executable, str(ROOT / "scripts/az_vs_hard.py"), "--config", cfg["net"]["config"],
                        "--checkpoint", str(ck), "--json", str(js)], check=True, cwd=ROOT)
        res = results(cfg)[1]
        res[k] = json.loads(js.read_text())
        transfer.save(path, res)


def cmd_report(cfg):
    from az_eval import fit_anchors, fit_one, load_ladder

    from parity_probe import wilson

    _, res = results(cfg)
    out = ROOT / cfg["data"]["out"]
    names = ["base", *cfg["train"]["arms"]]
    tc, ec = cfg["train"], cfg["eval"]
    L = ["# Part 3 (a): the hand learns parity from the judge", "",
         f"Generated by `scripts/parity_hand.py report` ({time.strftime('%Y-%m-%d %H:%M')}), "
         "configs/parity_hand.yaml, plan in docs/plan_part3.md, training data in docs/parity_hand_data.md. "
         f"**base** = Dot-2 ({cfg['net']['checkpoint']}, the app's 10x7 network). Arms, fine-tuned for "
         f"{tc['steps']} steps (Adam, lr {tc['lr']}; each step {tc['parity_batch']} parity positions of one size "
         f"+ {tc['selfplay_batch']} self-play positions from dot2c's replay buffers with their MCTS targets): "
         + "; ".join(f"**{a}** " + ("policy head only" if c["train"] == "head" else
                                    f"whole net, body and value heads at {c['body_lr_scale']}x lr, "
                                    "self-play value loss kept") for a, c in tc["arms"].items()) + ".", ""]

    def ci(x, n):
        lo, hi = wilson(round(x * n / 100), n)
        return f"{x:.1f} [{lo:.0f}-{hi:.0f}]"

    L += ["## Parity probe (held-out sets)", "",
          "Policy: the raw argmax is a correct move (no search). MCTS: "
          f"{ec['probe_sims']} simulations. V after the move: balanced share of right signs (50 = blind). "
          "95% Wilson intervals. 5x5: docs/parity_probe_positions.json (1-6 safe lines); 10x7: "
          "docs/transfer_parity_10x7.json (1-2 safe lines); baselines on the same sets: 5x5 hard 69%, "
          "greedy 61%, strong 97%; 10x7 greedy 66%, hard 64% (docs/transfer.md, docs/parity_curves.md).", "",
          "| net | board | n | policy % | MCTS % | V after the move % (win / lose) | validation (train split) % |",
          "|---|---|---:|---|---|---|---|"]
    for n in names:
        tj = out / f"{n}_train.json"
        val = json.loads(tj.read_text())[-1]["val"] if tj.exists() else {}
        for b in cfg["data"]["boards"]:
            r = res.get(f"probe/{n}/{b}")
            if r:
                L.append(f"| {n} | {b} | {r['n']} | {ci(r['policy'], r['n'])} | {ci(r['mcts'], r['n'])} | "
                         f"{r['v_child']:.1f} ({r['v_child_win']:.0f} / {r['v_child_lose']:.0f}) | "
                         f"{val.get(b, float('nan')):.1f} |")
    L += ["", "## Elo without search", "",
          f"Raw policy argmax, {ec['elo']['games']} games per opponent, win % (ties half); Elo by Bradley-Terry "
          "against fixed anchors (5x5: the az_eval ladder; 10x7: docs/ladder_10x7*.json).", ""]
    for b, bc in ec["elo"]["boards"].items():
        anchors = fit_anchors(*load_ladder(bc["ladder"])) if bc["ladder"] else fit_anchors()
        opps = ec["elo"]["opponents"]
        L += [f"{b}: anchors " + ", ".join(f"{a} {anchors[a]:.0f}" for a in opps if a in anchors) + ".", "",
              "| net | " + " | ".join(opps) + " | Elo |", "|---|" + "---:|" * (len(opps) + 1)]
        for n in names:
            r = {o: res[f"elo/{n}/{b}/{o}"] for o in opps if f"elo/{n}/{b}/{o}" in res}
            if not r:
                continue
            cells = [f"{100 * (r[o][0] + r[o][2] / 2) / sum(r[o]):.1f}" if o in r else "—" for o in opps]
            L.append(f"| {n} | " + " | ".join(cells) + f" | {fit_one(r, anchors):.0f} |")
        L.append("")
    rc_ = ec["routs"]
    L += ["## Routs vs Hard on 10x7", "",
          f"Simulator: {rc_['games']} games vs the simulator's Hard, MCTS {rc_['sims']} (the app's lowest), "
          f"first mover alternating. Bridge: {ec['bridge']['games']} games vs the real KSquares Hard "
          "(scripts/az_vs_hard.py, MCTS 200, srand 11 + game). Rout = a loss by "
          f"{rc_['margin']}+ boxes of 70.", "",
          "| net | sim: wins-losses-ties | sim: routs | sim: worst | bridge: wins-losses-ties | bridge: routs |",
          "|---|---:|---:|---:|---:|---:|"]
    for n in names:
        r, br = res.get(f"routs/{n}"), res.get(f"bridge/{n}")
        if not (r or br):
            continue
        cells = ["—"] * 5
        if r:
            ms = r["margins"]
            cells[:3] = ["-".join(map(str, r["wlt"])), f"{sum(m <= -rc_['margin'] for m in ms)}", f"{min(ms):+d}"]
        if br:
            cells[3] = f"{br['wins']}-{br['losses']}-{br['ties']}"
            cells[4] = f"{sum(g['net'] - g['hard'] <= -rc_['margin'] for g in br['games'])}"
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    concl = ROOT / "docs/parity_hand_conclusions.md"  # written by hand from the tables
    if concl.exists():
        L += ["", concl.read_text().rstrip()]
    (ROOT / "docs/parity_hand.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["data", "stats", "train", "probe", "elo", "routs", "bridge", "report"])
    ap.add_argument("--config", default=str(ROOT / "configs/parity_hand.yaml"))
    ap.add_argument("--nets", nargs="+", help="arms to train / nets to evaluate (default: all; 'base' = Dot-2)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.cmd in ("data", "stats", "report"):
        {"data": cmd_data, "stats": cmd_stats, "report": cmd_report}[a.cmd](cfg)
    elif a.cmd == "train":
        cmd_train(cfg, a.nets or list(cfg["train"]["arms"]))
    else:
        names = a.nets or ["base", *cfg["train"]["arms"]]
        {"probe": cmd_probe, "elo": cmd_elo, "routs": cmd_routs, "bridge": cmd_bridge}[a.cmd](cfg, names)


if __name__ == "__main__":
    main()
