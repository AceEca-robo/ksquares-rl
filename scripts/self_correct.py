"""Part 3 (d): the hand learns from its own mistakes, checked by the judge.

    python scripts/self_correct.py collect     # Dot-2 vs Hard and vs itself on 10x7 until enough losses
    python scripts/self_correct.py judge       # the middle-game judge from the end backwards per loss
    python scripts/self_correct.py train       # head-only fine-tune per variant (mistake share of the batch)
    python scripts/self_correct.py noise       # routs of every net on more seeds (the count's spread)
    python scripts/self_correct.py report      # docs/self_correction.md

Evaluation of the variants: scripts/parity_hand.py probe / elo / routs /
bridge --config configs/self_correct.yaml --nets ...

collect: dot2c gen 16 (MCTS ``sims`` each) plays the simulator's Hard (net
first alternating, temperature 0) and itself (temperature 1 for the first
``temp_moves`` lines); per (opponent, sims) games are played in batches
until ``losses`` games are lost (vs Hard: the net lost; vs itself: every
decided game, the loser's side) or ``max_games`` are played. Seeds differ
from the evaluation's.

judge: the loser's positions from the last backwards, ``mid_judge.outcome``
within ``time_limit`` s each, until the first that is not lost (*found*:
the mistake is the line played there; the correct moves are those
``classify`` gives the best result) or the first out of time (*wall*).

train: as arm a1 of configs/parity_hand.yaml (policy head only; body,
value and score heads frozen, BatchNorm statistics as trained): each step
``batch`` positions, a ``share`` of them mistakes (loss -log policy mass on
the judge's correct moves), the rest from dot2c's replay buffers with their
MCTS targets.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from _common import ROOT, load_config

import transfer

_G = {}


def out_dir(cfg):
    p = ROOT / cfg["out"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def board_of(moves):
    from sim.board import Board

    b = Board(10, 7)
    for ln in moves:
        b.step(ln)
    return b


# --- collect ---------------------------------------------------------------------------

def play_batch(ev, rc, opp, sims, n, pool, rng, temp_moves, _opp_move):
    """n games on 10x7; returns [{opp, sims, net_seat, moves, score}].
    net_seat None: the net plays both sides."""
    from agents.az_mcts import choose_move
    from agents.az_play import make_mcts
    from sim.board import Board

    mcts = make_mcts(ev, rc["mcts"], rng, leaves_per_tree=1)
    live = [{"b": Board(10, 7), "moves": [], "seat": None if opp == "self" else g % 2} for g in range(n)]
    done = []
    while live:
        net_turn = [g for g in live if g["seat"] is None or g["b"].player == g["seat"]]
        opp_turn = [g for g in live if not (g["seat"] is None or g["b"].player == g["seat"])]
        async_res = pool.map_async(_opp_move, [("hard", g["b"]) for g in opp_turn], chunksize=1) if opp_turn else None
        if net_turn:
            roots = mcts.search([g["b"] for g in net_turn], sims)
            for g, r in zip(net_turn, roots):
                t = 1.0 if g["seat"] is None and g["b"].n_drawn < temp_moves else 0.0
                # a decided game (the winner is fixed): the search returns no tree; the policy moves
                ln = int(np.argmax(ev([g["b"]])[0][0])) if r.terminal else choose_move(r, t, rng)
                g["b"].step(ln)
                g["moves"].append(ln)
        if async_res is not None:
            for g, ln in zip(opp_turn, async_res.get()):
                g["b"].step(ln)
                g["moves"].append(int(ln))
        nxt = []
        for g in live:
            if g["b"].done:
                done.append({"opp": opp, "sims": sims, "net_seat": g["seat"], "moves": g["moves"],
                             "score": [int(x) for x in g["b"].score]})
            else:
                nxt.append(g)
        live = nxt
    return done


def loser_seat(g):
    """The seat whose mistakes count: the net's if it lost (vs Hard); the
    loser in self-play; None if the net did not lose / a tie."""
    s = g["score"]
    if s[0] == s[1]:
        return None
    lose = 0 if s[0] < s[1] else 1
    return lose if g["net_seat"] is None or g["net_seat"] == lose else None


def cmd_collect(cfg):
    import torch

    import parity_hand
    from az_eval import _oinit, _opp_move

    cc = cfg["collect"]
    ev, rc = parity_hand.evaluator({"device": cfg["device"]}, (cfg["net"]["config"], ROOT / cfg["net"]["checkpoint"]))
    torch.manual_seed(cfg["seed"])
    path = out_dir(cfg) / "games.jsonl"
    games = read_jsonl(path)
    with get_context("spawn").Pool(cfg["processes"], initializer=_oinit, initargs=({}, cc["seed"])) as pool, \
            open(path, "a") as fh:
        for opp, sims, want in cc["arms"]:
            rng = np.random.default_rng([cc["seed"], sims, len(opp), len(games)])
            while True:
                mine = [g for g in games if g["opp"] == opp and g["sims"] == sims]
                lost = sum(loser_seat(g) is not None for g in mine)
                if lost >= want or len(mine) >= cc["max_games"]:
                    break
                t0 = time.time()
                new = play_batch(ev, rc, opp, sims, cc["batch"], pool, rng, cc["temp_moves"], _opp_move)
                for g in new:
                    fh.write(json.dumps(g) + "\n")
                fh.flush()
                games += new
                mine += new
                print(f"[{time.strftime('%H:%M:%S')}] {opp} MCTS-{sims}: {len(mine)} games, "
                      f"{sum(loser_seat(g) is not None for g in mine)} lost ({time.time() - t0:.0f} s)", flush=True)


def losses(cfg):
    """The lost games to judge: per arm the first ``want`` losses, in order."""
    games = read_jsonl(out_dir(cfg) / "games.jsonl")
    out = []
    for opp, sims, want in cfg["collect"]["arms"]:
        mine = [(i, g) for i, g in enumerate(games) if g["opp"] == opp and g["sims"] == sims
                and loser_seat(g) is not None]
        out += mine[:want]
    return out


# --- judge -----------------------------------------------------------------------------

def _jinit(tt_bits):
    sys.setrecursionlimit(10000)
    from sim import mid_judge as mj

    _G["j"] = mj.Judge(10, 7, tt_bits=tt_bits)


def _judge(task):
    from agents.baselines import safe_moves
    from sim import mid_judge as mj

    gid, g, limit = task
    j = _G["j"]
    seat = loser_seat(g)
    moves = g["moves"]
    b = board_of([])
    pos = []
    for k, ln in enumerate(moves):
        if b.player == seat:
            pos.append((k, len(safe_moves(b))))
        b.step(ln)
    r = {"gid": gid, "opp": g["opp"], "sims": g["sims"], "seat": seat, "score": g["score"], "walk": []}
    for k, s in reversed(pos):
        b = board_of(moves[:k])
        j.clear()
        t0 = time.perf_counter()
        try:
            o = mj.outcome(j, b, limit)
        except mj.Timeout:
            r["walk"].append({"k": k, "s": s, "outcome": None})
            r.update(status="wall", k=k, s=s)
            return r
        r["walk"].append({"k": k, "s": s, "outcome": o, "t": round(time.perf_counter() - t0, 2)})
        if o >= 0:
            j.clear()
            try:
                lab = {int(m): v for m, v in mj.classify(j, b, 4 * limit).items()}
            except mj.Timeout:
                r.update(status="found-unlabelled", k=k, s=s, line=moves[k], outcome_before=o)
                return r
            best = max(lab.values())
            r.update(status="found", k=k, s=s, line=moves[k], outcome_before=o, played_label=lab.get(moves[k]),
                     correct=sorted(m for m, v in lab.items() if v == best), n_legal=len(lab))
            return r
    r.update(status="lost from the start")
    return r


def cmd_judge(cfg, name="judge"):
    jc = cfg["judge"]
    path = out_dir(cfg) / f"{name}.jsonl"
    done = {r["gid"] for r in read_jsonl(path)}
    tasks = [(i, g, jc["time_limit"]) for i, g in losses(cfg) if i not in done]
    print(f"judge: {len(tasks)} lost games", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(cfg["processes"], initializer=_jinit, initargs=(jc["tt_bits"],)) as pool, \
            open(path, "a") as fh:
        for i, r in enumerate(pool.imap_unordered(_judge, tasks)):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if (i + 1) % 10 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(tasks)} ({time.time() - t0:.0f} s)", flush=True)


def mistakes(cfg):
    """[{moves, correct, ...}] of the found mistakes, split train / validation."""
    rs = [r for r in read_jsonl(out_dir(cfg) / "judge.jsonl") if r["status"] == "found"]
    games = read_jsonl(out_dir(cfg) / "games.jsonl")
    ps = [{"moves": games[r["gid"]]["moves"][:r["k"]], "correct": r["correct"], "played": r["line"],
           "gid": r["gid"], "s": r["s"]} for r in sorted(rs, key=lambda r: r["gid"])]
    random.Random(f"{cfg['seed']}/split").shuffle(ps)
    k = int(round(len(ps) * cfg["train"]["val_share"]))
    return ps[k:], ps[:k]


# --- train -----------------------------------------------------------------------------

def cmd_train(cfg, variants):
    import torch
    import torch.nn.functional as F

    import parity_hand as ph
    from agents.az_net import Codec, N_PLANES_2
    from az_train import Buffer

    tc = cfg["train"]
    dev = cfg["device"]
    tr, va = mistakes(cfg)
    print(f"mistakes: {len(tr)} train, {len(va)} validation", flush=True)
    bufs = {}
    for bname, path in tc["selfplay_buffers"].items():
        c = Codec(*transfer.wh_of(bname), planes=N_PLANES_2)
        bufs[bname] = Buffer(1, c.n_lines, c.n_boxes)
        d = np.load(ROOT / path)
        bufs[bname].a = {k: d[k] for k in bufs[bname].a}
        bufs[bname].n = int(d["n"])
    bw = np.array([tc["selfplay_weights"][b] for b in bufs], dtype=float)
    bw /= bw.sum()
    codecs = {b: Codec(*transfer.wh_of(b), planes=N_PLANES_2) for b in bufs}
    pcfg = {"net": cfg["net"]}
    for name in variants:
        share = tc["variants"][name]
        nm = int(round(share * tc["batch"]))
        torch.manual_seed(cfg["seed"])
        rng = np.random.default_rng([cfg["seed"], int(100 * share)])
        net, _ = ph.load_base(pcfg)
        mtr = ph.parity_arrays("10x7", tr)
        mva = ph.parity_arrays("10x7", va) if va else (mtr[0], mtr[1][:0], mtr[2][:0])
        frozen0 = {k: v.clone() for k, v in net.state_dict().items() if k.split(".")[0] not in ph.HEAD}
        net.to(dev).eval()
        head = [p for n, p in net.named_parameters() if n.split(".")[0] in ph.HEAD]
        for n, p in net.named_parameters():
            if n.split(".")[0] not in ph.HEAD:
                p.requires_grad_(False)
        opt = torch.optim.Adam(head, lr=tc["lr"])

        def logp_of(x, codec):
            xt = torch.from_numpy(x.reshape(len(x), -1, codec.rows, codec.cols)).to(dev)
            logits, v, sc = net(xt)
            legal = xt[:, 1].flatten(1)[:, net.line_cells(codec.rows, codec.cols)] > 0
            return F.log_softmax(logits.masked_fill(~legal, -1e9), 1)

        def acc(arr):
            codec, x, mask = arr
            if not len(x):
                return float("nan")
            with torch.no_grad():
                am = logp_of(x, codec).argmax(1).cpu().numpy()
            return 100 * float(np.mean(mask[np.arange(len(am)), am]))

        log = [{"step": 0, "train_acc": acc(mtr), "val": acc(mva)}]
        print(f"{name}: step 0 {log[-1]}", flush=True)
        run = {"mistake": 0.0, "policy": 0.0}
        t0 = time.time()
        for step in range(1, tc["steps"] + 1):
            loss = 0.0
            if nm:
                codec, X, M = mtr
                idx = rng.integers(0, len(X), nm)
                x, m = ph.augment(rng, codec, X[idx], M[idx], on=tc["symmetries"])
                lp = logp_of(x, codec)
                lm = -torch.logsumexp(lp.masked_fill(~torch.from_numpy(m).to(dev), -1e9), 1).mean()
                loss = loss + share * lm
                run["mistake"] += lm.item()
            j = list(bufs)[int(rng.choice(len(bufs), p=bw))]
            s = bufs[j].sample(rng, tc["batch"] - nm)
            c = codecs[j]
            xs = c.encode(s["lines"], s["sides"], s["owner"], s["diff"].astype(np.float32), s["first"])
            xs, pi = ph.augment(rng, c, xs.reshape(len(xs), c.planes, -1), s["pi"].astype(np.float32),
                                on=tc["symmetries"])
            lpol = -(torch.from_numpy(pi).to(dev) * logp_of(xs, c)).sum(1).mean()
            loss = loss + (1 - share) * lpol
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run["policy"] += lpol.item()
            if step % tc["log_every"] == 0:
                rec = {"step": step, **{k: v / tc["log_every"] for k, v in run.items()},
                       "train_acc": acc(mtr), "val": acc(mva), "seconds": round(time.time() - t0)}
                log.append(rec)
                run = dict.fromkeys(run, 0.0)
                print(f"{name}: {rec}", flush=True)
        net.cpu()
        par = {"10x7": (None, mva if len(mva[1]) else mtr)}
        log.append({"step": "frozen", **ph.frozen_check(pcfg, net, frozen0, par)})
        print(f"{name}: frozen check {log[-1]}", flush=True)
        torch.save(net.state_dict(), out_dir(cfg) / f"{name}.pt")
        (out_dir(cfg) / f"{name}_train.json").write_text(json.dumps(log, indent=1))


# --- noise ---------------------------------------------------------------------------

def cmd_noise(cfg):
    """Routs (scripts/parity_hand.py routs) of every net on more seeds: how much
    the 200-game count moves between game sets. -> {out}/noise.json"""
    import copy

    import parity_hand as ph

    path = out_dir(cfg) / "noise.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for seed in cfg["noise"]["seeds"]:
        c = copy.deepcopy(cfg)
        c["seed"] = seed
        c["data"]["out"] = str(Path(cfg["out"]) / f"noise_{seed}")
        (ROOT / c["data"]["out"]).mkdir(parents=True, exist_ok=True)
        ph.cmd_routs(c, list(cfg["nets"]))
        r = json.loads((ROOT / c["data"]["out"] / "eval.json").read_text())
        for n in cfg["nets"]:
            res[f"{seed}/{n}"] = r[f"routs/{n}"]
        path.write_text(json.dumps(res))


# --- report ----------------------------------------------------------------------------

def cmd_report(cfg):
    from az_eval import fit_anchors, fit_one, load_ladder

    out = out_dir(cfg)
    games = read_jsonl(out / "games.jsonl")
    js = read_jsonl(out / "judge.jsonl")
    cc, ec = cfg["collect"], cfg["eval"]
    L = ["# Part 3 (d): the hand learns from its own mistakes, checked by the judge", "",
         f"Generated by `scripts/self_correct.py report` ({time.strftime('%Y-%m-%d %H:%M')}), "
         "configs/self_correct.yaml. Net: Dot-2 (dot2c gen 16, the app's 10x7 network).", "",
         "## 1. Mistakes from lost games", "",
         f"Games on 10x7: vs the simulator's Hard (net first alternating, temperature 0) and vs itself "
         f"(temperature 1 for the first {cc['temp_moves']} lines). Per arm games until {cc['arms'][0][2]} "
         f"losses (cap {cc['max_games']} games). Judge: sim/mid_judge.py, {cfg['judge']['time_limit']} s per "
         "position, the loser's positions from the end backwards; *found* = the first position (going back) "
         "that is not lost: the line played there is the mistake, the judge's best lines the target; "
         "*wall* = out of time on a still lost position.", "",
         "| opponent | MCTS | games | lost | judged | found | wall | other | found: safe lines median (range) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for opp, sims, _ in cc["arms"]:
        mine = [g for g in games if g["opp"] == opp and g["sims"] == sims]
        rs = [r for r in js if r["opp"] == opp and r["sims"] == sims]
        f = [r["s"] for r in rs if r["status"] == "found"]
        w = sum(r["status"] == "wall" for r in rs)
        L.append(f"| {opp} | {sims} | {len(mine)} | {sum(loser_seat(g) is not None for g in mine)} | {len(rs)} | "
                 f"{len(f)} | {w} | {len(rs) - len(f) - w} | "
                 + (f"{np.median(f):.0f} ({min(f)}-{max(f)})" if f else "—") + " |")
    found = [r for r in js if r["status"] == "found"]
    walls = [r for r in js if r["status"] == "wall"]
    L += [f"| all | | {len(games)} | | {len(js)} | {len(found)} | {len(walls)} | "
          f"{len(js) - len(found) - len(walls)} | |", ""]
    edges = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 99)]
    L += ["Safe lines at the found mistake / at the wall:", "",
          "| safe lines | " + " | ".join(f"{a}-{b}" if b < 99 else f"{a}+" for a, b in edges) + " |",
          "|---|" + "---:|" * len(edges),
          "| found | " + " | ".join(str(sum(a <= r["s"] <= b for r in found)) for a, b in edges) + " |",
          "| wall | " + " | ".join(str(sum(a <= r["s"] <= b for r in walls)) for a, b in edges) + " |", ""]
    if found:
        L += [f"Found mistakes: before the move {sum(r['outcome_before'] > 0 for r in found)} won, "
              f"{sum(r['outcome_before'] == 0 for r in found)} drawn; median {np.median([len(r['correct']) for r in found]):.0f} "
              f"correct lines of median {np.median([r['n_legal'] for r in found]):.0f} legal.", ""]
    tr, va = mistakes(cfg)
    tc = cfg["train"]
    L += ["## 2. Fine-tuning", "",
          f"Policy head only (as arm a1: body, value and score heads frozen, BatchNorm statistics kept; the "
          f"frozen check below), {tc['steps']} steps, Adam lr {tc['lr']}, {tc['batch']} positions per step: a "
          f"share of them mistakes ({len(tr)} train, {len(va)} validation; loss -log policy mass on the "
          "judge's correct lines), the rest dot2c replay-buffer positions with their MCTS targets. Accuracy = "
          "the raw policy argmax is a correct line.", "",
          "| variant | mistake share | train acc % (start -> end) | validation acc % (start -> end) | "
          "weights / V unchanged |", "|---|---:|---|---|---|"]
    for name, share in tc["variants"].items():
        p = out / f"{name}_train.json"
        if not p.exists():
            continue
        lg = json.loads(p.read_text())
        st = [r for r in lg if r["step"] != "frozen"]
        fz = [r for r in lg if r["step"] == "frozen"][0]
        L.append(f"| {name} | {100 * share:.0f}% | {st[0]['train_acc']:.1f} -> {st[-1]['train_acc']:.1f} | "
                 f"{st[0]['val']:.1f} -> {st[-1]['val']:.1f} | {fz['weights_unchanged']} / {fz['v_bitwise_equal']} |")
    L.append("")
    res_p = out / "eval.json"
    res = json.loads(res_p.read_text()) if res_p.exists() else {}
    names = ["base", *tc["variants"]]
    rc_ = ec["routs"]
    bc = ec["elo"]["boards"]["10x7"]
    anchors = fit_anchors(*load_ladder(bc["ladder"]))
    opps = ec["elo"]["opponents"]
    L += ["## 3. Evaluation", "",
          f"Routs (main metric): {rc_['games']} games vs the simulator's Hard on 10x7, MCTS {rc_['sims']}, "
          f"first mover alternating; rout = a loss by {rc_['margin']}+ boxes. Elo: raw policy, "
          f"{ec['elo']['games']} games per opponent on 10x7 ({', '.join(opps)}), Bradley-Terry against "
          "docs/ladder_10x7*.json. Probe: docs/transfer_parity_10x7.json (policy argmax / MCTS 200 correct). "
          f"Bridge: {ec['bridge']['games']} games vs the real KSquares Hard (MCTS 200).", "",
          "| net | routs / 200 | sim W-L-T | Elo (raw) | probe policy % | probe MCTS % | bridge W-L-T | bridge routs |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        r = res.get(f"routs/{n}")
        el = {o: res[f"elo/{n}/10x7/{o}"] for o in opps if f"elo/{n}/10x7/{o}" in res}
        pr = res.get(f"probe/{n}/10x7")
        br = res.get(f"bridge/{n}")
        if not (r or el or pr or br):
            continue
        c = [n, "—", "—", "—", "—", "—", "—", "—"]
        if r:
            c[1], c[2] = str(sum(m <= -rc_["margin"] for m in r["margins"])), "-".join(map(str, r["wlt"]))
        if el:
            c[3] = f"{fit_one(el, anchors):.0f}"
        if pr:
            c[4], c[5] = f"{pr['policy']:.1f}", f"{pr['mcts']:.1f}"
        if br:
            c[6] = f"{br['wins']}-{br['losses']}-{br['ties']}"
            c[7] = str(sum(g["net"] - g["hard"] <= -rc_["margin"] for g in br["games"]))
        L.append("| " + " | ".join(c) + " |")
    L.append("")
    nz = out / "noise.json"
    if nz.exists():
        nd = json.loads(nz.read_text())
        seeds = [cfg["seed"], *cfg["noise"]["seeds"]]
        L += ["Routs on more game sets (`self_correct.py noise`: the same 200-game match with other seeds; "
              f"the first column is the table above). Total = routs / losses of {200 * len(seeds)} games.", "",
              "| net | " + " | ".join(f"seed {s}" for s in seeds) + " | total routs | per 200 | total losses |",
              "|---|" + "---:|" * (len(seeds) + 3)]
        for n in names:
            rr = [res.get(f"routs/{n}")] + [nd.get(f"{s}/{n}") for s in cfg["noise"]["seeds"]]
            if any(r is None for r in rr):
                continue
            ro = [sum(m <= -rc_["margin"] for m in r["margins"]) for r in rr]
            L.append(f"| {n} | " + " | ".join(map(str, ro)) + f" | {sum(ro)} | {200 * sum(ro) / (200 * len(ro)):.1f} | "
                     f"{sum(r['wlt'][1] for r in rr)} |")
        L.append("")
    concl = ROOT / "docs/self_correction_conclusions.md"
    if concl.exists():
        L += [concl.read_text().rstrip(), ""]
    (ROOT / "docs/self_correction.md").write_text("\n".join(L))
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["collect", "judge", "train", "noise", "report"])
    ap.add_argument("--config", default=str(ROOT / "configs/self_correct.yaml"))
    ap.add_argument("--variants", nargs="+")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.cmd == "collect":
        cmd_collect(cfg)
    elif a.cmd == "judge":
        cmd_judge(cfg)
    elif a.cmd == "noise":
        cmd_noise(cfg)
    elif a.cmd == "train":
        cmd_train(cfg, a.variants or list(cfg["train"]["variants"]))
    else:
        cmd_report(cfg)


if __name__ == "__main__":
    main()
