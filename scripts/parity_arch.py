"""Part 3, experiment (b): architecture arms with a middle-game teacher.

    python scripts/parity_arch.py mid        # 10x7 middle-game positions from dot2c self-play
    python scripts/parity_arch.py teacher    # MCTS-2000 of Dot-2 on them: visits, move, root value
    python scripts/parity_arch.py check      # the same teacher on late judge positions: its accuracy
    python scripts/parity_arch.py finetune   # b1 / b2 from dot2c gen 16 -> runs/parity_arch/{arm}_ft.pt
    python scripts/parity_arch.py midprobe   # raw policy vs the teacher on the validation positions
    python scripts/parity_arch.py report     # the part (b) section of docs/parity_hand.md

configs/parity_arch.yaml, plan in docs/plan_part3.md. Middle game = 10x7
positions with 20-40 safe lines left, taken from dot2c gen 16 self-play
(its settings: sims 100-300, Dirichlet noise, temperature for 10 lines);
training and validation positions come from different games. The teacher
is Dot-2's own MCTS at 2000 simulations (no noise). After ``finetune`` the
arms play the self-play rounds with scripts/az_train.py
(configs/parity_b1.yaml, configs/parity_b2.yaml), and are measured with
``scripts/parity_hand.py probe|elo|routs|bridge --config configs/parity_arch.yaml``.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context

import numpy as np

from _common import ROOT, load_config

import parity_hand
import transfer


def board_from_state(w, h, lines, sides, owner, diff, first):
    """A sim Board from the stored state (owner relative to the mover)."""
    from sim.board import Board

    b = Board(w, h)
    b.lines[:] = bytes(lines)
    b.sides[:] = bytes(sides)
    b.n_drawn = int(lines.sum())
    me = 0 if first else 1
    b.player = me
    b.owner = [me if o == 1 else 1 - me if o == -1 else -1 for o in owner]
    b.score = [sum(o == 0 for o in b.owner), sum(o == 1 for o in b.owner)]
    assert b.score[me] - b.score[1 - me] == diff
    return b


def load_net(rc_path, ck, device):
    import torch

    from agents.az_net import make_net

    rc = load_config(ROOT / rc_path)
    net = make_net(rc)
    net.load_state_dict(torch.load(ROOT / ck, map_location="cpu"))
    return net, rc


# --- middle-game positions ----------------------------------------------------------

_W = {}


def _winit(cfg):
    import torch

    from agents.az_net import Evaluator

    torch.set_num_threads(1)
    tn = cfg["teacher_net"]
    net, rc = load_net(tn["config"], tn["checkpoint"], cfg["device"])
    _W.update(cfg=cfg, rc=rc, ev=Evaluator(net, cfg["device"]))


def _selfplay(args):
    from agents.az_play import selfplay

    n, seed = args
    rc = _W["rc"]
    samples, _ = selfplay(_W["ev"], rc, n, n, seed, size=(10, 7))
    return samples


def cmd_mid(cfg):
    from agents.az_net import Codec, N_PLANES_3

    mc = cfg["mid"]
    out = ROOT / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    tasks = [(len(part), int(rng.integers(2**31))) for part in np.array_split(np.arange(mc["games"]), mc["workers"])]
    with get_context("spawn").Pool(mc["workers"], initializer=_winit, initargs=(cfg,)) as pool:
        parts = pool.map(_selfplay, tasks)
    s = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    drawn = s["lines"].sum(1).astype(int)
    game = np.cumsum(np.r_[1, (np.diff(drawn) <= 0).astype(int)]) - 1  # drawn lines grow within a game
    game += np.repeat(np.arange(len(parts)), [len(p["z"]) for p in parts]) * 10**6
    _, game = np.unique(game, return_inverse=True)
    codec = Codec(10, 7, N_PLANES_3)
    safe = codec.chain_hint(s["lines"], s["sides"])[3]
    keep = (safe >= mc["safe"][0]) & (safe <= mc["safe"][1])
    idx = np.flatnonzero(keep)
    _, first = np.unique(s["lines"][idx], axis=0, return_index=True)  # distinct line sets
    idx = np.sort(idx[first])
    games = np.unique(game[idx])
    rng.shuffle(games)
    val_games, cnt = set(), 0
    for g in games:  # whole games to validation until n_val positions
        if cnt >= mc["n_val"]:
            break
        val_games.add(int(g))
        cnt += int((game[idx] == g).sum())
    is_val = np.isin(game[idx], list(val_games))
    tr, va = idx[~is_val], idx[is_val]
    tr = rng.choice(tr, min(mc["n"], len(tr)), replace=False)
    va = va[:mc["n_val"]]
    sel = np.r_[tr, va]
    np.savez(out / "mid_states.npz", **{k: s[k][sel] for k in ("lines", "sides", "owner", "diff", "first")},
             game=game[sel], safe=safe[sel], val=np.r_[np.zeros(len(tr), bool), np.ones(len(va), bool)])
    print(f"{len(parts)} workers, {game.max() + 1} games, {len(s['z'])} positions, {keep.sum()} with "
          f"{mc['safe'][0]}-{mc['safe'][1]} safe lines ({len(idx)} distinct) -> {len(tr)} train, {len(va)} "
          f"validation ({time.time() - t0:.0f} s)", flush=True)


# --- the teacher -------------------------------------------------------------------

def _teach(args):
    from agents.az_mcts import visit_policy
    from agents.az_play import make_mcts

    boards, seed = args
    tc = _W["cfg"]["teacher"]
    rng = np.random.default_rng(seed)
    mcts = make_mcts(_W["ev"], _W["rc"]["mcts"], rng, leaves_per_tree=tc["leaves_per_tree"])
    out = []
    for i in range(0, len(boards), tc["trees"]):
        roots = mcts.search(boards[i:i + tc["trees"]], tc["sims"])
        for r, b in zip(roots, boards[i:i + tc["trees"]]):
            pi = visit_policy(r, b.geo.n_lines)
            q = float(r.W.sum() / max(r.N.sum(), 1)) if len(r.moves) > 1 else float(r.value)
            out.append((pi, int(np.argmax(pi)), q))
        del roots
    return out


def run_teacher(cfg, boards):
    tc = cfg["teacher"]
    rng = np.random.default_rng([cfg["seed"], 2])
    chunks = [list(c) for c in np.array_split(np.array(boards, dtype=object), tc["workers"] * 4)]
    t0 = time.time()
    with get_context("spawn").Pool(tc["workers"], initializer=_winit, initargs=(cfg,)) as pool:
        res = [r for part in pool.map(_teach, [(c, int(rng.integers(2**31))) for c in chunks]) for r in part]
    print(f"teacher: {len(boards)} positions x {tc['sims']} sims, {time.time() - t0:.0f} s", flush=True)
    return res


def mid_boards(cfg):
    d = np.load(ROOT / cfg["out"] / "mid_states.npz")
    boards = [board_from_state(10, 7, *(d[k][i] for k in ("lines", "sides", "owner", "diff", "first")))
              for i in range(len(d["val"]))]
    return d, boards


def cmd_teacher(cfg):
    d, boards = mid_boards(cfg)
    res = run_teacher(cfg, boards)
    np.savez(ROOT / cfg["out"] / "mid_teacher.npz", pi=np.array([r[0] for r in res], dtype=np.float32),
             move=np.array([r[1] for r in res]), q=np.array([r[2] for r in res], dtype=np.float32))


def check_sets(cfg):
    """Late 10x7 positions the judge solved: the held-out probe set and the
    validation split of part 3 (a)'s judge set."""
    pcfg = load_config(ROOT / cfg["finetune"]["parity_config"])
    sets = {}
    for path in cfg["teacher"]["check"]:
        d, boards = transfer.load_boards(path)
        sets[path] = (boards, [set(p["correct"]) for p in d["positions"]])
    _, va = parity_hand.split_data(pcfg, "10x7")
    from sim.board import Board

    boards = []
    for p in va:
        b = Board(10, 7)
        for ln in p["moves"]:
            b.step(ln)
        boards.append(b)
    sets["judge 10x7 validation (part 3 a)"] = (boards, [set(p["correct"]) for p in va])
    return sets


def cmd_check(cfg):
    from parity_probe import wilson

    out = {}
    for name, (boards, correct) in check_sets(cfg).items():
        res = run_teacher(cfg, boards)
        ok = [int(r[1] in c) for r, c in zip(res, correct)]
        k, n = sum(ok), len(ok)
        out[name] = {"n": n, "correct": k, "pct": 100 * k / n, "ci": wilson(k, n)}
        print(f"{name}: MCTS-{cfg['teacher']['sims']} correct {k}/{n} = {100 * k / n:.1f}%", flush=True)
    (ROOT / cfg["out"] / "teacher_check.json").write_text(json.dumps(out, indent=1))


# --- arms --------------------------------------------------------------------------

def init_arm(cfg, arm):
    """The arm's net with dot2c gen 16 loaded so that it plays as dot2c."""
    import torch

    from agents.az_net import make_net, widen_from

    rc = load_config(ROOT / cfg["finetune"]["arms"][arm]["config"])
    net = make_net(rc)
    tn = cfg["teacher_net"]
    widen_from(net, torch.load(ROOT / tn["checkpoint"], map_location="cpu"))
    return net, rc


def cmd_finetune(cfg, arms):
    import torch
    import torch.nn.functional as F

    from agents.az_net import Codec
    from az_train import Buffer

    fc = cfg["finetune"]
    pcfg = load_config(ROOT / fc["parity_config"])
    ptc = pcfg["train"]
    dev = cfg["device"]
    out = ROOT / cfg["out"]
    md = np.load(out / "mid_states.npz")
    mt = np.load(out / "mid_teacher.npz")
    tr = ~md["val"]
    judge = {b: parity_hand.split_data(pcfg, b) for b in pcfg["data"]["boards"]}
    bufs = {}
    for bname, path in ptc["selfplay_buffers"].items():
        c = Codec(*transfer.wh_of(bname))
        bufs[bname] = Buffer(1, c.n_lines, c.n_boxes)
        d = np.load(ROOT / path)
        bufs[bname].a = {k: d[k] for k in bufs[bname].a}
        bufs[bname].n = int(d["n"])
    bw = np.array([ptc["selfplay_weights"][b] for b in bufs], dtype=float)
    bw /= bw.sum()

    for arm in arms:
        torch.manual_seed(cfg["seed"])
        rng = np.random.default_rng([cfg["seed"], len(arm), 5])
        net, rc = init_arm(cfg, arm)
        planes = net.planes
        codecs = {b: Codec(*transfer.wh_of(b), planes) for b in ("5x5", "7x7", "10x7")}
        par = {}
        for bname, (ptr, pva) in judge.items():
            par[bname] = []
            for ps in (ptr, pva):
                c, _, mask = parity_hand.parity_arrays(bname, ps)
                bs = [_replay(c.w, c.h, p["moves"]) for p in ps]
                par[bname].append((codecs[bname], codecs[bname].encode_boards(bs).reshape(len(bs), planes, -1), mask))
        c10 = codecs["10x7"]
        xm = c10.encode(md["lines"], md["sides"], md["owner"], md["diff"].astype(np.float32),
                        md["first"].astype(np.float32)).reshape(len(md["val"]), planes, -1)
        mid_tr = (xm[tr], mt["pi"][tr], mt["q"][tr])
        mid_va = (xm[~tr], mt["move"][~tr])

        net.to(dev).eval()  # BatchNorm statistics stay as they are
        fast = [p for n, p in net.named_parameters() if n.split(".")[0] in parity_hand.HEAD + ("stem",)]
        slow = [p for n, p in net.named_parameters() if n.split(".")[0] not in parity_hand.HEAD + ("stem",)]
        opt = torch.optim.Adam([{"params": fast, "lr": fc["lr"]}, {"params": slow, "lr": fc["lr"] * fc["body_lr_scale"]}])

        def fwd(x, codec):
            xt = torch.from_numpy(x.reshape(len(x), -1, codec.rows, codec.cols)).to(dev)
            logits, v, sc = net(xt)
            legal = xt[:, 1].flatten(1)[:, net.line_cells(codec.rows, codec.cols)] > 0
            return F.log_softmax(logits.masked_fill(~legal, -1e9), 1), v, sc

        def validate():
            r = {}
            with torch.no_grad():
                for bname, (_, (codec, x, mask)) in par.items():
                    am = fwd(x, codec)[0].argmax(1).cpu().numpy()
                    r[bname] = 100 * float(np.mean(mask[np.arange(len(am)), am]))
                am = fwd(mid_va[0], c10)[0].argmax(1).cpu().numpy()
                r["mid"] = 100 * float(np.mean(am == mid_va[1]))
            return r

        log = [{"step": 0, "val": validate()}]
        print(f"{arm}: step 0 validation {log[-1]['val']}", flush=True)
        acc = dict.fromkeys(("parity", "mid_policy", "mid_value", "policy", "value"), 0.0)
        sizes = list(par)
        sw = rc["train"].get("score_weight", 0.0)
        t0 = time.time()
        for step in range(1, fc["steps"] + 1):
            bname = sizes[step % len(sizes)]
            codec, X, M = par[bname][0]
            i = rng.integers(0, len(X), fc["parity_batch"])
            x, m = parity_hand.augment(rng, codec, X[i], M[i])
            lp = fwd(x, codec)[0]
            lpar = -torch.logsumexp(lp.masked_fill(~torch.from_numpy(m).to(dev), -1e9), 1).mean()
            i = rng.integers(0, len(mid_tr[0]), fc["mid_batch"])
            x, pi = parity_hand.augment(rng, c10, mid_tr[0][i], mid_tr[1][i])
            lp, v, _ = fwd(x, c10)
            lmp = -(torch.from_numpy(pi).to(dev) * lp).sum(1).mean()
            lmv = F.mse_loss(v, torch.from_numpy(mid_tr[2][i]).to(dev))
            j = list(bufs)[int(rng.choice(len(bufs), p=bw))]
            s = bufs[j].sample(rng, fc["selfplay_batch"])
            c = codecs[j]
            xs = c.encode(s["lines"], s["sides"], s["owner"], s["diff"].astype(np.float32), s["first"])
            xs, pi = parity_hand.augment(rng, c, xs.reshape(len(xs), planes, -1), s["pi"].astype(np.float32))
            lp, v, sc = fwd(xs, c)
            lpol = -(torch.from_numpy(pi).to(dev) * lp).sum(1).mean()
            lv = F.mse_loss(v, torch.from_numpy(s["z"].astype(np.float32)).to(dev))
            mg = torch.from_numpy(s["margin"].astype(np.float32) / c.n_boxes).to(dev)
            loss = lpar + lmp + lmv + lpol + lv + sw * F.mse_loss(sc, mg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for k, t in zip(acc, (lpar, lmp, lmv, lpol, lv)):
                acc[k] += t.item()
            if step % fc["log_every"] == 0:
                rec = {"step": step, **{k: a / fc["log_every"] for k, a in acc.items()}, "val": validate(),
                       "seconds": time.time() - t0}
                log.append(rec)
                acc = dict.fromkeys(acc, 0.0)
                print(f"{arm}: step {step} " + " ".join(f"{k} {rec[k]:.3f}" for k in acc)
                      + f" validation {rec['val']} ({rec['seconds']:.0f} s)", flush=True)
        net.cpu()
        torch.save(net.state_dict(), out / f"{arm}_ft.pt")
        (out / f"{arm}_ft_train.json").write_text(json.dumps(log, indent=1))


def _replay(w, h, moves):
    from sim.board import Board

    b = Board(w, h)
    for ln in moves:
        b.step(ln)
    return b


# --- middle-game probe -------------------------------------------------------------

def cmd_midprobe(cfg, names):
    """Raw policy argmax = the teacher's move on the validation positions
    (other games than training); and the sign of V vs the teacher's."""
    import torch

    from agents.az_net import Codec

    out = ROOT / cfg["out"]
    md, mt = np.load(out / "mid_states.npz"), np.load(out / "mid_teacher.npz")
    va = md["val"]
    path, res = parity_hand.results(cfg)
    for name in names:
        k = f"midprobe/{name}"
        nc = cfg["nets"][name]
        if not (ROOT / nc["checkpoint"]).exists():
            continue
        net, _ = load_net(nc["config"], nc["checkpoint"], cfg["device"])
        net.to(cfg["device"]).eval()
        c = Codec(10, 7, net.planes)
        x = torch.from_numpy(c.encode(md["lines"][va], md["sides"][va], md["owner"][va],
                                      md["diff"][va].astype(np.float32), md["first"][va].astype(np.float32)))
        with torch.no_grad():
            logits, v, _ = net(x.to(cfg["device"]))
            legal = x[:, 1].flatten(1)[:, net.line_cells(c.rows, c.cols).cpu()] > 0
            am = logits.cpu().masked_fill(~legal, -1e9).argmax(1).numpy()
            v = v.cpu().numpy()
        q = mt["q"][va]
        res[k] = {"n": int(va.sum()), "agree": 100 * float(np.mean(am == mt["move"][va])),
                  "teacher_mass": 100 * float(np.mean(mt["pi"][va][np.arange(len(am)), am])),
                  "v_sign": 100 * float(np.mean(np.sign(v) == np.sign(q))), "v_mae": float(np.mean(np.abs(v - q)))}
        transfer.save(path, res)
        print(f"{k}: {res[k]}", flush=True)


# --- report ------------------------------------------------------------------------

def cmd_report(cfg):
    from parity_probe import wilson

    out = ROOT / cfg["out"]
    _, res = parity_hand.results(cfg)
    names = list(cfg["nets"])
    mc, tc, fc = cfg["mid"], cfg["teacher"], cfg["finetune"]
    md = np.load(out / "mid_states.npz")
    chk = json.loads((out / "teacher_check.json").read_text()) if (out / "teacher_check.json").exists() else {}

    def ci(x, n):
        lo, hi = wilson(round(x * n / 100), n)
        return f"{x:.1f} [{lo:.0f}-{hi:.0f}]"

    L = ["# Part 3 (b): architecture arms with a middle-game teacher", "",
         f"Generated by `scripts/parity_arch.py report` ({time.strftime('%Y-%m-%d %H:%M')}), "
         "configs/parity_arch.yaml. **base** = Dot-2 (dot2c gen 16, its numbers from part (a)). "
         "**b1** = pooling in all six blocks, 96 channels (configs/parity_b1.yaml); **b2** = chain hint planes "
         "on the input: per box chain / loop length, long chains formed, long chain rule parity relative to "
         "the mover, safe lines left (configs/parity_b2.yaml). Both start from dot2c gen 16 "
         "(agents/az_net.py widen_from: same outputs as dot2c), are fine-tuned "
         f"({fc['steps']} steps; judge positions of part (a) + middle-game positions with the teacher's visits "
         "and value + dot2c's replay buffers; head and stem at lr "
         f"{fc['lr']}, the rest at {fc['body_lr_scale']}x) -> **_ft**, then play "
         f"{cfg['generations']} self-play rounds as dot2c phase 2 (scripts/az_train.py, gating 55%) -> the last "
         "accepted net.", "",
         "## The middle-game teacher", "",
         f"{int((~md['val']).sum())} training + {int(md['val'].sum())} validation positions of 10x7 with "
         f"{mc['safe'][0]}-{mc['safe'][1]} safe lines left (median {int(np.median(md['safe']))}), from "
         f"{mc['games']} self-play games of dot2c gen 16; validation from other games. Teacher: Dot-2's MCTS at "
         f"{tc['sims']} simulations (its move = most visits, its value = root mean value).", "",
         "Teacher accuracy where the judge knows the answer (late 10x7 positions, 1-2 safe lines):", "",
         "| set | n | MCTS-" + str(tc["sims"]) + " correct % |", "|---|---:|---|"]
    for name, r in chk.items():
        L.append(f"| {name} | {r['n']} | {ci(r['pct'], r['n'])} |")
    L += ["", "For comparison on the probe set: MCTS-200 92.4%, raw policy 67.4% (part a, base).", "",
          "## Results", "",
          "Probe: raw policy argmax correct on the held-out judge sets (5x5 1000, 10x7 304). Middle game: raw "
          "policy argmax = teacher's move on the validation positions, and the sign of V = the sign of the "
          "teacher's value. Elo: raw policy, 200 games per opponent (Bradley-Terry, fixed anchors as in "
          "part a). Routs: 200 games vs the simulator's Hard at MCTS 50, loss by 20+ boxes. Bridge: 20 games "
          "vs the real KSquares Hard, MCTS 200.", "",
          "| net | probe 5x5 % | probe 10x7 % | mid: = teacher % | mid: V sign % | Elo 5x5 | Elo 10x7 | "
          "sim vs Hard (W-L-T) | routs / 200 | bridge (W-L-T) | bridge routs |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    from az_eval import fit_anchors, fit_one, load_ladder

    ec = cfg["eval"]
    anchors = {b: (fit_anchors(*load_ladder(bc["ladder"])) if bc["ladder"] else fit_anchors())
               for b, bc in ec["elo"]["boards"].items()}
    for n in names:
        cells = []
        for b in ("5x5", "10x7"):
            r = res.get(f"probe/{n}/{b}")
            cells.append(ci(r["policy"], r["n"]) if r else "—")
        m = res.get(f"midprobe/{n}")
        cells += [f"{m['agree']:.1f}", f"{m['v_sign']:.1f}"] if m else ["—", "—"]
        for b in ("5x5", "10x7"):
            r = {o: res[f"elo/{n}/{b}/{o}"] for o in ec["elo"]["opponents"] if f"elo/{n}/{b}/{o}" in res}
            cells.append(f"{fit_one(r, anchors[b]):.0f}" if len(r) == len(ec["elo"]["opponents"]) else "—")
        r, br = res.get(f"routs/{n}"), res.get(f"bridge/{n}")
        if r:
            cells += ["-".join(map(str, r["wlt"])), str(sum(x <= -ec["routs"]["margin"] for x in r["margins"]))]
        else:
            cells += ["—", "—"]
        if br:
            cells += [f"{br['wins']}-{br['losses']}-{br['ties']}",
                      str(sum(g["net"] - g["hard"] <= -ec["routs"]["margin"] for g in br["games"]))]
        else:
            cells += ["—", "—"]
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    L += ["", "Self-play rounds (gating vs the current best, 100 games, MCTS 200):", ""]
    for arm in fc["arms"]:
        st = ROOT / f"runs/parity_{arm}/state.json"
        if st.exists():
            h = json.loads(st.read_text())["history"]
            L.append(f"- {arm}: " + ", ".join(f"g{x['gen']} {100 * x['score']:.0f}%{'+' if x['accepted'] else ''}"
                                              for x in h) + " (+ = accepted)")
    concl = ROOT / "docs/parity_arch_conclusions.md"
    if concl.exists():
        L += ["", concl.read_text().rstrip()]
    section = "\n".join(L) + "\n"
    (ROOT / "docs/parity_arch.md").write_text(section)
    # docs/parity_hand.md: part (a) as generated, then this section
    ph = ROOT / "docs/parity_hand.md"
    text = ph.read_text()
    cut = text.find("\n# Part 3 (b)")
    ph.write_text((text if cut < 0 else text[:cut]).rstrip() + "\n\n" + section)
    print(section)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["mid", "teacher", "check", "finetune", "midprobe", "report"])
    ap.add_argument("--config", default=str(ROOT / "configs/parity_arch.yaml"))
    ap.add_argument("--nets", nargs="+")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.cmd == "finetune":
        cmd_finetune(cfg, a.nets or list(cfg["finetune"]["arms"]))
    elif a.cmd == "midprobe":
        cmd_midprobe(cfg, a.nets or list(cfg["nets"]))
    else:
        {"mid": cmd_mid, "teacher": cmd_teacher, "check": cmd_check, "report": cmd_report}[a.cmd](cfg)


if __name__ == "__main__":
    main()
