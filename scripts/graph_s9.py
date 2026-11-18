"""Part 4, series 9 of docs/plan_part4.md: the graph net (Gc) plays 5x5 by AlphaZero (run g5, configs/g5.yaml).

    python scripts/graph_s9.py probe        # 5x5 parity probe of every evaluated accepted g5 generation (after_eval)
    python scripts/graph_s9.py report       # docs/graph_s9.md, docs/graph_s9.png: g5 by round and by hours next to Dot
    python scripts/graph_s9.py match        # 3 g5 picks vs Dot gen 54: MCTS 200 both, raw both, equal time per move
    python scripts/graph_s9.py recheck      # gens 45-60 again on the ladder, 300 games per cell (runs/g5/eval300)
    python scripts/graph_s9.py bench --workers 6 --games 192   # self-play evals/s with N parallel workers

Training is scripts/az_train.py, the ladder scripts/az_eval.py (it calls probe and report after each evaluation).
Dot = az5 (gens 1-31) then az5b (its gens continue as rounds 32-...): one run in two stages, round = generation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np

from _common import ROOT, load_config

from az_eval import anchors_for, fit_one

CFG = "configs/g5.yaml"


def read_jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def hours(rc) -> dict:
    """{gen: hours of training up to the end of gen} from log.jsonl (self-play + train + gate seconds)."""
    secs = {}
    for e in read_jsonl(ROOT / rc["run_dir"] / "log.jsonl"):
        if e.get("phase") in ("selfplay", "train", "gate") and "seconds" in e:
            secs[e["gen"]] = secs.get(e["gen"], 0.0) + e["seconds"]
    out, t = {}, 0.0
    for g in sorted(secs):
        t += secs[g]
        out[g] = t / 3600
    return out


def win_share(r):
    return (r[0] + r[2] / 2) / sum(r)


def rows_of(rc, offset, parity, pkey, opps_elo, anchors):
    """One row per evaluated accepted generation: round, gen, hours, Elo (raw / mcts on opps_elo), mcts vs hard,
    parity (policy %, V child %, MCTS %)."""
    run = ROOT / rc["run_dir"]
    st = json.loads((run / "state.json").read_text()) if (run / "state.json").exists() else {"history": []}
    hrs = hours(rc)
    out = []
    for h in st["history"]:
        p = run / "eval" / f"gen_{h['gen']:04d}.json"
        if not (h["accepted"] and p.exists()):
            continue
        e = json.loads(p.read_text())
        pr = parity.get(pkey(h["gen"]))
        out.append({"round": offset + h["gen"], "gen": h["gen"], "hours": hrs.get(h["gen"]),
                    "raw_elo": fit_one({o: e["raw"][o] for o in opps_elo if o in e["raw"]}, anchors),
                    "mcts_elo": fit_one({o: e["mcts"][o] for o in opps_elo if o in e["mcts"]}, anchors),
                    "hard": win_share(e["mcts"]["hard"]), "raw_hard": win_share(e["raw"]["hard"]),
                    "eval": e,
                    "policy": None if pr is None else 100 * np.mean(pr["policy"]),
                    "v_child": None if pr is None else pr["v_child"],
                    "p_mcts": None if pr is None else 100 * np.mean(pr["mcts"])})
    return out


def dot_rows(cfg, anchors, opps):
    """Dot = az5 then az5b: rounds and hours continue from one stage to the next."""
    parity = json.loads((ROOT / "runs/parity_probe/curves.json").read_text())
    rows, off, h0 = [], 0, 0.0
    for path in cfg["series9"]["dot"]["configs"]:
        rc = load_config(ROOT / path)
        name = rc["run_dir"].split("/")[-1]
        for x in rows_of(rc, off, parity, lambda g, n=name: f"{n}/{g}", opps, anchors):
            x["hours"] = (x["hours"] or 0.0) + h0
            x["run"] = name
            rows.append(x)
        hs = hours(rc)
        h0 += max(hs.values()) if hs else 0.0
        off += json.loads((ROOT / rc["run_dir"] / "state.json").read_text())["gen"]
    return rows


def g5_rows(cfg, anchors, opps):
    p = ROOT / cfg["run_dir"] / "parity.json"
    parity = json.loads(p.read_text()) if p.exists() else {}
    return rows_of(cfg, 0, parity, str, opps, anchors)


def first_round(rows, key, thr):
    return next((r["round"] for r in rows if r[key] is not None and r[key] >= thr), None)


def at_hours(rows, h):
    """The last evaluated accepted generation finished within h hours of training."""
    got = [r for r in rows if r["hours"] is not None and r["hours"] <= h]
    return got[-1] if got else None


def best_gen(rows):
    """Best g5: the highest MCTS Elo (Dot's opponent set); a tie goes to the later generation."""
    return max(rows, key=lambda r: (r["mcts_elo"], r["gen"])) if rows else None


def with_recheck(cfg, rows, anchors, opps):
    """Rows with the Elo of the 300-games re-evaluation (runs/g5/eval300) where it exists: the most precise
    estimate of each generation (second amendment)."""
    d = ROOT / cfg["run_dir"] / "eval300"
    out = []
    for r in rows:
        p = d / f"gen_{r['gen']:04d}.json"
        if p.exists():
            e = json.loads(p.read_text())
            r = dict(r, raw_elo=fit_one({o: e["raw"][o] for o in opps}, anchors),
                     mcts_elo=fit_one({o: e["mcts"][o] for o in opps}, anchors), games=e["games"])
        out.append(r)
    return out


def picks(cfg, anchors, opps):
    """The g5 generations of the final matches (second amendment): {label: gen}, in order; one generation that is
    several picks plays once (labels joined)."""
    rows = with_recheck(cfg, g5_rows(cfg, anchors, opps), anchors, opps)
    st = json.loads((ROOT / cfg["run_dir"] / "state.json").read_text())
    got = {"best MCTS Elo": best_gen(rows)["gen"],
           "best Elo without thinking": max(rows, key=lambda r: (r["raw_elo"], r["gen"]))["gen"],
           "last accepted": max(h["gen"] for h in st["history"] if h["accepted"])}
    by = {}
    for k, g in got.items():
        by.setdefault(g, []).append(k)
    return {" + ".join(v): g for g, v in by.items()}


def elo_interval(res, anchors, n=400, seed=0):
    """95 % parametric bootstrap interval of fit_one: every cell resampled binomially (ties kept)."""
    rng = np.random.default_rng(seed)
    xs = []
    for _ in range(n):
        b = {}
        for o, (w, l_, t) in res.items():
            k = w + l_
            w2 = int(rng.binomial(k, w / k)) if k else 0
            b[o] = [w2, k - w2, t]
        xs.append(fit_one(b, anchors))
    return float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))


# --- probe ------------------------------------------------------------------------

def cmd_probe(cfg):
    import torch

    import parity_probe as pp
    import strategy_probe as sp
    from agents.az_net import Evaluator, make_net

    pc = load_config(ROOT / cfg["series9"]["parity"])
    c = pp.merged(pc)
    positions, boards = sp.load_positions(c)
    run = ROOT / cfg["run_dir"]
    path = run / "parity.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    st = json.loads((run / "state.json").read_text())
    for h in st["history"]:
        g = h["gen"]
        if not h["accepted"] or str(g) in res or not (run / "eval" / f"gen_{g:04d}.json").exists():
            continue
        t0 = time.time()
        net = make_net(cfg)
        net.load_state_dict(torch.load(run / f"gen_{g:04d}.pt", map_location="cpu"))
        r = pp.gen_curves(Evaluator(net, cfg["device"]), cfg, boards, positions, c["agents"]["sims"], pc["seed"])
        res[str(g)] = r
        print(f"g5 gen {g}: policy {sum(r['policy'])}, V child {r['v_child']:.1f}%, mcts {sum(r['mcts'])} / "
              f"{len(boards)} ({time.time() - t0:.0f} s)", flush=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(res))
        os.replace(tmp, path)


# --- report -----------------------------------------------------------------------

def fmt(x, f="{:.0f}"):
    return "—" if x is None else f.format(x)


def cmd_report(cfg):
    s9 = cfg["series9"]
    anchors = anchors_for(cfg)
    dot_cfg = load_config(ROOT / s9["dot"]["configs"][0])
    opps = [o for o in cfg["eval"]["opponents"] if o in dot_cfg["eval"]["opponents"]]
    G, D = g5_rows(cfg, anchors, opps), dot_rows(cfg, anchors, opps)
    run = ROOT / cfg["run_dir"]
    st = json.loads((run / "state.json").read_text()) if (run / "state.json").exists() else {"history": [], "gen": 0}
    hist = st["history"]
    log = read_jsonl(run / "log.jsonl")
    sp_ = [e for e in log if e.get("phase") == "selfplay"]
    stop = next((e["stop"] for e in log if "stop" in e), None)
    L = ["# Part 4, series 9: the graph net plays 5x5 (run g5)", "",
         f"Generated by `scripts/graph_s9.py report` ({time.strftime('%Y-%m-%d %H:%M')}). Protocol: docs/plan_part4.md "
         "(series 9), configs/g5.yaml (= configs/az5b.yaml apart from the net, from scratch, and the run plumbing). "
         "Dot = az5 (rounds 1-31) then az5b (rounds 32-119) on the same ladder; round = generation (accepted or not).",
         "",
         f"State: {st['gen']} rounds done, {sum(h['accepted'] for h in hist)} accepted"
         + (f", stopped by the rule: {stop}" if stop else ", running") + ". Training hours: "
         + fmt(G[-1]["hours"] if G else None, "{:.1f}") + " (to the last evaluated generation).", "",
         "Workers and limits: " + "; ".join(s9.get("slots", [])) + ".", ""]
    sc = cfg.get("stop") or {}
    if sc.get("min_rounds") and G:
        w = sc.get("plateau_window", 20)
        r = G[-1]["round"] if not [h for h in hist if h["accepted"] and h["gen"] > G[-1]["gen"]] else \
            min(h["gen"] for h in hist if h["accepted"] and h["gen"] > G[-1]["gen"]) - 1
        best = max(x["mcts_elo"] for x in G if x["round"] <= r)
        before = max((x["mcts_elo"] for x in G if x["round"] <= r - w), default=None)
        L += [f"Stop rule (amendment, 2026-11-17, round 16): at least {sc['min_rounds']} rounds and the best MCTS Elo "
              f"up by <= {sc.get('plateau_gain', 20)} over the last {w} rounds; 6 rejects in a row as the fallback; "
              f"hard cap {sc.get('max_rounds')} rounds. Now: judged at round {r} of {sc['min_rounds']}, best MCTS Elo "
              f"{best:.0f}, " + (f"+{best - before:.0f} over the last {w} rounds." if before is not None
                                 else f"fewer than {w} rounds of history."), ""]
    if sp_:
        L += [f"Pace: self-play {np.mean([e['seconds'] for e in sp_[-5:]]) / 60:.0f} min per round, "
              f"{np.mean([e['sims_per_s'] for e in sp_[-5:]]):.0f} evals/s (last 5 rounds).", ""]
    hr, vs = s9["hard_share"], s9["v_sign"]
    fr = {k: (first_round(G, *a), first_round(D, *a)) for k, a in
          (("hard", ("hard", hr)), ("v", ("v_child", vs)))}
    L += ["## Comparison by rounds", "",
          "| milestone | g5 round | Dot round (from the data) |", "|---|---:|---:|",
          f"| Hard >= {100 * hr:.0f}% with thinking (MCTS 200) | {fmt(fr['hard'][0])} | {fmt(fr['hard'][1])} (user: 17) |",
          f"| V sign on the parity probe >= {vs:.0f}% | {fmt(fr['v'][0])} | {fmt(fr['v'][1])} (user: 8) |", "",
          "## Comparison by time", "",
          "Elo (MCTS 200 / without thinking) of the last accepted, evaluated generation within H hours of training "
          f"(self-play + train + gate seconds). Elo on the opponents of both ladders ({', '.join(opps)}).", "",
          "| hours | g5 gen | g5 Elo MCTS / raw | g5 best MCTS Elo so far | Dot round | Dot Elo MCTS / raw | Dot best MCTS Elo so far |",
          "|---:|---:|---|---:|---:|---|---:|"]
    dh = D[-1]["hours"] if D else 0
    for h in s9["hours"]:
        g, d = at_hours(G, h), at_hours(D, h)
        gh = G[-1]["hours"] if G else 0
        bg = max([r["mcts_elo"] for r in G if r["hours"] is not None and r["hours"] <= h], default=None)
        bd = max([r["mcts_elo"] for r in D if r["hours"] is not None and r["hours"] <= h], default=None)
        L.append(f"| {h} | {fmt(g and g['gen'])}{'' if gh >= h else ' (not yet)'} | "
                 f"{fmt(g and g['mcts_elo'])} / {fmt(g and g['raw_elo'])} | {fmt(bg)} | {fmt(d and d['round'])}"
                 f"{' (Dot ended at %.1f h)' % dh if dh < h else ''} | {fmt(d and d['mcts_elo'])} / "
                 f"{fmt(d and d['raw_elo'])} | {fmt(bd)} |")
    L.append("")
    rc = s9.get("recheck")
    R = [r for r in with_recheck(cfg, G, anchors, opps) if "games" in r]
    if rc:
        L += [f"## Re-evaluation of gens {rc['gens'][0]}-{rc['gens'][1]}, {rc['games']} games per cell", "",
              "Second amendment (user, 2026-11-18, round 61): is the drop with MCTS after gen 56 noise? The same ladder, "
              f"{rc['games']} new games per cell (fresh seeds), raw and MCTS 200; Elo on the common opponents with a "
              "95 % parametric bootstrap interval; 100-games Elo for comparison. "
              + (f"{len(R)} generations done." if R else "Running."), ""]
        if R:
            L += ["| round | Elo MCTS (300) | 95 % | Elo MCTS (100) | Elo raw (300) | 95 % | Elo raw (100) | "
                  "MCTS vs hard / strong (300) |", "|---:|---:|---|---:|---:|---|---:|---|"]
            old = {r["gen"]: r for r in G}
            for r in R:
                e = json.loads((run / "eval300" / f"gen_{r['gen']:04d}.json").read_text())
                im = elo_interval({o: e["mcts"][o] for o in opps}, anchors)
                ir = elo_interval({o: e["raw"][o] for o in opps}, anchors)
                L.append(f"| {r['round']} | {r['mcts_elo']:.0f} | {im[0]:.0f}-{im[1]:.0f} | "
                         f"{old[r['gen']]['mcts_elo']:.0f} | {r['raw_elo']:.0f} | {ir[0]:.0f}-{ir[1]:.0f} | "
                         f"{old[r['gen']]['raw_elo']:.0f} | {100 * win_share(e['mcts']['hard']):.0f} / "
                         f"{100 * win_share(e['mcts']['strong']):.0f} |")
        L.append("")
    m = run / "match.json"
    if m.exists():
        mj = json.loads(m.read_text())
        if "picks" in mj:
            L += ["## Matches: g5 vs Dot gen 54", "", "Picks (second amendment; Elo with the 300-games "
                  "re-evaluation where it exists): " + "; ".join(f"{k} = gen {g}" for k, g in mj["picks"].items())
                  + f". {mj['games']} games per match, first mover alternating, the first {mj['temp_moves']} lines "
                  "at temperature 1 (opening variety), then the best move.", "",
                  "| g5 gen | pick | match | g5 | Dot | g5 wins | Dot wins | ties | g5 score % |",
                  "|---:|---|---|---|---|---:|---:|---:|---:|"]
            for k, r in mj["matches"].items():
                w, l_, t = r["result"][:3]
                L.append(f"| {r['gen']} | {r['pick']} | {r['mode']} | {r['g5']} | {r['dot']} | {w} | {l_} | {t} | "
                         f"{100 * (w + t / 2) / (w + l_ + t):.1f} |")
            if "equal_time" in mj:
                e = mj["equal_time"]
                L += ["", f"Equal time: single-game search (leaves_per_tree 8, as the app) on {e['positions']} "
                      f"positions: Dot 200 sims {1000 * e['dot_s']:.1f} ms per move; g5 {e['g5_sims']} sims "
                      f"{1000 * e['g5_s']:.1f} ms per move (g5 at 200 sims: {1000 * e['g5_200_s']:.1f} ms)."]
            L.append("")
    b = run / "bridge.json"
    if b.exists():
        bj = json.loads(b.read_text())
        L += ["## Bridge: 20 games against the real KSquares Hard", "", f"{json.dumps(bj.get('total', bj))}", ""]
    L += ["## Rounds (accepted, evaluated generations)", "",
          "Ladder: win % with MCTS 200 / without thinking, 100 games per cell. Parity probe 5x5 (1000 positions): "
          "policy argmax correct %, V child (sign of V after every move, balanced) %, MCTS 200 correct %.", "",
          "| round | hours | gating % | " + " | ".join(f"vs {o}" for o in cfg["eval"]["opponents"])
          + " | Elo MCTS / raw | policy | V sign | MCTS |", "|---:|---:|---:|" + "---|" * len(cfg["eval"]["opponents"])
          + "---|---:|---:|---:|"]
    hs = {h["gen"]: h for h in hist}
    for r in G:
        e = r["eval"]
        L.append(f"| {r['round']} | {fmt(r['hours'], '{:.1f}')} | {100 * hs[r['gen']]['score']:.0f} | "
                 + " | ".join(f"{100 * win_share(e['mcts'][o]):.0f} / {100 * win_share(e['raw'][o]):.0f}"
                              for o in cfg["eval"]["opponents"])
                 + f" | {r['mcts_elo']:.0f} / {r['raw_elo']:.0f} | {fmt(r['policy'], '{:.1f}')} | "
                   f"{fmt(r['v_child'], '{:.1f}')} | {fmt(r['p_mcts'], '{:.1f}')} |")
    L += ["", "![g5 next to Dot](graph_s9.png)", ""]
    plot(G, D, s9)
    concl = ROOT / "docs/graph_s9_notes.md"
    if concl.exists():
        L += [concl.read_text().rstrip(), ""]
    (ROOT / "docs/graph_s9.md").write_text("\n".join(L))
    print("\n".join(L[:30]))


def plot(G, D, s9):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for rows, lab, col in ((D, "Dot (az5 + az5b)", "#2a78d6"), (G, "g5 (graph net)", "#d62a6e")):
        if not rows:
            continue
        x = [r["round"] for r in rows]
        ax[0].plot(x, [r["mcts_elo"] for r in rows], marker="o", ms=3, color=col, label=f"{lab}, MCTS 200")
        ax[0].plot(x, [r["raw_elo"] for r in rows], marker="o", ms=3, ls="--", color=col, label=f"{lab}, raw")
        pr = [r for r in rows if r["v_child"] is not None]
        ax[1].plot([r["round"] for r in pr], [r["v_child"] for r in pr], marker="o", ms=3, color=col, label=f"{lab}, V sign")
        ax[1].plot([r["round"] for r in pr], [r["p_mcts"] for r in pr], marker="o", ms=3, ls=":", color=col,
                   label=f"{lab}, MCTS")
        hr = [r for r in rows if r["hours"] is not None]
        ax[2].plot([r["hours"] for r in hr], [r["mcts_elo"] for r in hr], marker="o", ms=3, color=col, label=f"{lab}, MCTS")
        ax[2].plot([r["hours"] for r in hr], [r["raw_elo"] for r in hr], marker="o", ms=3, ls="--", color=col,
                   label=f"{lab}, raw")
    ax[0].set(xlabel="round", ylabel="Elo (random = 0)", title="Ladder Elo by round")
    ax[1].axhline(s9["v_sign"], color="grey", lw=0.8, ls="--")
    ax[1].set(xlabel="round", ylabel="% (parity probe 5x5)", title="Parity probe: V sign and MCTS", ylim=(0, 100))
    ax[2].set(xlabel="hours of training", ylabel="Elo", title="Ladder Elo by hours")
    for a in ax:
        a.grid(alpha=0.25)
        a.legend(fontsize=7, frameon=False, loc="lower right")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "docs/graph_s9.png", dpi=110)


# --- matches ----------------------------------------------------------------------

def duel(evs, sims, cfg, n_games, temp_moves, seed, concurrent=200):
    """evs[0] vs evs[1]; sims[k] = 0: policy argmax (sampled from the policy in the first temp_moves lines),
    else MCTS with that many simulations. Game g: evs[g % 2] moves first. Returns [wins 0, wins 1, ties, margin 0]."""
    from agents.az_mcts import choose_move, result_for_mover
    from agents.az_play import make_mcts
    from sim.board import Board

    w, h = cfg["board"]
    rng = np.random.default_rng(seed)
    searchers = [make_mcts(e, cfg["mcts"], rng) for e in evs]
    out = [0, 0, 0, 0]
    started, live = 0, []
    while live or started < n_games:
        while len(live) < concurrent and started < n_games:
            live.append([Board(w, h), started % 2, 0])
            started += 1
        for k in (0, 1):
            grp = [g for g in live if (g[0].player == 0) == (g[1] == k) and result_for_mover(g[0]) is None]
            if not grp:
                continue
            if sims[k] == 0:
                pri, _ = evs[k]([g[0] for g in grp])
                for g, p in zip(grp, pri):
                    p = np.asarray(p, dtype=np.float64)
                    mv = int(rng.choice(len(p), p=p / p.sum())) if g[2] < temp_moves else int(np.argmax(p))
                    g[0].step(mv)
                    g[2] += 1
            else:
                for g, r in zip(grp, searchers[k].search([g[0] for g in grp], sims[k])):
                    g[0].step(choose_move(r, 1.0 if g[2] < temp_moves else 0.0, rng))
                    g[2] += 1
        nxt = []
        for g in live:
            b, first, _ = g
            if result_for_mover(b) is None:
                nxt.append(g)
                continue
            s0 = b.score[0] if first == 0 else b.score[1]
            s1 = sum(b.score) - s0
            out[3] += s0 - s1
            out[0 if s0 > s1 else 1 if s1 > s0 else 2] += 1
        live = nxt
    return out


def per_move_seconds(ev, cfg, boards, sims):
    from agents.az_mcts import choose_move
    from agents.az_play import make_mcts

    rng = np.random.default_rng(0)
    m = make_mcts(ev, cfg["mcts"], rng, leaves_per_tree=cfg["mcts"]["leaves_per_tree"])
    for b in boards[:5]:
        m.search([b], sims)                                 # warm-up (compile, cudnn)
    t0 = time.perf_counter()
    for b in boards:
        choose_move(m.search([b], sims)[0], 0.0, rng)
    return (time.perf_counter() - t0) / len(boards)


def cmd_match(cfg):
    """Second amendment: every pick (best MCTS Elo, best Elo without thinking, last accepted; Elo with the 300-games
    re-evaluation where it exists) against Dot gen 54 in the three modes. The equal-time sims are calibrated once
    (all picks are the same net size). Resumable: runs/g5/match.json."""
    import torch

    import parity_probe as pp
    import strategy_probe as sp
    from agents.az_net import Evaluator, make_net

    s9 = cfg["series9"]
    anchors = anchors_for(cfg)
    dot_cfg = load_config(ROOT / s9["dot"]["configs"][1])
    opps = [o for o in cfg["eval"]["opponents"] if o in dot_cfg["eval"]["opponents"]]
    pk = picks(cfg, anchors, opps)
    run = ROOT / cfg["run_dir"]
    dot = make_net(dot_cfg)
    dot.load_state_dict(torch.load(ROOT / s9["dot"]["match"], map_location="cpu"))
    dev = Evaluator(dot, cfg["device"])

    def g5_ev(gen):
        g5 = make_net(cfg)
        g5.load_state_dict(torch.load(run / f"gen_{gen:04d}.pt", map_location="cpu"))
        return Evaluator(g5, cfg["device"])

    mc = s9["match"]
    tm = cfg["gating"]["temp_moves"]
    path = run / "match.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    if "picks" not in res:
        res = {"games": mc["games"], "temp_moves": tm, "matches": {}}
    res["picks"] = pk
    et = s9["equal_time"]
    if "equal_time" not in res:
        ev = g5_ev(next(iter(pk.values())))
        _, boards = sp.load_positions(pp.merged(load_config(ROOT / s9["parity"])))
        bs = [boards[i] for i in np.random.default_rng(et["seed"]).choice(len(boards), et["positions"], replace=False)]
        td = per_move_seconds(dev, cfg, bs, mc["sims"])
        tg = per_move_seconds(ev, cfg, bs, mc["sims"])
        s = max(8, int(round(mc["sims"] * td / tg)))
        ts = per_move_seconds(ev, cfg, bs, s)
        s = max(8, int(round(s * td / ts)))                 # one correction step
        res["equal_time"] = {"positions": len(bs), "dot_s": td, "g5_200_s": tg, "g5_sims": s,
                             "g5_s": per_move_seconds(ev, cfg, bs, s)}
        print(res["equal_time"], flush=True)
        path.write_text(json.dumps(res, indent=1))
    plan = {"MCTS 200 both": (mc["sims"], mc["sims"]), "without thinking both": (0, 0),
            "equal time per move": (res["equal_time"]["g5_sims"], mc["sims"])}
    for label, gen in pk.items():
        ev = None
        for k, (a, b) in plan.items():
            key = f"gen {gen}: {k}"
            if key in res["matches"]:
                continue
            ev = ev or g5_ev(gen)
            t0 = time.time()
            r = duel([ev, dev], (a, b), cfg, mc["games"], tm, mc["seed"])
            res["matches"][key] = {"gen": gen, "pick": label, "mode": k, "g5": f"{a} sims" if a else "raw",
                                   "dot": f"{b} sims" if b else "raw", "result": r, "seconds": time.time() - t0}
            print(key, r, flush=True)
            path.write_text(json.dumps(res, indent=1))
    path.write_text(json.dumps(res, indent=1))


# --- recheck ----------------------------------------------------------------------

def cmd_recheck(cfg):
    """Second amendment: accepted generations in series9.recheck.gens [lo, hi] once more on the full ladder with
    recheck.games games per cell (raw and MCTS), fresh seeds [seed, gen, 7, games]; alongside the training
    (recheck.processes opponent processes). Resumable: one file per generation."""
    import torch

    import az_eval
    from agents.az_net import Evaluator, make_net

    torch.set_num_threads(2)
    rc = cfg["series9"]["recheck"]
    lo, hi = rc["gens"]
    run = ROOT / cfg["run_dir"]
    out_dir = run / "eval300"
    out_dir.mkdir(exist_ok=True)
    c = dict(cfg, eval=dict(cfg["eval"], games=rc["games"]))
    st = json.loads((run / "state.json").read_text())
    todo = [h["gen"] for h in st["history"] if h["accepted"] and lo <= h["gen"] <= hi
            and not (out_dir / f"gen_{h['gen']:04d}.json").exists()]
    ctx = get_context("spawn")
    with ctx.Pool(rc["processes"], initializer=az_eval._oinit,
                  initargs=(cfg["eval"].get("params") or {}, cfg["seed"] + 300)) as pool:
        for gen in todo:
            net = make_net(cfg)
            net.load_state_dict(torch.load(run / f"gen_{gen:04d}.pt", map_location="cpu"))
            ev = Evaluator(net, cfg["device"])
            rng = np.random.default_rng([cfg["seed"], gen, 7, rc["games"]])
            out = {"gen": gen, "games": rc["games"]}
            for mode in ("raw", "mcts"):
                t0 = time.time()
                out[mode] = az_eval.play(ev, mode, cfg["eval"]["opponents"], c, pool, rng)
                out[f"{mode}_seconds"] = time.time() - t0
                print(f"[{time.strftime('%H:%M:%S')}] gen {gen} {mode} ({rc['games']} games): "
                      + ", ".join(f"{o} {r[0]}-{r[1]}" for o, r in out[mode].items())
                      + f" ({out[f'{mode}_seconds']:.0f} s)", flush=True)
            p = out_dir / f"gen_{gen:04d}.json"
            p.with_suffix(".tmp").write_text(json.dumps(out, indent=1))
            os.replace(p.with_suffix(".tmp"), p)
            cmd_report(cfg)


# --- bench ------------------------------------------------------------------------

def _bench_one(args):
    import torch

    from agents.az_net import Evaluator, make_net
    from agents.az_play import selfplay

    cfg, n, seed = args
    torch.set_num_threads(1)
    torch.manual_seed(0)
    ev = Evaluator(make_net(cfg), cfg["device"])
    selfplay(ev, cfg, 16, 16, seed + 1000)                 # warm-up (compile)
    t0 = time.time()
    _, st = selfplay(ev, cfg, n, cfg["selfplay"]["games_per_worker"], seed)
    return st["evals"], time.time() - t0, st["games"]


def cmd_bench(cfg, workers, games):
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        t0 = time.time()
        res = pool.map(_bench_one, [(cfg, games, i) for i in range(workers)])
    ev = sum(r[0] for r in res)
    span = max(r[1] for r in res)
    print(json.dumps({"workers": workers, "games_per_worker": games, "evals": ev, "seconds": span,
                      "evals_per_s": ev / span, "wall": time.time() - t0, "evals_per_game": ev / (workers * games)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "report", "match", "recheck", "bench"])
    ap.add_argument("--config", default=CFG)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--games", type=int, default=192)
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    if a.cmd == "bench":
        cmd_bench(cfg, a.workers, a.games)
    else:
        {"probe": cmd_probe, "report": cmd_report, "match": cmd_match, "recheck": cmd_recheck}[a.cmd](cfg)


if __name__ == "__main__":
    main()
