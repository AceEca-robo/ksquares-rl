"""Step (b): Dot-2 trained on 5x5 only, on 10x7 and 7x7 without training.

    python scripts/transfer.py positions   # parity positions per size (judge with a time limit)
    python scripts/transfer.py play        # ladder games: every net x size x mode x opponent
    python scripts/transfer.py probe       # parity probe: policy, V after the move, MCTS
    python scripts/transfer.py report      # docs/transfer.md

configs/transfer.yaml. Positions: games of ladder agents; in each, up to
``parity_per_game`` positions without a capture and with 1..``max_safe``
safe lines are offered to the judge (strong engine, exact to the end,
``time_limit`` per position); only solved positions where the move decides
the result count (as scripts/strategy_probe.py ``classify``: "parity").
Probe metrics as scripts/parity_probe.py ``gen_curves``: policy argmax
correct; V child: the sign of V after every legal move right (balanced
over winning and losing moves); MCTS: the search's move correct.
Results in runs/transfer/*.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from multiprocessing import get_context

import numpy as np

from _common import ROOT, load_config

OUT = ROOT / "runs/transfer"


def wh_of(name: str) -> tuple[int, int]:
    w, h = name.split("x")
    return int(w), int(h)


# --- positions --------------------------------------------------------------------

_G = {}


def _ginit(cfg):
    sys.setrecursionlimit(10000)
    _G["cfg"] = cfg


def _gen_game(task):
    from agents.az_mcts import result_for_mover
    from agents.baselines import safe_moves
    from agents.registry import make_agent
    from agents.strong import Engine, exact_move_values
    from sim.board import Board
    from strategy_probe import capture_available, classify

    wh, a, b, seed, first, pc = task
    agents = [make_agent(a, seed), make_agent(b, seed + 1)]
    seat = [first, 1 - first]
    board = Board(*wh)
    eng = Engine(*wh)
    moves, found, tried, solved = [], [], 0, 0
    while result_for_mover(board) is None and len(found) < pc["parity_per_game"]:
        if not capture_available(board) and 0 < len(safe_moves(board)) <= pc["max_safe"]:
            tried += 1
            if len(eng.tt) > 1_500_000:
                eng.tt.clear()
            vals = exact_move_values(eng, board, None, pc["time_limit"])
            if vals is not None:
                solved += 1
                cat, correct, _ = classify(board, vals)
                if cat == "parity":
                    found.append({"moves": list(moves), "score": list(board.score), "player": board.player,
                                  "undrawn": board.geo.n_lines - board.n_drawn,
                                  "values": {str(k): v for k, v in vals.items()}, "correct": correct,
                                  "safe": len(safe_moves(board)), "source": f"{a}-{b}"})
        ln = agents[seat[board.player]].act(board)
        board.step(ln)
        moves.append(ln)
    return found, tried, solved


def cmd_positions(cfg):
    pc = cfg["positions"]
    for name, bc in cfg["boards"].items():
        path = ROOT / bc["positions"]
        if path.exists():
            continue
        wh = wh_of(name)
        rng = random.Random(cfg["seed"])
        gpc = dict(pc, **(bc.get("gen") or {}))
        tasks = [(wh, *pc["pairs"][g % len(pc["pairs"])], rng.randrange(2**31), g % 2, gpc)
                 for g in range(pc["games"])]
        uniq, tried, solved = {}, 0, 0
        t0 = time.time()
        with get_context("spawn").Pool(pc.get("processes", cfg["processes"]), initializer=_ginit,
                                       initargs=(cfg,)) as pool:
            for i, (found, t, s) in enumerate(pool.imap_unordered(_gen_game, tasks, chunksize=2)):
                tried += t
                solved += s
                for p in found:
                    uniq.setdefault((tuple(sorted(p["moves"])), tuple(p["score"]), p["player"]), p)
                if (i + 1) % 100 == 0:
                    print(f"{name}: {i + 1}/{len(tasks)} games, {time.time() - t0:.0f} s, parity {len(uniq)}, "
                          f"judge solved {solved}/{tried}", flush=True)
        ps = list(uniq.values())
        rng.shuffle(ps)
        out = ps[:pc["n"]]
        path.write_text(json.dumps({"board": list(wh), "time_limit": gpc["time_limit"], "max_safe": gpc["max_safe"],
                                    "tried": tried,
                                    "solved": solved, "distinct": len(uniq), "positions": out}))
        print(f"{name}: {len(uniq)} distinct parity positions, judge solved {solved}/{tried}; saved {len(out)}",
              flush=True)


def load_boards(path):
    from sim.board import Board

    d = json.loads((ROOT / path).read_text())
    boards = []
    for p in d["positions"]:
        b = Board(*d["board"])
        for ln in p["moves"]:
            b.step(ln)
        assert b.player == p["player"] and list(b.score) == p["score"]
        boards.append(b)
    return d, boards


# --- nets -------------------------------------------------------------------------

def load_net(cfg, name):
    import torch

    from agents.az_net import Evaluator, make_net

    nc = cfg["nets"][name]
    rc = load_config(ROOT / nc["config"])
    if nc.get("checkpoint"):
        net = make_net(rc)
        net.load_state_dict(torch.load(ROOT / nc["checkpoint"], map_location="cpu"))
    else:
        torch.manual_seed(nc.get("init_seed", 0))
        net = make_net(rc)
    return Evaluator(net, cfg["device"]), rc


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def cmd_play(cfg):
    import torch

    from az_eval import _oinit, play

    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    pc = cfg["play"]
    for bname, bc in cfg["boards"].items():
        if "ladder" not in bc:
            continue
        with get_context("spawn").Pool(cfg["processes"], initializer=_oinit,
                                       initargs=(bc["params"], cfg["seed"])) as pool:
            for net in cfg["nets"]:
                path = OUT / f"play_{net}_{bname}.json"
                res = json.loads(path.read_text()) if path.exists() else {}
                ev, rc = load_net(cfg, net)
                opps = [o for o in pc["opponents"] if o not in (pc.get("skip") or {}).get(net, [])]
                ecfg = {"board": list(wh_of(bname)), "mcts": rc["mcts"],
                        "eval": {"games": pc["games"], "sims": pc["sims"]}}
                for mode in ("raw", "mcts"):
                    for o in opps:  # one opponent at a time: results saved as they come
                        if o in res.get(mode, {}):
                            continue
                        t0 = time.time()
                        rng = np.random.default_rng([cfg["seed"], len(o), 7])
                        r = play(ev, mode, [o], ecfg, pool, rng)[o]
                        res.setdefault(mode, {})[o] = r
                        res.setdefault("seconds", {})[f"{mode}/{o}"] = time.time() - t0
                        save(path, res)
                        print(f"[{time.strftime('%H:%M:%S')}] {net} {bname} {mode} vs {o}: {r[0]}-{r[1]}-{r[2]} "
                              f"({time.time() - t0:.0f} s)", flush=True)


def cmd_probe(cfg):
    from agents.registry import make_agent
    from parity_probe import gen_curves

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "probe.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for bname, bc in cfg["boards"].items():
        d, boards = load_boards(bc["positions"])
        positions = d["positions"]
        correct = [set(p["correct"]) for p in positions]
        res[f"chance/{bname}"] = float(np.mean([len(p["correct"]) / len(p["values"]) for p in positions]))
        for base in cfg["probe"]["baselines"]:
            k = f"{base}/{bname}"
            if k in res:
                continue
            params = (bc.get("params") or {}).get(base, {"node_budget": 23000} if base == "strong" else {})
            moves = [make_agent(base, cfg["seed"] + i, **params).act(b) for i, b in enumerate(boards)]
            res[k] = 100 * float(np.mean([m in c for m, c in zip(moves, correct)]))
            save(path, res)
            print(f"{k}: {res[k]:.1f}%", flush=True)
        for net in cfg["nets"]:
            k = f"{net}/{bname}"
            if k in res:
                continue
            t0 = time.time()
            ev, rc = load_net(cfg, net)
            r = gen_curves(ev, rc, boards, positions, cfg["probe"]["sims"], cfg["seed"])
            res[k] = {"policy": 100 * float(np.mean(r["policy"])), "mcts": 100 * float(np.mean(r["mcts"])),
                      "v_greedy": 100 * float(np.mean(r["v_greedy"])), "v_child": float(r["v_child"]),
                      "v_child_win": float(r["v_child_win"]), "v_child_lose": float(r["v_child_lose"]),
                      "n": len(boards), "policy_each": r["policy"], "mcts_each": r["mcts"],
                      "v_greedy_each": r["v_greedy"]}
            save(path, res)
            print(f"{k}: {res[k]} ({time.time() - t0:.0f} s)", flush=True)


def cmd_report(cfg):
    from az_eval import fit_anchors, fit_one, load_ladder

    from parity_probe import wilson

    probe = json.loads((OUT / "probe.json").read_text()) if (OUT / "probe.json").exists() else {}
    L = ["# Transfer without training: Dot-2 (5x5 only) on 7x7 and 10x7", "",
         "Goal of part 2: does the understanding of parity transfer from 5x5 to 10x7 without further "
         f"training? Generated by `scripts/transfer.py` ({time.strftime('%Y-%m-%d %H:%M')}), "
         "configs/transfer.yaml. Nets: **dot2** = runs/dot2a/gen_0001.pt (Dot-2 at Dot's level on 5x5, never "
         "saw another size); **untrained** = Dot-2 with random weights (what the architecture gives for free).",
         ""]
    for bname, bc in cfg["boards"].items():
        if "ladder" not in bc or not all((ROOT / p).exists() for p in bc["ladder"]):
            continue
        anchors = fit_anchors(*load_ladder(bc["ladder"]))
        opps = cfg["play"]["opponents"]
        L += [f"## Ladder on {bname}", "",
              f"Win % of the net, {cfg['play']['games']} games per cell (ties half), first mover alternating; "
              f"raw = policy argmax, mcts = {cfg['play']['sims']} simulations. Elo: Bradley-Terry against fixed "
              f"anchors from {', '.join(bc['ladder'])}: " + ", ".join(f"{a} {v:.0f}" for a, v in anchors.items())
              + ".", "",
              "| net | mode | " + " | ".join(opps) + " | Elo |", "|---|---|" + "---:|" * (len(opps) + 1)]
        for net in cfg["nets"]:
            p = OUT / f"play_{net}_{bname}.json"
            if not p.exists():
                continue
            res = json.loads(p.read_text())
            for mode in ("raw", "mcts"):
                r = res.get(mode, {})
                cells = [f"{100 * (r[o][0] + r[o][2] / 2) / sum(r[o]):.1f}" if o in r else "—" for o in opps]
                elo = fit_one({o: v for o, v in r.items() if o in anchors}, anchors) if r else float("nan")
                L.append(f"| {net} | {mode} | " + " | ".join(cells) + f" | {elo:.0f} |")
        L.append("")
    bridge = sorted(OUT.glob("bridge_*.json")) + sorted(OUT.glob("leaves1_*.json"))
    if bridge:
        L += ["## Against the real KSquares Hard (bridge)", "",
              "`scripts/az_vs_hard.py`, MCTS 200, first mover alternating, the game's srand = 11 + game.", "",
              "| net | board | leaves per round | wins-losses-ties | score % |", "|---|---|---:|---:|---:|"]
        for p in bridge:
            d = json.loads(p.read_text())
            n = d["wins"] + d["losses"] + d["ties"]
            L.append(f"| {p.stem.split('_')[1]} | {d['board'][0]}x{d['board'][1]} | {d.get('leaves', 8)} | "
                     f"{d['wins']}-{d['losses']}-"
                     f"{d['ties']} | {100 * (d['wins'] + d['ties'] / 2) / n:.0f} |")
        L.append("")
    L += ["## Parity probe", "",
          "Positions where nothing can be taken, safe lines remain and the move decides the result "
          "(long chain rule). 10x7 / 7x7: new sets from games of ladder agents with "
          f"1-{cfg['positions']['max_safe']} safe lines, exact values by the judge (strong engine to the end; "
          + ", ".join(f"{b} {(bc.get('gen') or {}).get('time_limit', cfg['positions']['time_limit'])} s"
                      for b, bc in cfg["boards"].items() if bc.get("gen"))
          + " per position; only positions it solved: with more safe lines, or the app's 1 s, it solves almost "
          "none on 10x7); 5x5: the 1000 positions of docs/parity_probe_positions.json (1-6 safe lines), "
          f"and its subset with 1-{cfg['positions']['max_safe']} safe lines. Policy: the argmax is a "
          "correct move. V after the move: the sign of the value head after every legal move is right "
          "(balanced: mean of the shares over winning and over losing moves; 50 = blind). MCTS: the move of "
          f"{cfg['probe']['sims']} simulations is correct. 95% Wilson intervals for policy and MCTS.", "",
          "| board | positions | net | policy % | V after the move % (win / lose) | MCTS % |",
          "|---|---:|---|---|---|---|"]
    def ci(x, n):
        lo, hi = wilson(round(x * n / 100), n)
        return f"{x:.1f} [{lo:.0f}-{hi:.0f}]"

    from agents.baselines import safe_moves
    for bname, bc in cfg["boards"].items():
        for net in cfg["nets"]:
            r = probe.get(f"{net}/{bname}")
            if not r:
                continue
            n = r["n"]
            L.append(f"| {bname} | {n} | {net} | {ci(r['policy'], n)} | {r['v_child']:.1f} ({r['v_child_win']:.0f} / "
                     f"{r['v_child_lose']:.0f}) | {ci(r['mcts'], n)} |")
            d, boards = load_boards(bc["positions"])
            sub = [i for i, b in enumerate(boards) if len(safe_moves(b)) <= cfg["positions"]["max_safe"]]
            if len(sub) < n:  # the 5x5 set: also its 1-2 safe line subset, comparable to the new sets
                pol = 100 * np.mean([r["policy_each"][i] for i in sub])
                mc = 100 * np.mean([r["mcts_each"][i] for i in sub])
                L.append(f"| {bname}, <= {cfg['positions']['max_safe']} safe lines | {len(sub)} | {net} | "
                         f"{ci(pol, len(sub))} | — | {ci(mc, len(sub))} |")
        base = [f"{b} {probe[f'{b}/{bname}']:.0f}%" for b in cfg["probe"]["baselines"] if f"{b}/{bname}" in probe]
        if f"chance/{bname}" in probe:
            L.append(f"| {bname} | | baselines: random move {100 * probe[f'chance/{bname}']:.0f}%, "
                     + ", ".join(base) + " | | | |")
    (ROOT / "docs/transfer.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["positions", "play", "probe", "report"])
    ap.add_argument("--config", default=str(ROOT / "configs/transfer.yaml"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"positions": cmd_positions, "play": cmd_play, "probe": cmd_probe, "report": cmd_report}[a.cmd](cfg)


if __name__ == "__main__":
    main()
