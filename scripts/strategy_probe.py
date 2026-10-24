"""Strategy probe: does the net know the long chain rule and double-dealing?

positions  Plays games between ladder agents and keeps positions whose
           result depends on the move, with the exact value of every legal
           move. The correct moves are those that reach the best attainable
           result (win > tie > loss) given the current score.
           Exact values: up to ``max_undrawn`` undrawn lines by full brute
           force (negamax over all lines, memo over line masks, no pruning
           or chain theory). Parity decisions on 5x5 fall at ~30-34 undrawn
           lines, out of reach for brute force; there the strong engine
           solves every child position to the end (no depth limit, no
           heuristic leaf; its only pruning is alpha-beta plus captures and
           chain openings that provably never lose value). The two are
           compared on every brute-forced position (``solver_check``).
           Categories:
             dd      a capture is available but every capturing move gives
                     a worse result than the best move: the mover must
                     decline (double-dealing / all-but-two, all-but-four).
             parity  no capture is available, safe lines (giving no box a
                     third side) remain, and the moves differ in result:
                     who has to open the first long chain is decided now
                     (long chain rule; a sacrifice may be the fix).
           Saved to ``positions.out`` (move list, scores, exact values).

evaluate   For every accepted generation of the configured runs: share of
           positions where the net without search (policy argmax), the net
           with MCTS (``agents.sims``, no noise) pick a correct move; plus
           strong (node budget), Hard and greedy once. Results in
           runs/strategy_probe/eval.json.

selfplay   Self-play games per generation (settings as in training). Every
           position of the loony phase (no safe line left) where a capture
           is available is solved exactly (strong
           engine, no depth limit) and the chosen move is judged. A DD move
           captures nothing although a capture is available and leaves a
           new domino (one line completing two boxes): the mover hands
           over two boxes. Per generation: positions where declining is
           required (dd) and how often the net got them right; DD moves
           played where taking was better (wrong hand-overs).

report     docs/strategy_probe.md and docs/strategy_probe.png.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import get_context

import numpy as np

from _common import ROOT, load_config

from agents.baselines import safe_moves
from agents.registry import make_agent
from sim.board import Board, geometry

OUT = ROOT / "runs/strategy_probe"

# --- brute force ------------------------------------------------------------------


class Brute:
    """Net future boxes for the side to move, by plain negamax over all lines."""

    def __init__(self, w, h):
        geo = geometry(w, h)
        self.n = geo.n_lines
        self.full = (1 << self.n) - 1
        self.bm = [sum(1 << ln for ln in bl) for bl in geo.box_lines]
        self.lb = [tuple(self.bm[b] for b in bs) for bs in geo.line_boxes]
        self.memo: dict[int, int] = {}

    def got(self, L2, ln):
        return sum(1 for m in self.lb[ln] if L2 & m == m)

    def value(self, L: int) -> int:
        if L == self.full:
            return 0
        v = self.memo.get(L)
        if v is not None:
            return v
        best = -10**9
        for ln in range(self.n):
            bit = 1 << ln
            if not L & bit:
                L2 = L | bit
                g = self.got(L2, ln)
                x = g + self.value(L2) if g else -self.value(L2)
                if x > best:
                    best = x
        self.memo[L] = best
        return best

    def move_values(self, L: int) -> dict[int, int]:
        out = {}
        for ln in range(self.n):
            bit = 1 << ln
            if not L & bit:
                L2 = L | bit
                g = self.got(L2, ln)
                out[ln] = g + self.value(L2) if g else -self.value(L2)
        return out


def mask(board) -> int:
    from agents.strong import line_mask

    return line_mask(board)


def engine_values(eng, board, node_limit):
    """Exact value of every legal move by the strong engine (None: too big)."""
    from agents.strong import exact_move_values

    return exact_move_values(eng, board, node_limit)


def classify(board, vals):
    """(category or None, correct moves, result per move)."""
    me, op = board.score[board.player], board.score[board.player ^ 1]
    res = {ln: int(np.sign(me - op + v)) for ln, v in vals.items()}
    best = max(res.values())
    if min(res.values()) == best:
        return None, [], res
    correct = sorted(ln for ln, r in res.items() if r == best)
    sides = board.sides
    geo = board.geo
    captures = [ln for ln in vals if any(sides[b] == 3 for b in geo.line_boxes[ln])]
    if captures:
        if all(res[ln] < best for ln in captures):
            return "dd", correct, res
        return None, correct, res
    if safe_moves(board):
        return "parity", correct, res
    return None, correct, res


_G = {}


def _gen_init(cfg):
    sys.setrecursionlimit(10000)
    _G["cfg"] = cfg


def _gen_game(task):
    a, b, seed, first = task
    cfg = _G["cfg"]
    w, h = cfg["board"]
    pc = cfg["positions"]
    from agents.az_mcts import result_for_mover
    agents = [make_agent(a, seed), make_agent(b, seed + 1)]
    seat = [first, 1 - first]
    board = Board(w, h)
    moves = []
    from agents.strong import Engine
    brute = Brute(w, h)
    eng = Engine(w, h)
    found = []
    checks = [0, 0]  # brute-forced positions, of which the engine agreed on every move
    n_parity = 0
    while result_for_mover(board) is None:
        undrawn = board.geo.n_lines - board.n_drawn
        rec = None
        if undrawn <= pc["max_undrawn"]:
            vals = brute.move_values(mask(board))
            ev = engine_values(eng, board, pc["node_limit"])
            checks[0] += 1
            checks[1] += ev == vals
            rec = vals, "brute force"
        elif (n_parity < pc["parity_per_game"] and not capture_available(board)
              and 0 < len(safe_moves(board)) <= pc["max_safe"]):
            vals = engine_values(eng, board, pc["node_limit"])
            if vals is not None:
                rec = vals, "engine"
        if rec:
            vals, how = rec
            cat, correct, res = classify(board, vals)
            if cat == "parity" and how == "engine":
                n_parity += 1
            if cat:
                found.append({"category": cat, "moves": list(moves), "score": list(board.score),
                              "player": board.player, "undrawn": undrawn, "values": vals,
                              "correct": correct, "source": f"{a}-{b}", "exact": how})
        ln = agents[seat[board.player]].act(board)
        board.step(ln)
        moves.append(ln)
    return found, checks


def cmd_positions(cfg):
    pc = cfg["positions"]
    rng = random.Random(cfg["seed"])
    tasks = []
    for g in range(pc["games"]):
        a, b = pc["pairs"][g % len(pc["pairs"])]
        tasks.append((a, b, rng.randrange(2**31), g % 2))
    found = []
    checks = [0, 0]
    t0 = time.time()
    with get_context("spawn").Pool(cfg["processes"], initializer=_gen_init, initargs=(cfg,)) as pool:
        for i, (f, c) in enumerate(pool.imap_unordered(_gen_game, tasks, chunksize=4)):
            found.extend(f)
            checks[0] += c[0]
            checks[1] += c[1]
            if (i + 1) % 250 == 0:
                print(f"{i + 1}/{len(tasks)} games, {time.time() - t0:.0f} s, "
                      f"dd {sum(p['category'] == 'dd' for p in found)}, "
                      f"parity {sum(p['category'] == 'parity' for p in found)}", flush=True)
    # one position per distinct state, then a balanced sample per category
    uniq = {}
    for p in found:
        uniq.setdefault((tuple(sorted(p["moves"])), tuple(p["score"]), p["player"]), p)
    out = []
    for cat in ("dd", "parity"):
        ps = [p for p in uniq.values() if p["category"] == cat]
        rng.shuffle(ps)
        out += ps[:pc["per_category"]]
    for p in out:
        p["values"] = {str(k): v for k, v in p["values"].items()}
    path = ROOT / pc["out"]
    path.write_text(json.dumps({"board": cfg["board"], "max_undrawn": pc["max_undrawn"],
                                "solver_check": {"positions": checks[0], "engine_agrees": checks[1]},
                                "positions": out}))
    print(f"{len(uniq)} distinct critical positions; saved {len(out)} to {path}; engine = brute force "
          f"on {checks[1]}/{checks[0]} brute-forced positions (all moves)")


# --- evaluation -------------------------------------------------------------------


def load_positions(cfg):
    d = json.loads((ROOT / cfg["positions"]["out"]).read_text())
    w, h = d["board"]
    boards = []
    for p in d["positions"]:
        b = Board(w, h)
        for ln in p["moves"]:
            b.step(ln)
        assert b.player == p["player"] and list(b.score) == p["score"]
        boards.append(b)
    return d["positions"], boards


def accepted(run_cfg):
    run = ROOT / run_cfg["run_dir"]
    st = json.loads((run / "state.json").read_text())
    return [h["gen"] for h in st["history"] if h["accepted"] and (run / f"gen_{h['gen']:04d}.pt").exists()]


def _agent_move(task):
    name, params, seed, moves, wh = task
    b = Board(*wh)
    for ln in moves:
        b.step(ln)
    return make_agent(name, seed, **params).act(b)


def cmd_evaluate(cfg):
    import torch

    from agents.az_mcts import choose_move
    from agents.az_net import Evaluator, make_net
    from agents.az_play import make_mcts

    positions, boards = load_positions(cfg)
    correct = [set(p["correct"]) for p in positions]
    cats = [p["category"] for p in positions]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "eval.json"
    res = json.loads(path.read_text()) if path.exists() else {}

    def score(moves):
        return [int(m in c) for m, c in zip(moves, correct)]

    wh = cfg["board"]
    with get_context("spawn").Pool(cfg["processes"]) as pool:
        for name in ("strong", "hard", "greedy"):
            if name in res:
                continue
            params = cfg["agents"]["strong"] if name == "strong" else {}
            moves = pool.map(_agent_move, [(name, params, cfg["seed"] + i, p["moves"], wh)
                                           for i, p in enumerate(positions)], chunksize=2)
            res[name] = score(moves)
            print(f"{name}: {sum(res[name])}/{len(moves)}", flush=True)
            path.write_text(json.dumps(res))
    for r in cfg["runs"]:
        rc = load_config(ROOT / r["config"])
        for gen in accepted(rc):
            key = f"{r['name']}/{gen}"
            if key in res:
                continue
            net = make_net(rc)
            net.load_state_dict(torch.load(ROOT / rc["run_dir"] / f"gen_{gen:04d}.pt", map_location="cpu"))
            ev = Evaluator(net, cfg["device"])
            pri, _ = ev(boards)
            raw = [int(np.argmax(p)) for p in pri]
            rng = np.random.default_rng(cfg["seed"])
            mcts = make_mcts(ev, rc["mcts"], rng)
            mc = [choose_move(root, 0.0, rng) for root in mcts.search(boards, cfg["agents"]["sims"])]
            res[key] = {"raw": score(raw), "mcts": score(mc)}
            print(f"{key}: raw {sum(res[key]['raw'])}, mcts {sum(res[key]['mcts'])} / {len(boards)}", flush=True)
            path.write_text(json.dumps(res))
    res["_categories"] = cats
    path.write_text(json.dumps(res))


# --- double-dealing in self-play ----------------------------------------------------

def dominoes(board) -> int:
    geo, sides, lines = board.geo, board.sides, board.lines
    return sum(1 for ln, bs in enumerate(geo.line_boxes)
               if not lines[ln] and len(bs) == 2 and sides[bs[0]] == 3 and sides[bs[1]] == 3)


def capture_available(board) -> bool:
    return any(s == 3 for s in board.sides)


def is_dd_move(board, ln) -> bool:
    """Captures nothing although a capture is available and leaves a new
    domino (one line completing two boxes): the two boxes are handed over."""
    if not capture_available(board) or any(board.sides[b] == 3 for b in board.geo.line_boxes[ln]):
        return False
    before = dominoes(board)
    b = board.clone()
    b.step(ln)
    return dominoes(b) > before


def selfplay_positions(ev, rc, sc, seed):
    """Self-play games; returns every loony-phase position (no safe line
    left) with a capture available as (moves so far, chosen line). That is
    where double-dealing decides; the engine solves these quickly."""
    from agents.az_mcts import choose_move, result_for_mover
    from agents.az_play import make_mcts

    w, h = rc["board"]
    rng = np.random.default_rng(seed)
    mcts = make_mcts(ev, rc["mcts"], rng, noise=True)
    live = [[Board(w, h), []] for _ in range(sc["games"])]
    out = []
    while live:
        roots = mcts.search([g[0] for g in live], sc["sims"], noise=True)
        nxt = []
        for g, root in zip(live, roots):
            b, moves = g
            ln = choose_move(root, 1.0 if len(moves) < sc["temp_moves"] else 0.0, rng)
            if capture_available(b) and not safe_moves(b):
                out.append((list(moves), ln))
            b.step(ln)
            moves.append(ln)
            if result_for_mover(b) is None:
                nxt.append(g)
        live = nxt
    return out


_E = {}


def _judge(task):
    """Exact verdict for one capture position: (category by result, chosen
    reaches the best result?, chosen is a DD move?, some DD move reaches the
    best result?, lines drawn, boxes left, declining needed for the most
    boxes?, chosen gets the most boxes?)."""
    moves, ln, wh, node_limit = task
    from agents.strong import Engine
    if "eng" not in _E or len(_E["eng"].tt) > 1_500_000:
        _E["eng"] = Engine(*wh)
    b = Board(*wh)
    for m in moves:
        b.step(m)
    vals = engine_values(_E["eng"], b, node_limit)
    if vals is None:
        return None
    cat, correct, res = classify(b, vals)
    best = max(res.values())
    dd_ok = any(is_dd_move(b, m) for m in correct)
    top = max(vals.values())
    caps = [m for m in vals if any(b.sides[x] == 3 for x in b.geo.line_boxes[m])]
    need_boxes = all(vals[m] < top for m in caps)
    return (cat, res[ln] == best, is_dd_move(b, ln), dd_ok, b.n_drawn, b.geo.n_boxes - sum(b.score),
            need_boxes, vals[ln] == top)


def cmd_selfplay(cfg, keys=None):
    import torch

    from agents.az_net import Evaluator, make_net

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "selfplay_dd.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    sc = cfg["selfplay_dd"]
    with get_context("spawn").Pool(cfg["processes"]) as pool:
        for r in cfg["runs"]:
            rc = load_config(ROOT / r["config"])
            gens = accepted(rc)
            for gen in ([0] if r is cfg["runs"][0] else []) + gens:
                key = f"{r['name']}/{gen}"
                if key in res or (keys and key not in keys):
                    continue
                net = make_net(rc)
                net.load_state_dict(torch.load(ROOT / rc["run_dir"] / f"gen_{gen:04d}.pt", map_location="cpu"))
                t0 = time.time()
                pos = selfplay_positions(Evaluator(net, cfg["device"]), rc, sc, cfg["seed"] + gen)
                verdicts = pool.map(_judge, [(m, ln, cfg["board"], sc["node_limit"]) for m, ln in pos], chunksize=8)
                v = [x for x in verdicts if x is not None]
                res[key] = {"games": sc["games"], "positions": len(pos), "solved": len(v), "verdicts": v}
                need = [x for x in v if x[6]]
                dd_wrong = [x for x in v if x[2] and not x[7]]
                print(f"{key}: {len(v)} loony capture positions, {len(need)} need a decline for the most boxes, "
                      f"net declined right in {sum(x[7] for x in need)}; {sum(x[2] for x in v)} DD moves, "
                      f"{len(dd_wrong)} of them cost boxes ({time.time() - t0:.0f} s)", flush=True)
                path.write_text(json.dumps(res))


# --- report -------------------------------------------------------------------------

def cmd_report(cfg):
    positions, boards = load_positions(cfg)
    meta = json.loads((ROOT / cfg["positions"]["out"]).read_text())
    ev = json.loads((OUT / "eval.json").read_text())
    sp = json.loads((OUT / "selfplay_dd.json").read_text()) if (OUT / "selfplay_dd.json").exists() else {}
    cats = ev["_categories"]
    idx = {c: [i for i, x in enumerate(cats) if x == c] for c in ("dd", "parity")}
    n = {c: len(v) for c, v in idx.items()}

    def pct(bits, c):
        return 100 * sum(bits[i] for i in idx[c]) / n[c]

    gens = [(r["name"], int(k.split("/")[1])) for r in cfg["runs"] for k in ev if k.startswith(r["name"] + "/")]
    gens.sort(key=lambda x: ([r["name"] for r in cfg["runs"]].index(x[0]), x[1]))
    und = [p["undrawn"] for p in positions]
    L = ["# Strategy probe: long chain rule and double-dealing (5x5)", "",
         f"Generated by `scripts/strategy_probe.py` (`configs/strategy_probe.yaml`), {time.strftime('%Y-%m-%d %H:%M')}.",
         "",
         f"Positions: {len(positions)} late 5x5 positions ({n['dd']} double-dealing, {n['parity']} parity) "
         f"from games between ladder agents ({', '.join('-'.join(p) for p in cfg['positions']['pairs'])}), "
         f"{min(und)}–{max(und)} undrawn lines (median {int(np.median(und))}). Each one is *critical*: "
         "some legal move reaches a better final result (win/tie/loss, given the score so far) than another; "
         "a move is correct if it reaches the best result. Exact values of every legal move: the dd positions "
         f"(≤ {meta['max_undrawn']} undrawn lines) by full brute force (negamax over all remaining lines, memo "
         "over line masks, no pruning). Parity decisions on 5x5 come at ~30–36 undrawn lines, out of reach "
         "for brute force; there every child position is solved to the end by the strong engine (no depth "
         "limit, no heuristic; alpha-beta plus captures and chain openings that provably never lose value). "
         f"The engine agreed with brute force on all moves of {meta['solver_check']['engine_agrees']} of "
         f"{meta['solver_check']['positions']} brute-forced positions met while generating. "
         "The positions are in `docs/strategy_probe_positions.json`.", "",
         "- **dd** (double-dealing): a box can be taken, but every capturing line loses (or ties) where the "
         "best move wins: the mover must decline the last two boxes of a chain (four of a loop) — the "
         "all-but-two / all-but-four trick.",
         "- **parity** (long chain rule): nothing can be taken, safe lines remain, and the choice decides "
         "who has to open the first long chain; the fix can be a safe line or a sacrifice.", "",
         "## Share of correct moves", "",
         "| player | dd | parity | all |", "|---|---:|---:|---:|"]
    tot = n["dd"] + n["parity"]
    for name, label in (("strong", f"strong ({cfg['agents']['strong']['node_budget']} nodes)"),
                        ("hard", "KSquares Hard"), ("greedy", "greedy")):
        b = ev[name]
        L.append(f"| {label} | {pct(b, 'dd'):.0f} | {pct(b, 'parity'):.0f} | {100 * sum(b) / tot:.0f} |")
    chance = {c: 100 * np.mean([len(positions[i]["correct"]) / len(positions[i]["values"]) for i in idx[c]])
              for c in idx}
    L.append(f"| uniform random legal move (expected) | {chance['dd']:.0f} | {chance['parity']:.0f} | "
             f"{(chance['dd'] * n['dd'] + chance['parity'] * n['parity']) / tot:.0f} |")
    L.append("| exact (brute force / solver) | 100 | 100 | 100 |")
    for run, g in gens[-1:]:
        e = ev[f"{run}/{g}"]
        L.append(f"| {run} gen {g}, net only | {pct(e['raw'], 'dd'):.0f} | {pct(e['raw'], 'parity'):.0f} | "
                 f"{100 * sum(e['raw']) / tot:.0f} |")
        L.append(f"| {run} gen {g}, net + MCTS {cfg['agents']['sims']} | {pct(e['mcts'], 'dd'):.0f} | "
                 f"{pct(e['mcts'], 'parity'):.0f} | {100 * sum(e['mcts']) / tot:.0f} |")
    L += ["", "## By generation", "",
          "Probe columns: % correct on the probe positions. Self-play columns (40 games per generation, "
          "training settings, every loony-phase position with a capture available solved exactly): "
          "*need* = positions where only declining gets the most boxes, *declined right* = share of those "
          "where the net's move got the most boxes, *DD/game* = moves handing over two boxes per game, "
          "*DD wrong* = share of those that cost boxes, *stage* = median lines already drawn (of 60) and boxes "
          "left (of 25) at the net's correct declines.", "",
          "| run | gen | raw dd | raw parity | mcts dd | mcts parity | self-play: need | declined right | DD/game | DD wrong | stage (lines / boxes left) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    curve = []
    keys = ([("az5", 0)] if "az5/0" in sp else []) + gens
    for run, g in keys:
        e = ev.get(f"{run}/{g}")
        s = sp.get(f"{run}/{g}")
        cells = [run, str(g)]
        if e:
            cells += [f"{pct(e['raw'], 'dd'):.0f}", f"{pct(e['raw'], 'parity'):.0f}",
                      f"{pct(e['mcts'], 'dd'):.0f}", f"{pct(e['mcts'], 'parity'):.0f}"]
        else:
            cells += ["—"] * 4
        if s:
            v = s["verdicts"]
            need = [x for x in v if x[6]]
            right = [x for x in need if x[7]]
            dd = [x for x in v if x[2]]
            cells += [str(len(need)), f"{100 * len(right) / len(need):.0f}" if need else "—",
                      f"{len(dd) / s['games']:.2f}",
                      f"{100 * sum(1 for x in dd if not x[7]) / len(dd):.0f}" if dd else "—",
                      f"{np.median([x[4] for x in right]):.0f} / {np.median([x[5] for x in right]):.0f}"
                      if right else "—"]
        else:
            cells += ["—"] * 5
        L.append("| " + " | ".join(cells) + " |")
        curve.append((run, g, e, s))
    (ROOT / "docs/strategy_probe.md").write_text("\n".join(L) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    xs = list(range(len(curve)))
    labels = [f"{'b' if r == 'az5b' else ''}{g}" for r, g, _, _ in curve]
    ax = axes[0]
    for cat, color in (("dd", "#c44e52"), ("parity", "#4c72b0")):
        for mode, ls in (("raw", "--"), ("mcts", "-")):
            pts = [(x, pct(e[mode], cat)) for x, (_, _, e, _) in zip(xs, curve) if e]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, marker="o", ms=3, color=color,
                    label=f"{cat}, {'net only' if mode == 'raw' else 'MCTS'}")
    for name, m in (("strong", "s"), ("hard", "^")):
        for cat, color in (("dd", "#c44e52"), ("parity", "#4c72b0")):
            ax.axhline(pct(ev[name], cat), color=color, lw=0.8, ls=":")
            ax.text(xs[-1], pct(ev[name], cat), f"{name} {cat}", fontsize=7, color=color, ha="right", va="bottom")
    ax.set_ylim(-3, 103)
    ax.set_ylabel("correct moves %")
    ax.set_title("probe positions (b = az5b generation)")
    ax.set_xticks(xs, labels, fontsize=7)
    ax.set_xlabel("generation")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    def share(s):
        need = [x for x in s["verdicts"] if x[6]]
        return 100 * sum(x[7] for x in need) / len(need) if need else None

    pts = [(x, share(s)) for x, (_, _, _, s) in zip(xs, curve) if s and share(s) is not None]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color="black", label="declined right (need)")
    pts = [(x, 100 * sum(1 for v in s["verdicts"] if v[2] and not v[7]) / max(1, sum(v[2] for v in s["verdicts"])))
           for x, (_, _, _, s) in zip(xs, curve) if s and any(v[2] for v in s["verdicts"])]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="s", ls="--", color="#c44e52",
            label="DD moves that cost boxes")
    ax.set_ylim(-3, 103)
    ax.set_xticks(xs, labels, fontsize=7)
    ax.set_xlabel("generation")
    ax.set_ylabel("%")
    ax.set_title("double-dealing in self-play (loony phase, exact)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "docs/strategy_probe.png", dpi=110)
    print((ROOT / "docs/strategy_probe.md").read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["positions", "evaluate", "selfplay", "report"])
    ap.add_argument("--config", default=str(ROOT / "configs/strategy_probe.yaml"))
    ap.add_argument("--keys", default=None, help="selfplay only these, e.g. az5/0,az5/29")
    ap.add_argument("--out", default=None, help="results dir (default runs/strategy_probe)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    global OUT
    if a.out:
        OUT = ROOT / a.out
    if a.cmd == "selfplay":
        cmd_selfplay(cfg, set(a.keys.split(",")) if a.keys else None)
    else:
        {"positions": cmd_positions, "evaluate": cmd_evaluate, "report": cmd_report}[a.cmd](cfg)


if __name__ == "__main__":
    main()
