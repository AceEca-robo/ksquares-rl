"""Self-play and net-vs-net matches for AlphaZero, many games in lock step.

Positions are stored compactly (lines, box sides, owners relative to the
mover, score difference, mover-is-first) and encoded with ``Codec.encode``
when training. Player 0 always moves first in these games.
"""

from __future__ import annotations

import time

import numpy as np

from agents.az_mcts import MCTS, choose_move, result_for_mover, visit_policy
from sim.board import Board


def state_arrays(board: Board):
    owner = np.array(board.owner, dtype=np.int8)
    rel = np.where(owner < 0, 0, np.where(owner == board.player, 1, -1)).astype(np.int8)
    me, op = board.score[board.player], board.score[board.player ^ 1]
    return (np.frombuffer(bytes(board.lines), dtype=np.uint8), np.frombuffer(bytes(board.sides), dtype=np.uint8),
            rel, me - op, int(board.player == 0))


def make_mcts(evaluate, mcfg: dict, rng, leaves_per_tree: int = 1, noise: bool = False) -> MCTS:
    return MCTS(evaluate, c_puct=mcfg["c_puct"], fpu_reduction=mcfg["fpu_reduction"],
                dirichlet_alpha=mcfg["dirichlet_alpha"], dirichlet_eps=mcfg["dirichlet_eps"] if noise else 0.0,
                leaves_per_tree=leaves_per_tree, virtual_loss=mcfg["virtual_loss"], rng=rng)


def selfplay(evaluate, cfg: dict, n_games: int, concurrent: int, seed: int, progress=None):
    """Play ``n_games`` self-play games; returns (samples dict, stats dict).
    ``progress(finished games, seconds)`` is called after every quarter."""
    w, h = cfg["board"]
    sp = cfg["selfplay"]
    rng = np.random.default_rng(seed)
    mcts = make_mcts(evaluate, cfg["mcts"], rng, noise=True)
    cols = {k: [] for k in ("lines", "sides", "owner", "diff", "first", "pi", "z", "margin")}
    # sims: a fixed count, or [lo, hi] drawn uniformly per game and move
    lo, hi = (sp["sims"], sp["sims"]) if isinstance(sp["sims"], int) else sp["sims"]
    started = 0
    live = []  # [board, records [(state, pi, player)], plies]
    t0 = time.perf_counter()
    n_moves = n_search = 0
    lengths, first_wins = [], 0
    while live or started < n_games:
        while len(live) < concurrent and started < n_games:
            live.append([Board(w, h), [], 0])
            started += 1
        roots = mcts.search([g[0] for g in live], rng.integers(lo, hi + 1, len(live)), noise=True)
        nxt = []
        for g, root in zip(live, roots):
            board, recs, plies = g
            if len(root.moves) > 1:
                recs.append((state_arrays(board), visit_policy(root, board.geo.n_lines), board.player))
                n_search += 1
            temp = 1.0 if plies < sp["temp_moves"] else 0.0
            board.step(choose_move(root, temp, rng))
            n_moves += 1
            g[2] += 1
            res = result_for_mover(board)
            if res is None:
                nxt.append(g)
                continue
            winner = board.player if res > 0 else board.player ^ 1 if res < 0 else -1
            # final score difference of player 0 when the winner became fixed
            # (games stop there, the rest of the boxes is not played out)
            m0 = board.score[0] - board.score[1]
            first_wins += winner == 0
            lengths.append(g[2])
            if progress and len(lengths) % max(1, n_games // 4) == 0:
                progress(len(lengths), time.perf_counter() - t0)
            for (lines, sides, owner, diff, first), pi, player in recs:
                cols["lines"].append(lines)
                cols["sides"].append(sides)
                cols["owner"].append(owner)
                cols["diff"].append(diff)
                cols["first"].append(first)
                cols["pi"].append(pi)
                cols["z"].append(0 if winner < 0 else 1 if winner == player else -1)
                cols["margin"].append(m0 if player == 0 else -m0)
        live = nxt
    dt = time.perf_counter() - t0
    samples = {
        "lines": np.array(cols["lines"], dtype=np.uint8), "sides": np.array(cols["sides"], dtype=np.uint8),
        "owner": np.array(cols["owner"], dtype=np.int8), "diff": np.array(cols["diff"], dtype=np.int8),
        "first": np.array(cols["first"], dtype=np.uint8), "pi": np.array(cols["pi"], dtype=np.float16),
        "z": np.array(cols["z"], dtype=np.int8), "margin": np.array(cols["margin"], dtype=np.int8),
    }
    stats = {"games": n_games, "positions": len(cols["z"]), "moves": n_moves, "searches": n_search,
             "evals": mcts.n_evals, "seconds": dt, "mean_len": float(np.mean(lengths)),
             "first_wins": int(first_wins)}
    return samples, stats


def match(evaluators, cfg: dict, sims: int, temp_moves: int, n_games: int, concurrent: int, seed: int,
          first_offset: int = 0):
    """evaluators[0] vs evaluators[1]; game g: evaluators[(g + first_offset) % 2]
    moves first. Returns (wins of 0, wins of 1, ties, box margin of 0)."""
    w, h = cfg["board"]
    rng = np.random.default_rng(seed)
    searchers = [make_mcts(e, cfg["mcts"], rng) for e in evaluators]
    started = 0
    live = []  # [board, first (evaluator index moving first), plies]
    out = [0, 0, 0, 0]
    while live or started < n_games:
        while len(live) < concurrent and started < n_games:
            live.append([Board(w, h), (started + first_offset) % 2, 0])
            started += 1
        for k in (0, 1):
            group = [g for g in live if (g[0].player == 0) == (g[1] == k) and result_for_mover(g[0]) is None]
            if not group:
                continue
            roots = searchers[k].search([g[0] for g in group], sims)
            for g, root in zip(group, roots):
                g[0].step(choose_move(root, 1.0 if g[2] < temp_moves else 0.0, rng))
                g[2] += 1
        nxt = []
        for g in live:
            board, first, _ = g
            if result_for_mover(board) is None:
                nxt.append(g)
                continue
            s0 = board.score[0] if first == 0 else board.score[1]  # evaluator 0's boxes
            s1 = sum(board.score) - s0
            out[3] += s0 - s1
            out[0 if s0 > s1 else 1 if s1 > s0 else 2] += 1
        live = nxt
    return tuple(out)
