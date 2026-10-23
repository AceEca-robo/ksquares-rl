"""Check the bridge and the AI port against the real game, move by move.

Every game is played twice with the same seeds: once entirely in the
simulator (``sim.ksquares_ai`` with ``GlibcRand(seed)``) and once in KSquares
over the bridge (``srand(seed)`` in the game). External seats are a seeded
random agent in both runs. The two move lists (line, player, completed
boxes) must be identical; the bridge additionally replays every reported
move in its own simulator mirror and compares the full state.
"""

from __future__ import annotations

import argparse
import random
import time

from _common import ROOT, load_config

from agents.baselines import RandomAgent
from env.bridge import KSquaresBridge
from sim.board import Board
from sim.ksquares_ai import GlibcRand, KSquaresAI

LEVELS = {"easy": 0, "medium": 1, "hard": 2}


def sim_game(w, h, ai_seats, difficulty, seed, agent_seed, hash_seed):
    """Play in the simulator; returns [(line, player, boxes)] and AI stats."""
    rand = GlibcRand(seed)
    ai = KSquaresAI(difficulty, rand, hash_seed)
    ext = RandomAgent(agent_seed)
    b = Board(w, h)
    moves = []
    while not b.done:
        p = b.player
        line = (ai if ai_seats[p] else ext).act(b)
        before = b.owner.copy()
        b.step(line)
        moves.append((line, p, sorted(i for i, (x, y) in enumerate(zip(before, b.owner)) if x != y)))
    return moves, ai.stats


def bridge_game(br, w, h, ai_seats, difficulty, seed, agent_seed):
    ext = RandomAgent(agent_seed)
    moves = []
    st = br.reset(w, h, ai=ai_seats, difficulty=difficulty, seed=seed)
    while True:
        moves += [(ln, p, sorted(bx)) for ln, p, bx in st["moves"]]
        if st["done"]:
            return moves
        st = br.step(ext.act(br.board))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/verify_port.yaml"))
    cfg = load_config(ap.parse_args().config)
    hash_seed = cfg["qt_hash_seed"]  # None: the game picks a random QHash seed
    port_hash_seed = 0 if hash_seed is None else hash_seed
    rng = random.Random(cfg["seed"])
    print("| board | level | seats | games | identical | moves compared | AI moves | ambiguous handouts | bridge s/game | sim s/game |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    failures = 0
    with KSquaresBridge(hash_seed=hash_seed) as br:
        for w, h in cfg["boards"]:
            for level in cfg["levels"]:
                for seats in cfg["seats"]:
                    ai_seats = [s == "ai" for s in seats]
                    n = cfg["games"]
                    ok = n_moves = n_ai = amb = 0
                    t_br = t_sim = 0.0
                    for g in range(n):
                        # alternate who moves first when one seat is external
                        seats_g = ai_seats if g % 2 == 0 else ai_seats[::-1]
                        seed = rng.randrange(1, 2**31)
                        agent_seed = rng.randrange(2**32)
                        t0 = time.perf_counter()
                        ref, stats = sim_game(w, h, seats_g, LEVELS[level], seed, agent_seed, port_hash_seed)
                        t1 = time.perf_counter()
                        got = bridge_game(br, w, h, seats_g, LEVELS[level], seed, agent_seed)
                        t2 = time.perf_counter()
                        t_sim += t1 - t0
                        t_br += t2 - t1
                        amb += stats["ambiguous_handout"]
                        if ref == got:
                            ok += 1
                        else:
                            failures += 1
                            k = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), min(len(ref), len(got)))
                            print(f"MISMATCH {w}x{h} {level} seats={seats_g} seed={seed} agent_seed={agent_seed} "
                                  f"at move {k}: sim {ref[k] if k < len(ref) else None} "
                                  f"game {got[k] if k < len(got) else None}", flush=True)
                        n_moves += len(got)
                        n_ai += sum(1 for _, p, _ in got if seats_g[p])
                    print(f"| {w}x{h} | {level} | {'-'.join(seats)} | {n} | {ok} | {n_moves} | {n_ai} | {amb} | "
                          f"{t_br / n:.3f} | {t_sim / n:.3f} |", flush=True)
    if failures:
        raise SystemExit(f"{failures} games differ")


if __name__ == "__main__":
    main()
