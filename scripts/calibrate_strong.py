"""Node budget for strong equivalent to its time limit (configs/az5b.yaml).

Plays strong (time limit) against Hard and records, for every search that
ran into the limit, the nodes searched and the CPU time of this process.
CPU time instead of wall clock, so that a loaded machine measures the
search rate of one free core (up to CPU frequency effects).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from _common import ROOT  # noqa: F401

from agents.registry import make_agent
from agents.strong import StrongAgent
from sim.board import Board


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--time-limit", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=99)
    a = ap.parse_args()
    rates, hit_nodes = [], []
    for g in range(a.games):
        strong = StrongAgent(a.seed + g, time_limit=a.time_limit)
        hard = make_agent("hard", a.seed + g)
        b = Board(5, 5)
        first = g % 2
        while not b.done:
            if b.player == first:
                c0, t0 = time.process_time(), time.perf_counter()
                ln = strong.act(b)
                cpu, wall = time.process_time() - c0, time.perf_counter() - t0
                eng = strong._engines[(5, 5)]
                if not strong.last[3] and wall >= 0.95 * a.time_limit:  # stopped by the limit
                    rates.append(eng.nodes / cpu)
                    hit_nodes.append(eng.nodes)
            else:
                ln = hard.act(b)
            b.step(ln)
    rates = np.array(rates)
    print(f"{len(rates)} searches hit the {a.time_limit} s limit; nodes per CPU second: "
          f"median {np.median(rates):.0f}, mean {rates.mean():.0f}, p10 {np.percentile(rates, 10):.0f}, "
          f"p90 {np.percentile(rates, 90):.0f}; nodes per search (wall clock, loaded): median {np.median(hit_nodes):.0f}")
    print(f"budget = median rate x {a.time_limit} s = {np.median(rates) * a.time_limit:.0f} nodes")


if __name__ == "__main__":
    main()
