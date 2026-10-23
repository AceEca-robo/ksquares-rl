"""MCTS speed for AlphaZero self-play: simulations per second of one process
for several numbers of concurrent games (= GPU batch), with the time split
into network (encode + forward + copy back) and tree (everything else).
An untrained net from configs/az5.yaml; positions from random play.
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

from _common import ROOT, load_config

from agents.az_net import Evaluator, make_net
from agents.az_play import make_mcts
from sim.board import Board


class Timed:
    def __init__(self, ev):
        self.ev, self.t, self.calls = ev, 0.0, 0

    def __call__(self, boards):
        t0 = time.perf_counter()
        out = self.ev(boards)
        self.t += time.perf_counter() - t0
        self.calls += 1
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/az5.yaml"))
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 64, 128])
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--leaves", type=int, default=1, help="leaves per tree per round")
    a = ap.parse_args()
    cfg = load_config(a.config)
    torch.set_num_threads(1)
    ev = Timed(Evaluator(make_net(cfg), cfg["device"]))
    w, h = cfg["board"]
    rng = random.Random(0)
    print(f"| games (batch) | leaves/tree | sims/s | net share | ms per net call | mean batch |")
    print("|---:|---:|---:|---:|---:|---:|")
    for nb in a.batches:
        boards = []
        for _ in range(nb):
            b = Board(w, h)
            for _ in range(rng.randrange(0, 40)):
                b.step(rng.choice(b.legal_moves()))
            boards.append(b)
        mcts = make_mcts(ev, cfg["mcts"], np.random.default_rng(0), leaves_per_tree=a.leaves)
        mcts.search(boards[:2], 8)  # warm-up
        ev.t, ev.calls, mcts.n_evals = 0.0, 0, 0
        t0 = time.perf_counter()
        reps = max(1, 64 // nb)
        for _ in range(reps):
            mcts.search(boards, a.sims)
        dt = time.perf_counter() - t0
        print(f"| {nb} | {a.leaves} | {mcts.n_evals / dt:.0f} | {100 * ev.t / dt:.0f}% | "
              f"{1000 * ev.t / ev.calls:.2f} | {mcts.n_evals / ev.calls:.1f} |", flush=True)


if __name__ == "__main__":
    main()
