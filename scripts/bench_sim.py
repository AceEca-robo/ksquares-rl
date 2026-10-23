"""Measure simulator throughput (games/s, moves/s) for baseline agents."""

from __future__ import annotations

import argparse
import time

from _common import ROOT, load_config

from agents.baselines import AGENTS
from sim.board import Board


def bench(w: int, h: int, agent_name: str, seconds: float, seed: int) -> tuple[float, float]:
    agents = [AGENTS[agent_name](seed), AGENTS[agent_name](seed + 1)]
    games = moves = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        b = Board(w, h)
        while not b.done:
            b.step(agents[b.player].act(b))
        games += 1
        moves += b.geo.n_lines
    dt = time.perf_counter() - t0
    return games / dt, moves / dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/bench_sim.yaml"))
    cfg = load_config(ap.parse_args().config)
    print("| board | agents | games/s | moves/s |")
    print("|---|---|---:|---:|")
    for w, h in cfg["boards"]:
        for name in cfg["agents"]:
            gps, mps = bench(w, h, name, cfg["seconds"], cfg["seed"])
            print(f"| {w}x{h} | {name} vs {name} | {gps:,.0f} | {mps:,.0f} |", flush=True)


if __name__ == "__main__":
    main()
