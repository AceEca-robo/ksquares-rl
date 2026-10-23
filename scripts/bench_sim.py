"""Measure simulator throughput (games/s, moves/s, ms/move) in self-play.

Each cell runs for at least ``seconds`` and at least one full game.
"""

from __future__ import annotations

import argparse
import time

from _common import ROOT, load_config

from agents.registry import make_agent
from sim.board import Board


def bench(w: int, h: int, agent_name: str, seconds: float, seed: int, params: dict) -> tuple[float, float]:
    agents = [make_agent(agent_name, seed, **params), make_agent(agent_name, seed + 1, **params)]
    games = moves = 0
    t0 = time.perf_counter()
    while games == 0 or time.perf_counter() - t0 < seconds:
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
    params = cfg.get("params") or {}
    print("| board | agents | games/s | moves/s | ms/move |")
    print("|---|---|---:|---:|---:|")
    for w, h in cfg["boards"]:
        for name in cfg["agents"]:
            gps, mps = bench(w, h, name, cfg["seconds"], cfg["seed"], params.get(name, {}))
            print(f"| {w}x{h} | {name} vs {name} | {gps:,.2f} | {mps:,.0f} | {1000 / mps:.3f} |", flush=True)


if __name__ == "__main__":
    main()
