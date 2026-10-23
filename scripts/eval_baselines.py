"""Play baseline matchups and print a markdown results table."""

from __future__ import annotations

import argparse
import math
import random

from _common import ROOT, load_config

from agents.baselines import AGENTS
from sim.board import Board


def play(w: int, h: int, agents: list, first: int) -> list[int]:
    """Play one game; ``agents[first]`` moves first. Returns scores per agent."""
    b = Board(w, h, first_player=first)
    while not b.done:
        b.step(agents[b.player].act(b))
    return b.score


def run_matchup(w, h, a_name, b_name, games, seed):
    rng = random.Random(seed)
    agents = [AGENTS[a_name](rng.randrange(2**32)), AGENTS[b_name](rng.randrange(2**32))]
    wins = [0, 0]
    ties = 0
    margin = 0
    first_wins = 0
    for g in range(games):
        first = g % 2
        s = play(w, h, agents, first)
        margin += s[0] - s[1]
        if s[0] == s[1]:
            ties += 1
        else:
            win = 0 if s[0] > s[1] else 1
            wins[win] += 1
            first_wins += win == first
    return wins, ties, margin / games, first_wins


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/eval_baselines.yaml"))
    cfg = load_config(ap.parse_args().config)
    n = cfg["games"]
    print("| board | A vs B | games | A wins | B wins | ties | A win % (±95%) | mean margin A−B | first-mover wins % |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, (w, h) in enumerate(cfg["boards"]):
        for j, (a, b) in enumerate(cfg["matchups"]):
            wins, ties, margin, first_wins = run_matchup(w, h, a, b, n, cfg["seed"] + 1000 * i + j)
            p = wins[0] / n
            ci = 1.96 * math.sqrt(p * (1 - p) / n) * 100
            decided = n - ties
            fm = 100 * first_wins / decided if decided else float("nan")
            print(f"| {w}x{h} | {a} vs {b} | {n} | {wins[0]} | {wins[1]} | {ties} | "
                  f"{100 * p:.1f} ± {ci:.1f} | {margin:+.2f} | {fm:.1f} |", flush=True)


if __name__ == "__main__":
    main()
