"""Agents by name: ``make_agent(name, seed, **params)``.

random, greedy  — agents.baselines
easy, medium, hard — the KSquares AI port (sim.ksquares_ai), each instance
    with its own glibc rand() state seeded with ``seed`` (in the real game
    all computer seats share one process-wide state)
strong — agents.strong (params: time_limit, max_depth, tt_limit)
"""

from __future__ import annotations

from agents.baselines import GreedyAgent, RandomAgent
from agents.strong import StrongAgent
from sim.ksquares_ai import GlibcRand, make_ai

NAMES = ("random", "greedy", "easy", "medium", "hard", "strong")


def make_agent(name: str, seed: int, **params):
    if name == "random":
        return RandomAgent(seed)
    if name == "greedy":
        return GreedyAgent(seed)
    if name in ("easy", "medium", "hard"):
        # glibc treats srand(0) as srand(1); keep seeds 0 and 1 apart
        return make_ai(name, GlibcRand(seed % (2**32 - 1) + 1))
    if name == "strong":
        return StrongAgent(seed, **params)
    raise ValueError(f"unknown agent {name!r}; known: {', '.join(NAMES)}")
