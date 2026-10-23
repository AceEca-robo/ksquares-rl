"""Baseline agents: uniform random and one-ply greedy."""

from __future__ import annotations

import random

from sim.board import Board


class RandomAgent:
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, board: Board) -> int:
        return self.rng.choice(board.legal_moves())


def capturing_moves(board: Board) -> list[int]:
    """Lines that complete at least one box right now."""
    geo, sides, lines = board.geo, board.sides, board.lines
    out = []
    for b in range(geo.n_boxes):
        if sides[b] == 3:
            for ln in geo.box_lines[b]:
                if not lines[ln]:
                    out.append(ln)
    return list(dict.fromkeys(out))


def safe_moves(board: Board) -> list[int]:
    """Lines that put no box at three sides (do not hand a box over)."""
    geo, sides, lines = board.geo, board.sides, board.lines
    return [
        ln
        for ln in range(geo.n_lines)
        if not lines[ln] and all(sides[b] != 2 for b in geo.line_boxes[ln])
    ]


class GreedyAgent:
    """Take a box if possible; else never draw a third side if avoidable."""

    name = "greedy"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, board: Board) -> int:
        moves = capturing_moves(board) or safe_moves(board) or board.legal_moves()
        return self.rng.choice(moves)


AGENTS = {"random": RandomAgent, "greedy": GreedyAgent}
