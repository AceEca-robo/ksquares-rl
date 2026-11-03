"""sim/chain_solver.py: the formula judge against exact search on small boards."""

import random

from agents.strong import Engine, exact_move_values
from sim import chain_solver as cs
from sim.board import Board


def _settled_positions(w, h, n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = Board(w, h)
        while True:
            safe = cs.safe_lines(b)
            if not safe:
                break
            b.step(rng.choice(safe))
            if cs.options(b) is not None and rng.random() < 0.5:
                break
        if cs.options(b) is not None:
            out.append(b)
    return out


def test_value_exact_without_safe_lines():
    """No safe line left: Berlekamp's loony-endgame recursion is exact."""
    for w, h in ((3, 3), (4, 3)):
        eng = Engine(w, h)
        n = 0
        for b in _settled_positions(w, h, 80, 1):
            if cs.safe_lines(b):
                continue
            n += 1
            assert cs.solve(b) == max(exact_move_values(eng, b, None).values())
        assert n > 20


def test_best_move_keeps_result_on_small_boards():
    eng = Engine(3, 3)
    ok = total = 0
    for b in _settled_positions(3, 3, 60, 2):
        vals = exact_move_values(eng, b, None)
        best = max(vals.values())
        r = cs.best_move(b)
        if r is None:
            continue
        total += 1
        ok += (vals[r[0]] > 0) == (best > 0) and (vals[r[0]] >= 0) == (best >= 0)
    assert total and ok / total >= 0.75, (ok, total)


def test_empty_board_is_not_settled():
    assert cs.options(Board(5, 5)) is None


def test_ply_values_exact_without_safe_lines():
    eng = Engine(3, 3)
    for b in _settled_positions(3, 3, 40, 3):
        if cs.safe_lines(b):
            continue
        assert cs.ply_values(b) == exact_move_values(eng, b, None)
