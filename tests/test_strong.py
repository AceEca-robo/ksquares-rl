import random
from functools import lru_cache

import pytest

from agents.strong import INF, Engine, StrongAgent
from sim.board import Board, geometry
from sim.ksquares_ai import GlibcRand, KSquaresAI


def brute_value(w, h):
    """Plain negamax over every line: net future boxes for the side to move."""
    geo = geometry(w, h)
    full = (1 << geo.n_lines) - 1
    bm = [sum(1 << ln for ln in bl) for bl in geo.box_lines]

    @lru_cache(maxsize=None)
    def v(L):
        if L == full:
            return 0
        best = -INF
        for ln in range(geo.n_lines):
            if not (L >> ln) & 1:
                L2 = L | (1 << ln)
                got = sum(1 for b in geo.line_boxes[ln] if L2 & bm[b] == bm[b])
                best = max(best, got + v(L2) if got else -v(L2))
        return best

    return v


@pytest.mark.parametrize("w,h,n", [(2, 2, 200), (3, 2, 100), (1, 4, 100)])
def test_exact_search_matches_brute_force(w, h, n):
    v = brute_value(w, h)
    eng = Engine(w, h)
    rng = random.Random(w * 100 + h)
    n_lines = geometry(w, h).n_lines
    for _ in range(n):
        lines = rng.sample(range(n_lines), rng.randrange(n_lines))
        L = sum(1 << ln for ln in lines)
        assert eng.search(L, 999, -INF, INF) == v(L)


def test_opened_loop_of_four_all_but_four():
    # 5x5 position from a game: an opened 4-loop is capturable at both ends;
    # declining with the middle line (all-but-four) is the only good move
    L = 0b101111100011111010000101111111011010111110110111110111110111
    eng = Engine(5, 5)
    assert eng.search(L, 999, -INF, INF) == 9
    line, value, _, exact = eng.best_move(L, 5.0, random.Random(0))
    assert (line, value, exact) == (24, 9, True)


def test_simple_endgame_values():
    eng = Engine(5, 5)
    assert eng.simple_value((3,), ()) == -3
    assert eng.simple_value((1, 3), ()) == 2     # give 1, then get 3
    assert eng.simple_value((3, 3), ()) == -2    # 3 + (2 - 3): all-but-two once
    assert eng.simple_value((), (4,)) == -4
    assert eng.simple_value((), (4, 4)) == 0     # controller declines 4, takes 4


def test_strong_beats_hard_quickly():
    wins = 0
    for g in range(4):
        st, ai = StrongAgent(seed=g, time_limit=0.2), KSquaresAI(2, GlibcRand(g + 1))
        seats = [st, ai] if g % 2 == 0 else [ai, st]
        b = Board(4, 4)
        while not b.done:
            b.step(seats[b.player].act(b))
        wins += b.score[seats.index(st)] > b.score[seats.index(ai)]
    assert wins >= 3


def test_node_budget_stops_search():
    rng = random.Random(0)
    b = Board(5, 5)
    for _ in range(12):
        b.step(rng.choice(b.legal_moves()))
    a = StrongAgent(1, time_limit=100.0, node_budget=2000)
    ln = a.act(b)
    assert not b.lines[ln]
    assert a._engines[(5, 5)].nodes <= 2001
    # deterministic: same seed and budget -> same move
    assert StrongAgent(1, node_budget=2000).act(b) == ln
