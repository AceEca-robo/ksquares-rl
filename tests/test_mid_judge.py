"""sim/mid_judge.py: C search == Python reference == exact search."""

import random

import pytest

from agents.strong import Engine, exact_move_values
from sim import mid_judge as mj
from sim.board import Board


def _positions(w, h, n, seed, lo=0.45, hi=0.85):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = Board(w, h)
        k = rng.randint(int(lo * b.geo.n_lines), int(hi * b.geo.n_lines))
        while b.n_drawn < k:
            free = [ln for ln in range(b.geo.n_lines) if not b.lines[ln]]
            b.step(rng.choice(free))
        if b.n_drawn < b.geo.n_lines:
            out.append(b)
    return out


def _sign(x):
    return (x > 0) - (x < 0)


@pytest.mark.parametrize("wh", [(3, 3), (4, 3), (3, 4)])
def test_symmetries_are_permutations(wh):
    sy = mj.symmetries(*wh)
    n = Board(*wh).geo.n_lines
    assert sy[0] == list(range(n))
    assert all(sorted(p) == list(range(n)) for p in sy)
    assert len(sy) == (8 if wh[0] == wh[1] else 4)


@pytest.mark.parametrize("wh", [(3, 3), (4, 3)])
def test_c_and_python_match_exact(wh):
    eng, cj, pj = Engine(*wh), mj.Judge(*wh, tt_bits=16), mj.PyJudge(*wh)
    for b in _positions(*wh, 40, 5):
        vals = exact_move_values(eng, b, None)
        best = max(vals.values())
        assert mj.value(cj, b) == best
        assert mj.value(pj, b) == best
        me = b.score[b.player] - b.score[b.player ^ 1]
        want = {ln: _sign(me + v) for ln, v in vals.items()}
        assert mj.classify(cj, b) == want
        assert mj.outcome(cj, b) == max(want.values())


def test_c_matches_exact_on_4x4_with_fresh_tt():
    eng = Engine(4, 4)
    for b in _positions(4, 4, 10, 8, 0.55, 0.8):
        cj = mj.Judge(4, 4, tt_bits=16)
        assert mj.value(cj, b) == max(exact_move_values(eng, b, None).values())
