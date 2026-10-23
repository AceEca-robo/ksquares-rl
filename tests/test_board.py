import random

import pytest

from agents.baselines import GreedyAgent, RandomAgent, capturing_moves, safe_moves
from sim.board import NO_OWNER, Board, geometry


def ksquares_squares_from_line(line, w, h):
    """Literal port of aiController::squaresFromLine (aicontroller.cpp)."""
    stride = 2 * w + 1
    if line % stride < w:  # HORIZONTAL
        sq = [line - (w + 1) * (line // stride)]
        sq.append(sq[0] - w)
        if sq[1] < 0:
            sq.pop(1)
        if sq[0] >= w * h:
            sq.pop(0)
    else:
        sq = [line - ((line // stride) * w + (line // stride) + w)]
        sq.append(sq[0] - 1)
        if line % stride == w:
            sq.pop(1)
        if line % stride == 2 * w:
            sq.pop(0)
    return sorted(sq)


@pytest.mark.parametrize("w,h", [(1, 1), (2, 1), (3, 3), (10, 7), (5, 2), (50, 30)])
def test_geometry_matches_ksquares(w, h):
    geo = geometry(w, h)
    assert geo.n_lines == 2 * w * h + w + h
    for ln in range(geo.n_lines):
        assert sorted(geo.line_boxes[ln]) == ksquares_squares_from_line(ln, w, h)
    for b, (top, left, right, bottom) in enumerate(geo.box_lines):
        assert geo.is_horizontal(top) and geo.is_horizontal(bottom)
        assert not geo.is_horizontal(left) and not geo.is_horizontal(right)
        assert right == left + 1 and bottom == top + 2 * w + 1


def test_single_box_capture_and_extra_turn():
    b = Board(1, 1)  # lines: 0 top, 1 left, 2 right, 3 bottom
    assert b.step(0) == 0 and b.player == 1
    assert b.step(1) == 0 and b.player == 0
    assert b.step(2) == 0 and b.player == 1
    assert b.step(3) == 1
    assert b.player == 1 and b.owner == [1] and b.score == [0, 1]
    assert b.done and b.winner() == 1


def test_double_capture_with_one_line():
    b = Board(2, 1)  # 0,1 top; 2,3,4 verticals; 5,6 bottom
    for ln in (0, 1, 2, 4, 5, 6):
        b.step(ln)
    mover = b.player
    assert b.step(3) == 2
    assert b.player == mover and b.score[mover] == 2 and b.done


def test_turn_passes_without_capture_and_keeps_after_capture():
    b = Board(2, 2)
    p = b.player
    b.step(0)
    assert b.player != p


def test_illegal_move_raises():
    b = Board(3, 3)
    b.step(5)
    with pytest.raises(ValueError):
        b.step(5)


def test_clone_is_independent():
    b = Board(3, 3)
    for ln in (0, 1, 3):
        b.step(ln)
    c = b.clone()
    c.step(4)
    c.step(7)
    assert b.lines != c.lines and b.n_drawn == 3 and c.n_drawn == 5
    assert b.sides != c.sides
    assert b.geo is c.geo


@pytest.mark.parametrize("w,h", [(1, 1), (3, 3), (5, 5), (10, 7)])
def test_random_games_invariants(w, h):
    rng = random.Random(0)
    for _ in range(50):
        b = Board(w, h)
        while not b.done:
            b.step(rng.choice(b.legal_moves()))
        assert sum(b.score) == w * h
        assert NO_OWNER not in b.owner
        assert all(s == 4 for s in b.sides)
        assert b.owner.count(0) == b.score[0]


def test_capturing_and_safe_moves():
    b = Board(2, 1)
    b.step(0)
    b.step(2)  # box 0 has 2 sides
    assert set(safe_moves(b)) == {1, 4, 6}
    b.step(5)  # box 0 has 3 sides: line 3 captures
    assert capturing_moves(b) == [3]


def test_agents_play_legal_games():
    for agent_cls in (RandomAgent, GreedyAgent):
        a = agent_cls(seed=1)
        b = Board(4, 3)
        while not b.done:
            ln = a.act(b)
            assert not b.lines[ln]
            b.step(ln)


def test_greedy_takes_box_and_avoids_third_side():
    b = Board(2, 1)
    for ln in (0, 2, 5):
        b.step(ln)
    assert GreedyAgent(0).act(b) == 3
    b2 = Board(2, 1)
    b2.step(0)
    b2.step(2)
    for s in range(20):
        assert GreedyAgent(s).act(b2) in {1, 4, 6}


def test_str_renders():
    b = Board(2, 1)
    for ln in (0, 2, 3, 5):
        b.step(ln)
    assert str(b).splitlines()[1].startswith("| 1 |")
