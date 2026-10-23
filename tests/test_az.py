import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from agents.az_mcts import MCTS, choose_move, result_for_mover  # noqa: E402
from agents.az_net import N_PLANES, AZNet, Codec, Evaluator  # noqa: E402
from sim.board import Board, geometry  # noqa: E402


def random_board(w, h, n, rng):
    b = Board(w, h)
    for _ in range(n):
        if b.done:
            break
        b.step(rng.choice(b.legal_moves()))
    return b


def test_codec_cells_match_geometry():
    c = Codec(5, 5)
    geo = geometry(5, 5)
    assert len(c.syms) == 8
    for i, lines in enumerate(geo.box_lines):
        r, col = divmod(int(c.box_cell[i]), c.cols)
        around = {(r - 1, col), (r, col - 1), (r, col + 1), (r + 1, col)}
        assert {tuple(c.line_rc[ln]) for ln in lines} == around


@pytest.mark.parametrize("wh", [(5, 5), (3, 3), (4, 2)])
def test_symmetries_map_boxes_to_boxes(wh):
    c = Codec(*wh)
    geo = geometry(*wh)
    boxes = {frozenset(bl) for bl in geo.box_lines}
    assert len(c.syms) == (8 if wh[0] == wh[1] else 4)
    for _, lperm in c.syms:
        assert sorted(lperm) == list(range(geo.n_lines))
        inv = np.argsort(lperm)  # original line -> transformed line
        assert {frozenset(int(inv[ln]) for ln in bl) for bl in geo.box_lines} == boxes


def test_encode_commutes_with_symmetry():
    rng = random.Random(1)
    c = Codec(5, 5)
    for _ in range(20):
        b = random_board(5, 5, rng.randrange(60), rng)
        x = c.encode_boards([b]).reshape(1, N_PLANES, -1)
        for cperm, lperm in c.syms:
            t = Board(5, 5)  # same drawn lines, transformed (owners ignored)
            for k in range(60):
                if b.lines[lperm[k]]:
                    t.lines[k] = 1
                    for bx in t.geo.line_boxes[k]:
                        t.sides[bx] += 1
            y = c.encode_boards([t]).reshape(1, N_PLANES, -1)
            for p in (0, 1, 10):
                assert np.array_equal(x[0, p, cperm], y[0, p])
            free = [i for i in range(25) if b.owner[i] < 0]
            for s in range(4):  # side counts of free boxes
                xs = x[0, 4 + s, cperm][c.box_cell]
                assert xs.sum() == sum(1 for i in free if b.sides[i] == s)


def test_net_shapes_and_mask():
    net = AZNet(5, 5, blocks=1, channels=8)
    assert net(torch.zeros(2, N_PLANES, 11, 11))[0].shape == (2, 60)
    ev = Evaluator(net, "cpu")
    b = random_board(5, 5, 30, random.Random(2))
    pri, val = ev([b, Board(5, 5)])
    assert pri.shape == (2, 60) and val.shape == (2,)
    assert np.allclose(pri.sum(1), 1, atol=1e-5)
    assert pri[0][np.frombuffer(bytes(b.lines), dtype=np.uint8) == 1].max() < 1e-6
    assert np.all(np.abs(val) <= 1)


def exact_result(board):
    """+1 / -1 / 0: game result for the side to move with perfect play."""
    v = result_for_mover(board)
    if v is not None:
        return v
    best = -2.0
    for ln in board.legal_moves():
        b = board.clone()
        b.step(ln)
        r = exact_result(b)
        best = max(best, r if b.player == board.player else -r)
        if best == 1:
            break
    return best


def rollout(boards, rng=random.Random(5)):
    """Uniform priors, value = result of one random playout (pure MCTS)."""
    vals = []
    for b in boards:
        c = b.clone()
        while result_for_mover(c) is None:
            c.step(rng.choice(c.legal_moves()))
        r = result_for_mover(c)
        vals.append(r if c.player == b.player else -r)
    return np.full((len(boards), boards[0].geo.n_lines), 1.0), np.array(vals)


@pytest.mark.parametrize("leaves", [1, 4])
def test_mcts_finds_exact_result_on_2x2(leaves):
    """Pure MCTS (random playouts) converges to minimax; the root value must
    have the sign of the exact result (extra turns included)."""
    rng = random.Random(3)
    checked = 0
    while checked < 25:
        b = random_board(2, 2, rng.randrange(5, 9), rng)
        if result_for_mover(b) is not None:
            continue
        ex = exact_result(b)
        if ex == 0:
            continue
        root = MCTS(rollout, fpu_reduction=0.0, leaves_per_tree=leaves, rng=np.random.default_rng(0)).search(
            [b], 3000)[0]
        if len(root.moves) == 1:
            continue
        q = root.W[np.argmax(root.N)] / root.N.max()
        assert np.sign(q) == ex, (str(b), q, ex)
        if ex == 1:  # the chosen move does not lose (averaging may settle for a tie)
            c = b.clone()
            c.step(choose_move(root, 0, np.random.default_rng(0)))
            r = exact_result(c)
            assert (r if c.player == b.player else -r) >= 0
        checked += 1


def test_selfplay_and_train_smoke(tmp_path):
    from agents.az_play import match, selfplay

    cfg = {"board": [2, 2], "device": "cpu", "seed": 0,
           "mcts": {"c_puct": 1.5, "fpu_reduction": 0.2, "dirichlet_alpha": 0.3, "dirichlet_eps": 0.25,
                    "virtual_loss": 1.0},
           "selfplay": {"sims": 16, "temp_moves": 3},
           "train": {"batch_size": 16, "symmetries": True}}
    ev = Evaluator(AZNet(2, 2, blocks=1, channels=8), "cpu")
    samples, stats = selfplay(ev, cfg, n_games=6, concurrent=4, seed=0)
    assert stats["games"] == 6 and len(samples["z"]) == stats["positions"] > 0
    assert set(np.unique(samples["z"])) <= {-1, 0, 1}
    assert np.allclose(samples["pi"].astype(np.float32).sum(1), 1, atol=1e-2)
    w, l, t, _ = match([ev, ev], cfg, sims=8, temp_moves=2, n_games=4, concurrent=3, seed=0)
    assert w + l + t == 4

    import sys
    sys.path.insert(0, "scripts")
    from az_train import Buffer, train_steps

    codec = Codec(2, 2)
    buf = Buffer(100, codec.n_lines, codec.n_boxes)
    buf.add(samples)
    net = AZNet(2, 2, blocks=1, channels=8)
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    loss = train_steps(net, opt, buf, codec, cfg, np.random.default_rng(0), 2)
    assert np.isfinite(loss["policy"]) and np.isfinite(loss["value"])
    buf.save(tmp_path / "b.npz")
    b2 = Buffer(100, codec.n_lines, codec.n_boxes)
    b2.load(tmp_path / "b.npz")
    assert b2.n == buf.n and np.array_equal(b2.a["pi"], buf.a["pi"])
