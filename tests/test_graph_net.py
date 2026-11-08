"""GraphNet: segments as sim/chains.py, equivariance, Evaluator, budget."""

import random

import numpy as np
import pytest
import torch

from agents.az_net import Codec, Evaluator
from agents.graph_net import GraphNet
from sim.board import Board
from sim.chains import analyse


def random_board(w, h, n, rng):
    b = Board(w, h)
    for _ in range(n):
        if b.done:
            break
        b.step(rng.choice(b.legal_moves()))
    return b


def test_segments_match_sim_chains():
    rng = random.Random(1)
    for w, h, top in [(5, 5, 60), (10, 7, 142), (3, 3, 24)]:
        c = Codec(w, h, 12)
        boards = [random_board(w, h, rng.randint(0, top - 1), rng) for _ in range(200)]
        x = torch.from_numpy(c.encode_boards(boards))
        t = GraphNet(1, 8, 2).topo(x.shape[2], x.shape[3])
        lines = torch.tensor(np.array([list(b.lines) for b in boards]))
        sides = torch.tensor(np.array([list(b.sides) for b in boards]))
        lab, f = GraphNet.find_segments(t, sides.long(), lines == 0)
        for i, b in enumerate(boards):
            a = analyse(b)
            size = (f[i, :, 0] * 10).round().long()
            loop = f[i, :, 1] > 0
            assert sorted(size[(size > 0) & ~loop].tolist(), reverse=True) == a["chains"]
            assert sorted(size[loop].tolist(), reverse=True) == a["loops"]
            assert int(f[i, :, 4].sum()) == a["long"]


@pytest.mark.parametrize("segments", [True, False])
def test_equivariant(segments):
    torch.manual_seed(0)
    net = GraphNet(3, 32, 4, segments).eval()
    rng = random.Random(2)
    c = Codec(5, 5, 12)
    x = c.encode_boards([random_board(5, 5, n, rng) for n in (10, 30, 45)])
    B, P = x.shape[:2]
    with torch.no_grad():
        p0, v0, _ = net(torch.from_numpy(x))
        for cperm, lperm in c.syms:
            xs = x.reshape(B, P, -1)[:, :, cperm].reshape(x.shape)
            p, v, _ = net(torch.from_numpy(xs))
            assert torch.allclose(p, p0[:, torch.as_tensor(lperm)], atol=1e-3, rtol=1e-3)
            assert torch.allclose(v, v0, atol=1e-4)


def test_evaluator_mixed_sizes():
    net = GraphNet(2, 16, 4)
    ev = Evaluator(net, "cpu")
    pri, val = ev([Board(5, 5), Board(10, 7)])
    assert len(pri[0]) == Board(5, 5).geo.n_lines and len(pri[1]) == Board(10, 7).geo.n_lines
    assert abs(sum(pri[0]) - 1) < 1e-4 and val.shape == (2,)


@pytest.mark.parametrize("cfg", ["configs/graph_gc.yaml", "configs/graph_ga.yaml", "configs/graph_gc16.yaml"])
def test_budget(cfg):
    from agents.az_net import make_net
    from scripts._common import load_config

    n = sum(p.numel() for p in make_net(load_config(cfg)).parameters())
    assert 475_000 <= n <= 525_000, n


def test_series1_options():
    """v_out linear returns the raw V (= v_pre), tanh its tanh; readout_init
    small gives a near-zero V at start; readout_norm keeps shapes."""
    rng = random.Random(3)
    c = Codec(5, 5, 12)
    x = torch.from_numpy(c.encode_boards([random_board(5, 5, n, rng) for n in (5, 20, 40)]))
    for kw in ({}, {"v_out": "linear"}, {"readout_norm": True}, {"readout_init": 1e-3}):
        torch.manual_seed(0)
        net = GraphNet(2, 16, 4, **kw).eval()
        with torch.no_grad():
            p, v, s = net(x)
        assert v.shape == (3,) and p.shape[0] == 3
        if kw.get("v_out") == "linear":
            assert torch.equal(v, net.v_pre)
        else:
            assert torch.allclose(v, torch.tanh(net.v_pre))
        if "readout_init" in kw:
            assert v.abs().max() < 0.05
