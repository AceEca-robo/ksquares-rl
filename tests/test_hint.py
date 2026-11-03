"""Part 3 (b): chain hint planes and widening a Dot-2 net."""

import random

import numpy as np
import torch

from agents.az_net import N_PLANES_3, AZNet2, Codec, widen_from
from agents.baselines import safe_moves
from sim.board import Board
from sim.chains import analyse


def random_boards(w, h, n, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        b = Board(w, h)
        for _ in range(rng.randrange(b.geo.n_lines)):
            if b.done:
                break
            b.step(rng.choice([ln for ln in range(b.geo.n_lines) if not b.lines[ln]]))
        out.append(b)
    return out


def test_chain_hint_matches_analyse():
    for w, h in ((5, 5), (10, 7), (3, 2)):
        c = Codec(w, h, N_PLANES_3)
        boards = random_boards(w, h, 300, seed=w * h)
        lines = np.array([list(b.lines) for b in boards], dtype=np.uint8)
        sides = np.array([list(b.sides) for b in boards], dtype=np.uint8)
        chain_len, loop_len, long_chains, safe = c.chain_hint(lines, sides)
        for i, b in enumerate(boards):
            a = analyse(b)
            heads = sorted(chain_len[i][chain_len[i] > 0].tolist(), reverse=True)
            # every box of a chain carries its length: k boxes of length k
            got = []
            for k in sorted(set(heads), reverse=True):
                got += [k] * (heads.count(k) // k)
            assert got == a["chains"], (i, got, a["chains"])
            lp = sorted(loop_len[i][loop_len[i] > 0].tolist(), reverse=True)
            gl = []
            for k in sorted(set(lp), reverse=True):
                gl += [k] * (lp.count(k) // k)
            assert gl == a["loops"]
            assert long_chains[i] == a["long"]
            assert safe[i] == len(safe_moves(b))


def test_widen_keeps_outputs():
    torch.manual_seed(0)
    small = AZNet2(6, 64).eval()
    for m in small.modules():  # non-trivial BatchNorm statistics and pooling biases
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2)
            m.weight.data.uniform_(0.5, 1.5)
    for b in small.body:
        if hasattr(b, "g"):
            b.g.weight.data.normal_(0, 0.05)
    big = AZNet2(6, 96, gpool_blocks="all", planes=N_PLANES_3).eval()
    widen_from(big, small.state_dict())
    boards = random_boards(10, 7, 16, seed=3)
    x3 = torch.from_numpy(Codec(10, 7, N_PLANES_3).encode_boards(boards))
    with torch.no_grad():
        p0, v0, s0 = small(x3[:, :12])
        p1, v1, s1 = big(x3)
    assert torch.allclose(p0, p1, atol=1e-4) and torch.allclose(v0, v1, atol=1e-5) and torch.allclose(s0, s1, atol=1e-5)


def test_hint_head_widened_is_identical_and_value_blind():
    """a3: Dot-2 widened into a hint-head net plays as Dot-2; V and the
    score never depend on the hint planes, even once the hint path is trained."""
    torch.manual_seed(0)
    old = AZNet2(blocks=2, channels=16, head_ch=8).eval()
    for m in old.modules():  # non-trivial BatchNorm statistics
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2)
    new = AZNet2(blocks=2, channels=16, head_ch=8, planes=N_PLANES_3, hint_head=True).eval()
    widen_from(new, old.state_dict())
    c = Codec(5, 5, N_PLANES_3)
    x = torch.from_numpy(c.encode_boards(random_boards(5, 5, 50, seed=3)))
    with torch.no_grad():
        p0, v0, s0 = old(x[:, :12])
        p1, v1, s1 = new(x)
        assert torch.equal(v0, v1) and torch.equal(s0, s1)
        assert torch.allclose(p0, p1, atol=1e-6)
        torch.nn.init.normal_(new.p_hint[2].weight)
        p2, v2, s2 = new(x)
        assert torch.equal(v0, v2) and torch.equal(s0, s2)
        assert not torch.allclose(p0, p2, atol=1e-3)
