"""Mid-run checkpoints of scripts/graph_hand.py: stop + resume equals an uninterrupted run, bit for bit."""

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from graph_hand import load_checkpoint, save_checkpoint  # noqa: E402


def _train(tmp, stop_at=None, steps=40):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.3), torch.nn.Linear(8, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(1)
    path = tmp / "run.ckpt"
    start, st = load_checkpoint(path, net, opt, sched, rng)
    trace = st["trace"] if st else []
    for step in range(start + 1, steps + 1):
        x = torch.from_numpy(rng.normal(size=(16, 4)).astype(np.float32)) + random.random()
        loss = net(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        trace.append(loss.item())
        if step % 10 == 0:
            save_checkpoint(path, step, net, opt, sched, rng, {"trace": trace})
            if step == stop_at:
                return None
    return trace, net.state_dict()


def test_resume_is_bit_exact(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    full = _train(a)
    assert _train(b, stop_at=20) is None
    assert not list(b.glob("*.tmp"))
    res = _train(b)
    assert res[0] == full[0]
    assert all(torch.equal(res[1][k], full[1][k]) for k in full[1])
