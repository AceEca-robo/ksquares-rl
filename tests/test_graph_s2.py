"""Series 2: CNN linear V, Fisher test and the 2d diagnosis rule."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from agents.az_net import AZNet2, Codec  # noqa: E402

import graph_s2  # noqa: E402


def test_aznet2_v_out():
    x = torch.from_numpy(Codec(3, 3, planes=12).encode_boards([__import__("sim.board").board.Board(3, 3)]))
    torch.manual_seed(0)
    a = AZNet2(blocks=2, channels=8, gpool_blocks=(1,), v_out="linear").eval()
    torch.manual_seed(0)
    b = AZNet2(blocks=2, channels=8, gpool_blocks=(1,)).eval()
    va, vb = a(x)[1], b(x)[1]
    assert va.item() == pytest.approx(a.v_pre.item())
    assert vb.item() == pytest.approx(torch.tanh(va).item())


def test_fisher():
    one, two = graph_s2.fisher(8, 10, 2, 10)
    assert one == pytest.approx(0.01151, abs=1e-4)
    assert two == pytest.approx(0.02301, abs=1e-4)


def row(loss, grad):
    return {"step": 20000, "train_on": "10x7", "loss": loss, "readout_grad": grad, "val": {"10x7": {"sat": 0.5}}}


@pytest.mark.parametrize("loss,grad,cls", [(0.1, 1e-5, "fits train without the rule"), (0.69, 1e-5, "frozen"),
                                           (0.69, 0.5, "does not find the rule")])
def test_diagnose(loss, grad, cls):
    cfg = {"diagnosis": {"grad_frozen": 1e-3, "loss_fit": 0.3}}
    r = {"switch": 500, "steps": 20000, "log": [{"step": 0}, row(loss, grad)]}
    assert graph_s2.diagnose(cfg, r)[0] == cls


def test_diagnose_no_switch():
    assert graph_s2.diagnose({}, {"switch": None})[0] == "no 10x7 phase"
