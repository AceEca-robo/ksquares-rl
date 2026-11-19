"""Series 10 (scripts/graph_s10.py): nets as in the protocol, the strongest-Dot rule, the reading, a tiny match."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import ROOT  # noqa: E402
from graph_s10 import config, reading, strongest, wilson  # noqa: E402


def test_config():
    cfg = config()
    assert cfg["games"] == 200 and cfg["temp_moves"] == 4 and cfg["sims"] == 200
    assert cfg["nets"]["dot54"][1].endswith("az5b/gen_0054.pt")
    import json
    st = json.loads((ROOT / "runs/az5b/state.json").read_text()) if (ROOT / "runs/az5b/state.json").exists() else None
    if st:
        assert cfg["nets"]["dot_last"][1].endswith(f"gen_{max(h['gen'] for h in st['history'] if h['accepted']):04d}.pt")
    assert set(cfg["round_robin"]["players"]) == {"g5_49", "g5_114", "dot54", "dot_last", "hard", "strong"}


def test_wilson():
    lo, hi = wilson(100, 200)
    assert lo < 0.5 < hi
    assert wilson(115, 200)[0] > 0.5 and wilson(113, 200)[0] < 0.5


def test_rule_and_reading():
    assert strongest({}) is None and reading({}) is None
    even = [100, 100, 0, 0]
    assert strongest({"1:mcts": even, "2:mcts": even}) == "dot54"
    assert reading({"1:mcts": even, "2:mcts": even}).startswith("no Dot beats")
    assert strongest({"1:mcts": [130, 70, 0, 0], "2:mcts": [120, 80, 0, 0]}) == "dot_last"
    assert strongest({"1:mcts": [118, 82, 0, 0], "2:mcts": [130, 70, 0, 0]}) == "dot_raw"
    assert reading({"1:mcts": [130, 70, 0, 0], "2:mcts": even}).startswith("the late Dot beats")
    assert "best-raw Dot does" in reading({"1:mcts": even, "2:mcts": [130, 70, 0, 0]})


def test_play_pair_tiny():
    from multiprocessing import get_context

    import torch

    import az_eval
    from agents.az_net import Evaluator, make_net
    from graph_s10 import play_pair

    torch.manual_seed(0)
    net = make_net({"net": {"arch": "graph", "layers": 1, "width": 8, "pool": 4, "segments": True}})
    ev = Evaluator(net, "cpu")
    cfg = {"board": [2, 2], "temp_moves": 2, "mcts": {"c_puct": 1.5, "fpu_reduction": 0.2, "dirichlet_alpha": 0.3,
           "dirichlet_eps": 0.25, "virtual_loss": 1.0, "leaves_per_tree": 8}}
    with get_context("spawn").Pool(1, initializer=az_eval._oinit, initargs=({}, 0)) as pool:
        r = play_pair([{"ev": ev, "sims": 8}, {"agent": "hard"}], 6, cfg, 1, pool)
        assert sum(r[:3]) == 6
        r2 = play_pair([{"ev": ev, "sims": 0}, {"ev": ev, "sims": 8}], 4, cfg, 1, pool)
        assert sum(r2[:3]) == 4
