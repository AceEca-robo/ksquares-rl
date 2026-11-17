"""Series 7 (scripts/graph_s7.py): net list as in the protocol, reading rule, bootstrap interval."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import ROOT, load_config  # noqa: E402
from graph_s7 import best_moves, interval, net_list, reading  # noqa: E402


def test_nets():
    cfg = load_config(ROOT / "configs/graph_s7.yaml")
    L = net_list(cfg)
    by = {}
    for g, name, _, arm, path in L:
        by.setdefault(g, []).append(name)
        assert arm == ("Ch" if g == "ch" else "Gc")
    assert len(by["gc_jump"]) == 9 and len(by["gc_nojump"]) == 5 and len(by["ch"]) == 5
    # the jump group = every series 6 run at >= 80%, the no-jump group none of them
    import json
    jump = set()
    for w in (8, 9, 6, 11):
        for t in range(101, 106):
            p = ROOT / f"runs/graph_s6/w{w}/s2_Gc_s{t}.json"
            if p.exists() and json.loads(p.read_text())["best_val_10x7"] >= 80:
                jump.add(f"Gc w{w}/{t}")
    if jump:
        assert set(by["gc_jump"]) == jump and not set(by["gc_nojump"]) & jump


def test_reading():
    assert reading(2.0, 0.5, 3.5).startswith("the jump carries")
    assert reading(1.0, -0.5, 2.5) == "no transfer to the middle game"
    assert reading(-3.0, -5.0, -1.0).startswith("no transfer; Gc with a jump is below")


def test_interval():
    rng = np.random.default_rng(0)
    b = (rng.random(500) < 0.5).astype(int)
    d, lo, hi = interval(np.stack([b, b]), b, 200, 1, 0.95)
    assert d == 0 and lo == 0 and hi == 0
    a = np.ones((3, 500), int)
    d, lo, hi = interval(a, b, 200, 1, 0.95)
    assert lo < d < hi and lo > 0


def test_best_moves():
    assert best_moves({"labels": {"3": -1, "5": 1, "7": 1, "9": 0}}) == [5, 7]
