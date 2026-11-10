"""Series 2': bins, candidate rules, run metrics and the verdict rule."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import graph_s2p  # noqa: E402


def test_bins():
    assert [graph_s2p.bin_long(n) for n in (0, 1, 2, 3, 7)] == ["0", "1", "2", "3+", "3+"]
    assert [graph_s2p.bin_len(n) for n in (0, 3, 5, 6, 9, 11, 12)] == ["0", "3-5", "3-5", "6-8", "9-11", "9-11", "12+"]
    assert [graph_s2p.bin_loops(n) for n in (0, 1, 2, 4)] == ["0", "1", "2+", "2+"]


def test_true_rule_is_long_parity():
    from graph_hand import settled_boards
    for b, t in settled_boards(5, 5, 50, random.Random(3)):
        f = graph_s2p.features(b)
        assert (1.0 if graph_s2p.rule_count(f, "long") % 2 == f["player"] else -1.0) == t


def log(*vals):
    return [{"step": 250 * i, "val": {"5x5": {"acc": a}, "10x7": {"acc": b}}} for i, (a, b) in enumerate(vals)]


def test_run_rows():
    cfg = {"hold_acc": 90.0}
    r = {"switch": 250, "log": log((50, 50), (96, 50), (80, 70), (91, 96), (92, 97))}
    a = graph_s2p.run_rows(cfg, [r], 500)[0]
    assert a["hold5"] is True and a["both"] is True
    b = graph_s2p.run_rows(cfg, [r], 0)[0]
    assert b["hold5"] is False


def c(k, n=10):
    return {"click10": k, "n": n}


def test_decide():
    assert "series 3" in graph_s2p.decide({}, {"a": c(8), "cC": c(0, 5), "cCh": c(0, 5)})[-2]
    assert "2''" in graph_s2p.decide({"_sep": True}, {"a": c(5), "cC": c(0, 5), "cCh": c(0, 5)})[-2]
    assert "stop" in graph_s2p.decide({}, {"a": c(5), "cC": c(0, 5), "cCh": c(0, 5)})[-2]
    assert "series 3" in graph_s2p.decide({}, {"a": c(2), "cC": c(3, 5), "cCh": c(0, 5)})[-2]
