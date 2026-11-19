"""Series 11 (scripts/graph_s11.py): candidates as in the protocol, the reading rule."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import ROOT  # noqa: E402
from graph_s11 import config, verdict  # noqa: E402


def test_config():
    cfg = config()
    assert cfg["board"] == [10, 7] and cfg["games"] == 200 and cfg["sims"] == 200 and cfg["temp_moves"] == 4
    assert cfg["nets"]["dot2c_16"][1].endswith("dot2c/gen_0016.pt") and "dot2c_16" not in cfg["matches"]
    p = ROOT / "runs/dot2c/state.json"
    if p.exists():
        acc = [h["gen"] for h in json.loads(p.read_text())["history"] if h["accepted"]]
        assert acc[-1] == 16 and {int(m.split("_")[1]) for m in cfg["matches"] if m.startswith("dot2c")} <= set(acc)


def test_verdict():
    cfg = {"matches": ["a_1", "b_2"]}
    even = [100, 100, 0, 0]
    assert verdict(cfg, {})[1] is None
    assert verdict(cfg, {"a_1": even, "b_2": even})[1] is None
    assert verdict(cfg, {"a_1": [70, 130, 0, 0], "b_2": [80, 120, 0, 0]})[1] == "a_1"
    assert verdict(cfg, {"a_1": [90, 110, 0, 0], "b_2": even})[1] is None
