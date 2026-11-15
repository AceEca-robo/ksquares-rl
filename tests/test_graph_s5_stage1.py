"""compare_stage1 of scripts/graph_s5.py: equal runs compare equal (wall time aside), one changed weight or log entry is found."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from graph_s5 import compare_stage1  # noqa: E402


def write(d, w, loss, seconds):
    d.mkdir()
    torch.save({"a": torch.ones(3), "b": torch.full((2,), w)}, d / "a_Gc_s8.pt")
    log = [{"step": s, "loss": loss if s == 500 else 0.5, "seconds": seconds * s,
            "val": {"5x5": {"acc": 99.0}, "10x7": {"acc": 50.0 + s / 100}}} for s in (0, 250, 500)]
    (d / "a_Gc_s8.json").write_text(json.dumps({"log": log, "switch": 250, "click5": 250, "click10": None, "hold10": None,
                                                "last_5x5": 99.0, "last_10x7": 55.0}))


def test_compare_stage1(tmp_path):
    write(tmp_path / "ref", 1.0, 0.5, 1.0)
    write(tmp_path / "same", 1.0, 0.5, 2.0)          # only the wall time differs
    write(tmp_path / "w", 1.0 + 1e-6, 0.5, 1.0)
    write(tmp_path / "log", 1.0, 0.25, 1.0)
    r = compare_stage1(tmp_path / "ref", tmp_path / "same", "a_Gc_s8")
    assert r["weights_equal"] and r["file_bytes_equal"] and r["log_equal"] and r["summary_equal"] and r["max_abs_diff"] == 0
    assert r["val_ref"] == r["val_new"] and r["val_ref"][2] == [500, 99.0, 55.0]
    r = compare_stage1(tmp_path / "ref", tmp_path / "w", "a_Gc_s8")
    assert not r["weights_equal"] and r["tensors_differ"] == 1 and 0 < r["max_abs_diff"] < 1e-5 and r["log_equal"]
    r = compare_stage1(tmp_path / "ref", tmp_path / "log", "a_Gc_s8")
    assert r["weights_equal"] and not r["log_equal"] and r["first_log_diff_step"] == 500
