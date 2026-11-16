"""Series 6 (scripts/graph_s6.py): config = series 5 apart from the series6 block, job list, reading rule."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import ROOT, load_config  # noqa: E402
from graph_s6 import groups, job_list, reading  # noqa: E402


def cfgs():
    return load_config(ROOT / "configs/graph_s6.yaml"), load_config(ROOT / "configs/graph_s5.yaml")


def test_config_is_series5():
    cfg, ref = cfgs()
    assert set(cfg) - set(ref) == {"series6"} and set(ref) <= set(cfg)
    for k in ref:
        if k not in ("out", "report", "jobs"):       # jobs: slots at once, not a training value
            assert cfg[k] == ref[k], k
    assert cfg["out"] == "runs/graph_s6" and cfg["series6"]["from"] == ref["out"]


def test_jobs():
    cfg, _ = cfgs()
    jobs = job_list(cfg)
    assert groups(cfg) == [("jump", 8), ("jump", 9), ("no_jump", 6), ("no_jump", 11)]
    sub, w, t = jobs[0]                                  # the control: series 5 names, series 5 seed
    assert (sub["out"], w, t) == ("runs/graph_s6/control", 8, 8) and sub["parts"]["a"]["seeds"] == [8]
    rest = [(s["out"], w, t) for s, w, t in jobs[1:]]
    assert len(rest) == 20 == len(set(rest))
    assert rest[:4] == [(f"runs/graph_s6/w{w}", w, 101) for w in (8, 9, 6, 11)]
    assert {(w, t) for _, w, t in rest} == {(w, t) for w in (8, 9, 6, 11) for t in range(101, 106)}
    for s, w, t in jobs:                                 # training keys are the series config, untouched
        assert all(s[k] == cfg[k] for k in ("stage2", "recipe", "arms", "limits", "seed", "data_seed", "n_train", "n_val",
                                             "parity_config", "device"))
        assert s["reuse_stage1"] is None


def test_reading():
    cfg, _ = cfgs()
    s6 = cfg["series6"]
    r = lambda a, b, na=10, nb=10: reading(a, b, na, nb, s6)  # noqa: E731
    assert "no reading" in r(5, 0, 9, 10)
    for a, b in ((8, 2), (10, 0), (9, 1), (8, 0)):
        assert "stage 1 weights decide" in r(a, b)
    for a, b in ((5, 5), (5, 3), (3, 5), (0, 0), (10, 8), (2, 4), (9, 7)):
        assert "course of stage 2" in r(a, b)
    for a, b in ((7, 2), (8, 3), (7, 4), (2, 8), (10, 7), (3, 0), (9, 3)):
        assert "decision is the user's" in r(a, b)
