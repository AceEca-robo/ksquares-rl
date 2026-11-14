"""Bootstrap of scripts/graph_s3.py: sign, bound below the difference, deterministic."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from graph_s3 import bootstrap  # noqa: E402


def test_bootstrap():
    rng = np.random.default_rng(0)
    a = (rng.random((5, 300)) < 0.8).astype(int)
    b = (rng.random((5, 300)) < 0.6).astype(int)
    d, lo = bootstrap(a, b, 2000, 1, 0.95)
    assert 15 < d < 25 and 0 < lo < d
    assert (d, lo) == bootstrap(a, b, 2000, 1, 0.95)
    d2, lo2 = bootstrap(b, b, 2000, 1, 0.95)
    assert d2 == 0 and lo2 < 0


def test_keep_share_pattern_and_sign_test():
    import math

    from graph_s2pp import sign_test

    for share, every in ((0.10, 10), (0.25, 4)):
        hits = [s for s in range(1, 401) if math.floor(s * share) > math.floor((s - 1) * share)]
        assert len(hits) == 400 // every and all(h % every == 0 for h in hits)
    assert sign_test([1] * 5 + [0] * 5, [0] * 10) == (5, 0, 1 / 32)


def test_series3p_stage0_share_steps():
    """Series 3': 40000 steps at share 0.25 = 30000 policy steps; policy step 5500 is the switch."""
    import math

    share, steps = 0.25, 40000
    s0 = [s for s in range(1, steps + 1) if math.floor(s * share) > math.floor((s - 1) * share)]
    assert len(s0) == 10000 and all(s % 4 == 0 for s in s0)
    pol = lambda s: s - math.floor(s * share)  # noqa: E731
    assert pol(steps) == 30000
    assert max(s for s in range(1, steps + 1) if pol(s) <= 5500) == 7333


def test_series5_snapshot_steps_and_verdict():
    """Series 5: kept weights = steps 250, 500, 1000 + validations 7000-14000; nothing without the option."""
    from _common import ROOT, load_config
    from graph_s3 import snapshot_steps
    from graph_s5 import verdict

    cfg = load_config(ROOT / "configs/graph_s5.yaml")
    ref = load_config(ROOT / "configs/graph_s3p.yaml")
    steps = snapshot_steps(cfg["stage2"])
    assert steps == {250, 500, 1000} | set(range(7000, 14001, 500)) and len(steps) == 18
    assert snapshot_steps(ref["stage2"]) == set()
    # training is series 3' unchanged: stage 1 keys and stage 2 without the snapshots option
    for k in ("recipe", "steps", "log_every", "pass_acc", "n_train", "n_val", "data_seed", "seed", "parity_config"):
        assert cfg[k] == ref[k], k
    assert {k: v for k, v in cfg["stage2"].items() if k != "snapshots"} == ref["stage2"]
    assert cfg["arms"]["Gc"] == ref["arms"]["Gc"] and cfg["parts"]["a"]["mix_steps"] == ref["parts"]["a"]["mix_steps"]
    assert cfg["parts"]["a"]["seeds"] == list(range(6, 26)) and cfg["parts"]["a"]["arms"] == ["Gc"]
    cr = cfg["criterion"]
    assert "reproducible" in verdict(5, 20, 20, cr) and "chance" in verdict(1, 20, 20, cr) and "chance" in verdict(0, 20, 20, cr)
    assert "user" in verdict(2, 20, 20, cr) and "user" in verdict(4, 20, 20, cr) and "no verdict" in verdict(7, 19, 20, cr)
