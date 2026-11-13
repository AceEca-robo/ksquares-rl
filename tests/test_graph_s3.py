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
