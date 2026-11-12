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
