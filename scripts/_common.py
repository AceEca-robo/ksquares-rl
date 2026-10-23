"""Shared helpers for scripts: repo root on sys.path, config loading."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
