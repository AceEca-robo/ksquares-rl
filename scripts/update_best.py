"""Point runs/best/best.pt at the last accepted generation of the run in
configs/app.yaml (``best``), or at ``best.gen`` when that pins one, and
describe it in runs/best/best.json.

    python scripts/update_best.py

Called by scripts/az_eval.py after every evaluated generation and by the
app at start-up; the app loads runs/best/best.pt with the run's config.
"""

from __future__ import annotations

import argparse
import json
import os

from _common import ROOT, load_config


def update(app_cfg: dict) -> dict | None:
    bc = app_cfg["best"]
    rc = load_config(ROOT / bc["run_config"])
    run = ROOT / rc["run_dir"]
    try:
        st = json.loads((run / "state.json").read_text())
    except OSError:
        return None
    gens = [h["gen"] for h in st["history"] if h["accepted"] and (run / f"gen_{h['gen']:04d}.pt").exists()]
    if bc.get("gen") is not None:
        # pinned: the chosen generation, whatever is accepted after it
        gens = [g for g in gens if g == bc["gen"]]
    if not gens:
        return None
    d = ROOT / bc["dir"]
    d.mkdir(parents=True, exist_ok=True)
    info = {"run": run.name, "gen": gens[-1], "config": bc["run_config"],
            "checkpoint": os.path.relpath(run / f"gen_{gens[-1]:04d}.pt", d)}
    link = d / "best.pt"
    tmp = d / "best.pt.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(info["checkpoint"])
    os.replace(tmp, link)
    (d / "best.json.tmp").write_text(json.dumps(info, indent=1))
    os.replace(d / "best.json.tmp", d / "best.json")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/app.yaml"))
    a = ap.parse_args()
    print(update(load_config(a.config)))


if __name__ == "__main__":
    main()
