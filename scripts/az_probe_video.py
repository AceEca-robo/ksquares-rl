"""Video of the probe snapshots over the generations (docs/probes).

Every docs/probes/{run}/gen_NN.png of the runs in configs/strategy_probe.yaml,
in run order then generation order, becomes one frame of an H.264 mp4
(``--seconds`` per generation, the last one held longer): how the policy,
the value and the MCTS move on the fixed probe positions change while the
net learns.

    python scripts/az_probe_video.py
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from _common import ROOT, load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-config", default=str(ROOT / "configs/strategy_probe.yaml"))
    ap.add_argument("--seconds", type=float, default=0.8, help="per generation")
    ap.add_argument("--hold", type=float, default=3.0, help="last frame")
    ap.add_argument("--out", default=str(ROOT / "docs/probes/dynamics.mp4"))
    a = ap.parse_args()
    cfg = load_config(a.probe_config)
    out = ROOT / cfg["maps"]["out"]
    frames = []
    for r in cfg["runs"]:
        frames += sorted((out / r["name"]).glob("gen_*.png"), key=lambda p: int(p.stem.split("_")[1]))
    if not frames:
        raise SystemExit("no snapshots")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for i, p in enumerate(frames):
            f.write(f"file '{p}'\nduration {a.hold if i == len(frames) - 1 else a.seconds}\n")
        f.write(f"file '{frames[-1]}'\n")  # concat demuxer: the last duration needs a following entry
        lst = f.name
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", "-r", "10",
                    "-c:v", "libx264", "-crf", "28", a.out], check=True)
    Path(lst).unlink()
    print(f"{a.out}: {len(frames)} frames ({frames[0].parent.name}/{frames[0].stem} .. "
          f"{frames[-1].parent.name}/{frames[-1].stem})")


if __name__ == "__main__":
    main()
