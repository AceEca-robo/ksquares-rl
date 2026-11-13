"""Series 3' data checks (user, 2026-11-13), no training:

    python scripts/graph_s3p_checks.py     # writes the "Data checks" section of docs/graph_s3p.md

1. Overlap of the stage 0 positions (series 2 data, 10x7: the auxiliary train set and its validation)
   with the judge's 10x7 positions (stage 2 train, validation, probe), by board hash under the grid
   symmetries: (a) drawn lines only, (b) the full input encoding (lines, boxes, owners, score, player).
2. The judge's 10x7 validation of series 3 and series 3' is the same: same file, same positions (hash of
   the split from each config), same metric (validate() of scripts/graph_s3.py recomputed from saved
   weights against the logged values).
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from _common import ROOT, load_config

import graph_s3
from graph_hand import judge_sets, probe_sets
from graph_s2 import splits

MARK = "## Data checks"


def canon(x, codec, lines_only):
    """Canonical bytes per position over the grid symmetries. x: (N, planes, cells)."""
    keys = []
    for cperm, _ in codec.syms:
        y = x[:, :, cperm]
        y = (y[:, 1, codec.line_cell] > 0) if lines_only else y
        keys.append([hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest() for r in y.reshape(len(y), -1)])
    return [min(k) for k in zip(*keys)]


def file_sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def positions_sha(ps):
    return hashlib.sha256(json.dumps([p["moves"] for p in ps]).encode()).hexdigest()[:16]


def overlap(cfg):
    from agents.az_net import Codec

    planes = 17
    z = np.load(ROOT / cfg["out"] / "data.npz")
    s0 = splits(cfg, {k: z[k] for k in z.files})["10x7"]
    a = {"stage 0 train (aux)": s0[0][:, :planes], "stage 0 validation": s0[2][:, :planes]}
    _, sp = judge_sets(cfg)
    tr, va = sp["10x7"]
    import parity_hand

    b = {"judge train": parity_hand.parity_arrays("10x7", tr, planes)[1],
         "judge validation": parity_hand.parity_arrays("10x7", va, planes)[1],
         "judge probe": probe_sets(cfg, planes)["10x7"][1]}
    codec = Codec(10, 7, planes=planes)
    L = []
    for lo, title in ((True, "drawn lines only"), (False, "full encoding (17 planes)")):
        ka = {k: set(canon(v.reshape(len(v), planes, -1), codec, lo)) for k, v in a.items()}
        kb = {k: set(canon(v.reshape(len(v), planes, -1), codec, lo)) for k, v in b.items()}
        L += [f"Key: {title}, minimum over the {len(codec.syms)} grid symmetries. Cells: shared distinct positions.", "",
              "| | " + " | ".join(f"{k} ({len(v)} distinct)" for k, v in kb.items()) + " |",
              "|---|" + "---:|" * len(kb)]
        for k, v in ka.items():
            L.append(f"| {k} ({len(v)} distinct) | " + " | ".join(str(len(v & w)) for w in kb.values()) + " |")
        L.append("")
    sa = [np.asarray(v[:, 1, :]).reshape(len(v), -1)[:, codec.line_cell] for v in a.values()]
    sb = [np.asarray(v.reshape(len(v), planes, -1)[:, 1, codec.line_cell]) for v in b.values()]
    undrawn = lambda s: (s > 0).sum(1)  # noqa: E731
    L += ["Undrawn lines per position (min / median / max): "
          + "; ".join(f"{k} {undrawn(s).min()} / {int(np.median(undrawn(s)))} / {undrawn(s).max()}"
                      for k, s in zip(list(a) + list(b), sa + sb)) + ".", ""]
    return L


def same_validation(cfg3p):
    import torch

    cfg3 = load_config(ROOT / "configs/graph_s3.yaml")
    pc3, sp3 = judge_sets(cfg3)
    pc3p, sp3p = judge_sets(cfg3p)
    f = ROOT / pc3["data"]["out"] / "data_10x7.jsonl"
    import gzip
    import os
    import time as _t

    same = gzip.decompress((ROOT / "data/parity_hand_10x7.jsonl.gz").read_bytes()) == f.read_bytes()
    L = [f"- Config: both series read parity_config = {cfg3['parity_config']} / {cfg3p['parity_config']}; "
         f"split seed {pc3['seed']} / {pc3p['seed']}, val_share {pc3['train']['val_share']} / {pc3p['train']['val_share']}.",
         f"- Positions are read from {f.relative_to(ROOT)} (sha256 {file_sha(f)}, last modified "
         f"{_t.strftime('%Y-%m-%d %H:%M', _t.localtime(os.path.getmtime(f)))}, before both series); byte-identical "
         f"to the git-tracked data/parity_hand_10x7.jsonl.gz (commit 2167ff1): {same}.",
         f"- 10x7 validation positions: series 3 {len(sp3['10x7'][1])}, hash {positions_sha(sp3['10x7'][1])}; "
         f"series 3' {len(sp3p['10x7'][1])}, hash {positions_sha(sp3p['10x7'][1])}: "
         + ("identical." if positions_sha(sp3['10x7'][1]) == positions_sha(sp3p['10x7'][1]) else "DIFFERENT."),
         "- Metric: the same validate() of scripts/graph_s3.py in both series (share of positions whose argmax "
         "legal move is one of the judge's correct moves). Recomputed now from saved weights:", "",
         "| net | weights | logged 10x7 val | recomputed |", "|---|---|---:|---:|"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.1)
    import parity_hand
    from graph_hand import correct_rate

    def val(cfg, arm, state):
        net, zero, _ = graph_s3.build_net(cfg, arm)
        net.load_state_dict(state)
        net.to(dev).eval()
        _, sp = judge_sets(cfg)
        c, x, m = parity_hand.parity_arrays("10x7", sp["10x7"][1], net.planes)
        return 100 * float(np.mean(correct_rate(net, c, graph_s3.zeroed(x, zero), m, dev)))

    for s in (1, 2):
        p = ROOT / f"runs/graph_s3/s2_Gc_s{s}.ckpt"
        if p.exists():
            ck = torch.load(p, map_location="cpu", weights_only=False)
            logged = next(x["val"]["10x7"] for x in ck["loop"]["log"] if x["step"] == ck["step"])
            best = max(x["val"]["10x7"] for x in ck["loop"]["log"] if x["step"] > cfg3["stage2"]["switch_at"])
            L.append(f"| series 3 Gc s{s} (paused) | checkpoint, step {ck['step']} (best so far {best:.1f}) | "
                     f"{logged:.1f} | {val(cfg3, 'Gc', ck['net']):.1f} |")
    for p in sorted((ROOT / cfg3p["out"]).glob("s2_Gc_s*.json")):
        r = json.loads(p.read_text())
        st = torch.load(p.with_suffix(".pt"), map_location="cpu")
        L.append(f"| series 3' {p.stem[3:]} | best, step {r['best_step']} | {r['best_val_10x7']:.1f} | "
                 f"{val(cfg3p, 'Gc', st):.1f} |")
    return L + [""]


def main() -> None:
    cfg = load_config(ROOT / "configs/graph_s3p.yaml")
    L = [MARK, "", "Written 2026-11-13 at the user's request while the series runs; no training, read-only on the runs.", "",
         "### 1. Overlap of stage 0 positions with the judge's 10x7 positions", ""] + overlap(cfg) + \
        ["### 2. Is the 10x7 validation of series 3 and series 3' the same", ""] + same_validation(cfg)
    rep = ROOT / cfg["report"]
    old = rep.read_text() if rep.exists() else "# Part 4, series 3'\n\n"
    old = re.split(r"^" + re.escape(MARK), old, flags=re.M)[0].rstrip() + "\n\n"
    rep.write_text(old + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
