"""The hand on the judge's labelled middle game: does it generalise?

    python scripts/mid_hand.py check    # head-only fine-tune on 70% of set mid, held-out accuracy before / after
    python scripts/mid_hand.py routs    # only if check passes: routs on 1200 games and the bridge

Positions: the solved positions of set mid (10-15 safe lines,
runs/mid_judge/labels_mid.jsonl), target = the lines that keep the best
result. Unlike scripts/rout_mistakes.py the split is by game: positions of
one game are close to each other. The trainer, the frozen check and the
rule (a variant up with p < 0.05, one-sided sign test) are rout_mistakes'.
Only decisive positions can change, so the test sees those; the accuracy
is given on all held-out positions and on the decisive ones.
"""

from __future__ import annotations

import argparse
import json
import random

from _common import ROOT, load_config

import rout_mistakes as rm
import self_correct as sc

CFG = ROOT / "configs/mid_hand.yaml"


def positions(cfg):
    ps = []
    for r in sc.read_jsonl(ROOT / cfg["labels"]):
        if r["labels"] is None:
            continue
        lab = {int(k): v for k, v in r["labels"].items()}
        best = max(lab.values())
        ps.append({"moves": r["moves"], "correct": sorted(k for k, v in lab.items() if v == best),
                   "gid": r["key"][1], "s": r["key"][0], "decisive": len(set(lab.values())) > 1})
    return ps


def mistakes(cfg):
    ps = positions(cfg)
    gids = sorted({p["gid"] for p in ps})
    random.Random(f"{cfg['seed']}/split").shuffle(gids)
    held = set(gids[:int(round(len(gids) * cfg["train"]["val_share"]))])
    return [p for p in ps if p["gid"] not in held], [p for p in ps if p["gid"] in held]


def cmd_check(cfg):
    rm.mistakes = mistakes
    (sc.out_dir(cfg)).mkdir(parents=True, exist_ok=True)
    res = rm.cmd_check(cfg)
    import numpy as np
    import torch

    import parity_hand as ph

    _, va = mistakes(cfg)
    dec = [p for p in va if p["decisive"]]
    base, _ = ph.load_base({"net": cfg["net"]})
    res["decisive"] = {"n": len(dec), "base": float(np.mean(rm.correct_vec(base, dec)))}
    for n in cfg["train"]["variants"]:
        net, _ = ph.load_base({"net": cfg["net"]})
        net.load_state_dict(torch.load(sc.out_dir(cfg) / f"{n}.pt", map_location="cpu"))
        res["decisive"][n] = float(np.mean(rm.correct_vec(net, dec)))
    (sc.out_dir(cfg) / "check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["decisive"]), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "routs"])
    a = ap.parse_args()
    cfg = load_config(CFG)
    {"check": cmd_check, "routs": rm.cmd_routs}[a.cmd](cfg)


if __name__ == "__main__":
    main()
