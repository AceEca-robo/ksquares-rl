"""The app's 10x7 network against Hard in the simulator, at several numbers of
MCTS simulations (configs/sims_vs_hard_10x7.yaml): is the app's "MCTS 50"
button too weak on the big board?

    .venv/bin/python scripts/sims_vs_hard.py [--config configs/sims_vs_hard_10x7.yaml]

Writes the JSON named by ``out`` ({sims: [wins, losses, ties], seconds}) and
prints a markdown table for docs/results.md. Games as in scripts/az_eval.py.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context

import numpy as np
import yaml

from _common import ROOT, load_config
from az_eval import _oinit, play


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/sims_vs_hard_10x7.yaml"))
    a = ap.parse_args()
    sc = yaml.safe_load(open(a.config))
    cfg = load_config(str(ROOT / sc["base"]))
    import torch

    from agents.az_net import Evaluator, make_net

    torch.set_num_threads(2)
    net = make_net(cfg)
    net.load_state_dict(torch.load(ROOT / sc["checkpoint"], map_location="cpu"))
    ev = Evaluator(net, cfg["device"])
    ec = cfg["eval"]
    out_path = ROOT / sc["out"]
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    ctx = get_context("spawn")
    with ctx.Pool(ec["processes"], initializer=_oinit, initargs=(ec.get("params") or {}, sc["seed"])) as pool:
        for sims in sc["sims"]:
            if str(sims) in out:
                continue
            run_cfg = dict(cfg, eval=dict(ec, sims=sims, games=sc["games"]))
            rng = np.random.default_rng([sc["seed"], sims])
            t0 = time.time()
            r = play(ev, "mcts", [sc["opponent"]], run_cfg, pool, rng)[sc["opponent"]]
            out[str(sims)] = {"wins": r[0], "losses": r[1], "ties": r[2], "seconds": round(time.time() - t0)}
            print(f"[{time.strftime('%H:%M:%S')}] sims {sims}: {r[0]}-{r[1]}-{r[2]} "
                  f"({time.time() - t0:.0f} s)", flush=True)
            out_path.write_text(json.dumps(out, indent=1))
    print(f"\n| sims | wins | losses | ties |\n|---:|---:|---:|---:|")
    for s in sc["sims"]:
        o = out[str(s)]
        print(f"| {s} | {o['wins']} | {o['losses']} | {o['ties']} |")


if __name__ == "__main__":
    main()
