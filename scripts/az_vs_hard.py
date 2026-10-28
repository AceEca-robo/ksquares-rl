"""A net + MCTS against the real KSquares Hard over the RL bridge (offscreen).

``bridge_eval`` in the run config: ``games`` (first mover alternating,
the net first in even games), ``sims``, ``seed`` (the game's srand per game:
seed + game). The net's moves: MCTS with ``mcts.leaves_per_tree``, no noise,
temperature 0. Prints the score per game and the total; ``--json`` saves it.

    python scripts/az_vs_hard.py --config configs/dot2a.yaml --checkpoint runs/dot2a/gen_0012.pt
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from _common import ROOT, load_config

from agents.az_mcts import choose_move
from agents.az_net import Evaluator, make_net
from agents.az_play import make_mcts
from env.bridge import KSquaresBridge


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--board", type=int, nargs=2, help="w h (default: the config's board)")
    ap.add_argument("--json")
    ap.add_argument("--leaves", type=int, help="leaves per tree per round (default: mcts.leaves_per_tree)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    bc = cfg["bridge_eval"]
    w, h = a.board or cfg["board"]
    net = make_net(cfg)
    net.load_state_dict(torch.load(ROOT / a.checkpoint, map_location="cpu"))
    ev = Evaluator(net, cfg["device"])
    rng = np.random.default_rng(bc["seed"])
    mcts = make_mcts(ev, cfg["mcts"], rng, leaves_per_tree=a.leaves or cfg["mcts"]["leaves_per_tree"])
    res = {"wins": 0, "losses": 0, "ties": 0, "games": []}
    with KSquaresBridge() as br:
        for g in range(bc["games"]):
            seat = g % 2
            ai = [True, True]
            ai[seat] = False
            st = br.reset(w, h, ai=ai, difficulty=2, seed=bc["seed"] + g)  # 2 = Hard
            b = br.board
            while not b.done:
                root = mcts.search([b], bc["sims"])[0]
                ln = int(np.argmax(ev([b])[0][0])) if root.terminal else choose_move(root, 0.0, rng)
                st = br.step(ln)
                b = br.board
            me, op = st["scores"][seat], st["scores"][1 - seat]
            res["wins" if me > op else "losses" if op > me else "ties"] += 1
            res["games"].append({"net_first": seat == 0, "net": me, "hard": op})
            print(f"game {g + 1}: net {'first' if seat == 0 else 'second'} {me}:{op}", flush=True)
    n = bc["games"]
    print(f"{w}x{h}, {a.checkpoint}, MCTS {bc['sims']}: {res['wins']}-{res['losses']}-{res['ties']} "
          f"({100 * (res['wins'] + res['ties'] / 2) / n:.0f}%) vs KSquares Hard over the bridge")
    if a.json:
        (ROOT / a.json).write_text(json.dumps({"board": [w, h], "checkpoint": a.checkpoint, **bc,
                                               "leaves": a.leaves or cfg["mcts"]["leaves_per_tree"], **res}, indent=1))


if __name__ == "__main__":
    main()
