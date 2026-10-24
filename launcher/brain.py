"""The network's side of the app: load a generation, think about a position.

``Brain.think(board, sims)`` returns everything the thought panel shows:
the policy over the lines, the value (and score head) for the side to move,
and after a search (sims > 0) the MCTS visits, Q per line, the chosen line
and the principal variation. sims = 0 plays the policy's argmax.

``Solver`` gives the exact verdict on a move in the chain fight (strong
engine solving to the end, under a node limit), for "the network sees /
does not see". Networks are loaded once per process and kept (_NETS).
"""

from __future__ import annotations

import threading

import numpy as np

from . import core

_NETS = {}   # (config, checkpoint, device) -> Evaluator
_LOCK = threading.Lock()


def device(cfg_device: str = "auto") -> str:
    if cfg_device != "auto":
        return cfg_device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluator(config: str, checkpoint: str, dev: str):
    import torch

    from agents.az_net import Evaluator, make_net

    key = (str(config), str(checkpoint), dev)
    with _LOCK:
        if key not in _NETS:
            rc = core.load_yaml(config)
            net = make_net(rc)
            net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            _NETS[key] = (Evaluator(net, dev), rc)
        return _NETS[key]


class Brain:
    """One network generation; ``label`` names it in the UI."""

    def __init__(self, config: str, checkpoint: str, label: str, dev: str, seed: int = 0):
        from agents.az_play import make_mcts

        self.ev, self.rc = evaluator(config, checkpoint, dev)
        self.label = label
        self.board_size = tuple(self.rc["board"])
        self.rng = np.random.default_rng(seed)
        self.mcts = make_mcts(self.ev, self.rc["mcts"], self.rng,
                              leaves_per_tree=self.rc["mcts"]["leaves_per_tree"])

    def fits(self, board) -> bool:
        return (board.geo.w, board.geo.h) == self.board_size

    def think(self, board, sims: int, pv_len: int = 10) -> dict:
        import torch

        from agents.az_mcts import choose_move, result_for_mover

        net = self.ev.net
        pri, val = self.ev([board])
        score = None
        if net.s_fc is not None:
            with torch.no_grad():
                x = torch.from_numpy(self.ev.codec.encode_boards([board])).to(self.ev.device)
                score = float(net(x)[2].float()[0]) * board.geo.n_boxes
        out = {"player": board.player, "policy": pri[0], "value": float(val[0]), "score": score,
               "sims": sims, "visits": None, "q": None, "pv": [], "move": int(np.argmax(pri[0])),
               # the winner is already fixed: nothing to search or to fight for
               "decided": result_for_mover(board) is not None}
        if sims > 0:
            root = self.mcts.search([board], sims)[0]
            if not root.terminal:
                n = board.geo.n_lines
                visits = np.zeros(n)
                q = np.full(n, np.nan)
                visits[root.moves] = root.N
                seen = root.N > 0
                q[root.moves[seen]] = root.W[seen] / root.N[seen]
                out.update(visits=visits, q=q, move=choose_move(root, 0.0, self.rng),
                           pv=_pv(root, pv_len), root_q=float(root.W.sum() / max(1, root.N.sum())))
        return out


def _pv(root, n):
    pv, node = [], root
    while node is not None and node.expanded and node.N.sum() > 0 and len(pv) < n:
        k = int(np.argmax(node.N))
        pv.append((int(node.moves[k]), int(node.player)))
        node = node.children[k]
    return pv


class Solver:
    """Exact verdicts in the chain fight; one engine (and its table) per game."""

    def __init__(self, w: int, h: int, max_undrawn: int, node_limit: int):
        import sys

        from agents.strong import Engine

        sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
        self.eng = Engine(w, h)
        self.max_undrawn, self.node_limit = max_undrawn, node_limit

    def correct(self, board):
        """(correct lines, all moves equal?) or None when not solved."""
        from agents.strong import exact_move_values

        undrawn = board.geo.n_lines - board.n_drawn
        if undrawn > self.max_undrawn:
            return None
        if len(self.eng.tt) > 1_500_000:
            self.eng.tt.clear()
        vals = exact_move_values(self.eng, board, self.node_limit)
        if vals is None:
            return None
        me, op = board.score[board.player], board.score[board.player ^ 1]
        res = {ln: int(np.sign(me - op + v)) for ln, v in vals.items()}
        best = max(res.values())
        return sorted(ln for ln, r in res.items() if r == best), min(res.values()) == best
