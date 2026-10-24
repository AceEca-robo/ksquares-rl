"""Batched PUCT search for AlphaZero over many game trees at once.

All trees advance in lock step: every round each tree descends to up to
``leaves_per_tree`` new leaves (virtual loss keeps them apart), the leaves
of all trees go to the network in one batch, then they are expanded and
backed up. Self-play runs many games with one leaf each; a single game
(evaluation, the az agent) uses several leaves per round.

Extra turns: drawing a line that completes a box gives the same player
another move. Every node stores the player to move, and a backed-up value
changes sign only on edges where the mover changes. Edge statistics live in
the parent (``N``, ``W`` per legal move, from the parent mover's view).

Terminal nodes: the game is over when all lines are drawn or one player owns
more than half of the boxes (then the winner is fixed; ``decided``). The
value is the exact result, +1 / -1 / 0 for the mover.
"""

from __future__ import annotations

import math

import numpy as np

from sim.board import Board


def result_for_mover(board: Board) -> float | None:
    """Exact game value for the side to move, or None if not decided."""
    me, op = board.score[board.player], board.score[board.player ^ 1]
    left = board.geo.n_boxes - me - op
    if me > op + left:
        return 1.0
    if op > me + left:
        return -1.0
    if left == 0:
        return 0.0
    return None


class Node:
    __slots__ = ("board", "player", "moves", "P", "N", "W", "children", "n", "value", "terminal")

    def __init__(self, board: Board):
        self.board = board
        self.player = board.player
        v = result_for_mover(board)
        self.terminal = v is not None
        self.value = v  # exact if terminal, else the network value once expanded
        self.moves = None  # legal lines (np.int64), set on expansion
        self.P = self.N = self.W = None
        self.children = None
        self.n = 0  # visits through this node (incl. virtual)

    @property
    def expanded(self) -> bool:
        return self.moves is not None

    def expand(self, priors: np.ndarray, value: float) -> None:
        self.moves = np.flatnonzero(np.frombuffer(bytes(self.board.lines), dtype=np.uint8) == 0)
        p = priors[self.moves].astype(np.float64)
        s = p.sum()
        self.P = p / s if s > 0 else np.full(len(p), 1.0 / len(p))
        self.N = np.zeros(len(self.moves))
        self.W = np.zeros(len(self.moves))
        self.children = [None] * len(self.moves)
        self.value = float(value)


class MCTS:
    """PUCT search; ``evaluate(boards) -> (priors (B, n_lines), values (B,))``."""

    def __init__(self, evaluate, c_puct: float = 1.5, fpu_reduction: float = 0.2,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25,
                 leaves_per_tree: int = 1, virtual_loss: float = 1.0, rng=None):
        self.evaluate = evaluate
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        self.alpha = dirichlet_alpha
        self.eps = dirichlet_eps
        self.leaves_per_tree = leaves_per_tree
        self.vl = virtual_loss
        self.rng = rng or np.random.default_rng()
        self.n_evals = 0

    # --- one descent ----------------------------------------------------------

    def _select(self, node: Node) -> int:
        N = node.N
        sq = math.sqrt(node.n) if node.n > 0 else 1.0
        fpu = node.value - self.fpu_reduction
        q = np.where(N > 0, node.W / np.maximum(N, 1), fpu)
        return int(np.argmax(q + self.c_puct * sq * node.P / (1.0 + N)))

    def _descend(self, root: Node):
        """Walk to a leaf adding virtual loss; returns (path, leaf) where path
        is [(node, action index)], leaf an unexpanded or terminal node, or
        None when the leaf is already queued in this round (collision)."""
        path = []
        node = root
        vl = self.vl
        while node.expanded and not node.terminal:
            a = self._select(node)
            path.append((node, a))
            node.n += 1
            node.N[a] += vl
            node.W[a] -= vl
            child = node.children[a]
            if child is None:
                b = node.board.clone()
                b.step(int(node.moves[a]))
                child = node.children[a] = Node(b)
                return path, child
            node = child
        if node is not root and not node.terminal:
            return path, None  # pending expansion from another descent
        return path, node

    # --- public ----------------------------------------------------------------

    def add_noise(self, root: Node) -> None:
        if self.eps > 0 and len(root.P) > 1:
            noise = self.rng.dirichlet([self.alpha] * len(root.P))
            root.P = (1 - self.eps) * root.P + self.eps * noise

    def search(self, boards: list[Board], sims, noise: bool = False) -> list[Node]:
        """Run ``sims`` simulations for every board (an int, or one count per
        board); returns the roots."""
        if isinstance(sims, (int, np.integer)):
            sims = [int(sims)] * len(boards)
        roots = [Node(b.clone()) for b in boards]
        todo = [r for r in roots if not r.terminal]
        if todo:
            pri, val = self.evaluate([r.board for r in todo])
            self.n_evals += len(todo)
            for r, p, v in zip(todo, pri, val):
                r.expand(p, v)
                r.n = 1
                if noise:
                    self.add_noise(r)
        done = [0] * len(roots)
        while True:
            pending = []  # (path, leaf, tree index)
            for i, root in enumerate(roots):
                if root.terminal or len(root.moves) == 1:
                    continue
                # the first descent of a round cannot collide, so every
                # round makes progress
                for _ in range(min(self.leaves_per_tree, sims[i] - done[i])):
                    path, leaf = self._descend(root)
                    if leaf is None:
                        self._undo(path)
                        continue
                    pending.append((path, leaf, i))
                    done[i] += 1
            if not pending:
                break
            to_eval = [(j, leaf) for j, (_, leaf, _) in enumerate(pending) if not leaf.terminal]
            if to_eval:
                pri, val = self.evaluate([leaf.board for _, leaf in to_eval])
                self.n_evals += len(to_eval)
                for (j, leaf), p, v in zip(to_eval, pri, val):
                    leaf.expand(p, v)
            for path, leaf, _ in pending:
                leaf.n += 1
                self._backup_path(path, leaf.value)
        return roots

    def _undo(self, path) -> None:
        vl = self.vl
        for node, a in path:
            node.n -= 1
            node.N[a] -= vl
            node.W[a] += vl

    def _backup_path(self, path, v: float) -> None:
        """v: value for the player to move at the leaf (end of path)."""
        vl = self.vl
        for node, a in reversed(path):
            if node.children[a].player != node.player:
                v = -v
            node.N[a] += 1 - vl
            node.W[a] += v + vl


def visit_policy(root: Node, n_lines: int) -> np.ndarray:
    pi = np.zeros(n_lines, dtype=np.float32)
    if root.moves is None:
        return pi
    if len(root.moves) == 1:
        pi[root.moves[0]] = 1
    else:
        pi[root.moves] = root.N / max(root.N.sum(), 1)
    return pi


def choose_move(root: Node, temperature: float, rng) -> int:
    if len(root.moves) == 1:
        return int(root.moves[0])
    N = root.N
    if temperature <= 0:
        best = np.flatnonzero(N == N.max())
        return int(root.moves[rng.choice(best)])
    p = N ** (1.0 / temperature)
    return int(root.moves[rng.choice(len(N), p=p / p.sum())])
