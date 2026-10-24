"""Strong reference player: alpha-beta with a transposition table and chain
analysis (long chain rule, all-but-two / all-but-four, hard-hearted handouts).

Search
    Negamax over the *net number of boxes still to be won* by the side to
    move. That value depends only on the drawn lines, so the transposition
    table is keyed by the line bitmask alone (past scores and the identity
    of the mover do not matter).

Captures (strings-and-coins reasoning, as in Berlekamp's book and
Barker & Korf 2012, "Solving Dots-And-Boxes")
    A capturable box is taken at once unless it is a decision point: the
    first of the last two boxes of a chain whose far end is the ground or a
    junction (then also try the all-but-two double-dealing move), or the
    first of four boxes of a chain open at both ends, i.e. an opened loop
    (then also try all-but-four). Taking any other capturable box never
    loses, so those captures are applied without branching.

Moves without captures
    Every safe line (gives no box a third side) plus one representative per
    chain or loop: the middle line of a 2-chain (hard-hearted handout, the
    opponent cannot decline it), an end line of longer chains, any line of a
    loop. Other ways to open the same chain only give the opponent extra
    options and are dominated.

Evaluation
    Positions without safe moves (the loony endgame) are solved exactly:
    by the closed-form recursion over independent chains and loops when no
    junction box (three or more undrawn sides) is left, by search otherwise.
    At the depth limit in positions that still have safe moves, the safe
    moves are played out in a pseudo-random order fixed by the position; the
    parity of their number decides who must open the first chain (the long
    chain rule in practice), and the resulting endgame is solved exactly.

Iterative deepening under a per-move time limit; ties between equally good
root moves are broken by a seeded RNG.
"""

from __future__ import annotations

import random
import time

from sim.board import Board, geometry

INF = 10**6
EXACT, LOWER, UPPER = 0, 1, 2
SOLVED = 1000  # TT depth of exactly solved entries


class Timeout(Exception):
    pass


class Engine:
    """Search tables for one board size; the TT lives as long as the engine."""

    def __init__(self, w: int, h: int, tt_limit: int = 2_000_000):
        geo = geometry(w, h)
        self.w, self.h = w, h
        self.n_lines, self.n_boxes = geo.n_lines, geo.n_boxes
        self.full = (1 << self.n_lines) - 1
        self.box_lines = geo.box_lines
        self.box_mask = tuple(sum(1 << ln for ln in bl) for bl in geo.box_lines)
        self.line_boxes = geo.line_boxes
        # the two ends of every line: boxes, -1 = ground
        self.ends = tuple(tuple(lb) + (-1,) * (2 - len(lb)) for lb in geo.line_boxes)
        self.tt: dict[int, tuple] = {}
        self.dp: dict[tuple, int] = {}
        self.tt_limit = tt_limit
        self.deadline = None
        self.node_limit = None
        self.nodes = 0

    # --- primitives --------------------------------------------------------

    def sides(self, L: int) -> list[int]:
        bm = self.box_mask
        return [(L & bm[b]).bit_count() for b in range(self.n_boxes)]

    def _undrawn(self, L, b, skip=-1):
        for ln in self.box_lines[b]:
            if not (L >> ln) & 1 and ln != skip:
                return ln
        return -1

    def _other(self, ln, b):
        e0, e1 = self.ends[ln]
        return e1 if e0 == b else e0

    def _walk(self, L, s, c, ln):
        """Walk from capturable box ``c`` through its missing line ``ln``.

        Returns (n, kind, dd): n boxes in the run (c included, plus the far
        capturable box if kind == "open"), kind "end" (ground or junction at
        the far end) or "open" (another capturable box), and the
        double-dealing line for the decision cases (else -1).
        """
        n, prev, cur = 1, ln, self._other(ln, c)
        lines = [ln]
        while True:
            if cur == -1:
                kind = "end"
                break
            d = 4 - s[cur]
            if d == 1:
                n += 1
                kind = "open"
                break
            if d >= 3:
                kind = "end"
                break
            n += 1
            nxt = self._undrawn(L, cur, prev)
            lines.append(nxt)
            prev, cur = nxt, self._other(nxt, cur)
        dd = -1
        if kind == "end" and n == 2:
            dd = lines[1]   # far side of the second box
        elif kind == "open" and n == 4:
            dd = lines[1]   # between the second and the third box
        return n, kind, dd

    def _draw(self, L, ln, s):
        """Draw ``ln`` updating ``s`` in place; returns (L, boxes completed)."""
        got = 0
        for b in self.line_boxes[ln]:
            s[b] += 1
            if s[b] == 4:
                got += 1
        return L | (1 << ln), got

    def forced(self, L: int, s: list[int]):
        """Apply all captures that never lose. Returns (L, gained, decisions)
        with decisions = [(capture line, double-deal line)] left open."""
        gained = 0
        while True:
            decisions = []
            for c in range(self.n_boxes):
                if s[c] == 3:
                    ln = self._undrawn(L, c)
                    n, kind, dd = self._walk(L, s, c, ln)
                    if dd < 0:
                        L, got = self._draw(L, ln, s)
                        gained += got
                        break
                    decisions.append((ln, dd))
            else:
                return L, gained, decisions

    def moves(self, L: int, s: list[int]):
        """(safe lines, chain representatives) of a position without captures."""
        safe = []
        n_lines, ends = self.n_lines, self.ends
        for ln in range(n_lines):
            if not (L >> ln) & 1:
                e0, e1 = ends[ln]
                if (e0 < 0 or s[e0] < 2) and (e1 < 0 or s[e1] < 2):
                    safe.append(ln)
        reps = []
        seen = set()
        for b in range(self.n_boxes):
            if s[b] != 2 or b in seen:
                continue
            # walk both ways from b through boxes with two drawn sides
            l1 = self._undrawn(L, b)
            l2 = self._undrawn(L, b, l1)
            chain = [b]
            seen.add(b)
            loop = False
            ends_lines = []
            arms = []
            for start in (l1, l2):
                arm_lines = [start]
                prev, cur = start, self._other(start, b)
                while cur >= 0 and s[cur] == 2:
                    if cur == b:
                        loop = True
                        break
                    seen.add(cur)
                    chain.append(cur)
                    nxt = self._undrawn(L, cur, prev)
                    arm_lines.append(nxt)
                    prev, cur = nxt, self._other(nxt, cur)
                arms.append(arm_lines)
                if loop:
                    break
                ends_lines.append(prev)
            if loop:
                reps.append(l1)
            elif len(chain) == 1:
                reps.append(l1)
            elif len(chain) == 2:
                # the line between the two boxes
                reps.append(arms[0][0] if len(arms[0]) == 2 else arms[1][0])
            else:
                reps.append(ends_lines[0])
        return safe, reps

    # --- loony endgame -------------------------------------------------------

    def _components(self, L, s):
        """Chains/loops of a junction-free loony position, or None."""
        chains, loops = [], []
        seen = set()
        for b in range(self.n_boxes):
            if s[b] == 4 or b in seen:
                continue
            if s[b] != 2:
                return None
            l1 = self._undrawn(L, b)
            l2 = self._undrawn(L, b, l1)
            n = 1
            seen.add(b)
            loop = False
            for start in (l1, l2):
                prev, cur = start, self._other(start, b)
                while cur >= 0:
                    if cur == b:
                        loop = True
                        break
                    if s[cur] != 2:
                        return None
                    seen.add(cur)
                    n += 1
                    nxt = self._undrawn(L, cur, prev)
                    prev, cur = nxt, self._other(nxt, cur)
                if loop:
                    break
            (loops if loop else chains).append(n)
        return tuple(sorted(chains)), tuple(sorted(loops))

    def simple_value(self, chains: tuple, loops: tuple) -> int:
        """Value for the mover of independent chains and loops, all loony."""
        if not chains and not loops:
            return 0
        key = (chains, loops)
        v = self.dp.get(key)
        if v is not None:
            return v
        best = -INF
        for i, c in enumerate(chains):
            if i and chains[i - 1] == c:
                continue
            r = self.simple_value(chains[:i] + chains[i + 1:], loops)
            if c <= 2:
                val = -(c + r)
            else:
                val = -max(c + r, c - 4 - r)
            best = max(best, val)
        for i, c in enumerate(loops):
            if i and loops[i - 1] == c:
                continue
            r = self.simple_value(chains, loops[:i] + loops[i + 1:])
            best = max(best, -max(c + r, c - 8 - r))
        self.dp[key] = best
        return best

    # --- search ------------------------------------------------------------

    def _tick(self):
        self.nodes += 1
        if self.node_limit is not None and self.nodes > self.node_limit:
            raise Timeout
        if self.deadline is not None and not self.nodes & 255 and time.perf_counter() > self.deadline:
            raise Timeout

    def search(self, L: int, depth: int, alpha: int, beta: int) -> int:
        """Net future boxes for the side to move (captures included)."""
        self._tick()
        s = self.sides(L)
        L, gained, decisions = self.forced(L, s)
        if L == self.full:
            return gained
        return gained + self._node(L, s, decisions, depth, alpha - gained, beta - gained)

    def _node(self, L, s, decisions, depth, alpha, beta):
        ent = self.tt.get(L)
        tt_move = -1
        if ent is not None:
            e_depth, e_flag, e_val, tt_move = ent
            if e_depth >= depth:
                if e_flag == EXACT:
                    return e_val
                if e_flag == LOWER and e_val >= beta:
                    return e_val
                if e_flag == UPPER and e_val <= alpha:
                    return e_val
        if decisions:
            cand = []
            for take, dd in decisions:
                cand.append((take, True))
                cand.append((dd, False))
            solved = True
        else:
            safe, reps = self.moves(L, s)
            if not safe:
                comp = self._components(L, s)
                if comp is not None:
                    v = self.simple_value(*comp)
                    self._store(L, SOLVED, EXACT, v, -1)
                    return v
                depth = SOLVED  # loony endgame with junctions: solve exactly
            elif depth <= 0:
                return self.heuristic(L, s)
            cand = [(ln, False) for ln in reps] + [(ln, False) for ln in safe]
            solved = not safe
        if tt_move >= 0:
            for i, (ln, _) in enumerate(cand):
                if ln == tt_move:
                    cand.insert(0, cand.pop(i))
                    break
        a0 = alpha
        best, best_move = -INF, -1
        child_depth = depth if solved else depth - 1
        for ln, _ in cand:
            s2 = s.copy()
            L2, got = self._draw(L, ln, s2)
            if got:
                v = got + self.search(L2, child_depth, alpha - got, beta - got)
            else:
                v = -self.search(L2, child_depth, -beta, -alpha)
            if v > best:
                best, best_move = v, ln
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        flag = UPPER if best <= a0 else LOWER if best >= beta else EXACT
        self._store(L, depth, flag, best, best_move)
        return best

    def _store(self, L, depth, flag, val, move):
        if len(self.tt) >= self.tt_limit:
            self.tt.clear()
        self.tt[L] = (depth, flag, val, move)

    def heuristic(self, L, s):
        """Play the safe moves out, then solve the endgame (see module doc)."""
        rng = random.Random(L)
        s = s.copy()
        k = 0
        ends = self.ends
        while True:
            safe = [ln for ln in range(self.n_lines) if not (L >> ln) & 1
                    and (ends[ln][0] < 0 or s[ends[ln][0]] < 2)
                    and (ends[ln][1] < 0 or s[ends[ln][1]] < 2)]
            if not safe:
                break
            L, _ = self._draw(L, rng.choice(safe), s)
            k += 1
        v = self.search(L, SOLVED, -INF, INF)
        return v if k % 2 == 0 else -v

    # --- root --------------------------------------------------------------

    def best_move(self, L: int, time_limit: float | None, rng: random.Random, max_depth: int = 64,
                  node_budget: int | None = None):
        """Returns (line, value, depth reached, exact?). The search stops at
        ``time_limit`` seconds (wall clock) or, if ``node_budget`` is set,
        after that many nodes (machine-independent and reproducible)."""
        s = self.sides(L)
        # forced captures at the root: just take one
        for c in range(self.n_boxes):
            if s[c] == 3:
                ln = self._undrawn(L, c)
                n, kind, dd = self._walk(L, s, c, ln)
                if dd < 0:
                    return ln, None, 0, True
        decisions = [(ln, dd) for c in range(self.n_boxes) if s[c] == 3
                     for ln in [self._undrawn(L, c)] for _, _, dd in [self._walk(L, s, c, ln)]]
        if decisions:
            cand = [ln for pair in decisions for ln in pair]
            loony = True
        else:
            safe, reps = self.moves(L, s)
            cand = reps + safe
            loony = not safe
        if len(cand) == 1:
            return cand[0], None, 0, True
        self.deadline = None if node_budget is not None else time.perf_counter() + time_limit
        self.node_limit = node_budget
        self.nodes = 0
        best_line, best_val, reached = rng.choice(cand), None, 0
        depths = [SOLVED] if loony else range(1, max_depth + 1)
        try:
            for depth in depths:
                vals = {}
                order = sorted(cand, key=lambda ln: -(vals.get(ln, 0)))
                if best_line in order:
                    order.insert(0, order.pop(order.index(best_line)))
                alpha = -INF
                for ln in order:
                    s2 = s.copy()
                    L2, got = self._draw(L, ln, s2)
                    child_depth = depth if loony else depth - 1
                    # full window for every root move: exact values for tie-breaks
                    if got:
                        v = got + self.search(L2, child_depth, -INF, INF)
                    else:
                        v = -self.search(L2, child_depth, -INF, INF)
                    vals[ln] = v
                    alpha = max(alpha, v)
                top = max(vals.values())
                best = [ln for ln in cand if vals[ln] == top]
                best_line, best_val, reached = rng.choice(best), top, depth
                if loony or depth >= self.n_lines - L.bit_count():
                    return best_line, best_val, depth, True
        except Timeout:
            pass
        finally:
            self.deadline = None
            self.node_limit = None
        return best_line, best_val, reached, False


class StrongAgent:
    name = "strong"

    def __init__(self, seed: int | None = None, time_limit: float = 1.0, max_depth: int = 64,
                 tt_limit: int = 2_000_000, node_budget: int | None = None):
        self.rng = random.Random(seed)
        self.time_limit = time_limit
        self.node_budget = node_budget  # if set, replaces the time limit
        self.max_depth = max_depth
        self.tt_limit = tt_limit
        self._engines: dict[tuple[int, int], Engine] = {}
        self.last = None  # (line, value, depth, exact) of the last move

    def act(self, board: Board) -> int:
        key = (board.w, board.h)
        eng = self._engines.get(key)
        if eng is None:
            eng = self._engines[key] = Engine(board.w, board.h, self.tt_limit)
        L = 0
        for i, x in enumerate(board.lines):
            if x:
                L |= 1 << i
        self.last = eng.best_move(L, self.time_limit, self.rng, self.max_depth, self.node_budget)
        return self.last[0]
