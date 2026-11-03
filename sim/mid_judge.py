"""Middle-game judge (part 3 c): exact search on the real board, by result sign.

``Judge`` wraps the C search agents/csrc/mid_judge.c (built with cffi on
first use, like agents/mcts_c.py); ``PyJudge`` is the Python reference with
the same algorithm on top of agents/strong.py Engine (TT of value bounds
keyed by the canonical position over the board's symmetries), for the tests.

Both search the net boxes still to win for the side to move; the leaves of
the loony endgame without junctions are Berlekamp's recursion (exact, the
formula judge sim/chain_solver.py at s = 0). ``classify`` asks only for the
sign of the final result (win / draw / loss for the mover, the boxes already
won included) with null-window searches, per legal move.
"""

from __future__ import annotations

import importlib
import os
import shutil
import time
from pathlib import Path

from sim.board import Board, geometry

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "agents" / "csrc" / "mid_judge.c"
CDEF = """
typedef struct judge_t judge_t;
judge_t *judge_new(int nl, int nb, const int16_t *box_lines, const int16_t *ends,
                   int nsym, const int16_t *perm, int tt_bits);
void judge_free(judge_t *j);
void judge_clear(judge_t *j);
int judge_search(judge_t *j, const uint8_t *lines, int alpha, int beta, double seconds,
                 int64_t node_limit, int *status, int64_t *nodes);
"""
MOD = "_mid_judge"


def _load():
    import cffi

    so = list((ROOT / "agents").glob(MOD + "*.so"))
    if not so or so[0].stat().st_mtime < SRC.stat().st_mtime:
        ffi = cffi.FFI()
        ffi.cdef(CDEF)
        ffi.set_source("agents." + MOD, SRC.read_text(), extra_compile_args=["-O3", "-march=native"])
        tmp = ROOT / "build" / "mid_judge_cffi" / str(os.getpid())
        lib_path = ffi.compile(tmpdir=str(tmp))
        for p in so:
            p.unlink()
        shutil.copy(lib_path, ROOT / "agents" / Path(lib_path).name)
        shutil.rmtree(tmp, ignore_errors=True)
    importlib.invalidate_caches()
    return importlib.import_module("agents." + MOD)


def symmetries(w: int, h: int) -> list[list[int]]:
    """Line permutations of the board's symmetries (identity first)."""
    geo = geometry(w, h)
    mid = {}
    for b, (top, left, right, bottom) in enumerate(geo.box_lines):
        r, c = divmod(b, w)
        mid[top] = (2 * c + 1, 2 * r)
        mid[left] = (2 * c, 2 * r + 1)
        mid[right] = (2 * c + 2, 2 * r + 1)
        mid[bottom] = (2 * c + 1, 2 * r + 2)
    at = {p: ln for ln, p in mid.items()}
    W, H = 2 * w, 2 * h
    maps = [lambda x, y: (x, y), lambda x, y: (W - x, y), lambda x, y: (x, H - y), lambda x, y: (W - x, H - y)]
    if w == h:
        maps += [lambda x, y: (y, x), lambda x, y: (W - y, x), lambda x, y: (y, H - x), lambda x, y: (W - y, H - x)]
    return [[at[f(*mid[ln])] for ln in range(geo.n_lines)] for f in maps]


class Timeout(Exception):
    pass


class Judge:
    """C search for one board size; the TT lives as long as the judge."""

    def __init__(self, w: int, h: int, tt_bits: int = 24):
        m = _load()
        self.ffi, self.lib = m.ffi, m.lib
        geo = geometry(w, h)
        self.w, self.h, self.geo = w, h, geo
        bl = [x for b in geo.box_lines for x in b]
        ends = [x for lb in geo.line_boxes for x in (tuple(lb) + (-1,) * (2 - len(lb)))]
        sy = symmetries(w, h)
        self._keep = [self.ffi.new("int16_t[]", bl), self.ffi.new("int16_t[]", ends),
                      self.ffi.new("int16_t[]", [x for p in sy for x in p])]
        self.j = self.lib.judge_new(geo.n_lines, geo.n_boxes, self._keep[0], self._keep[1], len(sy),
                                    self._keep[2], tt_bits)
        self.buf = self.ffi.new("uint8_t[]", geo.n_lines)
        self.st = self.ffi.new("int *")
        self.nn = self.ffi.new("int64_t *")
        self.nodes = 0
        self.deadline = None

    def __del__(self):
        if getattr(self, "j", None) is not None:
            self.lib.judge_free(self.j)
            self.j = None

    def clear(self):
        self.lib.judge_clear(self.j)

    def search(self, lines, alpha: int, beta: int) -> int:
        """Value for the mover at the line set within (alpha, beta), fail-soft."""
        for i, x in enumerate(lines):
            self.buf[i] = 1 if x else 0
        left = 0.0
        if self.deadline is not None:
            left = self.deadline - time.perf_counter()
            if left <= 0:
                raise Timeout
        v = self.lib.judge_search(self.j, self.buf, alpha, beta, left, 0, self.st, self.nn)
        self.nodes += self.nn[0]
        if self.st[0]:
            raise Timeout
        return v


class PyJudge:
    """The same search in Python (agents/strong.py Engine with a bounds TT
    keyed by the canonical position); slow, for checking the C version."""

    def __init__(self, w: int, h: int):
        from agents.strong import Engine

        self.eng = Engine(w, h)
        self.geo = geometry(w, h)
        self.sym = symmetries(w, h)
        self.tt = {}
        self.nodes = 0
        self.deadline = None

    def clear(self):
        self.tt.clear()

    def _canon(self, L):
        best = None
        for s, p in enumerate(self.sym):
            k = 0
            x = L
            while x:
                low = x & -x
                k |= 1 << p[low.bit_length() - 1]
                x ^= low
            if best is None or k < best[0]:
                best = (k, s)
        return best

    def _search(self, L, alpha, beta):
        eng = self.eng
        self.nodes += 1
        if self.deadline is not None and not self.nodes & 1023 and time.perf_counter() > self.deadline:
            raise Timeout
        s = eng.sides(L)
        L, gained, decisions = eng.forced(L, s)
        if L == eng.full:
            return gained
        return gained + self._node(L, s, decisions, alpha - gained, beta - gained)

    def _node(self, L, s, decisions, alpha, beta):
        eng = self.eng
        key, sym = self._canon(L)
        lo, hi = self.tt.get(key, (-127, 127))
        if lo >= beta:
            return lo
        if hi <= alpha:
            return hi
        if lo == hi:
            return lo
        alpha, beta = max(alpha, lo), min(beta, hi)
        if decisions:
            cand = [x for take, dd in decisions for x in (take, dd)]
        else:
            safe, reps = eng.moves(L, s)
            if not safe:
                comp = eng._components(L, s)
                if comp is not None:
                    v = eng.simple_value(*comp)
                    self.tt[key] = (v, v)
                    return v
            cand = reps + safe
        a0, best = alpha, -1000
        for ln in cand:
            s2 = s.copy()
            L2, got = eng._draw(L, ln, s2)
            v = got + self._search(L2, alpha - got, beta - got) if got else -self._search(L2, -beta, -alpha)
            if v > best:
                best = v
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        lo2, hi2 = (-127, best) if best <= a0 else (best, 127) if best >= beta else (best, best)
        olo, ohi = self.tt.get(key, (-127, 127))
        self.tt[key] = (max(lo2, olo), min(hi2, ohi))
        return best

    def search(self, lines, alpha, beta):
        L = 0
        for i, x in enumerate(lines):
            if x:
                L |= 1 << i
        return self._search(L, alpha, beta)


# --- result signs -------------------------------------------------------------------

def value(judge, board: Board) -> int:
    """Exact net future boxes for the mover (full window)."""
    return judge.search(board.lines, -1000, 1000)


def _at_least(judge, board: Board, ln: int, t: int) -> bool:
    """Does move ``ln`` give the mover at least ``t`` net future boxes?"""
    b2 = board.clone()
    got = b2.step(ln)
    if b2.n_drawn == b2.geo.n_lines:
        return got >= t
    if got:  # the mover moves again
        return judge.search(b2.lines, t - got - 1, t - got) >= t - got
    return judge.search(b2.lines, -t, -t + 1) <= -t


def classify(judge, board: Board, time_limit: float | None = None, moves=None):
    """{line: result sign for the mover (1 win, 0 draw, -1 loss)} for every
    legal line (or ``moves``), the score so far included; raises Timeout."""
    judge.deadline = None if time_limit is None else time.perf_counter() + time_limit
    try:
        me = board.score[board.player] - board.score[board.player ^ 1]
        out = {}
        lines = moves if moves is not None else [ln for ln in range(board.geo.n_lines) if not board.lines[ln]]
        for ln in lines:
            if _at_least(judge, board, ln, 1 - me):
                out[ln] = 1
            elif board.geo.n_boxes % 2 == 0 and _at_least(judge, board, ln, -me):
                out[ln] = 0
            else:
                out[ln] = -1
        return out
    finally:
        judge.deadline = None


def outcome(judge, board: Board, time_limit: float | None = None) -> int:
    """Result sign of the position for the mover with best play; Timeout."""
    judge.deadline = None if time_limit is None else time.perf_counter() + time_limit
    try:
        me = board.score[board.player] - board.score[board.player ^ 1]
        if judge.search(board.lines, -me, -me + 1) >= 1 - me:
            return 1
        if board.geo.n_boxes % 2 == 0 and judge.search(board.lines, -me - 1, -me) >= -me:
            return 0
        return -1
    finally:
        judge.deadline = None
