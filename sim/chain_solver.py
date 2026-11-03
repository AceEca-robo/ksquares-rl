"""Formula judge: Dots and Boxes outcome by chain / loop counting, no search.

Theory (Berlekamp, "The Dots-and-Boxes Game"; Berlekamp, Conway & Guy,
"Winning Ways", ch. 16): once the board is cut into independent chains and
loops and the number of safe moves left is fixed, the game is a sum of
simple pieces and its value is arithmetic.

Settled positions
    A position without a capturable box is *settled* when

    1. the safe lines (lines that give no box a third side) cannot spoil
       each other: every box has ``drawn sides + safe lines on it <= 2``.
       Then every safe line stays safe until it is drawn, so the safe phase
       lasts exactly ``s`` moves whatever the order (sides only grow, so no
       new safe line ever appears), and
    2. the *final* board F (every safe line drawn) is junction-free: every
       free box has two drawn sides. F is then a set of chains (both ends
       on the ground / the edge of the board) and loops.

    Otherwise the formula has no answer (``None``): the safe-move count can
    still be fought over, or the endgame has junctions.

Value (net boxes still to win for the side to move), state (s, chains, loops)::

    g(0, {}) = 0
    g(s, C)  = max of
        -g(s - 1, C)                                  play a safe move (s > 0)
        -(n + g(s, C - c))                            open a chain of n <= 2
                                                      (hard-hearted handout: the
                                                      opponent must take it)
        -max(n + g(s, C - c), n - 4 - g(s, C - c))   open a chain of n >= 3: the
                                                      opponent takes all and moves,
                                                      or all but two (double-dealing)
        -max(n + g(s, C - c), n - 8 - g(s, C - c))   open a loop of n: all, or all
                                                      but four

    With s = 0 this is the loony endgame recursion of agents/strong.py
    (``Engine.simple_value``); s > 0 adds the safe moves as tempo moves, so
    the fight for control (long chain rule: who must open the first long
    chain) and the parity-changing sacrifice of a short chain are both in
    it. The chains are those of F, also when one is opened before all safe
    lines are drawn: that is the approximation the judge is tested on
    (scripts/chain_solver_eval.py, docs/chain_solver.md).

Moves
    ``best_move`` turns the best option into a line: any safe line; the
    middle line of a 2-chain; an end line of a longer chain; any line of a
    loop. With boxes to capture it decides the double-cross question: take
    everything and move, or leave the last two boxes of a chain (four of a
    loop) to keep control.
"""

from __future__ import annotations

from functools import lru_cache

from sim.board import Board

SAFE, CHAIN, LOOP = "safe", "chain", "loop"


# --- board analysis ----------------------------------------------------------------

def safe_lines(board: Board) -> list[int]:
    s, geo = board.sides, board.geo
    return [ln for ln in range(geo.n_lines)
            if not board.lines[ln] and all(s[b] < 2 for b in geo.line_boxes[ln])]


def capturable(board: Board) -> list[int]:
    return [b for b in range(board.geo.n_boxes) if board.sides[b] == 3]


def safe_phase_fixed(board: Board, safe: list[int]) -> bool:
    """Every safe line stays safe whatever is drawn before it."""
    k = [0] * board.geo.n_boxes
    for ln in safe:
        for b in board.geo.line_boxes[ln]:
            k[b] += 1
    return all(board.sides[b] + k[b] <= 2 for b in range(board.geo.n_boxes) if k[b])


class Unsettled(Exception):
    pass


def graph(board: Board):
    """Strings-and-coins structure of a board without safe lines and boxes
    to take: junction boxes (one drawn side or none) with their number of
    undrawn sides, arms (runs of boxes with two drawn sides between two
    ends, a junction or the ground = -1; both ends on the ground: a plain
    chain) and loops. Returns (arms, loops, deg, lines) with arms as sorted
    (u, v, n), loops as sizes, deg as sorted (junction, undrawn sides) and
    lines: arm or ('loop', n) -> its undrawn lines in order (one example
    per kind); raises Unsettled on anything else."""
    geo, lines, sides = board.geo, board.lines, board.sides
    free = [b for b in range(geo.n_boxes) if sides[b] < 4]
    if any(sides[b] == 3 for b in free):
        raise Unsettled("box to take")
    junc = {b: 4 - sides[b] for b in free if sides[b] <= 1}

    def undrawn(b):
        return [ln for ln in geo.box_lines[b] if not lines[ln]]

    def across(ln, b):
        return next((o for o in geo.line_boxes[ln] if o != b), -1)

    used, arms, loops, where = set(), [], [], {}

    def walk(u, ln):
        """From end u (junction or -1) over line ln: (end, boxes, lines)."""
        boxes, lns = [], [ln]
        cur = across(ln, u) if u >= 0 else next(iter(geo.line_boxes[ln]))
        prev = ln
        while cur >= 0 and cur not in junc:
            boxes.append(cur)
            nxt = next(x for x in undrawn(cur) if x != prev)
            lns.append(nxt)
            prev, cur = nxt, across(nxt, cur)
        return cur, boxes, lns

    def add(u, v, boxes, lns):
        if not boxes:
            raise Unsettled("a line between ends")  # would be a safe line
        used.update(lns)
        a = (min(u, v), max(u, v), len(boxes))
        arms.append(a)
        where.setdefault(a, lns)

    for j in junc:
        for ln in undrawn(j):
            if ln not in used:
                v, boxes, lns = walk(j, ln)
                add(j, v, boxes, lns)
    for b in free:
        if b in junc or all(ln in used for ln in undrawn(b)):
            continue
        # a link box not reached from a junction: a plain chain or a loop
        l1, l2 = undrawn(b)
        e1, bx1, ln1 = _walk_from(b, l1, junc, undrawn, across)
        if e1 == b:
            used.update(ln1)
            loops.append(len(bx1) + 1)
            where.setdefault(("loop", len(bx1) + 1), ln1)
            continue
        e2, bx2, ln2 = _walk_from(b, l2, junc, undrawn, across)
        add(-1, -1, bx1[::-1] + [b] + bx2, ln1[::-1] + ln2)
    return tuple(sorted(arms)), tuple(sorted(loops)), tuple(sorted(junc.items())), where


def _walk_from(b, ln, junc, undrawn, across):
    boxes, lns = [], [ln]
    prev, cur = ln, across(ln, b)
    while cur >= 0 and cur != b:
        if cur in junc:
            raise AssertionError("a link run from a junction was missed")
        boxes.append(cur)
        nxt = next(x for x in undrawn(cur) if x != prev)
        lns.append(nxt)
        prev, cur = nxt, across(nxt, cur)
    return cur, boxes, lns


def final_board(board: Board, safe: list[int]) -> Board:
    f = board.clone()
    for ln in safe:
        f.step(ln)
    return f


# --- the value -----------------------------------------------------------------------

def open_arm(arms: tuple, deg: tuple, a: tuple):
    """The opponent takes arm ``a`` (and whatever that makes takeable):
    (boxes taken, arms, new loops, deg) of what is left. A junction left
    with two undrawn sides joins its two arms into one (a loop if they are
    the same arm); with one or none it is taken too, and so is the arm it
    still has."""
    arms = list(arms)
    arms.remove(a)
    d = dict(deg)
    taken = a[2]
    loops = []
    todo = [e for e in a[:2] if e >= 0]
    while todo:
        for e in todo:
            d[e] -= 1
        nxt = []
        for j in sorted(set(todo)):
            if j not in d:
                continue
            if d[j] >= 3:
                continue
            inc = [x for x in arms if j in x[:2]]
            if d[j] == 2:
                del d[j]
                if len(inc) == 1:  # one arm with both ends here: a loop
                    arms.remove(inc[0])
                    loops.append(inc[0][2] + 1)
                else:
                    x, y = inc
                    arms.remove(x)
                    arms.remove(y)
                    ox = x[1] if x[0] == j else x[0]
                    oy = y[1] if y[0] == j else y[0]
                    arms.append((min(ox, oy), max(ox, oy), x[2] + y[2] + 1))
            else:  # 1 or 0 undrawn sides left: taken, with the arm it has
                del d[j]
                taken += 1
                for x in inc:
                    arms.remove(x)
                    taken += x[2]
                    nxt += [e for e in x[:2] if e >= 0 and e != j]
        todo = nxt
    return taken, tuple(sorted(arms)), tuple(loops), tuple(sorted(d.items()))


class _Budget:
    left = 0


_budget = _Budget()


@lru_cache(maxsize=2_000_000)
def _options(s: int, arms: tuple, loops: tuple, deg: tuple):
    """((value, kind, what), ...) best first; value = net boxes still to
    win for the side to move (Berlekamp's recursion, module docstring)."""
    _budget.left -= 1
    if _budget.left < 0:
        raise Unsettled("budget")
    opts = []
    if s > 0:
        opts.append((-value(s - 1, arms, loops, deg), SAFE, None))
    for i, a in enumerate(arms):
        if i and arms[i - 1] == a:
            continue
        n, arms2, new_loops, deg2 = open_arm(arms, deg, a)
        r = value(s, arms2, tuple(sorted(loops + new_loops)), deg2)
        opts.append((-(n + r) if n <= 2 else -max(n + r, n - 4 - r), CHAIN, a))
    for i, n in enumerate(loops):
        if i and loops[i - 1] == n:
            continue
        r = value(s, arms, loops[:i] + loops[i + 1:], deg)
        opts.append((-max(n + r, n - 8 - r), LOOP, n))
    if not opts:
        return ((0, None, None),)
    opts.sort(key=lambda o: -o[0])
    return tuple(opts)


def value(s: int, arms: tuple, loops: tuple, deg: tuple) -> int:
    return _options(s, arms, loops, deg)[0][0]


def settle(board: Board):
    """(safe lines, graph of F) if the position is settled; raises Unsettled."""
    if capturable(board):
        raise Unsettled("box to take")
    safe = safe_lines(board)
    if not safe_phase_fixed(board, safe):
        raise Unsettled("safe phase not fixed")
    return safe, graph(final_board(board, safe) if safe else board)


def options(board: Board, budget: int = 200_000):
    """Formula options at a settled position, best first, or None."""
    try:
        safe, (arms, loops, deg, where) = settle(board)
        _budget.left = budget
        return safe, where, _options(len(safe), arms, loops, deg)
    except Unsettled:
        return None


def solve(board: Board, budget: int = 200_000):
    """Net boxes still to win for the side to move, or None (not settled;
    positions with a box to take: see ``capture_decision``)."""
    r = options(board, budget)
    return None if r is None else r[2][0][0]


def why_not(board: Board) -> str | None:
    """None if the formula answers, else the reason."""
    try:
        safe, (arms, loops, deg, _) = settle(board)
        _budget.left = 200_000
        _options(len(safe), arms, loops, deg)
    except Unsettled as e:
        return str(e)
    return None


def best_move(board: Board, budget: int = 200_000):
    """(line, value, kind) by the formula, or None. With a box to take:
    ``capture_decision``."""
    if capturable(board):
        return capture_decision(board, budget)
    r = options(board, budget)
    if r is None:
        return None
    safe, where, opts = r
    v, kind, what = opts[0]
    if kind is None:
        return None
    if kind == SAFE:
        return safe[0], v, SAFE
    if kind == LOOP:
        return where[("loop", what)][0], v, LOOP
    lns = where[what]
    if what[2] == 2:
        return lns[1], v, CHAIN   # middle: hard-hearted handout
    if what[0] < 0 <= what[1]:
        return lns[-1], v, CHAIN  # the ground end (walked from the junction)
    return lns[0], v, CHAIN       # an end


def capture_decision(board: Board, budget: int = 200_000):
    """With boxes to take (the opponent opened something): take all and
    move on, or decline the last two (chain) / four (loop) boxes. Returns
    (line, value, 'all' | 'decline') or None if the board after taking
    everything is not settled. The line is the next one to draw."""
    geo = board.geo
    b2 = board.clone()
    k = 0
    while True:
        cap = capturable(b2)
        if not cap:
            break
        b = cap[0]
        ln = next(ln for ln in geo.box_lines[b] if not b2.lines[ln])
        k += b2.step(ln)
    r = solve(b2, budget)
    if r is None:
        return None
    take_all = k + r
    # the run the decline would be made on: from a box to take along boxes
    # with two drawn sides; a loop if it ends at another box to take
    best = (take_all, "all")
    runs = _runs(board)
    if len(runs) == 2:  # a chain opened in the middle: decline on the longer side
        runs.sort(key=lambda r: len(r[1]))
    if len(runs) == 1 or (len(runs) == 2 and runs[0][0] == runs[1][0] == CHAIN):
        kind, boxes, lns = runs[-1]
        give = 4 if kind == LOOP else 2
        if len(boxes) >= give:
            best = max(best, (k - 2 * give - r, "decline"))
    v, how = best
    if how == "all":
        b = capturable(board)[0]
        return next(ln for ln in geo.box_lines[b] if not board.lines[ln]), v, how
    kind, boxes, lns = runs[-1]
    if len(runs) == 2 or len(boxes) > (4 if kind == LOOP else 2):
        # first take the boxes that are taken either way
        b = runs[0][1][0] if len(runs) == 2 else boxes[0]
        return next(ln for ln in geo.box_lines[b] if not board.lines[ln]), v, how
    # the double-dealing move itself: the far line of a chain's last two
    # boxes (a domino left), the middle line of a loop's last four (two)
    return (lns[1] if kind == LOOP else lns[-1]), v, how


def _runs(board: Board):
    """Runs starting at a box to take: (kind, boxes, lines crossed), the
    lines from the first box outwards; a loop run is listed once."""
    geo, lines, sides = board.geo, board.lines, board.sides

    def across(ln, b):
        return next((o for o in geo.line_boxes[ln] if o != b), -1)

    out, used = [], set()
    for b in capturable(board):
        if b in used:
            continue
        used.add(b)
        boxes, lns = [b], []
        prev = None
        cur = b
        while True:
            nxt = [ln for ln in geo.box_lines[cur] if not lines[ln] and ln != prev]
            if not nxt:
                break
            ln = nxt[0]
            lns.append(ln)
            o = across(ln, cur)
            if o < 0 or sides[o] < 2:
                out.append((CHAIN, boxes, lns))
                break
            if sides[o] == 3:
                used.add(o)
                boxes.append(o)
                out.append((LOOP, boxes, lns))
                break
            boxes.append(o)
            prev, cur = ln, o
        else:
            pass
        if not out or out[-1][1] is not boxes:
            out.append((CHAIN, boxes, lns))
    return out


# --- one real ply, then the formula ----------------------------------------------

def _after(board: Board, budget: int):
    """Net boxes for the side to move at ``board`` (it may have boxes to
    take), by the formula; None if it has no answer."""
    if capturable(board):
        r = capture_decision(board, budget)
        return None if r is None else r[1]
    if board.n_drawn == board.geo.n_lines:
        return 0
    return solve(board, budget)


def ply_values(board: Board, budget: int = 200_000):
    """{line: net future boxes for the mover} with every real legal move
    played on the actual board and the position after it valued by the
    formula (so a sacrifice is the one the board offers now, not a chain of
    the final board F); None if any answer is missing. For a position
    without boxes to take."""
    out = {}
    for ln in range(board.geo.n_lines):
        if board.lines[ln]:
            continue
        b2 = board.clone()
        got = b2.step(ln)
        v = _after(b2, budget)
        if v is None:
            return None
        out[ln] = got + v if got else -v
    return out


def ply_move(board: Board, budget: int = 200_000):
    """(line, value, 'ply') by ``ply_values``, or None."""
    if capturable(board):
        return None
    vals = ply_values(board, budget)
    if not vals:
        return None
    ln = max(vals, key=vals.get)
    return ln, vals[ln], "ply"
