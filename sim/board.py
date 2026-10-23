"""Dots and Boxes board, two players, numbered exactly like KSquares.

Board of ``w`` x ``h`` boxes (KSquares: width x height). Lines are numbered
row by row with stride ``2w+1``: in row ``r`` the ``w`` horizontal lines
come first (top edges of the boxes in row ``r``), then the ``w+1`` vertical
lines (left/right edges). The last row has only the ``w`` bottom lines.
Total ``2wh + w + h`` lines. Box ``i`` (row ``i // w``, column ``i % w``)
has the lines (see KSquaresGame::checkForNewSquares)::

    top    = (i // w) * (2w+1) + i % w
    left   = top + w
    right  = left + 1
    bottom = right + w

Rules as in KSquaresGame: drawing a line that completes one or two boxes
gives them to the mover, who moves again; otherwise the turn passes.
The game ends when every box is owned (then every line is drawn).
"""

from __future__ import annotations

from functools import lru_cache

NO_OWNER = -1


class Geometry:
    """Static index tables for a w x h board (shared between boards)."""

    __slots__ = ("w", "h", "n_lines", "n_boxes", "box_lines", "line_boxes")

    def __init__(self, w: int, h: int):
        if w < 1 or h < 1:
            raise ValueError(f"bad board size {w}x{h}")
        self.w, self.h = w, h
        self.n_lines = 2 * w * h + w + h
        self.n_boxes = w * h
        stride = 2 * w + 1
        box_lines = []
        line_boxes: list[list[int]] = [[] for _ in range(self.n_lines)]
        for i in range(self.n_boxes):
            top = (i // w) * stride + i % w
            left = top + w
            right = left + 1
            bottom = right + w
            box_lines.append((top, left, right, bottom))
            for ln in (top, left, right, bottom):
                line_boxes[ln].append(i)
        self.box_lines = tuple(box_lines)
        self.line_boxes = tuple(tuple(b) for b in line_boxes)

    def is_horizontal(self, line: int) -> bool:
        return line % (2 * self.w + 1) < self.w


@lru_cache(maxsize=None)
def geometry(w: int, h: int) -> Geometry:
    return Geometry(w, h)


class Board:
    """Mutable game state. ``player`` is the side to move (0 or 1)."""

    __slots__ = ("geo", "lines", "sides", "owner", "score", "player", "n_drawn")

    def __init__(self, w: int, h: int, first_player: int = 0):
        self.geo = geometry(w, h)
        self.lines = bytearray(self.geo.n_lines)
        self.sides = bytearray(self.geo.n_boxes)
        self.owner = [NO_OWNER] * self.geo.n_boxes
        self.score = [0, 0]
        self.player = first_player
        self.n_drawn = 0

    @property
    def w(self) -> int:
        return self.geo.w

    @property
    def h(self) -> int:
        return self.geo.h

    def clone(self) -> "Board":
        b = Board.__new__(Board)
        b.geo = self.geo
        b.lines = bytearray(self.lines)
        b.sides = bytearray(self.sides)
        b.owner = self.owner.copy()
        b.score = self.score.copy()
        b.player = self.player
        b.n_drawn = self.n_drawn
        return b

    @property
    def done(self) -> bool:
        return self.n_drawn == self.geo.n_lines

    def legal_moves(self) -> list[int]:
        lines = self.lines
        return [i for i in range(len(lines)) if not lines[i]]

    def step(self, line: int) -> int:
        """Draw ``line``; return the number of boxes completed (0, 1, 2).

        The mover keeps the turn iff the return value is non-zero.
        """
        if self.lines[line]:
            raise ValueError(f"line {line} is already drawn")
        self.lines[line] = 1
        self.n_drawn += 1
        captured = 0
        sides = self.sides
        for b in self.geo.line_boxes[line]:
            sides[b] += 1
            if sides[b] == 4:
                self.owner[b] = self.player
                captured += 1
        if captured:
            self.score[self.player] += captured
        else:
            self.player ^= 1
        return captured

    def winner(self) -> int | None:
        """0 or 1 for the leader (final result when done), None on a tie."""
        if self.score[0] == self.score[1]:
            return None
        return 0 if self.score[0] > self.score[1] else 1

    def __str__(self) -> str:
        w, h = self.w, self.h
        stride = 2 * w + 1
        rows = []
        for r in range(h + 1):
            s = "+"
            for c in range(w):
                s += ("---" if self.lines[r * stride + c] else "   ") + "+"
            rows.append(s)
            if r == h:
                break
            s = ""
            for c in range(w + 1):
                s += "|" if self.lines[r * stride + w + c] else " "
                if c < w:
                    o = self.owner[r * w + c]
                    s += f" {o} " if o != NO_OWNER else "   "
            rows.append(s)
        return "\n".join(rows)
