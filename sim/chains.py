"""Chains and loops on the board as it stands, and the long chain rule.

A free box with exactly two undrawn sides is a link: it passes play on
through one side to the other. Maximal runs of links are chains (their
ends are the ground or a box with another number of undrawn sides) or
loops (closed runs). A chain of three or more boxes is long.

Long chain rule (Berlekamp, "The Dots-and-Boxes Game"): the first player
wants the number of dots plus the number of long chains at the end of the
fight to be even, the second player wants it odd. ``analyse`` counts the
long chains already formed, so early in the game the count still changes;
it says which player the count favours now.
"""

from __future__ import annotations

from sim.board import Board


def analyse(board: Board) -> dict:
    geo, sides = board.geo, board.sides
    free = [b for b in range(geo.n_boxes) if sides[b] < 4]
    link = {b for b in free if sides[b] == 2}

    def undrawn(b):
        return [ln for ln in geo.box_lines[b] if not board.lines[ln]]

    def across(ln, b):
        bs = geo.line_boxes[ln]
        return next((o for o in bs if o != b), -1)  # -1: the ground

    seen, chains, loops = set(), [], []
    for b in sorted(link):
        if b in seen:
            continue
        seen.add(b)
        n, closed = 1, False
        for start in undrawn(b):
            prev_line, cur = start, across(start, b)
            while cur in link and cur not in seen:
                seen.add(cur)
                n += 1
                nxt = next(ln for ln in undrawn(cur) if ln != prev_line)
                prev_line, cur = nxt, across(nxt, cur)
            if cur == b:
                closed = True
                break
        (loops if closed else chains).append(n)
    long_chains = sum(1 for c in chains if c >= 3)
    dots = (geo.w + 1) * (geo.h + 1)
    return {"chains": sorted(chains, reverse=True), "loops": sorted(loops, reverse=True),
            "long": long_chains, "dots": dots,
            # the player the long chain rule favours with the chains formed so far
            "favoured": 0 if (dots + long_chains) % 2 == 0 else 1}
