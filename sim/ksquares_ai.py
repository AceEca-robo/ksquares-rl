"""Port of the KSquares 21.12.3 computer player (aicontroller.cpp).

``KSquaresAI(difficulty, rand).act(board)`` returns exactly the line that
``aiController::chooseLine()`` returns for the same position, difficulty
(0 Easy, 1 Medium, 2 Hard) and ``rand()`` state, including the quirks:

- the random index ``(short)((float)rand() / (RAND_MAX + 1.0) * (n - 1))``
  with its float32 roundings (the last candidate is chosen only when
  ``(float)rand()`` rounds up to 2^31 or the product rounds up to n-1),
- candidate lists with duplicates (one entry per adjacent box),
- ``QMultiMap::values()`` / ``begin()`` order (latest insertion first),
- the iteration order of ``QSet<int>::values()`` (Qt 5 QHash buckets) in the
  hard-hearted handout, which depends on the process-wide QHash seed
  (``QT_HASH_SEED``; random per process unless set),
- the walk order of ``squaresFromLine`` (horizontal: box below, box above;
  vertical: box right, box left), which decides where chains are followed.

``GlibcRand`` reproduces glibc's ``rand()``/``srand()`` (TYPE_3 additive
feedback generator); one instance stands for the game's process-wide state,
so all computer seats of one game share it. The code mirrors the C++
structure on purpose (it is checked against the real game over the bridge).
"""

from __future__ import annotations

import struct

from sim.board import Board, geometry

RAND_MAX = 2147483647
_F32 = struct.Struct("f")


def _f32(x: float) -> float:
    """Round a double to the nearest float32 (as a C float assignment)."""
    return _F32.unpack(_F32.pack(x))[0]


class GlibcRand:
    """glibc ``rand()`` with ``srand(seed)``; unseeded it behaves as srand(1)."""

    __slots__ = ("_r", "_i", "calls")

    def __init__(self, seed: int = 1):
        self.srand(seed)

    def srand(self, seed: int) -> None:
        seed &= 0xFFFFFFFF
        if seed == 0:
            seed = 1
        word = seed - (1 << 32) if seed >= 1 << 31 else seed  # int32_t
        r = [word]
        for _ in range(1, 31):
            hi = int(word / 127773)  # C division truncates towards zero
            lo = word - hi * 127773
            word = 16807 * lo - 2836 * hi
            if word < 0:
                word += 2147483647
            r.append(word)
        r = [x & 0xFFFFFFFF for x in r]
        # r[31..33] = r[0..2], so the ring of the last 31 values (r[j] at
        # index j % 31) is already r[0..30]; generation starts at i = 34
        self._r = r
        self._i = 34
        self.calls = 0
        for _ in range(310):
            self._next()

    def _next(self) -> int:
        r, i = self._r, self._i
        v = (r[i % 31] + r[(i - 3) % 31]) & 0xFFFFFFFF
        r[i % 31] = v
        self._i = i + 1
        return v >> 1

    def rand(self) -> int:
        self.calls += 1
        return self._next()

    def getstate(self):
        return (tuple(self._r), self._i, self.calls)

    def setstate(self, state) -> None:
        r, self._i, self.calls = state
        self._r = list(r)


def random_index(r: int, n: int) -> int:
    """``(short)((float)r / (RAND_MAX + 1.0) * (n - 1))`` as compiled."""
    return int(_f32(_f32(float(r)) / (RAND_MAX + 1.0) * (n - 1)))


# --- QHash<int> iteration order (Qt 5.15) --------------------------------

_PRIME_DELTAS = (0, 0, 1, 3, 1, 5, 3, 3, 1, 9, 7, 5, 3, 17, 27, 3,
                 1, 29, 3, 21, 7, 17, 15, 9, 43, 35, 15, 0, 0, 0, 0, 0)
_MIN_NUM_BITS = 4


def qset_values(keys, seed: int = 0) -> list[int]:
    """Order of ``QSet<int>::values()`` after inserting ``keys`` in order.

    Emulates QHashData: buckets ``h % numBuckets`` with ``h = key ^ seed``,
    new nodes appended to the bucket chain, ``rehash(numBits + 1)`` (at
    least 4 bits, ``numBuckets = 2^bits + prime_delta``) whenever an insert
    finds ``size >= numBuckets``, rehash re-appending nodes in old order.
    """
    num_bits, buckets, size = 0, [], 0
    present = set()
    for k in keys:
        if k in present:
            continue
        h = (k ^ seed) & 0xFFFFFFFF
        if size >= len(buckets):
            num_bits = max(num_bits + 1, _MIN_NUM_BITS)
            nb = (1 << num_bits) + _PRIME_DELTAS[num_bits]
            new = [[] for _ in range(nb)]
            for chain in buckets:
                for hk, kk in chain:
                    new[hk % nb].append((hk, kk))
            buckets = new
        buckets[h % len(buckets)].append((h, k))
        present.add(k)
        size += 1
    return [k for chain in buckets for _, k in chain]


# --- the AI ----------------------------------------------------------------


class _Ctx:
    """Board geometry in aiController's terms (w, h, squaresFromLine order)."""

    __slots__ = ("w", "h", "n_lines", "n_sq", "box_lines", "sfl")

    def __init__(self, w: int, h: int):
        geo = geometry(w, h)
        self.w, self.h = w, h
        self.n_lines, self.n_sq = geo.n_lines, geo.n_boxes
        self.box_lines = geo.box_lines  # top, left, right, bottom (linesFromSquare)
        self.sfl = tuple(self._squares_from_line(ln) for ln in range(self.n_lines))

    def _squares_from_line(self, line: int):
        w, h = self.w, self.h
        stride = 2 * w + 1
        if line % stride < w:  # HORIZONTAL
            sq = [line - (w + 1) * (line // stride)]
            sq.append(sq[0] - w)
            if sq[1] < 0:
                sq.pop(1)
            if sq[0] >= w * h:
                sq.pop(0)
        else:
            sq = [line - ((line // stride) * w + (line // stride) + w)]
            sq.append(sq[0] - 1)
            if line % stride == w:
                sq.pop(1)
            if line % stride == 2 * w:
                sq.pop(0)
        return tuple(sq)


_CTX: dict[tuple[int, int], _Ctx] = {}


def _ctx(w: int, h: int) -> _Ctx:
    c = _CTX.get((w, h))
    if c is None:
        c = _CTX[(w, h)] = _Ctx(w, h)
    return c


class KSquaresAI:
    """KSquares computer player. ``rand`` is the (shared) GlibcRand.

    stats: counters of rare situations (``ambiguous_handout``: the
    hard-hearted handout had several candidate lines, so its choice depends
    on the QHash seed).
    """

    def __init__(self, difficulty: int, rand: GlibcRand | None = None, hash_seed: int = 0):
        if difficulty not in (0, 1, 2):
            raise ValueError("difficulty must be 0, 1 or 2")
        self.difficulty = difficulty
        self.rand = rand if rand is not None else GlibcRand()
        self.hash_seed = hash_seed
        self.name = ("easy", "medium", "hard")[difficulty]
        self.stats = {"ambiguous_handout": 0}

    def act(self, board: Board) -> int:
        return self.choose_line(list(map(bool, board.lines)), board.w, board.h)

    # aiController::chooseLine
    def choose_line(self, lines: list[bool], w: int, h: int) -> int:
        c = _ctx(w, h)
        sfl, box_lines = c.sfl, c.box_lines

        def cnt(sq):
            t, l, r, b = box_lines[sq]
            return lines[t] + lines[l] + lines[r] + lines[b]

        choice = []
        for i in range(c.n_lines):
            if not lines[i]:
                for sq in sfl[i]:
                    if cnt(sq) == 3:
                        choice.append(i)
        if choice:
            if self.difficulty == 2:
                open_lines = [i for i in range(c.n_lines) if not lines[i]]
                choices = self._least_damaging(c, lines, open_lines)
                if choices:
                    return choices[0]
            return choice[random_index(self.rand.rand(), len(choice))]

        choice = self._safe_moves(c, lines)
        if choice:
            return choice[random_index(self.rand.rand(), len(choice))]

        choice = []
        for i in range(c.n_lines):
            if not lines[i]:
                for sq in sfl[i]:
                    if cnt(sq) == 2:
                        choice.append(i)
        if self.difficulty >= 1:
            good = self._least_damaging(c, lines, choice)
            if good:
                return good[random_index(self.rand.rand(), len(good))]
        if choice:
            return choice[random_index(self.rand.rand(), len(choice))]
        return 0

    @staticmethod
    def _safe_moves(c: _Ctx, lines) -> list[int]:
        sfl, box_lines = c.sfl, c.box_lines
        out = []
        for i in range(c.n_lines):
            if not lines[i]:
                for sq in sfl[i]:
                    t, l, r, b = box_lines[sq]
                    if lines[t] + lines[l] + lines[r] + lines[b] == 2:
                        break
                else:
                    out.append(i)
        return out

    # aiController::chooseLeastDamaging
    def _least_damaging(self, c: _Ctx, lines, choice_list) -> list[int]:
        sfl, box_lines, n_sq = c.sfl, c.box_lines, c.n_sq
        hard = self.difficulty > 1

        def cnt(sq, ls):
            t, l, r, b = box_lines[sq]
            return ls[t] + ls[l] + ls[r] + ls[b]

        damage = []   # QMultiMap<int,int> as (count, line) in insertion order
        chains = {}   # QMap<int, QSet<int>>: line -> keys in insertion order
        own_chains = []
        own_moves = []
        own_lines_cnt = 0
        own_squares_cnt = 0
        my_lines = None

        if hard:
            my_lines = list(lines)
            while True:  # complete all chains the opponent gave to the AI
                chain_found = False
                for cur in range(n_sq):
                    own_chain = []
                    chain_sq = cur
                    while True:
                        square_found = False
                        sides = box_lines[chain_sq]
                        if cnt(chain_sq, my_lines) == 3:
                            for s in sides:
                                if not my_lines[s]:
                                    own_lines_cnt += 1
                                    nxt = -1
                                    for a in sfl[s]:
                                        k = cnt(a, my_lines)
                                        if chain_sq != a and k == 3:
                                            own_squares_cnt += 1
                                        if k == 2:
                                            nxt = a
                                    my_lines[s] = True
                                    if nxt >= 0:
                                        chain_sq = nxt
                                    own_chain.append(s)
                            square_found = True
                            chain_found = True
                            own_squares_cnt += 1
                        if not square_found:
                            break
                    if chain_found:
                        own_chains.append(own_chain)
                        break
                if not chain_found:
                    break
            best = c.w * c.h * 10
            for oc in own_chains:
                if best > len(oc):
                    best = len(oc)
                    own_moves = oc

        for cand in choice_list:
            if hard:
                lc = list(my_lines)
                if lc[cand]:
                    continue  # already covered: the AI takes this line itself
            else:
                lc = list(lines)
            lc[cand] = True
            count = 0
            chain = [cand]
            while True:
                square_found = False
                for cs in range(n_sq):
                    sides = box_lines[cs]
                    if cnt(cs, lc) == 3:
                        count += 1
                        square_found = True
                        for s in sides:
                            if hard and not lc[s]:
                                if s not in chain:
                                    chain.append(s)
                                for a in sfl[s]:
                                    if a == cs:
                                        continue
                                    if cnt(a, my_lines) == 3:  # sic: myLines
                                        count += 1
                            lc[s] = True
                        break
                if not square_found:
                    break
            damage.append((count, cand))
            chains[cand] = chain

        # linePointDamage.begin() on an empty map is UB in KSquares; it cannot
        # be reached: Medium and Hard phase 3 always have candidates, and Hard
        # phase 1 returns an own move before begin() is used
        min_key = min(k for k, _ in damage) if damage else None

        def values(key):
            return [ln for k, ln in reversed(damage) if k == key]

        if not hard:
            if not damage:
                raise RuntimeError("linePointDamage empty (UB in KSquares)")
            return values(min_key)

        chain_set = {}  # unique chains, QMap key order
        for key in sorted(chains):
            ch = chains[key]
            s = set(ch)
            if all(set(v) != s for v in chain_set.values()):
                chain_set[key] = ch
        long_chains = sum(1 for v in chain_set.values() if len(v) > 3)

        if ((own_lines_cnt == 2 or (own_lines_cnt == 3 and own_squares_cnt == 4))
                and long_chains > 0 and not self._safe_moves(c, lines)):
            magic = -1
            for i in range(len(own_moves)):
                ml = list(lines)
                ml[own_moves[i]] = True
                for j in range(len(own_moves)):
                    if i == j:
                        continue
                    left = sum(1 for a in sfl[own_moves[j]] if cnt(a, ml) == 3)
                    if left == 2:
                        magic = i
            if len(own_moves) > 1:
                k = magic if 0 <= magic < len(own_moves) else 1
                return [own_moves[k]]

        if own_moves:
            return [own_moves[0]]

        if not damage:
            raise RuntimeError("linePointDamage empty (UB in KSquares)")
        if min_key == 2:
            first = values(min_key)[0]  # linePointDamage.begin().value()
            opp = []
            for key in sorted(chain_set):
                if first in chain_set[key]:
                    opp = qset_values(chain_set[key], self.hash_seed)
            handout = -1
            qualifying = set()
            for i in range(len(opp)):
                ml = list(lines)
                ml[opp[i]] = True
                for j in range(len(opp)):
                    if i == j:
                        continue
                    bad = sum(1 for a in sfl[opp[j]] if cnt(a, ml) != 3)
                    if bad == 0:
                        handout = i
                        qualifying.add(opp[i])
            if handout >= 0:
                if len(qualifying) > 1:
                    self.stats["ambiguous_handout"] += 1
                return [opp[handout]]

        return values(min_key)


def make_ai(name: str, rand: GlibcRand | None = None, hash_seed: int = 0) -> KSquaresAI:
    return KSquaresAI({"easy": 0, "medium": 1, "hard": 2}[name], rand, hash_seed)
