import ctypes

import pytest

from sim.board import Board
from sim.ksquares_ai import GlibcRand, KSquaresAI, qset_values, random_index

# Move lists of computer-vs-computer games recorded from the real KSquares
# 21.12.3 over the bridge (srand(seed) at reset, QT_HASH_SEED=0).
GOLDEN = {
    (3, 3, 2, 99): ([10, 20, 21, 14, 6, 16, 2, 12, 1, 15, 4, 3, 22, 23, 19, 17, 18, 11, 7, 0, 8, 5, 9, 13], [7, 2]),
    (5, 5, 1, 5): ([16, 2, 58, 5, 10, 6, 21, 30, 33, 52, 1, 22, 56, 19, 31, 20, 34, 24, 40, 32, 38, 43, 42, 9,
                    3, 55, 57, 41, 49, 54, 23, 59, 18, 13, 29, 28, 27, 35, 46, 51, 39, 44, 45, 50, 8, 7, 12, 14,
                    17, 11, 0, 25, 36, 47, 53, 48, 37, 26, 15, 4], [20, 5]),
    (4, 3, 0, 3): ([16, 6, 12, 13, 7, 4, 22, 27, 28, 9, 1, 24, 15, 26, 29, 30, 0, 5, 10, 18, 14, 19, 23, 20, 11,
                    2, 25, 21, 17, 3, 8], [5, 7]),
    (5, 5, 2, 123): ([3, 46, 12, 21, 22, 8, 26, 18, 1, 53, 41, 30, 27, 52, 31, 34, 49, 56, 43, 16, 50, 13, 10, 0,
                      5, 54, 4, 39, 38, 29, 42, 15, 20, 14, 9, 37, 32, 59, 48, 28, 33, 44, 55, 23, 17, 11, 6, 2,
                      7, 51, 57, 45, 40, 35, 24, 19, 25, 36, 47, 58], [19, 6]),
}


@pytest.mark.parametrize("seed", [0, 1, 2, 12345, 2**31 - 1, 2**31, 2**32 - 1])
def test_glibc_rand_matches_libc(seed):
    libc = ctypes.CDLL("libc.so.6")
    libc.srand(ctypes.c_uint(seed))
    g = GlibcRand(seed)
    assert [g.rand() for _ in range(1000)] == [libc.rand() for _ in range(1000)]


def test_unseeded_rand_is_seed_one():
    assert GlibcRand().rand() == 1804289383


def test_random_index_quirks():
    assert random_index(0, 1) == 0
    assert random_index(2**31 - 1, 2) == 1      # (float) rounds up to 2^31
    assert random_index(2**31 - 1000, 2) == 0   # otherwise never the last one
    assert random_index(2**30, 5) == 2
    assert all(random_index(r, 7) < 6 for r in range(0, 2**31 - 200, 2**31 // 1000))


def test_qset_order():
    # outputs of a Qt 5.15.3 program (QSet<int>::values(), QT_HASH_SEED=0)
    assert qset_values([5, 3, 22, 1]) == [1, 3, 5, 22]  # 22 % 17 == 5, after 5
    keys = list(range(0, 36, 2))  # 18 keys: grows to 37 buckets
    assert qset_values(keys) == sorted(keys, key=lambda k: (k % 37, k))
    assert qset_values([40, 6, 23]) == [40, 6, 23]  # all in bucket 6: insertion order


@pytest.mark.parametrize("key", list(GOLDEN))
def test_matches_recorded_games(key):
    w, h, difficulty, seed = key
    moves, scores = GOLDEN[key]
    ai = KSquaresAI(difficulty, GlibcRand(seed))
    b = Board(w, h)
    got = []
    while not b.done:
        ln = ai.act(b)
        got.append(ln)
        b.step(ln)
    assert got == moves
    assert b.score == scores
