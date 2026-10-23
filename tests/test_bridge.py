"""Bridge tests against the real game (skipped when build/ is not there)."""

import sys
from pathlib import Path

import pytest

from env.bridge import DEFAULT_BINARY, KSquaresBridge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

pytestmark = pytest.mark.skipif(not DEFAULT_BINARY.exists(), reason="KSquares not built")


@pytest.fixture(scope="module")
def bridge():
    with KSquaresBridge() as br:
        yield br


@pytest.mark.parametrize("w,h,difficulty", [(3, 3, 2), (5, 5, 2), (5, 5, 1), (4, 6, 0)])
def test_game_matches_sim_move_by_move(bridge, w, h, difficulty):
    from verify_port import bridge_game, sim_game

    for g, seats in enumerate(([False, True], [True, False], [True, True])):
        seed, agent_seed = 1000 + g, 2000 + g
        ref, _ = sim_game(w, h, seats, difficulty, seed, agent_seed, hash_seed=0)
        assert bridge_game(bridge, w, h, seats, difficulty, seed, agent_seed) == ref


def test_illegal_moves_are_rejected(bridge):
    st = bridge.reset(3, 3, ai=[False, True], difficulty=2, seed=1)
    assert st["player"] == 0 and not st["moves"]
    st = bridge.step(0)
    with pytest.raises(ValueError, match="illegal"):
        bridge.step(0)
    with pytest.raises(ValueError, match="illegal"):
        bridge.step(999)
    # still usable afterwards
    while not st["done"]:
        st = bridge.step(bridge.board.legal_moves()[0])
    assert sum(st["scores"]) == 9
    with pytest.raises(ValueError, match="not an external"):
        bridge.step(0)
