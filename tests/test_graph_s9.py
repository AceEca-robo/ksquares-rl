"""Series 9 (configs/g5.yaml): Gc in AlphaZero. Config = az5b apart from the listed keys; Python and C trees
give the same search on Gc; the compiled evaluator gives the eager fp32 outputs; a run resumed after a phase
and a run with another worker count end with the same weights, buffer and history."""

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import ROOT, load_config  # noqa: E402

from agents.az_mcts import MCTS  # noqa: E402
from agents.az_net import Evaluator, make_net  # noqa: E402
from sim.board import Board  # noqa: E402

PLUMBING = {"run_dir", "net", "selfplay", "phase_seeds", "stop", "eval", "probes", "update_best", "after_eval",
            "limits", "bridge_eval", "series9", "init_from"}


def test_config_is_az5b():
    g, a = load_config(ROOT / "configs/g5.yaml"), load_config(ROOT / "configs/az5b.yaml")
    for k in set(g) | set(a):
        if k not in PLUMBING:
            assert g.get(k) == a.get(k), k
    assert "init_from" not in g and g["net"]["arch"] == "graph"
    for k in ("games_per_worker", "games_per_gen", "sims", "temp_moves"):
        assert g["selfplay"][k] == a["selfplay"][k], k
    assert {k: v for k, v in g["eval"].items() if k != "opponents"} == {k: v for k, v in a["eval"].items()
                                                                       if k != "opponents"}
    assert set(g["eval"]["opponents"]) - set(a["eval"]["opponents"]) == {"easy"}
    assert g["stop"] == {"max_rejects": 6} and g["update_best"] is False


def small_gc():
    torch.manual_seed(0)
    return make_net({"net": {"arch": "graph", "layers": 2, "width": 16, "pool": 4, "segments": True}})


def random_board(w, h, n, rng):
    b = Board(w, h)
    for _ in range(n):
        legal = [ln for ln in range(b.geo.n_lines) if not b.lines[ln]]
        if not legal:
            break
        b.step(rng.choice(legal))
    return b


@pytest.mark.parametrize("leaves", [1, 8])
def test_c_tree_matches_python_on_gc(leaves):
    from agents.mcts_c import load
    if load() is None:
        pytest.skip("no C compiler / cffi")
    ev = Evaluator(small_gc(), "cpu")
    rng = random.Random(leaves)
    boards = [random_board(5, 5, rng.randrange(0, 40), rng) for _ in range(6)]
    sims = [50, 100, 200, 7, 120, 64]
    out = []
    for backend in ("py", "c"):
        m = MCTS(ev, leaves_per_tree=leaves, rng=np.random.default_rng(1), backend=backend)
        roots = m.search([b.clone() for b in boards], sims, noise=True)
        out.append(([None if r.moves is None else (r.N.tolist(), r.W.tolist(), r.n) for r in roots], m.n_evals))
    assert out[0] == out[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_compiled_evaluator_equals_eager_fp32():
    cfg = load_config(ROOT / "configs/g5.yaml")
    torch.manual_seed(0)
    net = make_net(cfg)
    assert net.compile_eval
    rng = random.Random(3)
    boards = [random_board(5, 5, rng.randrange(0, 55), rng) for _ in range(150)]
    ev = Evaluator(net, "cuda")
    assert ev.compiled
    p1, v1 = ev(boards)
    p0, v0 = Evaluator(net, "cuda", compile=False)._eager_fp32(boards)
    assert np.abs(p1 - p0).max() < 1e-5 and np.abs(v1 - v0).max() < 1e-5
    assert (p1.argmax(1) == p0.argmax(1)).all()
    p2, _ = ev(boards[:37])                                # another batch size, same outputs
    assert np.abs(p2 - p1[:37]).max() < 1e-5


def tiny_cfg(tmp, name, workers):
    cfg = {"seed": 5, "board": [2, 2], "run_dir": str(tmp / name), "device": "cpu",
           "net": {"arch": "graph", "layers": 1, "width": 8, "pool": 4, "segments": True},
           "mcts": {"c_puct": 1.5, "fpu_reduction": 0.2, "dirichlet_alpha": 0.3, "dirichlet_eps": 0.25,
                    "virtual_loss": 1.0, "leaves_per_tree": 8},
           "selfplay": {"workers": workers, "tasks": 3, "games_per_worker": 4, "games_per_gen": 9, "sims": [4, 12],
                        "temp_moves": 3},
           "phase_seeds": True, "buffer_size": 1000,
           "train": {"batch_size": 16, "steps_per_gen": 3, "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0001,
                     "score_weight": 0.25, "symmetries": True, "min_buffer": 10},
           "gating": {"games": 4, "sims": 8, "temp_moves": 2, "tasks": 2, "threshold": 0.0},
           "eval_during": False}
    p = tmp / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def state_of(run):
    import json
    st = json.loads((run / "state.json").read_text())
    tr = torch.load(run / "trainer.pt", map_location="cpu")
    buf = np.load(run / "buffer.npz")
    return [(h["gen"], h["wins"], h["losses"], h["ties"]) for h in st["history"]], tr["net"], {k: buf[k] for k in buf.files}


def same(a, b):
    assert a[0] == b[0]
    assert all(torch.equal(a[1][k], b[1][k]) for k in a[1])
    assert all(np.array_equal(a[2][k], b[2][k]) for k in a[2])


def test_resume_and_worker_count(tmp_path, monkeypatch):
    import az_train

    full = az_train.Run(tiny_cfg(tmp_path, "full", 2), 2)
    full.loop()
    ref = state_of(tmp_path / "full")
    assert len(ref[0]) == 2

    # stopped after the self-play of gen 1 and after the training of gen 2, resumed each time
    stops = iter([("train", 1), ("gate", 2)])
    nxt = [next(stops)]

    def stop_once(self):
        if nxt[0] and (self.state["phase"], self.state["gen"] + 1) == nxt[0]:
            nxt[0] = next(stops, None)
            return True
        return False

    monkeypatch.setattr(az_train.Run, "stop_requested", stop_once)
    cfg = tiny_cfg(tmp_path, "resumed", 2)
    for _ in range(3):
        r = az_train.Run(cfg, None)
        if r.state["gen"] >= 2:
            break
        r.max_gens = 2 - r.state["gen"]
        r.loop()
    monkeypatch.undo()
    same(state_of(tmp_path / "resumed"), ref)

    one = az_train.Run(tiny_cfg(tmp_path, "one_worker", 1), 2)
    one.loop()
    same(state_of(tmp_path / "one_worker"), ref)
