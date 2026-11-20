"""Series 12 (scripts/graph_s12.py): nets and splits as in the protocol, overlap key, statistics, reading."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import ROOT, load_config  # noqa: E402
import graph_s12 as s12  # noqa: E402

CFG = load_config(ROOT / "configs/graph_s12.yaml")


def test_nets():
    L = s12.net_list(CFG)
    names = [n for _, n, _, _ in L]
    assert names == ["Dot-G w8/101", "Dot-G w8/102", "Dot-G w8/103", "Dot-2", "C+h s1", "C+h s2", "C+h s3"]
    assert len({s12.file_name(n, v) for n in names for v in CFG["train"]["variants"]}) == 14
    kinds = {n: k for _, n, k, _ in L}
    assert kinds["Dot-G w8/101"] == "graph" and kinds["Dot-2"] == "dot2" and kinds["C+h s1"] == "dot2"
    for _, _, _, spec in L:
        if not s12.weights_path(spec).exists():
            pytest.skip("weights not present")


def test_hyperparameters_are_mid_hand_r30():
    mh = load_config(ROOT / CFG["data"]["mid_hand"])["train"]
    tc = CFG["train"]
    for k in ("steps", "batch", "lr", "symmetries", "selfplay_buffers", "selfplay_weights"):
        assert tc[k] == mh[k], k
    assert tc["share"] == mh["variants"]["r30"]
    assert CFG["seed"] == load_config(ROOT / CFG["data"]["mid_hand"])["seed"]
    assert tc["variants"]["frozen"]["rest_lr"] == 0 and tc["variants"]["lr01"]["rest_lr"] == pytest.approx(0.1 * tc["lr"])


def test_heads():
    for _, name, kind, spec in s12.net_list(CFG):
        if not s12.weights_path(spec).exists():
            pytest.skip("weights not present")
        net = s12.load_net(spec)
        top = {n.split(".")[0] for n, _ in net.named_parameters()}
        assert s12.head_names(CFG, kind) <= top
        assert "v_fc" in top and "v_fc" not in s12.head_names(CFG, kind)


def test_splits():
    if not (ROOT / "runs/mid_judge/labels_mid.jsonl").exists() or not (ROOT / "runs/rout_mistakes/games.jsonl").exists():
        pytest.skip("data not present")
    D = s12.data(CFG)
    assert (len(D["mid_train"]), len(D["mid_held"])) == (1387, 538)
    assert sum(p["decisive"] for p in D["mid_held"]) == 199
    assert (len(D["mis_train"]), len(D["mis_held"])) == (56, 24)
    assert not {p["gid"] for p in D["mid_train"]} & {p["gid"] for p in D["mid_held"]}
    assert not {p["gid"] for p in D["mis_train"]} & {p["gid"] for p in D["mis_held"]}


def test_canon_lines_equals_canon():
    from agents.az_net import Codec
    from graph_s3p_checks import canon
    from sim.board import Board

    codec = Codec(10, 7, planes=17)
    rng = np.random.default_rng(0)
    ps = []
    for _ in range(20):
        b, mv = Board(10, 7), []
        for _ in range(int(rng.integers(0, 60))):
            ln = int(rng.choice([i for i in range(codec.n_lines) if not b.lines[i]]))
            b.step(ln)
            mv.append(ln)
        ps.append({"moves": mv})
    boards = []
    for p in ps:
        b = Board(10, 7)
        for ln in p["moves"]:
            b.step(ln)
        boards.append(b)
    x = codec.encode_boards(boards).reshape(len(boards), 17, -1)
    assert s12.canon_lines(s12.lines_of(ps), codec) == canon(x, codec, True)
    # a position and its mirror image share the key
    lp = codec.syms[1][1]
    L = s12.lines_of(ps)
    assert s12.canon_lines(L, codec) == s12.canon_lines(L[:, lp], codec)


def test_statistics():
    rng = np.random.default_rng(0)
    b = (rng.random(300) < 0.5).astype(int)
    d, lo, hi = s12.interval(np.stack([b, b, b]), b, 200, 1, 0.95)
    assert d == 0 and lo == 0 and hi == 0
    d, lo, hi = s12.interval(np.ones((3, 300)), b, 200, 1, 0.95)
    assert lo < d < hi and lo > 0
    d, lo, hi = s12.gain(np.ones((3, 300)), np.stack([b, b, b]), 200, 1, 0.95)
    assert lo > 0
    d, lo, hi = s12.gain(np.stack([b, b]), np.stack([b, b]), 200, 1, 0.95)
    assert d == lo == hi == 0


def test_reading():
    assert s12.reading(0.5, 3.0) == "the graph net learns the middle game better"
    assert s12.reading(-0.5, 3.0) == "no difference seen"
    assert s12.reading(-3.0, -0.5) == "the graph net learns the middle game worse"


def test_series_b_config():
    b = load_config(ROOT / "configs/graph_s12b.yaml")
    for k in ("seed", "eval_device", "limits", "data", "nets", "heads", "bootstrap"):
        assert b[k] == CFG[k], k
    tb, tc = b["train"], CFG["train"]
    for k in ("batch", "lr", "symmetries", "variants"):
        assert tb[k] == tc[k], k
    assert tb["share"] == 1.0 and not tb["selfplay_buffers"]
    seen = tc["steps"] * int(round(tc["share"] * tc["batch"]))
    assert seen == 460000 and tb["steps"] == round(seen / tb["batch"]) == 1198
    assert b["out"] != CFG["out"]


def test_scene6_phrase():
    p12 = {"frozen": (-4.2, -9.9, 1.5), "lr01": (-6.2, -10.6, -2.2)}
    ph, better = s12.scene6_phrase(p12, {"frozen": (1.0, -2.0, 4.0), "lr01": (-1.0, -3.0, 1.0)})
    assert ph.startswith("те же ответы судьи") and not better
    ph, better = s12.scene6_phrase(p12, {"frozen": (5.0, 1.0, 9.0), "lr01": (-1.0, -3.0, 1.0)})
    assert ph.startswith("на одних ответах") and better == ["frozen"]


def test_judge_only_training_runs():
    """A few steps of the judge-only trainer on a tiny pool: no replay needed, head-only keeps the rest."""
    import copy

    import torch
    spec = CFG["nets"]["dot2"]
    if not s12.weights_path(spec).exists():
        pytest.skip("weights not present")
    b = load_config(ROOT / "configs/graph_s12b.yaml")
    b = {**b, "device": "cpu", "train": {**b["train"], "steps": 3, "log_every": 1, "batch": 8}}
    net = s12.load_net(spec)
    net0 = copy.deepcopy(net)
    pool = [{"moves": [0, 1, 2], "correct": [5, 6]}, {"moves": [10], "correct": [3]}]
    log = s12.train_one(b, net, "dot2", pool, {}, "frozen")
    assert len(log) == 4 and log[-1]["policy"] == 0.0
    sd, sd0 = net.state_dict(), net0.state_dict()
    assert all(torch.equal(sd[k], sd0[k]) for k in sd0 if k.split(".")[0] not in CFG["heads"]["dot2"])
    assert not torch.equal(sd["p_out.2.weight"], sd0["p_out.2.weight"])
