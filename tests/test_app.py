"""The app (scripts/launcher.py, launcher/): strings, chains, the network on
other board sizes, the sessions over the bridge and a clean exit."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from launcher import core, i18n
from sim.board import Board
from sim.chains import analyse

ROOT = Path(core.ROOT)


def test_strings_complete():
    for key, v in i18n.STRINGS.items():
        assert set(v) == set(i18n.LANGS), key
    assert i18n.t("ru", "mode_self") == "Сеть против себя"
    assert "10×7" in i18n.t("uk", "err_size", w=10, h=7, tw=5, th=5)


def test_desktop_entry():
    from launcher.desktop import entry

    e = entry("/x y/ksquares-rl")
    assert 'Exec="/x y/ksquares-rl"' in e and "Name[ru]=" in e and "Icon=ksquares-rl" in e


def test_download_best(tmp_path, monkeypatch):
    import hashlib

    src = tmp_path / "release.pt"
    src.write_bytes(b"weights" * 1000)
    cfg = core.app_config()
    dl = cfg["best"]["download"]
    dl.update(url=src.as_uri(), size=src.stat().st_size,
              sha256=hashlib.sha256(src.read_bytes()).hexdigest())
    monkeypatch.setattr(core, "ROOT", str(tmp_path / "clone"))
    seen = []
    core.download_best(cfg, lambda done, total: seen.append(done))
    best = tmp_path / "clone" / cfg["best"]["dir"]
    assert (best / "best.pt").read_bytes() == src.read_bytes()
    assert seen[-2:] == [src.stat().st_size, None]
    assert core.json.loads((best / "best.json").read_text())["gen"] == dl["gen"]
    (best / "best.pt").unlink()
    dl["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        core.download_best(cfg, lambda *a: None)
    assert not (best / "best.pt").exists() and not (best / "best.pt.part").exists()


def test_chains():
    # 3x1 strip with tops and bottoms drawn: one chain of three
    b = Board(3, 1)
    for ln in (0, 1, 2, 7, 8, 9):   # tops and bottoms
        b.step(ln)
    a = analyse(b)
    assert a["chains"] == [3] and a["long"] == 1 and a["loops"] == []
    assert a["dots"] == 8 and a["favoured"] == 1   # 8 + 1 odd: the second player
    # 2x2 with only the border drawn: the four boxes form a loop
    b = Board(2, 2)
    for ln in (0, 1, 2, 4, 7, 9, 10, 11):   # the outer border
        b.step(ln)
    a = analyse(b)
    assert a["loops"] == [4] and a["chains"] == [] and a["long"] == 0


def test_net_tied_to_board_size(tmp_path):
    """The heads are fully connected over the 11 x 11 grid: a 5x5 net cannot
    evaluate 10x7, and the app says so before a game starts."""
    torch = pytest.importorskip("torch")
    from agents.az_net import AZNet, Codec

    net = AZNet(5, 5, blocks=1, channels=8)
    x = torch.from_numpy(Codec(10, 7).encode_boards([Board(10, 7)]))
    with pytest.raises(RuntimeError):
        net(x)
    cfg = core.app_config()
    if core.best(cfg) is not None:
        main_only = {**cfg, "best_sizes": {}}   # 10x7 has its own network (best_sizes): without it
        msg = core.check("play", (10, 7), "en", main_only)
        assert msg and "10×7" in msg
        rc = tmp_path / "run.yaml"   # a run that never started: no link is made
        rc.write_text((ROOT / "configs/dot2s.yaml").read_text().replace("run_dir: runs/dot2s", f"run_dir: {tmp_path / 'run'}"))
        missing = {**cfg, "best_sizes": {"10x7": {"run_config": str(rc), "gen": None, "dir": str(tmp_path / "best")}}}
        assert str(tmp_path / "best") in core.check("play", (10, 7), "en", missing)
    b = core.best(cfg, (10, 7))
    if b is not None:
        assert core.check("play", (10, 7), "en", cfg) is None
        assert core.load_yaml(b["config"])["board"] == [10, 7]


def _have_ksquares():
    return os.access(core.GAME, os.X_OK) and all(shutil.which(p) for p in core.TOOLS)


def _children(pid):
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout
    return [int(p) for p in out.split()]


def _session(mode, **settings):
    from launcher.session import Session

    cfg = core.app_config()
    cfg.update(ai_delay_ms=0, net_delay=0, human_review=0, device="cpu", selfplay_delays=[0.0])
    best = core.best(cfg)
    if best is None:
        pytest.skip("no trained network (runs/best/best.pt)")
    st = core.load_settings(cfg)
    st.update(sims=0, level=0, net_first=True, size=0, speed=0, show_mine=False)
    st.update(settings)
    spec = (best["config"], best["path"], "best")
    return Session(mode, cfg, st, [spec, spec] if mode == "self" else [spec], seed=1)


def _wait(cond, timeout):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        time.sleep(0.05)
    return cond()


@pytest.mark.skipif(not _have_ksquares(), reason="KSquares not built or Xvfb / xdotool missing")
def test_ksquares_session_bridge_and_clean_stop():
    """The real game over the bridge on a private Xvfb: moves arrive, the
    window is grabbed, and stopping mid-game closes the game and the Xvfb."""
    s = _session("ksquares")
    assert _wait(lambda: s.snap and s.snap["board"].n_drawn >= 6, 90), s.error
    assert _wait(lambda: s.frame is not None, 30)
    assert s.frame.ndim == 3 and s.frame.shape[2] == 3
    assert s.snap["thought"]["label"] == "best"
    procs = [r for r in s._res]
    s.stop()
    s.thread.join(20)
    assert not s.thread.is_alive() and s.error is None
    xvfb, bridge = procs
    assert xvfb.proc.poll() is not None
    assert bridge._res.get("proc") is None   # waited for and gone


def test_play_session_clicks_and_illegal_lines():
    s = _session("play", net_first=False, show_mine=True)
    assert _wait(lambda: s.snap and s.snap["waiting"], 60), s.error
    assert s.snap["thought"]["for_human"]
    s.click(999)            # not a line: ignored
    s.click(0)
    assert _wait(lambda: s.snap["board"].lines[0] and s.snap["human"] is not None, 60)
    assert s.snap["human"]["line"] == 0 and 1 <= s.snap["human"]["rank"] <= 60
    assert _wait(lambda: s.snap["waiting"] and s.snap["board"].n_drawn >= 2, 60)
    s.click(0)              # already drawn: ignored, still waiting
    time.sleep(0.3)
    assert s.snap["waiting"]
    s.stop()
    s.thread.join(10)
    assert not s.thread.is_alive() and s.error is None


def test_self_session_pause_and_step():
    s = _session("self")
    s.pause(True)
    assert _wait(lambda: s.snap and s.snap["thought"] is not None, 60), s.error
    n = s.snap["board"].n_drawn
    time.sleep(0.5)
    assert s.snap["board"].n_drawn == n          # paused
    s.step_once()
    assert _wait(lambda: s.snap["board"].n_drawn == n + 1, 30)
    s.pause(False)
    assert _wait(lambda: s.over, 120)
    assert sum(s.snap["board"].score) == 25 and len(s.timeline) == 60
    s.thread.join(10)
    assert not s.thread.is_alive()


def test_app_clean_exit():
    """The app quits by itself after a short game and leaves no child
    processes behind (headless, "self" with no search)."""
    if core.best(core.app_config()) is None:
        pytest.skip("no trained network")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", XDG_CONFIG_HOME=str(ROOT / "build" / "test-config"))
    p = subprocess.Popen([sys.executable, str(ROOT / "scripts/launcher.py"), "--headless", "--mode", "self", "--board", "5x5",
                          "--sims", "0", "--moves", "8", "--linger", "0.2"], env=env)
    assert p.wait(timeout=180) == 0
    assert _children(p.pid) == []


def test_menu_rows_follow_the_mode(tmp_path, monkeypatch):
    """Only the selected mode's settings are on the menu; a click on another
    card selects it, a second click (or Start) would start it."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import importlib.util

    spec = importlib.util.spec_from_file_location("launcher_app", ROOT / "scripts/launcher.py")
    app_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_mod)
    app = app_mod.App(app_mod.parse(["--lang", "en"]), headless=True)
    try:
        rows = {}
        for mode in app_mod.MODES:
            app.settings["mode"] = mode
            app.frame()
            rows[mode] = {n.split(":")[0] for n in app.buttons}
        assert {"level", "first", "sims", "size"} <= rows["ksquares"] and not rows["ksquares"] & {"mine", "gen_a"}
        assert {"first", "sims", "size", "mine"} <= rows["play"] and not rows["play"] & {"level", "gen_a"}
        assert {"gen_a", "gen_b", "sims", "size"} <= rows["self"] and not rows["self"] & {"level", "first", "mine"}
        app.settings["mode"] = "self"
        app.menu_click("mode:play")
        assert app.settings["mode"] == "play" and app.screen_name == "menu"
    finally:
        app.pg.quit()


MOCK_TRAINER = r'''
import json, os, sys, time
run = sys.argv[1]
os.makedirs(os.path.join(run, "eval"), exist_ok=True)
hist = []
for gen in (1, 2, 3, 4, 5, 6):
    if os.path.exists(os.path.join(run, "STOP")):
        print("stopped cleanly", flush=True)
        break
    ok = gen != 2
    hist.append({"gen": gen, "accepted": ok, "score": 0.6 if ok else 0.4, "wins": 60 if ok else 40,
                 "losses": 40 if ok else 60, "ties": 0, "margin": 1.0, "time": time.time()})
    with open(os.path.join(run, "log.jsonl"), "a") as f:
        for ph, sec in (("selfplay", 600.0), ("train", 30.0), ("gate", 40.0)):
            f.write(json.dumps({"time": time.time(), "gen": gen, "phase": ph, "seconds": sec, "vs": gen - 1}) + "\n")
    if ok:
        with open(os.path.join(run, "eval", f"gen_{gen:04d}.json"), "w") as f:
            json.dump({"gen": gen, "raw": {"hard": [10 * gen, 100 - 10 * gen, 0]},
                       "mcts": {"hard": [20 * gen, 100 - 20 * gen, 0]}}, f)
        open(os.path.join(run, f"gen_{gen:04d}.pt"), "w").close()
    with open(os.path.join(run, "state.json"), "w") as f:
        json.dump({"gen": gen, "best": gen, "phase": "selfplay", "history": hist}, f)
    print(f"gen {gen} done", flush=True)
    time.sleep(0.3)
'''


def test_train_mode_with_a_mock_trainer(tmp_path, monkeypatch):
    """"Train your own Dot" without training: the run's config, one trainer
    at a time, the progress read from its files, the safe stop, and the
    accepted rounds in the generation lists."""
    from launcher import train

    monkeypatch.setattr(train, "RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(train, "LOCK", str(tmp_path / "runs" / ".training.json"))
    cfg = core.app_config()
    assert not train.valid_name("az5b") and not train.valid_name("../x") and train.valid_name("my_dot")
    m = train.machine(cfg)
    assert m["running"] is None and m["disk_gb"] > 0
    script = tmp_path / "mock_trainer.py"
    script.write_text(MOCK_TRAINER)
    run = tmp_path / "runs" / "my_dot"
    p = train.start("my_dot", 3, cfg, device="cpu", argv=[sys.executable, str(script), str(run)])
    try:
        rc = core.load_yaml(str(run / "train.yaml"))
        assert rc["init_from"] is None and rc["publish"] is False and rc["device"] == "cpu"
        assert rc["run_dir"].endswith("my_dot") and rc["net"] == core.load_yaml(cfg["train"]["base_config"])["net"]
        assert train.running() == "my_dot"
        with pytest.raises(RuntimeError):
            train.start("other", 1, cfg, device="cpu", argv=[sys.executable, "-c", "pass"])
        t0 = time.time()
        while time.time() - t0 < 20 and train.progress("my_dot", cfg)["gen"] < 3:
            time.sleep(0.1)
        pr = train.progress("my_dot", cfg)
        assert pr["running"] and pr["rounds"] == 3 and pr["gen"] >= 3
        assert [h["accepted"] for h in pr["history"][:3]] == [True, False, True]
        assert pr["history"][1]["vs"] == 1 and pr["history"][0]["seconds"] == 670.0
        assert pr["round_s"] == 670.0 and pr["eta_s"] is not None
        assert pr["ladder"][:2] == [(1, 10.0, 20.0), (3, 30.0, 60.0)]
        train.stop("my_dot")
        assert p.wait(timeout=20) == 0
        pr = train.progress("my_dot", cfg)
        assert not pr["running"] and train.running() is None and pr["gen"] < 6
        assert "stopped cleanly" in (run / "train.log").read_text()
    finally:
        if p.poll() is None:
            p.kill()
    # the accepted rounds join "network vs itself" and "play the network"
    labels = [g["label"] for g in core.generations(cfg)]
    assert "my_dot gen 1" in labels and "my_dot gen 3" in labels and "my_dot gen 2" not in labels


def test_train_screen_draws(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import importlib.util

    spec = importlib.util.spec_from_file_location("launcher_app", ROOT / "scripts/launcher.py")
    app_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_mod)
    app = app_mod.App(app_mod.parse(["--lang", "ru"]), headless=True)
    try:
        app.start("train")
        app.frame()
        assert app.screen_name == "train" and "rounds:10" in app.buttons and "name" in app.buttons
        app.train_click("name")
        app.settings["train_name"] = ""
        app.train_text("a/b")   # not a valid name: no Start
        app.frame()
        assert "train_start" not in app.buttons
    finally:
        app.pg.quit()


def test_train_progress_inside_a_round(tmp_path, monkeypatch):
    """The round's inside from the trainer's progress.jsonl and live.json (as
    az_train.py writes them; the trainer is mocked by this process), drawn."""
    import json
    import os

    from launcher import train

    sys.path.insert(0, str(ROOT / "scripts"))
    import az_train

    monkeypatch.setattr(train, "RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(train, "LOCK", str(tmp_path / "runs" / ".training.json"))
    cfg = core.app_config()
    run = tmp_path / "runs" / "my_dot"
    run.mkdir(parents=True)
    (run / "train.yaml").write_text("run_dir: runs/my_dot\nboard: [5, 5]\n")
    (run / "train.log").write_text("".join(f"line {k}\n" for k in range(15)))
    hist = [{"gen": 1, "accepted": True, "score": 0.7, "wins": 70, "losses": 30, "ties": 0}]
    (run / "state.json").write_text(json.dumps({"gen": 1, "best": 1, "phase": "selfplay", "history": hist}))
    (tmp_path / "runs" / ".training.json").write_text(json.dumps(
        {"name": "my_dot", "pid": os.getpid(), "rounds": 5, "first_gen": 0, "started": time.time()}))
    # self-play of round 2: two worker tasks, one followed on the board
    az_train.phase_start(run, 2, "selfplay", 2500)
    for task, done in ((0, 300), (1, 410), (0, 512)):
        az_train.Progress(run, 2, "selfplay", task)(done, force=True)
    with open(run / "progress.jsonl", "a") as f:
        f.write('{"gen": 2, "phase": "selfp')   # a line being written
    az_train.save_text(run / "live.json", json.dumps({"gen": 2, "game": 6, "moves": [0, 30, 5], "w": 5, "h": 5,
                                                      "time": time.time()}))
    pr = train.progress("my_dot", cfg)
    assert pr["running"] and pr["inner"] == {"phase": "selfplay", "done": 922, "of": 2500, "score": None}
    assert pr["live"]["moves"] == [0, 30, 5] and pr["quiet_s"] < 5
    assert pr["log"] == [f"line {k}" for k in range(5, 15)]
    # the match: the score summed over the tasks; the stale live game is dropped
    (run / "state.json").write_text(json.dumps({"gen": 1, "best": 1, "phase": "gate", "history": hist}))
    az_train.phase_start(run, 2, "gate", 100)
    az_train.Progress(run, 2, "gate", 0)(20, force=True, score=[12, 8, 0])
    az_train.Progress(run, 2, "gate", 1)(17, force=True, score=[9, 7, 1])
    pr = train.progress("my_dot", cfg)
    assert pr["inner"] == {"phase": "gate", "done": 37, "of": 100, "score": [21, 15, 1]} and pr["live"] is None
    # a quiet log: the warning
    old = time.time() - 300
    for f in ("train.log", "progress.jsonl"):
        os.utime(run / f, (old, old))
    assert train.progress("my_dot", cfg)["quiet_s"] > cfg["train"]["stale_s"]
    # the screen, with the live board
    (run / "state.json").write_text(json.dumps({"gen": 1, "best": 1, "phase": "selfplay", "history": hist}))
    az_train.phase_start(run, 2, "selfplay", 2500)
    import importlib.util

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    spec = importlib.util.spec_from_file_location("launcher_app", ROOT / "scripts/launcher.py")
    app_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_mod)
    app = app_mod.App(app_mod.parse(["--lang", "ru"]), headless=True)
    try:
        app.settings["train_name"] = "my_dot"
        app.start("train")
        app.frame()
        assert app._live[1].n_drawn == 3
        # round 1 accepted: the buttons into the modes with it; the network lists
        assert {"go:play", "go:self", "go:ksquares"} <= set(app.buttons)
        monkeypatch.setattr(core, "generations", lambda cfg: [
            {"run": "az5b", "gen": 54, "config": "configs/az5b.yaml", "path": "x", "label": "az5b gen 54",
             "own": False, "size": (5, 5)},
            {"run": "my_dot", "gen": 1, "config": "runs/my_dot/train.yaml", "path": "y", "label": "my_dot gen 1",
             "own": True, "size": (5, 5)}])
        started = []
        monkeypatch.setattr(app, "start", started.append)
        app.best = {"run": "az5b", "gen": 54}
        monkeypatch.setattr(core, "best", lambda cfg, size=None: app.best)
        app.train_click("go:self")
        assert started == ["self"] and (app.settings["gen_a"], app.settings["gen_b"]) == ("my_dot gen 1", "az5b gen 54")
        app.train_click("go:ksquares")
        assert started[-1] == "ksquares" and app.settings["gen_play"] == "my_dot gen 1"
        assert app.play_labels() == [None, "my_dot gen 1"]   # the released Dot and the own runs only
        assert "gen_play" in app.MODE_ROWS["ksquares"] and "gen_play" in app.MODE_ROWS["play"]
    finally:
        app.pg.quit()
