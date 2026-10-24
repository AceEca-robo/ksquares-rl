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


def test_net_tied_to_board_size():
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
        msg = core.check("play", (10, 7), "en", cfg)
        assert msg and "10×7" in msg


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
