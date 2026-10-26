"""The app's logic without the UI: paths, settings, the networks on disk,
the checks before a game starts, the private X server and what a failed
game means. scripts/launcher.py draws the window, launcher/session.py runs
the games, launcher/brain.py thinks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import select
import shutil
import subprocess
import time
import urllib.request

from . import i18n

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME = os.path.join(ROOT, "build", "src", "ksquares")
APP_CONFIG = os.path.join(ROOT, "configs", "app.yaml")
MODES = ("ksquares", "play", "self")

# program -> Debian / Ubuntu package (the private Xvfb, finding and grabbing
# KSquares' window on it)
TOOLS = {"Xvfb": "xvfb", "xdotool": "xdotool", "xwininfo": "x11-utils"}


def _xdg(var, default):
    return os.path.join(os.environ.get(var) or os.path.expanduser(default), "ksquares-rl")


CONFIG_DIR = _xdg("XDG_CONFIG_HOME", "~/.config")
CACHE_DIR = _xdg("XDG_CACHE_HOME", "~/.cache")
SETTINGS = os.path.join(CONFIG_DIR, "settings.json")
LOG = os.path.join(CACHE_DIR, "last.log")


def load_yaml(path):
    import yaml

    with open(os.path.join(ROOT, path)) as f:
        return yaml.safe_load(f)


def app_config():
    return load_yaml(APP_CONFIG)


# --- settings ---------------------------------------------------------------

def defaults(cfg):
    return {"lang": i18n.DEFAULT, "level": 2, "net_first": True, "sims": cfg["sims_default"],
            "size": 0, "show_mine": False, "gen_a": None, "gen_b": None,
            "speed": cfg["selfplay_delay_default"], "mode": MODES[0]}


def load_settings(cfg):
    s = defaults(cfg)
    try:
        with open(SETTINGS) as f:
            s.update({k: v for k, v in json.load(f).items() if k in s})
    except (OSError, ValueError):
        pass
    if s["lang"] not in i18n.LANGS:
        s["lang"] = i18n.DEFAULT
    if s["level"] not in (0, 1, 2):
        s["level"] = 2
    if s["sims"] not in cfg["sims_choices"]:
        s["sims"] = cfg["sims_default"]
    if not isinstance(s["size"], int) or not 0 <= s["size"] < len(cfg["board_sizes"]):
        s["size"] = 0
    if s["mode"] not in MODES:
        s["mode"] = MODES[0]
    if not isinstance(s["speed"], int) or not 0 <= s["speed"] < len(cfg["selfplay_delays"]):
        s["speed"] = cfg["selfplay_delay_default"]
    s["net_first"], s["show_mine"] = bool(s["net_first"]), bool(s["show_mine"])
    return s


def save_settings(s):
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        tmp = SETTINGS + ".tmp"
        with open(tmp, "w") as f:
            json.dump(s, f, indent=1)
        os.replace(tmp, SETTINGS)
    except OSError:
        pass   # not remembering the choice is no reason to fail


# --- networks -----------------------------------------------------------------

def best(cfg):
    """{"run", "gen", "config", "path"} of runs/best/best.pt (refreshed from
    the run's state first), or None."""
    import sys

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        from update_best import update
        update(cfg)
    except Exception:
        pass   # a read-only checkout: use the link as it is
    d = os.path.join(ROOT, cfg["best"]["dir"])
    try:
        with open(os.path.join(d, "best.json")) as f:
            info = json.load(f)
    except (OSError, ValueError):
        return None
    info["path"] = os.path.join(d, "best.pt")
    return info if os.path.isfile(info["path"]) else None


class Cancelled(Exception):
    pass


def download_best(cfg, progress, cancelled=lambda: False):
    """Download the release's network (best.download in configs/app.yaml)
    over https into runs/best/best.pt and describe it in best.json.
    progress(done, total) is called per chunk, with done = None while the
    checksum is checked. Raises Cancelled, ValueError (checksum mismatch)
    or OSError."""
    dl = cfg["best"]["download"]
    d = os.path.join(ROOT, cfg["best"]["dir"])
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "best.pt")
    part = path + ".part"
    h = hashlib.sha256()
    try:
        req = urllib.request.Request(dl["url"], headers={"User-Agent": "ksquares-rl-launcher"})
        with urllib.request.urlopen(req, timeout=30) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or dl["size"])
            done = 0
            while chunk := r.read(1 << 16):
                if cancelled():
                    raise Cancelled
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                progress(done, total)
        progress(None, total)
        if h.hexdigest() != dl["sha256"]:
            raise ValueError("sha256 mismatch")
        os.replace(part, path)
    finally:
        if os.path.exists(part):
            os.remove(part)
    info = {"run": dl["run"], "gen": dl["gen"], "config": cfg["best"]["run_config"],
            "checkpoint": dl["url"]}
    with open(os.path.join(d, "best.json.tmp"), "w") as f:
        json.dump(info, f, indent=1)
    os.replace(os.path.join(d, "best.json.tmp"), os.path.join(d, "best.json"))


def generations(cfg):
    """Every accepted generation on disk, in order: [{"run", "gen", "config",
    "path", "label"}]; the best network too when its run is not on disk (a
    fresh clone with the downloaded one)."""
    out = []
    for c in cfg["generations"]:
        rc = load_yaml(c)
        run = os.path.join(ROOT, rc["run_dir"])
        try:
            with open(os.path.join(run, "state.json")) as f:
                hist = json.load(f)["history"]
        except (OSError, ValueError):
            continue
        name = os.path.basename(rc["run_dir"])
        for h in hist:
            p = os.path.join(run, f"gen_{h['gen']:04d}.pt")
            if h["accepted"] and os.path.isfile(p):
                out.append({"run": name, "gen": h["gen"], "config": c, "path": p,
                            "label": f"{name} gen {h['gen']}"})
    b = best(cfg)
    if b is not None and not any(g["run"] == b["run"] and g["gen"] == b["gen"] for g in out):
        out.append({**b, "label": f"{b['run']} gen {b['gen']}"})
    return out


# --- checks -------------------------------------------------------------------

NO_NET = "no network"   # check()'s answer when runs/best/best.pt is missing: offer the download


def check(mode, size, lang, cfg):
    """The message that stops `mode` (ksquares / play / self) on board `size`,
    NO_NET, or None."""
    for name in ("torch", "yaml", "PIL"):
        if importlib.util.find_spec(name) is None:
            return i18n.t(lang, "err_no_module", name={"yaml": "PyYAML", "PIL": "Pillow"}.get(name, name))
    b = best(cfg)
    if b is None:
        return NO_NET
    trained = tuple(load_yaml(b["config"])["board"])
    if tuple(size) != trained:
        return i18n.t(lang, "err_size", w=size[0], h=size[1], tw=trained[0], th=trained[1])
    if mode == "ksquares":
        if not os.access(GAME, os.X_OK):
            return i18n.t(lang, "err_no_game")
        missing = [p for p in TOOLS if shutil.which(p) is None]
        if missing:
            return i18n.t(lang, "err_no_tool", tools=", ".join(missing),
                          pkgs=" ".join(sorted({TOOLS[p] for p in missing})))
    return None


def explain(error, lang):
    """The menu's message for a session that failed: error is (text,
    traceback); the traceback goes to LOG."""
    text, tb = error
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(LOG, "w") as f:
            f.write(tb)
    except OSError:
        pass
    if "out of memory" in text.lower():
        return i18n.t(lang, "err_gpu_memory")
    if any(s in text for s in ("game did not connect", "Xvfb did not start", "window not found",
                               "game closed the connection")):
        return i18n.t(lang, "err_game_start", log=LOG)
    return i18n.t(lang, "err_crash", error=text.splitlines()[0][:160], log=LOG)


# --- the private X server ---------------------------------------------------------

class PrivateX:
    """A private Xvfb for KSquares' window: nothing covers it there, so it can
    be grabbed at any time, and the person's desktop stays untouched."""

    def __init__(self, size=(1024, 900), timeout=10.0):
        from env.bridge import _die_with_parent

        r, w = os.pipe()
        self.proc = subprocess.Popen(
            ["Xvfb", "-displayfd", str(w), "-screen", "0", f"{size[0]}x{size[1]}x24", "-nolisten", "tcp"],
            pass_fds=(w,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, preexec_fn=_die_with_parent)
        os.close(w)
        try:
            ready = select.select([r], [], [], timeout)[0]
            num = os.read(r, 16).decode().strip() if ready else ""
        finally:
            os.close(r)
        if not num:
            self.close()
            raise RuntimeError("Xvfb did not start")
        self.name = ":" + num

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def find_window(display, pid, size, timeout=20.0):
    """KSquares' main window on `display`, resized to `size` and put at the
    origin; returns its box (x0, y0, x1, y1) for grabbing."""
    env = dict(os.environ, DISPLAY=display)
    t0 = time.time()
    win = None
    while time.time() - t0 < timeout:
        r = subprocess.run(["xdotool", "search", "--onlyvisible", "--pid", str(pid)],
                           capture_output=True, text=True, env=env)
        ids = r.stdout.split()
        if ids:
            win = ids[-1]
            break
        time.sleep(0.2)
    if win is None:
        raise RuntimeError("window not found")
    for cmd in (["windowsize", win, str(size[0]), str(size[1])], ["windowmove", win, "0", "0"]):
        subprocess.run(["xdotool", *cmd], env=env, capture_output=True)
    time.sleep(0.5)
    info = subprocess.run(["xwininfo", "-id", win], capture_output=True, text=True, env=env).stdout
    geo = {k.strip(): v.strip() for k, v in (ln.split(":", 1) for ln in info.splitlines() if ":" in ln)}
    x, y = int(geo["Absolute upper-left X"]), int(geo["Absolute upper-left Y"])
    return x, y, x + int(geo["Width"]), y + int(geo["Height"])
