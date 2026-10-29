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
MODES = ("ksquares", "play", "self", "train")

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
            "speed": cfg["selfplay_delay_default"], "mode": MODES[0], "gen_play": None,
            "train_name": cfg["train"]["default_name"], "train_rounds": cfg["train"]["rounds_default"]}


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
    if s["train_rounds"] not in cfg["train"]["rounds_choices"]:
        s["train_rounds"] = cfg["train"]["rounds_default"]
    if not isinstance(s["train_name"], str):
        s["train_name"] = cfg["train"]["default_name"]
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

def best_section(cfg, size=None):
    """configs/app.yaml's section of the best network for a board size:
    ``best_sizes["WxH"]``, else ``best``."""
    if size is not None:
        bc = (cfg.get("best_sizes") or {}).get(f"{size[0]}x{size[1]}")
        if bc is not None:
            return bc
    return cfg["best"]


def best(cfg, size=None):
    """{"run", "gen", "config", "path"} of runs/best/best.pt, or of the
    board size's own link (``best_sizes``; refreshed from the run's state
    first), or None."""
    import sys

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    try:
        from update_best import update
        update(cfg)
    except Exception:
        pass   # a read-only checkout: use the link as it is
    d = os.path.join(ROOT, best_section(cfg, size)["dir"])
    try:
        with open(os.path.join(d, "best.json")) as f:
            info = json.load(f)
    except (OSError, ValueError):
        return None
    info["path"] = os.path.join(d, "best.pt")
    return info if os.path.isfile(info["path"]) else None


class Cancelled(Exception):
    pass


def release_items(cfg):
    """The release's networks: [{"download", "dir", "file", "config"}] for
    ``best``, every ``best_sizes`` entry with a ``download``, and the
    ``release_generations`` (extra generations offered in the menus)."""
    out = [{"download": cfg["best"]["download"], "dir": cfg["best"]["dir"], "file": "best.pt",
            "config": cfg["best"]["run_config"]}]
    for bc in (cfg.get("best_sizes") or {}).values():
        if bc.get("download"):
            out.append({"download": bc["download"], "dir": bc["dir"], "file": "best.pt", "config": bc["run_config"]})
    for g in cfg.get("release_generations") or []:
        out.append({"download": g["download"], "dir": g["dir"], "file": g["file"], "config": g["config"]})
    return out


def missing_downloads(cfg):
    """The release items not on disk (a fresh clone: all of them)."""
    return [it for it in release_items(cfg) if not os.path.isfile(os.path.join(ROOT, it["dir"], it["file"]))]


def download_best(cfg, progress, cancelled=lambda: False):
    """Download the release's missing networks (missing_downloads) over https,
    each into its dir and described in <file stem>.json (best.json for a
    best.pt). progress(done, total) is called per chunk with the bytes of all
    of them, with done = None while the checksums are checked. Raises
    Cancelled, ValueError (checksum mismatch) or OSError."""
    items = missing_downloads(cfg)
    total = sum(it["download"]["size"] for it in items)
    done = 0
    for it in items:
        dl = it["download"]
        d = os.path.join(ROOT, it["dir"])
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, it["file"])
        part = path + ".part"
        h = hashlib.sha256()
        try:
            req = urllib.request.Request(dl["url"], headers={"User-Agent": "ksquares-rl-launcher"})
            with urllib.request.urlopen(req, timeout=30) as r, open(part, "wb") as f:
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
        info = {"run": dl["run"], "gen": dl["gen"], "config": it["config"], "checkpoint": dl["url"]}
        meta = os.path.join(d, os.path.splitext(it["file"])[0] + ".json")
        with open(meta + ".tmp", "w") as f:
            json.dump(info, f, indent=1)
        os.replace(meta + ".tmp", meta)


def generations(cfg):
    """Every accepted generation on disk, in order: [{"run", "gen", "config",
    "path", "label", "own", "size"}] (size: the run's board; own: a run of "Train your own Dot"): the project's runs, the best network too when its run
    is not on disk (a fresh clone with the downloaded one), then the app's
    own runs ("Train your own Dot", runs/<name>/train.yaml)."""
    from . import train

    out = []
    own = [os.path.relpath(os.path.join(train.run_dir(n), "train.yaml"), ROOT) for n in train.own_runs()]
    for c in cfg["generations"] + own:
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
                            "label": f"{name} gen {h['gen']}", "own": c in own, "size": tuple(rc["board"])})
    # the downloaded networks of runs not on disk (a fresh clone)
    extra = [best(cfg)] + [best(cfg, tuple(int(v) for v in k.split("x"))) for k in cfg.get("best_sizes") or {}]
    for g in cfg.get("release_generations") or []:
        p = os.path.join(ROOT, g["dir"], g["file"])
        if os.path.isfile(p):
            extra.append({"run": g["download"]["run"], "gen": g["download"]["gen"], "config": g["config"], "path": p})
    k = sum(1 for g in out if g["config"] in cfg["generations"])
    for b in extra:
        if b is not None and not any(g["run"] == b["run"] and g["gen"] == b["gen"] for g in out):
            out.insert(k, {**b, "label": f"{b['run']} gen {b['gen']}", "own": False,
                           "size": tuple(load_yaml(b["config"])["board"])})
            k += 1
    return out


# --- checks -------------------------------------------------------------------

NO_NET = "no network"   # check()'s answer when runs/best/best.pt is missing: offer the download


def check(mode, size, lang, cfg):
    """The message that stops `mode` (ksquares / play / self / train) on board
    `size`, NO_NET, or None."""
    for name in ("torch", "yaml", "PIL"):
        if importlib.util.find_spec(name) is None:
            return i18n.t(lang, "err_no_module", name={"yaml": "PyYAML", "PIL": "Pillow"}.get(name, name))
    if mode == "train":
        return None   # needs no network: launcher/train.py checks the machine
    b = best(cfg, size)
    if b is None and best_section(cfg, size) is not cfg["best"] and not best_section(cfg, size).get("download"):
        return i18n.t(lang, "err_size_net", w=size[0], h=size[1], dir=best_section(cfg, size)["dir"])
    if b is None:
        return NO_NET
    if tuple(load_yaml(b["config"])["board"]) != tuple(size):   # no network of this board configured
        return i18n.t(lang, "err_size_net", w=size[0], h=size[1], dir=best_section(cfg, size)["dir"])
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
