#!/usr/bin/env python3
"""The app: one 1920x1080 pygame window (F11: full screen) that starts with
the menu and shows every mode in place:

    Network vs KSquares   the real KSquares (Easy / Medium / Hard) on a private
                          Xvfb, its window grabbed into the centre; the network
                          moves through the RL bridge
    Play the network      a board of your own: click a line to draw it
    Network vs itself     two generations (or one) against each other; Space
                          pauses, → plays one move, +/− change the speed
    Train your own Dot    checks the GPU and the disk, starts scripts/az_train.py
                          from scratch into runs/<name>/ as a process of its own
                          (it outlives the app) and shows its progress; its
                          accepted rounds join the generations of the modes above

The thought panel on the right shows the network's side: the policy as a
heat map over the lines, V and a "who is winning" bar, the MCTS visits
with the principal variation numbered on the board, the long chains on the
board with the side the long chain rule favours, and whether the network's
move keeps the win (exact solving in the chain fight). Below, V move by
move. Search: none (the policy's argmax) or 50 / 200 / 800 simulations.

The network is runs/best/best.pt: az5b gen 54, pinned by best.gen in
configs/app.yaml (without the pin, the last accepted generation;
scripts/update_best.py keeps the link current); in a fresh clone, without
runs/, the menu offers to download it from the v0.1 release (best.download
in configs/app.yaml). "network vs itself" can pick any accepted generation
of az5 and az5b on disk. Esc goes back to the menu
from any mode (the game, its Xvfb and the worker are closed), Esc in the
menu quits. English, Russian or Ukrainian (launcher/i18n.py), switched in
the menu's corner; the choices are kept in ~/.config/ksquares-rl/.

    python scripts/launcher.py
    python scripts/launcher.py --mode self --gen-a "az5 gen 1" --record docs/self.mp4
    python scripts/launcher.py --mode ksquares --screenshot docs/app_ksquares.png --moves 30
    python scripts/launcher.py --mode train --screenshot docs/app_train.png
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from launcher import core, i18n, train  # noqa: E402
from launcher.i18n import t  # noqa: E402

W, H = 1920, 1080
BG = (16, 18, 24)
PANEL = (26, 29, 38)
PANEL_HI = (38, 42, 56)
FG = (226, 229, 236)
DIM = (140, 146, 160)
FAINT = (78, 83, 96)
ACCENT = (255, 196, 0)
GREEN = (80, 200, 120)
RED = (235, 80, 70)
SEAT = ((240, 96, 80), (86, 156, 255))       # first / second player, as in KSquares (red, blue)
HEAT = (255, 196, 0)                          # policy
VISIT = (110, 220, 160)                       # MCTS visits
MODES = core.MODES
MAIN = (40, 110, 1000, 760)                   # x, y, w, h: the game
SIDE = (1080, 40, 800, 860)                   # the thought panel
TIMELINE = (40, 912, 1840, 140)


class Download:
    """The release network's download in a thread; the menu polls its fields."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.done = self.total = 0
        self.verifying = False
        self.result = None   # None while running, then "ok", "cancelled", "hash" or the error
        self.cancel = False
        threading.Thread(target=self.run, daemon=True).start()

    def progress(self, done, total):
        if done is None:
            self.verifying = True
        else:
            self.done, self.total = done, total

    def run(self):
        try:
            core.download_best(self.cfg, self.progress, lambda: self.cancel)
            self.result = "ok"
        except core.Cancelled:
            self.result = "cancelled"
        except ValueError:
            self.result = "hash"
        except Exception as e:   # network errors come as several types
            self.result = str(getattr(e, "reason", None) or e) or type(e).__name__


class Recorder:
    """The window's frames -> mp4 or gif through ffmpeg (gif: one palette
    for the whole clip)."""

    def __init__(self, path, fps, size=(1280, 720)):
        self.size, self.fps = size, fps
        vf = "split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer" \
            if path.endswith(".gif") else "format=yuv420p"
        codec = [] if path.endswith(".gif") else ["-c:v", "libx264", "-crf", "22", "-movflags", "+faststart"]
        self.proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-vf", vf, *codec, path],
            stdin=subprocess.PIPE)
        self.next = 0.0

    def frame(self, pg, screen, now):
        if now < self.next:
            return
        self.next = (self.next or now) + 1.0 / self.fps
        small = pg.transform.smoothscale(screen, self.size)
        self.proc.stdin.write(pg.image.tobytes(small, "RGB"))

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


class BoardView:
    """Geometry of a w x h board drawn into a rectangle."""

    def __init__(self, w, h, rect):
        from agents.az_net import Codec

        self.w, self.h = w, h
        self.codec = Codec(w, h)
        x, y, rw, rh = rect
        self.cell = min(rw / (w + 0.6), rh / (h + 0.6))
        self.x0 = x + (rw - self.cell * w) / 2
        self.y0 = y + (rh - self.cell * h) / 2

    def pt(self, gx, gy):
        """Grid (11 x 11 on 5x5) coordinates -> screen; dots at even ones."""
        return self.x0 + gx / 2 * self.cell, self.y0 + gy / 2 * self.cell

    def ends(self, ln):
        r, c = self.codec.line_rc[ln]
        if r % 2 == 0:
            return self.pt(c - 1, r), self.pt(c + 1, r)
        return self.pt(c, r - 1), self.pt(c, r + 1)

    def mid(self, ln):
        r, c = self.codec.line_rc[ln]
        return self.pt(c, r)

    def line_at(self, pos, board):
        best, dist = None, self.cell * 0.28
        for ln in range(board.geo.n_lines):
            if board.lines[ln]:
                continue
            (ax, ay), (bx, by) = self.ends(ln)
            dx, dy = bx - ax, by - ay
            k = max(0.0, min(1.0, ((pos[0] - ax) * dx + (pos[1] - ay) * dy) / (dx * dx + dy * dy)))
            d = math.hypot(pos[0] - ax - k * dx, pos[1] - ay - k * dy)
            if d < dist:
                best, dist = ln, d
        return best


class App:
    def __init__(self, args, headless=False):
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        os.environ.setdefault("SDL_VIDEO_X11_WMCLASS", "ksquares-rl")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        import pygame
        self.pg = pygame
        pygame.init()
        flags = 0
        if not headless:
            flags = pygame.SCALED
            sizes = pygame.display.get_desktop_sizes()
            if sizes and (sizes[0][0] < W or sizes[0][1] < H):
                flags |= pygame.FULLSCREEN
        self.screen = pygame.display.set_mode((W, H), flags)
        pygame.display.set_caption("KSquares RL")
        try:
            pygame.display.set_icon(pygame.image.load(os.path.join(ROOT, "launcher", "ksquares-rl.svg")))
        except (pygame.error, FileNotFoundError):
            pass
        self.args = args
        self.cfg = core.app_config()
        self.settings = core.load_settings(self.cfg)
        self.save = not headless
        for k in ("lang", "level", "sims", "net_first", "show_mine"):
            if getattr(args, k, None) is not None:
                self.settings[k] = type(self.settings[k])(getattr(args, k))
        if args.board:
            self.settings["size"] = [f"{bw}x{bh}" for bw, bh in self.cfg["board_sizes"]].index(args.board)
        if args.gen_a:
            self.settings["gen_a"] = args.gen_a
        if args.gen_b:
            self.settings["gen_b"] = args.gen_b
        self.find_networks()
        self.offer = False      # the download dialog is up
        self.download = None    # its Download once started
        self.fonts = {}
        self.buttons = {}
        self.hover = None
        self.message = None
        self.screen_name = "menu"
        self.mode = None
        self.session = None
        self.view = None
        self.quit = False
        self.recorder = None
        self.over_at = None
        self.machine = None     # "train": launcher/train.machine(), checked when the mode opens
        self.name_focus = False
        self.train_polled = (0.0, None)

    def find_networks(self):
        self.gens = core.generations(self.cfg)
        self.best = core.best(self.cfg)
        labels = [g["label"] for g in self.gens]
        if self.settings["gen_a"] not in labels:
            self.settings["gen_a"] = labels[0] if labels else None
        if self.settings["gen_b"] not in labels:
            self.settings["gen_b"] = labels[-1] if labels else None
        if self.settings["gen_play"] not in self.play_labels():
            self.settings["gen_play"] = None   # the released network

    def play_labels(self):
        """The networks of "vs KSquares" and "play": None (the released one,
        runs/best) and the accepted rounds of the app's own runs."""
        return [None] + [g["label"] for g in self.gens if g["own"]]

    # --- drawing helpers ---------------------------------------------------------

    def font(self, size, mono=False):
        key = (size, mono)
        if key not in self.fonts:
            name = self.pg.font.match_font("dejavusansmono" if mono else "dejavusans")
            self.fonts[key] = self.pg.font.Font(name, size)
        return self.fonts[key]

    def text(self, s, pos, size=22, color=FG, anchor="topleft", mono=False):
        img = self.font(size, mono).render(s, True, color)
        r = img.get_rect(**{anchor: pos})
        self.screen.blit(img, r)
        return r

    def wrap(self, s, pos, width, size=20, color=DIM):
        words, line, y = s.split(), "", pos[1]
        f = self.font(size)
        for w in words:
            if f.size(line + " " + w)[0] > width and line:
                self.text(line, (pos[0], y), size, color)
                y += size + 6
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            self.text(line, (pos[0], y), size, color)
            y += size + 6
        return y

    def button(self, name, rect, label, active=False, size=22, hint=None, bg=PANEL):
        pg = self.pg
        r = pg.Rect(rect)
        hot = self.hover == name
        pg.draw.rect(self.screen, PANEL_HI if hot or active else bg, r, border_radius=10)
        if active:
            pg.draw.rect(self.screen, ACCENT, r, 2, border_radius=10)
        self.text(label, (r.x + 18, r.y + (14 if hint else r.h / 2 - size * 0.62)), size, ACCENT if active else FG)
        if hint:
            self.wrap(hint, (r.x + 18, r.y + 22 + size), r.w - 36, 18)
        self.buttons[name] = r
        return r

    def L(self, key, **kw):
        return t(self.settings["lang"], key, **kw)

    # --- menu ---------------------------------------------------------------------

    # the settings each mode uses, in this order
    MODE_ROWS = {"ksquares": ("gen_play", "level", "first", "search", "board"),
                 "play": ("gen_play", "first", "search", "board", "show_mine"),
                 "self": ("gen_a", "gen_b", "search", "board"),
                 "train": ()}

    def draw_menu(self):
        pg = self.pg
        s = self.settings
        self.text("KSquares RL", (W / 2, 36), 56, ACCENT, "midtop")
        self.text(self.L("subtitle"), (W / 2, 112), 22, DIM, "midtop")
        for i, lang in enumerate(i18n.LANGS):
            self.button(f"lang:{lang}", (W - 300 + i * 90, 30, 80, 46), i18n.LABELS[lang], s["lang"] == lang)
        # modes
        self.text(self.L("modes_title"), (160, 176), 28, FG)
        for i, m in enumerate(MODES):
            self.button(f"mode:{m}", (160 + i * 400, 220, 380, 160), self.L(f"mode_{m}"), s["mode"] == m,
                        size=28, hint=self.L(f"mode_{m}_hint"))
        # settings of the selected mode
        y = 420
        self.text(self.L("settings_title"), (160, y), 28, FG)
        pg.draw.line(self.screen, FAINT, (160, y + 46), (W - 180, y + 46), 1)
        y += 70
        for key in self.MODE_ROWS[s["mode"]]:
            if key in ("gen_a", "gen_b", "gen_play"):
                self.text(self.L({"gen_a": "gen_first", "gen_b": "gen_second"}.get(key, key)), (160, y + 10), 22, DIM)
                self.button(f"{key}:-1", (700, y, 60, 48), "‹", size=24)
                label = s[key] or "—"
                if key == "gen_play" and s[key] is None:
                    label = self.L("gen_best", gen=self.best["gen"]) if self.best else "—"
                self.text(label, (930, y + 10), 22, FG, "midtop")
                self.button(f"{key}:1", (1100, y, 60, 48), "›", size=24)
            else:
                self.text(self.L(key), (160, y + 10), 22, DIM)
                for j, (name, label, act) in enumerate(self.options(key)):
                    self.button(name, (700 + j * 220, y, 200, 48), label, act, size=20)
            y += 62
        self.button("start", (160, y + 20, 360, 60), self.L("start"), True, size=24)
        if self.message:
            self.wrap(self.message, (560, y + 26), W - 740, 20, RED)
        # common
        pg.draw.line(self.screen, FAINT, (160, 930), (W - 180, 930), 1)
        if s["mode"] in ("ksquares", "play") and s["gen_play"]:
            self.text(self.L("network", label=s["gen_play"]), (160, 948), 22, DIM)
        elif self.best:
            self.text(self.L("network", label=f"{self.best['run']} gen {self.best['gen']}"), (160, 948), 22, DIM)
        else:
            self.text(self.L("network_none"), (160, 948), 22, DIM)
        self.text(self.L("menu_keys"), (W / 2, H - 30), 20, FAINT, "midbottom")
        if self.offer or self.download is not None:
            self.draw_download()

    def draw_download(self):
        """The dialog that fetches the network: offer, progress, error."""
        pg = self.pg
        self.buttons = {}   # modal: only its own buttons
        shade = pg.Surface((W, H), pg.SRCALPHA)
        shade.fill((0, 0, 0, 170))
        self.screen.blit(shade, (0, 0))
        p = pg.Rect(W / 2 - 440, H / 2 - 200, 880, 400)
        pg.draw.rect(self.screen, PANEL, p, border_radius=20)
        pg.draw.rect(self.screen, ACCENT, p, 3, border_radius=20)
        dl = self.cfg["best"]["download"]
        self.text(self.L("dl_title"), (p.x + 40, p.y + 34), 34, ACCENT)
        self.wrap(self.L("dl_text", label=f"{dl['run']} gen {dl['gen']}", size=f"{dl['size'] / 1e6:.1f}",
                         url=dl["url"]), (p.x + 40, p.y + 94), p.w - 80, 22, FG)
        d = self.download
        buttons = []
        if d is None or d.result not in (None, "ok"):
            if d is not None and d.result != "cancelled":
                err = self.L("dl_err_hash") if d.result == "hash" else self.L("dl_err_net", error=d.result)
                self.wrap(err, (p.x + 40, p.y + 230), p.w - 80, 20, RED)
            buttons = [("dl_yes", self.L("dl_yes" if d is None else "dl_retry")), ("dl_no", self.L("dl_no"))]
        elif d.result is None:
            bar = pg.Rect(p.x + 40, p.y + 240, p.w - 80, 18)
            pg.draw.rect(self.screen, BG, bar, border_radius=9)
            frac = d.done / d.total if d.total else 0.0
            if frac:
                pg.draw.rect(self.screen, ACCENT, (bar.x, bar.y, max(bar.h, bar.w * frac), bar.h), border_radius=9)
            s = (self.L("dl_verifying") if d.verifying else
                 self.L("dl_progress", done=f"{d.done / 1e6:.1f}", total=f"{d.total / 1e6:.1f}"))
            self.text(s, (bar.x, bar.bottom + 12), 20, DIM)
            buttons = [("dl_cancel", self.L("dl_cancel"))]
        x = p.right - 40
        for name, label in reversed(buttons):
            w = self.font(24).size(label)[0] + 50
            x -= w
            self.button(name, (x, p.bottom - 80, w, 52), label, name == "dl_yes", size=24)
            x -= 16

    def poll_download(self):
        """A finished download closes the dialog and starts the chosen mode."""
        d = self.download
        if d is not None and d.result == "ok":
            self.download = None
            self.find_networks()
            self.start(self.settings["mode"])

    def options(self, key):
        """(button name, label, active) for one settings row."""
        s = self.settings
        if key == "level":
            return [(f"level:{k}", self.L(f"level_{k}"), s["level"] == k) for k in range(3)]
        if key == "first":
            return [("first:1", self.L("first_net"), s["net_first"]),
                    ("first:0", self.L("first_you" if s["mode"] == "play" else "first_other"), not s["net_first"])]
        if key == "search":
            return [(f"sims:{n}", self.L("sims", n=n) if n else self.L("no_search"), s["sims"] == n)
                    for n in self.cfg["sims_choices"]]
        if key == "board":
            return [(f"size:{i}", f"{bw}×{bh}", s["size"] == i) for i, (bw, bh) in enumerate(self.cfg["board_sizes"])]
        return [("mine:1", self.L("on"), s["show_mine"]), ("mine:0", self.L("off"), not s["show_mine"])]

    def menu_click(self, name):
        s = self.settings
        kind, _, val = name.partition(":")
        if kind == "dl_yes":
            self.offer, self.download = False, Download(self.cfg)
        elif kind == "dl_no":
            self.offer, self.download = False, None
        elif kind == "dl_cancel":
            self.download.cancel = True
        elif kind == "lang":
            s["lang"] = val
        elif kind == "mode":
            if s["mode"] == val:
                self.start(val)   # a second click on the selected card starts it
            else:
                s["mode"] = val
                self.message = None
        elif kind == "start":
            self.start(s["mode"])
        elif kind == "level":
            s["level"] = int(val)
        elif kind == "first":
            s["net_first"] = val == "1"
        elif kind == "sims":
            s["sims"] = int(val)
        elif kind == "size":
            s["size"] = int(val)
        elif kind == "mine":
            s["show_mine"] = val == "1"
        elif kind in ("gen_a", "gen_b") and self.gens:
            labels = [g["label"] for g in self.gens]
            i = labels.index(s[kind]) if s[kind] in labels else 0
            s[kind] = labels[(i + int(val)) % len(labels)]
        elif kind == "gen_play":
            labels = self.play_labels()
            i = labels.index(s[kind]) if s[kind] in labels else 0
            s[kind] = labels[(i + int(val)) % len(labels)]
        if self.save:
            core.save_settings(s)

    # --- sessions -------------------------------------------------------------------

    def start(self, mode):
        from launcher.session import Session
        from launcher.brain import Brain  # noqa: F401  (torch import before the game)

        size = self.cfg["board_sizes"][self.settings["size"]]
        self.settings["mode"] = mode   # the card stays selected in the menu
        if self.save:
            core.save_settings(self.settings)
        if mode == "train":
            self.open_train()
            return
        err = core.check(mode, size, self.settings["lang"], self.cfg)
        if err == core.NO_NET:
            self.offer = True   # the download dialog says why
            return
        if err:
            self.message = err
            return
        self.best = core.best(self.cfg)
        if mode == "self":
            by = {g["label"]: g for g in self.gens}
            specs = [(by[self.settings[k]]["config"], by[self.settings[k]]["path"], self.settings[k])
                     for k in ("gen_a", "gen_b")]
        elif mode in ("play", "ksquares") and self.settings["gen_play"]:
            g = {g["label"]: g for g in self.gens}[self.settings["gen_play"]]
            specs = [(g["config"], g["path"], g["label"])]
        else:
            specs = [(self.best["config"], self.best["path"], f"{self.best['run']} gen {self.best['gen']}")]
        prev = self.session
        if prev is not None:
            prev.stop()
        self.session = Session(mode, self.cfg, self.settings, specs, seed=self.args.seed, after=prev)
        self.mode = mode
        self.view = BoardView(*size, MAIN)
        self.screen_name = "game"
        self.message = None
        self.over_at = None

    def stop_session(self):
        if self.session is not None:
            self.session.stop()
            self.session.thread.join(timeout=15)
        self.session = None
        self.screen_name = "menu"

    # --- "Train your own Dot" -----------------------------------------------------------

    def open_train(self):
        self.machine = train.machine(self.cfg)
        if self.machine["running"]:
            self.settings["train_name"] = self.machine["running"]   # show the run in progress
        self.mode = "train"
        self.screen_name = "train"
        self.message = None
        self.name_focus = False
        self.train_polled = (0.0, None)

    def train_progress(self):
        """train.progress() of the shown run, read at most once a second."""
        t0, pr = self.train_polled
        name = self.settings["train_name"]
        if pr is None or time.time() - t0 > 1.0 or pr.get("_name") != name:
            pr = train.progress(name, self.cfg) if train.valid_name(name) else None
            if pr is not None:
                pr["_name"] = name
            self.train_polled = (time.time(), pr)
            if self.machine is not None:
                self.machine["running"] = train.running()
        return pr

    @staticmethod
    def hms(sec):
        sec = int(sec)
        return f"{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"

    def draw_train(self):
        pg = self.pg
        s = self.settings
        m = self.machine
        tc = self.cfg["train"]
        name = s["train_name"]
        pr = self.train_progress()
        busy = m["running"]
        self.text(self.L("mode_train"), (40, 30), 40, ACCENT)
        self.text(self.L("keys_train"), (40, H - 30), 18, FAINT, "bottomleft")
        # left: the machine and the settings
        x, y, w = 40, 100, 860
        pg.draw.rect(self.screen, PANEL, (x, y, w, 930), border_radius=14)
        x += 24
        self.text(self.L("tr_machine"), (x, y + 18), 26, FG)
        yy = y + 62
        if m["gpu"]:
            self.text(self.L("tr_gpu", name=m["gpu"]["name"], mem=m["gpu"]["mem_gb"]), (x, yy), 22, GREEN)
            yy += 34
            if m["low_vram"]:
                yy = self.wrap(self.L("tr_low_vram", mem=m["gpu"]["mem_gb"], need=tc["min_vram_gb"]), (x, yy),
                               w - 48, 20, ACCENT)
        else:
            yy = self.wrap(self.L("tr_no_gpu", k=tc["cpu_slowdown"]), (x, yy), w - 48, 22, ACCENT)
        self.text(self.L("tr_disk", gb=m["disk_gb"]), (x, yy + 4), 22, RED if m["low_disk"] else FG)
        yy += 38
        if m["low_disk"]:
            yy = self.wrap(self.L("tr_low_disk", need=tc["min_disk_gb"]), (x, yy), w - 48, 20, RED)
        if busy and busy != name:
            yy = self.wrap(self.L("tr_busy", name=busy), (x, yy + 4), w - 48, 20, ACCENT)
        yy += 30
        pg.draw.line(self.screen, FAINT, (x, yy), (x + w - 48, yy), 1)
        yy += 24
        # name
        self.text(self.L("tr_name"), (x, yy + 10), 22, DIM)
        field = pg.Rect(x + 220, yy, 360, 48)
        pg.draw.rect(self.screen, BG, field, border_radius=8)
        pg.draw.rect(self.screen, ACCENT if self.name_focus else FAINT, field, 2, border_radius=8)
        caret = "▏" if self.name_focus and int(time.time() * 2) % 2 else ""
        self.text(name + caret, (field.x + 14, field.y + 10), 22, FG)
        self.buttons["name"] = field
        ok = train.valid_name(name)
        if not ok:
            yy = self.wrap(self.L("tr_name_bad"), (x, yy + 58), w - 48, 18, RED) - 58
        yy += 66
        # rounds
        self.text(self.L("tr_rounds"), (x, yy + 10), 22, DIM)
        for j, n in enumerate(tc["rounds_choices"]):
            bw = 190 if n == 0 else 80
            bx = x + 220 + sum(190 if c == 0 else 80 for c in tc["rounds_choices"][:j]) + 12 * j
            self.button(f"rounds:{n}", (bx, yy, bw, 48), str(n) if n else self.L("tr_unlimited"),
                        s["train_rounds"] == n, size=20, bg=BG)
        yy += 80
        # start / stop
        mine = busy == name
        if mine:
            if pr and pr["stopping"]:
                self.text(self.L("tr_stopping"), (x, yy + 16), 22, ACCENT)
            else:
                self.button("train_stop", (x, yy, 300, 60), self.L("tr_stop"), True, size=24)
        elif not busy and ok:
            again = bool(pr and pr["gen"])
            self.button("train_start", (x, yy, 300, 60), self.L("tr_resume" if again else "tr_start"), True, size=24)
        yy += 84
        # after the first accepted round: straight into a mode with this Dot
        mine_gen = self.my_dot(name)
        if mine_gen:
            bw = (w - 48 - 2 * 12) // 3
            for j, go in enumerate(("play", "self", "ksquares")):
                self.button(f"go:{go}", (x + j * (bw + 12), yy, bw, 50), self.L(f"tr_go_{go}"), size=18, bg=BG)
            self.text(self.L("tr_go_note", label=mine_gen), (x, yy + 58), 18, DIM)
            yy += 94
        if self.message:
            yy = self.wrap(self.message, (x, yy), w - 48, 20, RED) + 8
        if ok:
            d = os.path.relpath(train.run_dir(name), ROOT)
            yy = self.wrap(self.L("tr_files", dir=d + "/", log=f"{d}/train.log"), (x, yy), w - 48, 18) + 6
            yy = self.wrap(self.L("tr_gens_note", name=name), (x, yy), w - 48, 18)
        left_end = yy
        # right: the progress
        x, y, w = 940, 100, 940
        pg.draw.rect(self.screen, PANEL, (x, y, w, 930), border_radius=14)
        if pr is None:
            return
        x += 24
        stale = tc["stale_s"]
        quiet = pr["quiet_s"]
        if pr["running"]:
            gen = pr["gen"] + 1
            head = (self.L("tr_round_of", gen=gen, total=self.train_last(pr)) if pr["rounds"]
                    else self.L("tr_round", gen=gen))
            phase = self.L(f"tr_phase_{pr['phase'] or 'selfplay'}")
            # the pulse: beats while the trainer writes, yellow when it has gone quiet
            ok = quiet is None or quiet <= stale
            beat = 0.5 + 0.5 * math.sin(time.time() * 4) if ok else 1.0
            pg.draw.circle(self.screen, GREEN if ok else ACCENT, (x + 9, y + 34), 5 + 3 * beat)
            self.text(f"{head}: {phase}", (x + 28, y + 18), 26, FG)
            since = self.hms(pr["since_s"] or 0)
            if pr["round_s"]:
                eta = self.hms(pr["eta_s"]) if pr["eta_s"] is not None else self.L("tr_eta_none")
                t = self.L("tr_times", r=self.hms(pr["round_s"]), s=since, eta=eta)
            else:
                t = self.L("tr_times_first", r=self.hms(tc["round_minutes_gpu"] * 60), s=since)
            self.text(t, (x, y + 58), 20, DIM)
            if quiet is not None:
                self.text(self.L("tr_quiet", s=int(quiet)) if ok else self.L("tr_stale", s=self.hms(quiet)),
                          (x, y + 86), 18, DIM if ok else ACCENT)
            self.draw_inner(pr["inner"], (x, y + 118, w - 48))
        else:
            self.text(self.L("tr_idle", gen=pr["gen"]), (x, y + 18), 26, FG)
        # the rounds: a timeline of dots, then the last ones in full
        yy = y + (178 if pr["running"] else 70)   # right under the header when there is no round
        self.text(self.L("tr_timeline"), (x, yy), 18, DIM)
        self.draw_rounds(pr["history"], (x, yy + 30, w - 48))
        yy += 86
        self.text(self.L("tr_history"), (x, yy), 18, DIM)
        yy += 28
        for h in pr["history"][-4:]:
            verdict = self.L("tr_accepted" if h["accepted"] else "tr_rejected")
            t = self.hms(h["seconds"]) if h["seconds"] else ""
            row = self.L("tr_row", gen=h["gen"], vs=h["vs"] if h["vs"] is not None else "?", w=h["wins"],
                         l=h["losses"], p=100 * h["score"], t=t)
            self.text(row, (x, yy), 20, FG, mono=True)
            self.text(verdict, (x + w - 48, yy), 20, GREEN if h["accepted"] else DIM, "topright")
            yy += 28
        # the ladder against Hard
        yy += 10
        self.text(self.L("tr_ladder"), (x, yy), 20, FG)
        self.draw_ladder(pr["ladder"], (x, yy + 30, w - 48, 260))
        # the log: the last lines, newest at the bottom, as wide as the panel
        yy = y + 712
        self.text(self.L("tr_log"), (x, yy), 18, DIM)
        cols = int((w - 48) / self.font(14, mono=True).size("M")[0])
        for k, ln in enumerate(pr["log"][-10:]):
            self.text(ln[:cols], (x, yy + 26 + k * 18), 14, FAINT, mono=True)
        # left, at the bottom: the self-play game followed right now
        if pr["live"]:
            top = max(690, left_end + 20)
            self.draw_live(pr["live"], (64, top, 812, 1010 - top))

    def draw_inner(self, inner, rect):
        """The phase's progress inside the round: a label and a bar."""
        pg = self.pg
        x, y, w = rect
        if inner is None:
            self.text(self.L("tr_in_wait"), (x, y), 20, DIM)
            return
        n, of = inner["done"], max(1, inner["of"])
        if inner["phase"] == "gate":
            w_, l_, _ = inner["score"] or (0, 0, 0)
            label = self.L("tr_in_gate", n=n, of=of, w=w_, l=l_)
        else:
            label = self.L(f"tr_in_{inner['phase']}", n=n, of=of)
        self.text(label, (x, y), 20, FG)
        bar = pg.Rect(x, y + 32, w, 16)
        pg.draw.rect(self.screen, BG, bar, border_radius=8)
        pg.draw.rect(self.screen, VISIT, (bar.x, bar.y, max(16, bar.w * min(1.0, n / of)), bar.h), border_radius=8)
        self.text(f"{100 * n / of:.0f}%", (x + w, y), 20, DIM, "topright")

    def draw_rounds(self, history, rect):
        """A dot per round, accepted green, rejected grey, the match % below."""
        pg = self.pg
        x, y, w = rect
        step = 46
        shown = history[-max(1, w // step):]
        for k, h in enumerate(shown):
            cx = x + 12 + k * step
            pg.draw.circle(self.screen, GREEN if h["accepted"] else FAINT, (cx, y + 20), 9)
            self.text(str(h["gen"]), (cx, y + 4), 13, DIM, "midbottom")
            self.text(f"{100 * h['score']:.0f}%", (cx, y + 34), 14, FG if h["accepted"] else DIM, "midtop")
        if not shown:
            self.text("—", (x, y + 10), 18, FAINT)

    def draw_live(self, live, rect):
        """The self-play game az_train.py follows (live.json), on a small board."""
        from sim.board import Board

        x, y, w, h = rect
        key = (live["gen"], live["game"], len(live["moves"]))
        if getattr(self, "_live", (None,))[0] != key:
            b = Board(live["w"], live["h"])
            for mv in live["moves"]:
                if b.lines[mv] or b.done:
                    break
                b.step(mv)
            self._live = (key, b, BoardView(live["w"], live["h"], (x, y + 34, h - 34, h - 34)))
        _, b, view = self._live
        self.text(self.L("tr_live", game=live["game"] + 1, gen=live["gen"]), (x, y), 20, DIM)
        self.draw_board(view, b, small=True)
        if live["moves"]:   # the last line, in the accent colour
            a, c = view.ends(live["moves"][-1])
            self.pg.draw.line(self.screen, ACCENT, a, c, 5)
        who = self.L("tr_red" if b.player == 0 else "tr_blue")
        info = self.L("tr_live_info", n=b.n_drawn, all=b.geo.n_lines, a=b.score[0], b=b.score[1], who=who)
        for k, ln in enumerate(info.split("\n")):
            self.text(ln, (x + h + 10, y + 60 + k * 32), 22, FG)

    def my_dot(self, name):
        """The label of the run's last accepted round (as in the generation
        lists), or None."""
        pr = self.train_polled[1]
        if not pr or pr.get("_name") != name:
            return None
        acc = [h["gen"] for h in pr["history"] if h["accepted"]]
        return f"{name} gen {acc[-1]}" if acc else None

    def go_with_my_dot(self, mode):
        """"Play my Dot" and the like: the mode with the last accepted round chosen."""
        label = self.my_dot(self.settings["train_name"])
        self.find_networks()
        labels = [g["label"] for g in self.gens]
        if label not in labels:
            return
        if mode == "self":
            rel = f"{self.best['run']} gen {self.best['gen']}" if self.best else None
            self.settings["gen_a"] = label
            self.settings["gen_b"] = rel if rel in labels else labels[-1]
        else:
            self.settings["gen_play"] = label
        self.start(mode)

    def train_last(self, pr):
        return train._planned_last(self.settings["train_name"], pr)

    def draw_ladder(self, ladder, rect):
        """% won against Hard per accepted round: intuition alone and with search."""
        pg = self.pg
        x, y, w, h = rect
        px0, px1, py0, py1 = x + 50, x + w - 10, y + 10, y + h - 50
        for v in (0, 50, 100):
            yv = py1 - (py1 - py0) * v / 100
            pg.draw.line(self.screen, FAINT, (px0, yv), (px1, yv), 1)
            self.text(f"{v}%", (px0 - 8, yv), 16, DIM, "midright")
        if not ladder:
            self.wrap(self.L("tr_ladder_none"), (px0 + 20, py0 + 30), px1 - px0 - 40, 20)
            return
        g0, g1 = ladder[0][0], max(ladder[-1][0], ladder[0][0] + 1)
        for k, (col, key) in enumerate(((HEAT, "tr_raw"), (VISIT, "tr_mcts"))):
            pts = [(px0 + (px1 - px0) * (g - g0) / (g1 - g0), py1 - (py1 - py0) * row[k] / 100)
                   for g, *row in ladder]
            if len(pts) > 1:
                pg.draw.lines(self.screen, col, False, pts, 3)
            for p in pts:
                pg.draw.circle(self.screen, col, p, 5)
            lx = px0 + k * 320
            pg.draw.line(self.screen, col, (lx, py1 + 34), (lx + 30, py1 + 34), 3)
            self.text(self.L(key), (lx + 40, py1 + 24), 18, col)
        self.text(f"gen {g0}", (px0, py1 + 4), 16, DIM)
        self.text(f"gen {ladder[-1][0]}", (px1, py1 + 4), 16, DIM, "topright")

    def train_click(self, name):
        s = self.settings
        kind, _, val = name.partition(":")
        self.name_focus = kind == "name"
        if kind == "rounds":
            s["train_rounds"] = int(val)
        elif kind == "train_start":
            self.machine = train.machine(self.cfg)
            try:
                train.start(s["train_name"], s["train_rounds"], self.cfg)
                self.message = None
            except Exception as e:
                self.message = self.L("tr_err_start", error=str(e))
            self.machine["running"] = train.running()
            self.train_polled = (0.0, None)
        elif kind == "train_stop":
            train.stop(s["train_name"])
            self.train_polled = (0.0, None)
        elif kind == "go":
            self.go_with_my_dot(val)
        if self.save:
            core.save_settings(s)

    def train_key(self, e):
        pg = self.pg
        if e.key == pg.K_ESCAPE:
            if self.name_focus:
                self.name_focus = False
            else:
                self.screen_name = "menu"
                self.find_networks()   # its accepted rounds join the generations
        elif self.name_focus and e.key == pg.K_BACKSPACE:
            self.settings["train_name"] = self.settings["train_name"][:-1]
        elif self.name_focus and e.key in (pg.K_RETURN, pg.K_KP_ENTER):
            self.name_focus = False

    def train_text(self, text):
        if self.name_focus and not self.machine["running"] == self.settings["train_name"]:
            self.settings["train_name"] = (self.settings["train_name"] + text)[:32]


    # --- the game screen ---------------------------------------------------------------

    def player_name(self, seat):
        s = self.session
        kind = s.seat_kind[seat]
        if kind == "net":
            return s.brain_specs[seat if s.mode == "self" else 0][2]
        if kind == "human":
            return self.L("you")
        return self.L("ksquares_player", level=self.L(f"level_{s.level}"))

    def draw_board(self, view, board, heat=None, heat_color=HEAT, pv=None, last=None, hover=None,
                   move=None, small=False):
        pg = self.pg
        geo = board.geo
        lw = max(3, int(view.cell * (0.06 if small else 0.08)))
        for b in range(geo.n_boxes):
            if board.owner[b] >= 0:
                r, c = divmod(int(view.codec.box_cell[b]), view.codec.cols)
                (x0, y0), (x1, y1) = view.pt(c - 1, r - 1), view.pt(c + 1, r + 1)
                col = SEAT[board.owner[b]]
                pg.draw.rect(self.screen, tuple(int(0.35 * v + 0.65 * PANEL[i]) for i, v in enumerate(col)),
                             (x0 + 3, y0 + 3, x1 - x0 - 6, y1 - y0 - 6), border_radius=6)
        hmax = float(max(heat)) if heat is not None and max(heat) > 0 else 1.0
        for ln in range(geo.n_lines):
            a, b = view.ends(ln)
            if board.lines[ln]:
                col = SEAT[last[1]] if last and last[0] == ln else (200, 204, 214)
                pg.draw.line(self.screen, col, a, b, lw + (2 if last and last[0] == ln else 0))
            else:
                k = 0.0 if heat is None else math.sqrt(max(0.0, float(heat[ln])) / hmax)
                col = tuple(int(FAINT[i] * (1 - k) * 0.6 + heat_color[i] * k) for i in range(3))
                pg.draw.line(self.screen, col, a, b, lw if k > 0.05 else max(2, lw // 2))
        if hover is not None:
            a, b = view.ends(hover)
            pg.draw.line(self.screen, ACCENT, a, b, lw + 2)
        for gy in range(0, 2 * view.h + 1, 2):
            for gx in range(0, 2 * view.w + 1, 2):
                pg.draw.circle(self.screen, FG, view.pt(gx, gy), max(3, lw * 0.9))
        if move is not None:
            pg.draw.circle(self.screen, ACCENT, view.mid(move), max(9, view.cell * 0.17), 3)
        if pv:
            for k, (ln, seat) in enumerate(pv, start=1):
                x, y = view.mid(ln)
                rad = max(9, view.cell * 0.15)
                pg.draw.circle(self.screen, SEAT[seat], (x, y), rad)
                self.text(str(k), (x, y + 1), int(rad * 1.25), (10, 10, 14), "center")

    def draw_game(self):
        pg = self.pg
        s = self.session
        snap = s.snap
        board = snap["board"] if snap else None
        # players and score
        for seat in (0, 1):
            x = MAIN[0] + seat * MAIN[2] / 2
            pg.draw.circle(self.screen, SEAT[seat], (x + 14, 64), 10)
            sc = board.score[seat] if board else 0
            turn = board is not None and not snap["over"] and board.player == seat
            self.text(f"{self.player_name(seat)}  {sc}", (x + 34, 48), 28, ACCENT if turn else FG)
        self.text(self.L(f"keys_{'over' if s.over else s.mode}"), (MAIN[0], 1040 - 140 - 26), 18, FAINT)
        if s.mode == "self":
            st = self.L("speed", s=s.delay) + ("   " + self.L("paused") if s.paused else "")
            self.text(st, (MAIN[0] + MAIN[2], 1040 - 140 - 26), 18, ACCENT if s.paused else DIM, "topright")
        # the game
        if s.mode == "ksquares":
            if s.frame is None:
                self.text(self.L("starting"), (MAIN[0] + MAIN[2] / 2, MAIN[1] + MAIN[3] / 2), 28, DIM, "center")
            else:
                img = pg.image.frombuffer(s.frame.tobytes(), (s.frame.shape[1], s.frame.shape[0]), "RGB")
                k = min(MAIN[2] / img.get_width(), MAIN[3] / img.get_height())
                img = pg.transform.smoothscale(img, (int(img.get_width() * k), int(img.get_height() * k)))
                self.screen.blit(img, img.get_rect(center=(MAIN[0] + MAIN[2] / 2, MAIN[1] + MAIN[3] / 2)))
        elif board is not None:
            hover = None
            if snap["waiting"]:
                hover = self.view.line_at(pg.mouse.get_pos(), board)
            elif self.reviewing(snap):
                hover = snap["human"]["line"]   # your line, before it is drawn
            self.draw_board(self.view, board, last=snap["last"], hover=hover)
            if snap["waiting"]:
                self.text(self.L("your_turn"), (MAIN[0] + MAIN[2] / 2, MAIN[1] + MAIN[3] + 4), 22, ACCENT, "midtop")
        if snap and snap["over"]:
            a, b = board.score
            w = 0 if a > b else 1
            msg = self.L("wins", who=self.player_name(w), a=max(a, b), b=min(a, b))
            r = self.text(msg, (MAIN[0] + MAIN[2] / 2, MAIN[1] + MAIN[3] / 2), 44, SEAT[w], "center")
            pg.draw.rect(self.screen, BG, r.inflate(40, 24), border_radius=12)
            pg.draw.rect(self.screen, SEAT[w], r.inflate(40, 24), 2, border_radius=12)
            self.text(msg, (MAIN[0] + MAIN[2] / 2, MAIN[1] + MAIN[3] / 2), 44, SEAT[w], "center")
        if s.error:
            self.message = core.explain(s.error, self.settings["lang"])
            self.stop_session()
            return
        self.draw_thoughts(snap)
        self.draw_timeline(snap)

    def draw_thoughts(self, snap):
        pg = self.pg
        x, y, w, h = SIDE
        pg.draw.rect(self.screen, PANEL, SIDE, border_radius=14)
        th = snap and snap["thought"]
        if not th:
            return
        board = th["board"]
        title = self.L("thoughts_mine") if th["for_human"] else self.L("thoughts", label=th["label"])
        self.text(title, (x + 24, y + 18), 26, FG)
        bw = (w - 72) / 2
        for i, (cap, heat, col) in enumerate(((self.L("policy"), th["policy"], HEAT),
                                              (self.L("visits"), th["visits"], VISIT))):
            bx = x + 24 + i * (bw + 24)
            self.text(cap, (bx, y + 62), 18, DIM)
            view = BoardView(board.geo.w, board.geo.h, (bx, y + 86, bw, bw))
            if i == 1 and heat is None:
                self.draw_board(view, board, small=True)
                self.wrap(self.L("visits_decided" if th["decided"] else "visits_none"), (bx, y + 92 + bw), bw, 16,
                          DIM if th["decided"] else ACCENT)
                continue
            self.draw_board(view, board, heat=heat, heat_color=col, small=True,
                            pv=th["pv"] if i == 1 else None,
                            move=th["move"] if i == 1 else int(heat.argmax()))
        yy = y + 140 + bw   # room for the no-search caption under the maps
        # value and the bar: the side to move's colour on its side of the bar
        v = th["value"]
        self.text(self.L("value", v=v), (x + 24, yy), 24, FG)
        if th["score"] is not None:
            self.text(self.L("score_head", s=th["score"]), (x + w - 24, yy + 4), 20, DIM, "topright")
        yy += 40
        v0 = v if board.player == 0 else -v
        bar = pg.Rect(x + 24, yy, w - 48, 22)
        split = bar.x + bar.w * (1 + v0) / 2
        pg.draw.rect(self.screen, SEAT[0], (bar.x, bar.y, split - bar.x, bar.h), border_radius=6)
        pg.draw.rect(self.screen, SEAT[1], (split, bar.y, bar.right - split, bar.h), border_radius=6)
        pg.draw.line(self.screen, BG, (bar.centerx, bar.y - 4), (bar.centerx, bar.bottom + 4), 2)
        self.text(self.L("who_wins"), (bar.centerx, bar.bottom + 4), 16, DIM, "midtop")
        yy += 52
        # chains and the rule, while the fight for them is on
        ch = th["chains"]
        if not snap["over"] and not th["decided"]:
            self.text(self.L("chains", n=ch["long"], chains=" ".join(map(str, ch["chains"])),
                             loops=" ".join(map(str, ch["loops"]))), (x + 24, yy), 22, FG)
            who = self.player_name(ch["favoured"])
            self.text(self.L("parity", dots=ch["dots"], n=ch["long"], who=who), (x + 24, yy + 32), 20,
                      SEAT[ch["favoured"]])
            yy += 74
        # the exact verdict
        vd = self.session.verdict(th["key"])
        if vd == "pending":
            self.text(self.L("solving"), (x + 24, yy), 22, DIM)
        elif vd is None:
            self.text(self.L("no_verdict"), (x + 24, yy), 22, DIM)
        elif vd["equal"]:
            self.text(self.L("equal"), (x + 24, yy), 22, DIM)
        elif not th["for_human"]:
            ok = th["move"] in vd["correct"]
            self.wrap(self.L("sees" if ok else "misses"), (x + 24, yy), w - 48, 22, GREEN if ok else RED)
            pol = int(th["policy"].argmax()) in vd["correct"]
            self.text(self.L("verdict_policy", ok=self.L("right" if pol else "wrong")), (x + 24, yy + 34), 18,
                      GREEN if pol else RED)
        hm = snap.get("human")
        if hm and th["for_human"] and hm["key"] == th["key"]:
            yy += 64
            self.text(self.L("human_move", p=100 * hm["prob"], rank=hm["rank"], n=hm["n"]), (x + 24, yy), 20, FG)
            if hm["q"] is not None:
                self.text(self.L("human_q", q=hm["q"]), (x + 24, yy + 28), 18, DIM)
            if isinstance(vd, dict) and not vd["equal"]:
                ok = hm["line"] in vd["correct"]
                self.text(self.L("human_right" if ok else "human_wrong"), (x + 24, yy + 54), 18,
                          GREEN if ok else RED)

    def draw_timeline(self, snap):
        pg = self.pg
        x, y, w, h = TIMELINE
        pg.draw.rect(self.screen, PANEL, TIMELINE, border_radius=12)
        self.text(self.L("timeline"), (x + 16, y + 10), 16, DIM)
        px0, px1 = x + 140, x + w - 30
        py0, py1 = y + 38, y + h - 12
        mid = (py0 + py1) / 2
        pg.draw.line(self.screen, FAINT, (px0, mid), (px1, mid), 1)
        self.text(self.L("player_first"), (px0 - 10, py0), 14, SEAT[0], "topright")
        self.text(self.L("player_second"), (px0 - 10, py1), 14, SEAT[1], "bottomright")
        if not snap:
            return
        n = snap["board"].geo.n_lines
        pts = [(px0 + (px1 - px0) * d / n, mid - (py1 - py0) / 2 * v, seat) for d, v, seat, _ in snap["timeline"]]
        if len(pts) > 1:
            pg.draw.lines(self.screen, DIM, False, [(a, b) for a, b, _ in pts], 2)
        for a, b, seat in pts:
            pg.draw.circle(self.screen, SEAT[seat], (a, b), 5)

    # --- events and the loop ---------------------------------------------------------

    @staticmethod
    def reviewing(snap):
        """The network's view of the person's last move is on the panel."""
        th, hm = snap["thought"], snap.get("human")
        return bool(th and hm and th["for_human"] and hm["key"] == th["key"])

    def on_key(self, e):
        pg = self.pg
        if e.key == pg.K_F11:
            pg.display.toggle_fullscreen()
        elif self.screen_name == "menu" and (self.offer or self.download is not None):
            if e.key == pg.K_ESCAPE:
                if self.download is not None and self.download.result is None:
                    self.download.cancel = True
                else:
                    self.offer, self.download = False, None
        elif self.screen_name == "train":
            self.train_key(e)
        elif self.screen_name == "menu":
            if e.key == pg.K_ESCAPE:
                self.quit = True
            elif e.key in (pg.K_RETURN, pg.K_KP_ENTER):
                self.start(self.settings["mode"])
            elif e.key in (pg.K_LEFT, pg.K_RIGHT):
                i = MODES.index(self.settings["mode"]) + (1 if e.key == pg.K_RIGHT else -1)
                self.settings["mode"] = MODES[i % len(MODES)]
        else:
            s = self.session
            if e.key == pg.K_ESCAPE:
                self.stop_session()
            elif s.over and e.key in (pg.K_RETURN, pg.K_KP_ENTER):
                self.start(s.mode)
            elif s.mode == "self":
                if e.key == pg.K_SPACE:
                    s.pause(not s.paused)
                elif e.key == pg.K_RIGHT:
                    s.pause(True)
                    s.step_once()
                elif e.key in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS, pg.K_MINUS, pg.K_KP_MINUS):
                    faster = e.key in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS)
                    d = self.cfg["selfplay_delays"]
                    i = max(0, min(len(d) - 1, self.settings["speed"] + (-1 if faster else 1)))
                    self.settings["speed"] = i
                    s.delay = d[i]

    def on_click(self, pos):
        if self.screen_name in ("menu", "train"):
            click = self.menu_click if self.screen_name == "menu" else self.train_click
            for name, r in self.buttons.items():
                if r.collidepoint(pos):
                    click(name)
                    return
            self.name_focus = False
        else:
            snap = self.session.snap
            if snap and snap["waiting"]:
                ln = self.view.line_at(pos, snap["board"])
                if ln is not None:
                    self.session.click(ln)

    def frame(self):
        self.poll_download()
        self.screen.fill(BG)
        self.buttons = {}
        if self.screen_name == "menu":
            self.draw_menu()
        elif self.screen_name == "train":
            self.draw_train()
        else:
            self.draw_game()

    def run(self):
        pg = self.pg
        a = self.args
        if a.record:
            self.recorder = Recorder(a.record, self.cfg["record_fps"],
                                     tuple(int(x) for x in a.record_size.split("x")))
        if a.mode:
            self.start(a.mode)
        clock = pg.time.Clock()
        human = None
        if a.human_agent:
            from agents.registry import make_agent
            human = make_agent(a.human_agent, a.seed or 0)
        try:
            while not self.quit:
                for e in pg.event.get():
                    if e.type == pg.QUIT:
                        self.quit = True
                    elif e.type == pg.KEYDOWN:
                        self.on_key(e)
                    elif e.type == pg.MOUSEMOTION:
                        self.hover = next((n for n, r in self.buttons.items() if r.collidepoint(e.pos)), None)
                    elif e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
                        self.on_click(e.pos)
                    elif e.type == pg.TEXTINPUT and self.screen_name == "train":
                        self.train_text(e.text)
                s = self.session
                if human is not None and s and s.snap and s.snap["waiting"] and s.snap.get("_asked") != s.version:
                    s.snap["_asked"] = s.version
                    s.click(human.act(s.snap["board"]))
                self.frame()
                pg.display.flip()
                now = time.time()
                if self.recorder:
                    self.recorder.frame(pg, self.screen, now)
                if a.mode and s is not None:
                    done = s.over or (a.moves and s.snap and s.snap["board"].n_drawn >= a.moves
                                      and s.snap["thought"] is not None
                                      and s.verdict(s.snap["thought"]["key"]) != "pending"
                                      and (not s.show_mine or self.reviewing(s.snap)))
                    if done and self.over_at is None:
                        self.over_at = now
                        if a.screenshot:
                            pg.image.save(self.screen, a.screenshot)
                    if self.over_at and now - self.over_at > a.linger:
                        self.quit = True
                if a.mode and self.screen_name == "menu" and (a.record or a.screenshot):
                    self.quit = True   # the game failed: the message is on the menu
                clock.tick(30)
            if a.screenshot and self.over_at is None:
                pg.image.save(self.screen, a.screenshot)
        finally:
            if self.recorder:
                self.recorder.close()
            if self.session is not None:
                self.session.stop()
                self.session.thread.join(timeout=15)
            pg.quit()
        if self.offer:
            self.message = self.L("err_no_net")
        if self.message:
            print(self.message, file=sys.stderr)
            return 1
        return 0


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=i18n.LANGS)
    ap.add_argument("--mode", choices=MODES, help="start this mode at once (and quit when the game ends "
                                                   "with --record / --screenshot)")
    ap.add_argument("--level", type=int, choices=[0, 1, 2])
    ap.add_argument("--sims", type=int)
    ap.add_argument("--gen-a", help='"network vs itself", first player, e.g. "az5 gen 1"')
    ap.add_argument("--gen-b", help="second player")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--board", help="board size, e.g. 5x5 (one of configs/app.yaml board_sizes)")
    ap.add_argument("--net-first", type=int, choices=[0, 1], help="1: the network moves first")
    ap.add_argument("--show-mine", type=int, choices=[0, 1], help="1: the network's view of your moves")
    ap.add_argument("--record", help="record the window into an .mp4 or .gif (with --mode)")
    ap.add_argument("--record-size", default="1280x720", help="WxH of the recording (the window is 1920x1080)")
    ap.add_argument("--screenshot", help="save the last frame here and quit (no window)")
    ap.add_argument("--moves", type=int, help="with --screenshot / --record: stop after this many lines")
    ap.add_argument("--linger", type=float, default=2.0, help="seconds kept after the game / --moves")
    ap.add_argument("--human-agent", help="'play' without a person: this agent clicks (screenshots, tests)")
    ap.add_argument("--headless", action="store_true", help="no window (SDL dummy driver)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    if a.screenshot and a.mode in (None, "train"):   # the menu, the training screen
        app = App(a, headless=True)
        if a.mode:
            app.start(a.mode)
        app.frame()
        app.pg.image.save(app.screen, a.screenshot)
        app.pg.quit()
        return 0
    return App(a, headless=a.headless or bool(a.screenshot)).run()


if __name__ == "__main__":
    sys.exit(main())
