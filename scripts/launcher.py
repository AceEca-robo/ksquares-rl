#!/usr/bin/env python3
"""The app: one 1920x1080 pygame window (F11: full screen) that starts with
the menu and shows every mode in place:

    Network vs KSquares   the real KSquares (Easy / Medium / Hard) on a private
                          Xvfb, its window grabbed into the centre; the network
                          moves through the RL bridge
    Play the network      a board of your own: click a line to draw it
    Network vs itself     two generations (or one) against each other; Space
                          pauses, → plays one move, +/− change the speed

The thought panel on the right shows the network's side: the policy as a
heat map over the lines, V and a "who is winning" bar, the MCTS visits
with the principal variation numbered on the board, the long chains on the
board with the side the long chain rule favours, and whether the network's
move keeps the win (exact solving in the chain fight). Below, V move by
move. Search: none (the policy's argmax) or 50 / 200 / 800 simulations.

The network is runs/best/best.pt, the last accepted generation of az5b
(scripts/update_best.py keeps the link current); "network vs itself" can
pick any accepted generation of az5 and az5b. Esc goes back to the menu
from any mode (the game, its Xvfb and the worker are closed), Esc in the
menu quits. English, Russian or Ukrainian (launcher/i18n.py), switched in
the menu's corner; the choices are kept in ~/.config/ksquares-rl/.

    python scripts/launcher.py
    python scripts/launcher.py --mode self --gen-a "az5 gen 1" --record docs/self.mp4
    python scripts/launcher.py --mode ksquares --screenshot docs/app_ksquares.png --moves 30
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from launcher import core, i18n  # noqa: E402
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
        self.gens = core.generations(self.cfg)
        self.best = core.best(self.cfg)
        labels = [g["label"] for g in self.gens]
        if args.gen_a:
            self.settings["gen_a"] = args.gen_a
        if args.gen_b:
            self.settings["gen_b"] = args.gen_b
        if self.settings["gen_a"] not in labels:
            self.settings["gen_a"] = labels[0] if labels else None
        if self.settings["gen_b"] not in labels:
            self.settings["gen_b"] = labels[-1] if labels else None
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

    def button(self, name, rect, label, active=False, size=22, hint=None):
        pg = self.pg
        r = pg.Rect(rect)
        hot = self.hover == name
        pg.draw.rect(self.screen, PANEL_HI if hot or active else PANEL, r, border_radius=10)
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

    def draw_menu(self):
        s = self.settings
        self.text("KSquares RL", (W / 2, 70), 64, ACCENT, "midtop")
        self.text(self.L("subtitle"), (W / 2, 160), 24, DIM, "midtop")
        for i, lang in enumerate(i18n.LANGS):
            self.button(f"lang:{lang}", (W - 300 + i * 90, 30, 80, 46), i18n.LABELS[lang], s["lang"] == lang)
        for i, m in enumerate(MODES):
            self.button(f"mode:{m}", (160 + i * 540, 230, 500, 170), self.L(f"mode_{m}"), s["mode"] == m,
                        size=30, hint=self.L(f"mode_{m}_hint"))
        y = 450
        rows = [
            ("level", [(f"level:{k}", self.L(f"level_{k}"), s["level"] == k) for k in range(3)]),
            ("first", [("first:1", self.L("first_net"), s["net_first"]),
                       ("first:0", self.L("first_other"), not s["net_first"])]),
            ("search", [(f"sims:{n}", self.L("sims", n=n) if n else self.L("no_search"), s["sims"] == n)
                        for n in self.cfg["sims_choices"]]),
            ("board", [(f"size:{i}", f"{bw}×{bh}", s["size"] == i)
                       for i, (bw, bh) in enumerate(self.cfg["board_sizes"])]),
            ("show_mine", [("mine:1", self.L("on"), s["show_mine"]), ("mine:0", self.L("off"), not s["show_mine"])]),
        ]
        for key, opts in rows:
            self.text(self.L(key), (160, y + 10), 22, DIM)
            for j, (name, label, act) in enumerate(opts):
                self.button(name, (700 + j * 220, y, 200, 48), label, act, size=20)
            y += 62
        for key, which in (("gen_first", "gen_a"), ("gen_second", "gen_b")):
            self.text(f"{self.L('mode_self')}: {self.L(key)}", (160, y + 10), 22, DIM)
            self.button(f"{which}:-1", (700, y, 60, 48), "‹", size=24)
            self.text(s[which] or "—", (900, y + 10), 22, FG, "midtop")
            self.button(f"{which}:1", (1040, y, 60, 48), "›", size=24)
            y += 62
        if self.best:
            self.text(self.L("network", label=f"{self.best['run']} gen {self.best['gen']}"), (160, y + 14), 22, DIM)
        if self.message:
            self.wrap(self.message, (160, y + 60), 1600, 22, RED)
        self.text(self.L("menu_keys"), (W / 2, H - 40), 20, FAINT, "midbottom")

    def menu_click(self, name):
        s = self.settings
        kind, _, val = name.partition(":")
        if kind == "lang":
            s["lang"] = val
        elif kind == "mode":
            self.start(val)
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
        err = core.check(mode, size, self.settings["lang"], self.cfg)
        if err:
            self.message = err
            return
        self.best = core.best(self.cfg)
        if mode == "self":
            by = {g["label"]: g for g in self.gens}
            specs = [(by[self.settings[k]]["config"], by[self.settings[k]]["path"], self.settings[k])
                     for k in ("gen_a", "gen_b")]
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
        if self.screen_name == "menu":
            for name, r in self.buttons.items():
                if r.collidepoint(pos):
                    self.menu_click(name)
                    return
        else:
            snap = self.session.snap
            if snap and snap["waiting"]:
                ln = self.view.line_at(pos, snap["board"])
                if ln is not None:
                    self.session.click(ln)

    def frame(self):
        self.screen.fill(BG)
        self.buttons = {}
        if self.screen_name == "menu":
            self.draw_menu()
        else:
            self.draw_game()

    def run(self):
        pg = self.pg
        a = self.args
        if a.record:
            self.recorder = Recorder(a.record, self.cfg["record_fps"])
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
    ap.add_argument("--screenshot", help="save the last frame here and quit (no window)")
    ap.add_argument("--moves", type=int, help="with --screenshot / --record: stop after this many lines")
    ap.add_argument("--linger", type=float, default=2.0, help="seconds kept after the game / --moves")
    ap.add_argument("--human-agent", help="'play' without a person: this agent clicks (screenshots, tests)")
    ap.add_argument("--headless", action="store_true", help="no window (SDL dummy driver)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    if a.screenshot and not a.mode:   # the menu
        app = App(a, headless=True)
        app.frame()
        app.pg.image.save(app.screen, a.screenshot)
        app.pg.quit()
        return 0
    return App(a, headless=a.headless or bool(a.screenshot)).run()


if __name__ == "__main__":
    sys.exit(main())
