"""One game in the app's window, played in a worker thread; the UI thread
draws ``.snap`` (the latest state, a dict) and never waits for the game.

    ksquares  the network against KSquares' own computer player: the real
              game over the RL bridge on a private Xvfb; a grabber thread
              keeps ``.frame``, the game's window, fresh
    play      a person (clicks, ``click``) against the network
    self      the network against itself, possibly two generations;
              ``pause``, ``step_once``, ``delay``

The network's thoughts (launcher/brain.py) are published with every move
it makes, and with the person's moves when ``show_mine`` is on. Exact
verdicts on the chain fight ("the network sees it or not") come from an
``Analyst`` thread, so a slow solve never holds up the game.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import traceback

import numpy as np

from sim.board import Board
from sim.chains import analyse

from . import brain, core

LEVELS = ("Easy", "Medium", "Hard")


class Analyst:
    """Solves the positions the network thought about, newest first; the
    verdicts land in ``results[key]``: {"correct", "equal"} or None."""

    def __init__(self, w, h, scfg):
        self.solver = None
        self.args = (w, h, *brain.solver_params(scfg, w, h))
        self.results = {}
        self.q = queue.LifoQueue()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="analyst", daemon=True)
        self.thread.start()

    def submit(self, key, board):
        self.q.put((key, board.clone()))

    def _run(self):
        while not self._stop.is_set():
            try:
                key, board = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            if key in self.results:
                continue
            if self.solver is None:
                self.solver = brain.Solver(*self.args)
            try:
                r = self.solver.correct(board)
            except RecursionError:
                r = None
            self.results[key] = None if r is None else {"correct": r[0], "equal": r[1]}

    def stop(self):
        self._stop.set()


class Session:
    def __init__(self, mode, cfg, settings, brains, seed=None, after=None):
        """brains: (config, checkpoint, label) of the network, for "self" one
        per seat [first, second]; loaded in the worker (launcher/brain.py)."""
        self.mode, self.cfg = mode, cfg
        self.w, self.h = cfg["board_sizes"][settings["size"]]
        self.level = settings["level"]
        self.sims = settings["sims"]
        self.show_mine = settings["show_mine"]
        self.delay = cfg["selfplay_delays"][settings["speed"]]
        self.brain_specs = brains
        self.brains = []
        if mode == "self":
            self.seat_kind = ("net", "net")
        else:
            net = 0 if settings["net_first"] else 1
            other = "ksquares" if mode == "ksquares" else "human"
            self.seat_kind = tuple("net" if s == net else other for s in (0, 1))
        self.seed = int.from_bytes(os.urandom(2), "little") if seed is None else seed
        self.snap = None            # see _publish
        self.version = 0            # bumped with every snap
        self.frame = None           # "ksquares": the game's window, an RGB array
        self.frame_version = 0
        self.over = False
        self.error = None           # (text, traceback) if the game failed
        self.paused = False
        self._step = threading.Event()
        self._click = queue.Queue()
        self._stop = threading.Event()
        self._after = after
        self._res = []              # closed at the end: the Xvfb, the bridge
        self.timeline = []          # (lines drawn, V for the first player, seat, label)
        self.analyst = Analyst(self.w, self.h, cfg["solver"])
        self.thread = threading.Thread(target=self._run, name=f"session-{mode}", daemon=True)
        self.thread.start()

    # --- UI thread ------------------------------------------------------------

    def click(self, line):
        """The person picked a line ("play")."""
        self._click.put(int(line))

    def pause(self, on):
        self.paused = on

    def step_once(self):
        self._step.set()

    def stop(self):
        self._stop.set()
        self._click.put(None)

    def running(self):
        return self.thread.is_alive()

    def verdict(self, key):
        return self.analyst.results.get(key, "pending")

    # --- worker thread ----------------------------------------------------------

    def _publish(self, board, **kw):
        prev = self.snap or {}
        snap = {"board": board.clone(), "thought": prev.get("thought"), "human": prev.get("human"),
                "waiting": False, "last": prev.get("last"), "over": False, "timeline": list(self.timeline)}
        snap.update(kw)
        self.snap = snap
        self.version += 1

    def _sleep(self, seconds):
        t_end = time.time() + seconds
        while time.time() < t_end and not self._stop.is_set():
            time.sleep(0.02)
        return not self._stop.is_set()

    def _gate(self):
        """"self": wait while paused, unless one step was asked for."""
        if self.mode != "self":
            return not self._stop.is_set()
        while self.paused and not self._stop.is_set():
            if self._step.wait(0.05):
                self._step.clear()
                return True
        return not self._stop.is_set()

    def _think(self, board, br, for_human=False):
        th = br.think(board, self.sims, self.cfg["pv_len"])
        th.update(board=board.clone(), chains=analyse(board), label=br.label, for_human=for_human,
                  key=(board.n_drawn, board.player, br.label))
        self.analyst.submit(th["key"], board)
        v0 = th["value"] if board.player == 0 else -th["value"]
        self.timeline.append((board.n_drawn, v0, board.player, br.label))
        return th

    def _brain(self, seat):
        return self.brains[seat] if self.mode == "self" else self.brains[0]

    def _run(self):
        if self._after is not None:
            self._after.thread.join()
        try:
            dev = brain.device(self.cfg["device"])
            self.brains = [brain.Brain(c, p, label, dev, seed=self.seed + i)
                           for i, (c, p, label) in enumerate(self.brain_specs)]
            if self._stop.is_set():
                return
            if self.mode == "ksquares":
                self._run_ksquares()
            else:
                self._run_sim()
        except BaseException as e:
            if not self._stop.is_set():
                self.error = (f"{type(e).__name__}: {e}", traceback.format_exc())
        finally:
            self.analyst.stop()
            for res in reversed(self._res):   # the game first, then its X server
                res.close()

    def _net_move(self, board, seat):
        """Think, show the thought, wait, return the line."""
        th = self._think(board, self._brain(seat))
        self._publish(board, thought=th)
        if not self._gate() or not self._sleep(self.delay if self.mode == "self" else self.cfg["net_delay"]):
            return None
        return th["move"]

    def _human_move(self, board, seat):
        th = None
        if self.show_mine:
            th = self._think(board, self.brains[0], for_human=True)
        self._publish(board, waiting=True, **({"thought": th} if th else {}))
        while True:
            ln = self._click.get()
            if ln is None or self._stop.is_set():
                return None
            if 0 <= ln < board.geo.n_lines and not board.lines[ln]:
                break
        if th is not None:
            p = th["policy"]
            legal = [i for i in range(board.geo.n_lines) if not board.lines[i]]
            rank = 1 + sum(p[i] > p[ln] for i in legal)
            q = None if th["q"] is None or np.isnan(th["q"][ln]) else float(th["q"][ln])
            self._publish(board, human={"line": ln, "prob": float(p[ln]), "rank": rank, "n": len(legal),
                                        "q": q, "key": th["key"]})
            self._sleep(self.cfg["human_review"])   # time to read the network's view of the move
        return ln

    def _run_sim(self):
        board = Board(self.w, self.h)
        self._publish(board)
        while not board.done and not self._stop.is_set():
            seat = board.player
            kind = self.seat_kind[seat]
            ln = self._net_move(board, seat) if kind == "net" else self._human_move(board, seat)
            if ln is None:
                return
            board.step(ln)
            self._publish(board, last=(ln, seat))
        if not self._stop.is_set():
            self._publish(board, over=True)
            self.over = True

    def _run_ksquares(self):
        from env.bridge import KSquaresBridge

        cfg = self.cfg
        xserver = core.PrivateX(tuple(cfg["xvfb_size"]))
        self._res.append(xserver)
        env = {"DISPLAY": xserver.name, "QT_QPA_PLATFORM": "xcb",
               "KSQUARES_RL_AI_DELAY_MS": str(cfg["ai_delay_ms"])}
        bridge = KSquaresBridge(core.GAME, env=env)
        self._res.append(bridge)
        grabber = threading.Thread(target=self._grab, args=(xserver.name, bridge._res["proc"].pid),
                                   name="grabber", daemon=True)
        grabber.start()
        net_seat = self.seat_kind.index("net")
        self._publish(Board(self.w, self.h))
        bridge.reset(self.w, self.h, ai=[s != net_seat for s in (0, 1)], difficulty=self.level, seed=self.seed)
        board = bridge.board
        while not board.done and not self._stop.is_set():
            self._publish(board)
            ln = self._net_move(board, net_seat)
            if ln is None:
                break
            bridge.step(ln)
            self._publish(board, last=(ln, net_seat))
        if not self._stop.is_set():
            self._sleep(0.5)   # the last frame
            self._publish(board, over=True)
            self.over = True

    def _grab(self, display, pid):
        from PIL import ImageGrab

        try:
            box = core.find_window(display, pid, tuple(self.cfg["ksquares_window"]))
        except Exception:
            return
        while not self._stop.is_set() and self.thread.is_alive():
            try:
                img = ImageGrab.grab(bbox=box, xdisplay=display).convert("RGB")
            except Exception:
                return   # the game or the X server is gone
            self.frame = np.asarray(img)
            self.frame_version += 1
            time.sleep(0.06)
