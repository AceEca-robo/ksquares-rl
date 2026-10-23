"""Client for the KSquares RL bridge (game/src/rlbridge.cpp).

The game (``build/src/ksquares`` with ``KSQUARES_RL_SOCKET=<path>``)
connects to a Unix socket served here and speaks one JSON object per line;
the protocol is documented in ``game/src/rlbridge.h``.

Every line the game reports is replayed in a ``sim.board.Board`` and the
whole state (lines, owners, scores, side to move) is compared after each
move, so any divergence between the game and the simulator raises
``BridgeMismatch`` at the move where it happens.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import weakref
from pathlib import Path

from sim.board import NO_OWNER, Board

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BINARY = ROOT / "build/src/ksquares"


class BridgeMismatch(AssertionError):
    """The game and the simulator disagree."""


def _die_with_parent():
    # the game gets SIGTERM if Python dies without close(); Ctrl+C is ours
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except OSError:
        pass


def _stop(res):
    for key in ("rfile", "conn"):
        if res.get(key) is not None:
            try:
                res[key].close()
            except OSError:
                pass
            res[key] = None
    proc = res.get("proc")
    if proc is not None:
        try:
            proc.wait(timeout=5)  # the game quits when the socket closes
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        res["proc"] = None
    if res.get("home"):
        shutil.rmtree(res["home"], ignore_errors=True)
        res["home"] = None


class KSquaresBridge:
    """One KSquares process driven over the bridge.

    ``reset(w, h, ai, difficulty, seed)`` starts a game; ``ai[p]`` says
    whether player ``p`` is a KSquares computer player, the others are
    external and move with ``step(line)``. Both return the bridge's state
    message (a dict); ``self.board`` is the simulator mirror of the game.

    hash_seed: value of ``QT_HASH_SEED`` for the game (Qt 5 otherwise picks a
    random QHash seed per process, which makes one rare Hard AI decision —
    the hard-hearted handout line — depend on the process; see
    docs/game_spec.md). None leaves the environment as it is.
    env: extra environment variables for the game process.
    """

    def __init__(self, binary: str | os.PathLike | None = None, connect_timeout: float = 30.0,
                 hash_seed: int | None = 0, verify: bool = True, env: dict | None = None):
        self.binary = Path(binary) if binary else DEFAULT_BINARY
        self.verify = verify
        self.board: Board | None = None
        self.n_players = 2
        self._res: dict = {}
        self._finalizer = weakref.finalize(self, _stop, self._res)
        self._start(connect_timeout, hash_seed, env or {})

    def _start(self, timeout, hash_seed, extra_env):
        if not self.binary.exists():
            raise FileNotFoundError(f"{self.binary} not built (cmake -S game -B build && cmake --build build -j)")
        tmp = tempfile.mkdtemp(prefix="ksquares-rl-")
        self._res["home"] = tmp
        path = os.path.join(tmp, "game.sock")
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            srv.bind(path)
            srv.listen(1)
            srv.settimeout(timeout)
            env = dict(os.environ)
            env.pop("DISPLAY", None)
            env.pop("WAYLAND_DISPLAY", None)
            env.update({
                "KSQUARES_RL_SOCKET": path,
                "QT_QPA_PLATFORM": "offscreen",
                # private config dir: the game's settings never touch ~/.config
                "XDG_CONFIG_HOME": os.path.join(tmp, "config"),
                "XDG_DATA_HOME": os.path.join(tmp, "data"),
                "XDG_CACHE_HOME": os.path.join(tmp, "cache"),
                "KDE_DEBUG": "1",  # no DrKonqi crash dialog
            })
            if hash_seed is not None:
                env["QT_HASH_SEED"] = str(hash_seed)
            env.update(extra_env)
            self._res["proc"] = subprocess.Popen(
                [str(self.binary)], env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=_die_with_parent)
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                raise RuntimeError(f"game did not connect, exit code {self._res['proc'].poll()}")
        finally:
            srv.close()
            os.unlink(path)
        conn.settimeout(None)
        self._res["conn"] = conn
        self._res["rfile"] = conn.makefile("rb")
        hello = self._recv()
        if hello.get("type") != "hello" or hello.get("version") != 1:
            raise RuntimeError(f"unexpected greeting {hello}")

    def _send(self, obj):
        self._res["conn"].sendall(json.dumps(obj).encode() + b"\n")

    def _recv(self):
        line = self._res["rfile"].readline()
        if not line:
            raise RuntimeError(f"game closed the connection, exit code {self._res['proc'].poll()}")
        return json.loads(line)

    def _call(self, obj):
        self._send(obj)
        msg = self._recv()
        if msg.get("type") == "error":
            raise ValueError(f"bridge error: {msg['msg']}")
        return msg

    # --- game API ----------------------------------------------------

    def reset(self, w: int, h: int, ai=(False, True), difficulty: int = 2, seed: int | None = None) -> dict:
        """seed: srand(seed) in the game before it starts (None: the glibc
        rand() sequence just continues from the previous game)."""
        ai = [bool(a) for a in ai]
        if len(ai) != 2 and self.verify:
            raise ValueError("verification needs 2 players (the simulator is 1v1)")
        self.n_players = len(ai)
        self.board = Board(w, h)
        cmd = {"cmd": "reset", "w": w, "h": h, "ai": ai, "difficulty": difficulty}
        if seed is not None:
            cmd["seed"] = int(seed)
        return self._apply(self._call(cmd))

    def step(self, line: int) -> dict:
        return self._apply(self._call({"cmd": "step", "line": int(line)}))

    def close(self):
        if self._res.get("conn") is not None:
            try:
                self._send({"cmd": "quit"})
            except OSError:
                pass
        self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- verification ------------------------------------------------

    def _apply(self, msg: dict) -> dict:
        if msg.get("type") != "state":
            raise RuntimeError(f"unexpected message {msg}")
        if not self.verify:
            return msg
        b = self.board
        for line, player, boxes in msg["moves"]:
            if player != b.player:
                raise BridgeMismatch(f"move {b.n_drawn}: game says player {player} drew {line}, "
                                     f"sim expects player {b.player}")
            before = b.owner.copy()
            b.step(line)
            got = sorted(i for i, (o0, o1) in enumerate(zip(before, b.owner)) if o0 != o1)
            if got != sorted(boxes):
                raise BridgeMismatch(f"move {b.n_drawn - 1} (line {line}): game completed boxes "
                                     f"{sorted(boxes)}, sim {got}")
        self.check(msg)
        return msg

    def check(self, msg: dict):
        b = self.board
        lines = "".join("1" if x else "0" for x in b.lines)
        if lines != msg["lines"]:
            raise BridgeMismatch(f"lines differ after {b.n_drawn} moves")
        owners = [o if o != NO_OWNER else -1 for o in b.owner]
        if owners != msg["owners"]:
            raise BridgeMismatch(f"owners differ after {b.n_drawn} moves: {owners} vs {msg['owners']}")
        if list(b.score) != msg["scores"]:
            raise BridgeMismatch(f"scores differ: sim {b.score} game {msg['scores']}")
        if b.done != msg["done"]:
            raise BridgeMismatch(f"done differs: sim {b.done} game {msg['done']}")
        if not b.done and b.player != msg["player"]:
            raise BridgeMismatch(f"side to move differs: sim {b.player} game {msg['player']}")
