"""AlphaZero network for Dots and Boxes: board encoding, symmetries, ResNet.

Encoding
    One (2h+1) x (2w+1) grid that holds dots, lines and boxes at once
    (11 x 11 on 5x5): dots at (even, even), horizontal lines at
    (even, odd), vertical lines at (odd, even), boxes at (odd, odd). The
    eight symmetries of a square board are then plain array rotations and
    flips, and a transpose turns horizontal lines into vertical ones by
    itself; no channel has to be swapped or padded. Planes, all from the
    point of view of the side to move:

    0 drawn line          1 undrawn line (= legal move)
    2 my box              3 opponent's box
    4-7 free box with 0, 1, 2, 3 drawn sides
    8 (my score - opponent's score) / boxes, constant
    9 1 if the side to move moved first in the game, constant
    10 ones (marks the board inside the zero padding)
    11 undrawn lines / all lines (game phase), constant; Dot-2 only

    "Whose turn" is plane 9 (the identity of the mover); everything else is
    relative to the mover, so value and policy need no colour flip.

Outputs
    Policy logits over the lines in KSquares order (illegal lines are masked
    by the caller), a value in [-1, 1] for the side to move and, with
    ``score_head``, a second value head: the expected final score
    difference for the mover divided by the number of boxes (regression,
    auxiliary; the search uses only the win/loss value). ``forward``
    returns (policy, value, score or None).

Networks
    ``AZNet`` (Dot): fully connected heads, one board size, 11 planes.
    ``AZNet2`` (Dot-2): any board size, 12 planes. Policy, value and score
    heads are convolutional plus global pooling (masked mean, max and
    mean x (sqrt(boxes) - 5) / 10, KataGo style); ``gpool_blocks`` residual
    blocks also add a bias from the pooled board after their first conv.
    The pooling biases start at zero, so a Dot body loaded into Dot-2
    computes the same features (``load_dot_body``).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sim.board import geometry

N_PLANES = 11      # Dot
N_PLANES_2 = 12    # Dot-2: + game phase


class Codec:
    """Index tables for one board size: grid cells, symmetries."""

    def __init__(self, w: int, h: int, planes: int = N_PLANES):
        geo = geometry(w, h)
        self.w, self.h = w, h
        self.planes = planes
        self.n_lines, self.n_boxes = geo.n_lines, geo.n_boxes
        self.rows, self.cols = 2 * h + 1, 2 * w + 1
        self.n_cells = self.rows * self.cols
        stride = 2 * w + 1
        rc = []
        for ln in range(self.n_lines):
            r, p = divmod(ln, stride)
            rc.append((2 * r, 2 * p + 1) if p < w else (2 * r + 1, 2 * (p - w)))
        self.line_rc = np.array(rc)
        self.line_cell = self.line_rc[:, 0] * self.cols + self.line_rc[:, 1]
        box_rc = np.array([(2 * (i // w) + 1, 2 * (i % w) + 1) for i in range(self.n_boxes)])
        self.box_cell = box_rc[:, 0] * self.cols + box_rc[:, 1]
        cell_line = -np.ones(self.n_cells, dtype=np.int64)
        cell_line[self.line_cell] = np.arange(self.n_lines)
        # symmetries that keep the grid shape: (cell perm, line perm), with
        # transformed[..., k] = original[..., perm[k]]
        self.syms = []
        grid = np.arange(self.n_cells).reshape(self.rows, self.cols)
        for k in range(8):
            g = np.rot90(grid, k % 4)
            if k >= 4:
                g = g.T
            if g.shape != grid.shape:
                continue
            cperm = g.reshape(-1).copy()
            lperm = cell_line[cperm[self.line_cell]]
            assert (lperm >= 0).all()
            self.syms.append((cperm, lperm))

    def encode(self, lines: np.ndarray, sides: np.ndarray, owner: np.ndarray,
               score_diff: np.ndarray, first: np.ndarray) -> np.ndarray:
        """Batch of states -> (B, N_PLANES, rows, cols) float32.

        lines (B, n_lines) 0/1; sides (B, n_boxes) 0..4; owner (B, n_boxes)
        +1 mine, -1 opponent's, 0 free; score_diff (B,) mine - opponent's;
        first (B,) 1 if the mover moved first.
        """
        B = lines.shape[0]
        x = np.zeros((B, self.planes, self.n_cells), dtype=np.float32)
        lc, bc = self.line_cell, self.box_cell
        x[:, 0, lc] = lines
        x[:, 1, lc] = 1 - lines
        x[:, 2, bc] = owner == 1
        x[:, 3, bc] = owner == -1
        free = owner == 0
        for s in range(4):
            x[:, 4 + s, bc] = free & (sides == s)
        x[:, 8] = (score_diff / self.n_boxes)[:, None]
        x[:, 9] = first[:, None]
        x[:, 10] = 1
        if self.planes > 11:
            x[:, 11] = (self.n_lines - lines.sum(1, dtype=np.float32))[:, None] / self.n_lines
        return x.reshape(B, self.planes, self.rows, self.cols)

    def encode_boards(self, boards) -> np.ndarray:
        """Encode a list of sim.board.Board for their sides to move."""
        B = len(boards)
        lines = np.frombuffer(b"".join(bytes(b.lines) for b in boards), dtype=np.uint8)
        sides = np.frombuffer(b"".join(bytes(b.sides) for b in boards), dtype=np.uint8)
        owner = np.array([b.owner for b in boards], dtype=np.int8)
        player = np.array([b.player for b in boards], dtype=np.int8)
        rel = np.where(owner < 0, 0, np.where(owner == player[:, None], 1, -1))
        score = np.array([b.score for b in boards])
        diff = np.where(player == 0, score[:, 0] - score[:, 1], score[:, 1] - score[:, 0])
        return self.encode(lines.reshape(B, -1), sides.reshape(B, -1), rel, diff,
                           (player == 0).astype(np.float32))


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(y)))


class AZNet(nn.Module):
    """ResNet body, AlphaZero-style policy and value heads."""

    def __init__(self, w: int, h: int, blocks: int = 6, channels: int = 64, score_head: bool = False):
        super().__init__()
        self.w, self.h = w, h
        codec = Codec(w, h)
        self.n_lines = codec.n_lines
        cells = codec.n_cells
        self.register_buffer("line_cell", torch.as_tensor(codec.line_cell), persistent=False)
        self.stem = nn.Sequential(nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.p_conv = nn.Sequential(nn.Conv2d(channels, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU())
        self.p_fc = nn.Linear(4 * cells, self.n_lines)
        self.v_conv = nn.Sequential(nn.Conv2d(channels, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU())
        self.v_fc = nn.Sequential(nn.Linear(2 * cells, 128), nn.ReLU(), nn.Linear(128, 1))
        self.s_fc = nn.Sequential(nn.Linear(2 * cells, 128), nn.ReLU(), nn.Linear(128, 1)) if score_head else None

    def forward(self, x):
        y = self.body(self.stem(x))
        p = self.p_fc(self.p_conv(y).flatten(1))
        vf = self.v_conv(y).flatten(1)
        v = torch.tanh(self.v_fc(vf)).squeeze(1)
        s = self.s_fc(vf).squeeze(1) if self.s_fc is not None else None
        return p, v, s


    def line_cells(self, rows: int, cols: int) -> torch.Tensor:
        return self.line_cell


# --- Dot-2: any board size ------------------------------------------------------

def global_pool(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, C, R, K), mask (B, 1, R, K) -> (B, 3C): masked mean, max and
    mean x (sqrt(boxes) - 5) / 10, boxes of the R x K grid (batches are not
    padded: one board size per batch)."""
    mean = (y * mask).sum((2, 3)) / mask.sum((2, 3))
    mx = y.masked_fill(mask == 0, float("-inf")).amax((2, 3))
    size = (math.sqrt((y.shape[2] - 1) // 2 * ((y.shape[3] - 1) // 2)) - 5) / 10
    return torch.cat([mean, mx, mean * size], 1)


class GPoolBlock(nn.Module):
    """ResBlock whose hidden layer gets a bias from the pooled board."""

    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)
        self.g = nn.Linear(3 * ch, ch)
        nn.init.zeros_(self.g.weight)
        nn.init.zeros_(self.g.bias)

    def forward(self, x, mask):
        y = F.relu(self.b1(self.c1(x)))
        y = y + self.g(global_pool(y, mask))[:, :, None, None]
        return F.relu(x + self.b2(self.c2(y)))


class AZNet2(nn.Module):
    """Dot-2: ResNet body as Dot with global pooling, size-free heads."""

    def __init__(self, blocks: int = 6, channels: int = 64, score_head: bool = True,
                 gpool_blocks=(2, 4), head_ch: int = 32):
        super().__init__()
        self.planes = N_PLANES_2
        self.stem = nn.Sequential(nn.Conv2d(N_PLANES_2, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.ModuleList([GPoolBlock(channels) if i in gpool_blocks else ResBlock(channels)
                                   for i in range(blocks)])
        self.p_conv = nn.Conv2d(channels, head_ch, 1, bias=False)
        self.g_conv = nn.Sequential(nn.Conv2d(channels, head_ch, 1, bias=False), nn.BatchNorm2d(head_ch), nn.ReLU())
        self.p_g = nn.Linear(3 * head_ch, head_ch)
        self.p_out = nn.Sequential(nn.BatchNorm2d(head_ch), nn.ReLU(), nn.Conv2d(head_ch, 1, 1))
        self.v_conv = nn.Sequential(nn.Conv2d(channels, head_ch, 1, bias=False), nn.BatchNorm2d(head_ch), nn.ReLU())
        self.v_fc = nn.Sequential(nn.Linear(3 * head_ch, 128), nn.ReLU(), nn.Linear(128, 1))
        self.s_fc = nn.Sequential(nn.Linear(3 * head_ch, 128), nn.ReLU(), nn.Linear(128, 1)) if score_head else None
        self._cells = {}

    def line_cells(self, rows: int, cols: int) -> torch.Tensor:
        key = (rows, cols)
        if key not in self._cells:
            c = Codec((cols - 1) // 2, (rows - 1) // 2)
            self._cells[key] = torch.as_tensor(c.line_cell)
        t = self._cells[key]
        dev = self.stem[0].weight.device
        if t.device != dev:
            t = self._cells[key] = t.to(dev)
        return t

    def forward(self, x):
        mask = x[:, 10:11]
        y = self.stem(x)
        for b in self.body:
            y = b(y, mask) if isinstance(b, GPoolBlock) else b(y)
        g = global_pool(self.g_conv(y), mask)
        p = self.p_conv(y) + self.p_g(g)[:, :, None, None]
        p = self.p_out(p).flatten(1)[:, self.line_cells(x.shape[2], x.shape[3])]
        vf = global_pool(self.v_conv(y), mask)
        v = torch.tanh(self.v_fc(vf)).squeeze(1)
        s = self.s_fc(vf).squeeze(1) if self.s_fc is not None else None
        return p, v, s


def load_dot_body(net2: AZNet2, dot_sd: dict) -> list[str]:
    """Stem and body weights of a Dot state dict into Dot-2 (the stem gets a
    zero weight for the new phase plane). Returns the Dot-2 keys left as
    they were (heads, pooling biases)."""
    own = net2.state_dict()
    loaded = set()
    for k, v in dot_sd.items():
        if not (k.startswith("stem.") or k.startswith("body.")) or k not in own:
            continue
        if k == "stem.0.weight":
            w = torch.zeros_like(own[k])
            w[:, :v.shape[1]] = v
            v = w
        own[k] = v.clone()
        loaded.add(k)
    net2.load_state_dict(own)
    return sorted(k for k in own if k not in loaded)


def net_planes(net) -> int:
    return getattr(net, "planes", N_PLANES)


def make_net(cfg: dict):
    n = cfg["net"]
    if n.get("arch") == "dot2":
        return AZNet2(n["blocks"], n["channels"], n.get("score_head", True), tuple(n.get("gpool_blocks", (2, 4))),
                      n.get("head_channels", 32))
    w, h = cfg["board"]
    return AZNet(w, h, n["blocks"], n["channels"], n.get("score_head", False))


class Evaluator:
    """Batched inference: boards -> (priors over lines, values), numpy.
    Boards of several sizes are split into one network call per size; the
    priors then come back as a list of arrays (one per board)."""

    def __init__(self, net, device: str):
        self.net = net.to(device).eval()
        self.device = device
        self.codecs = {}
        # the one-size nets' codec (launcher, probes); Dot-2: codec_for(w, h)
        self.codec = Codec(net.w, net.h) if hasattr(net, "w") else None
        if self.codec is not None:
            self.codecs[(net.w, net.h)] = self.codec

    def codec_for(self, w: int, h: int) -> Codec:
        c = self.codecs.get((w, h))
        if c is None:
            c = self.codecs[(w, h)] = Codec(w, h, net_planes(self.net))
        return c

    @torch.no_grad()
    def _one_size(self, boards):
        g = boards[0].geo
        c = self.codec_for(g.w, g.h)
        x = torch.from_numpy(c.encode_boards(boards)).to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
            logits, v, _ = self.net(x)
        logits = logits.float()
        legal = x[:, 1].flatten(1)[:, self.net.line_cells(c.rows, c.cols)] > 0
        logits = logits.masked_fill(~legal, -1e9)
        return torch.softmax(logits, 1).cpu().numpy(), v.float().cpu().numpy()

    def __call__(self, boards):
        sizes = {(b.geo.w, b.geo.h) for b in boards}
        if len(sizes) == 1:
            return self._one_size(boards)
        pri, val = [None] * len(boards), np.zeros(len(boards), dtype=np.float32)
        for wh in sizes:
            idx = [i for i, b in enumerate(boards) if (b.geo.w, b.geo.h) == wh]
            p, v = self._one_size([boards[i] for i in idx])
            for j, i in enumerate(idx):
                pri[i] = p[j]
            val[idx] = v
        return pri, val
