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

    "Whose turn" is plane 9 (the identity of the mover); everything else is
    relative to the mover, so value and policy need no colour flip.

Outputs
    Policy logits over the lines in KSquares order (illegal lines are masked
    by the caller), a value in [-1, 1] for the side to move and, with
    ``score_head``, a second value head: the expected final score
    difference for the mover divided by the number of boxes (regression,
    auxiliary; the search uses only the win/loss value). ``forward``
    returns (policy, value, score or None).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sim.board import geometry

N_PLANES = 11


class Codec:
    """Index tables for one board size: grid cells, symmetries."""

    def __init__(self, w: int, h: int):
        geo = geometry(w, h)
        self.w, self.h = w, h
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
        x = np.zeros((B, N_PLANES, self.n_cells), dtype=np.float32)
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
        return x.reshape(B, N_PLANES, self.rows, self.cols)

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


def make_net(cfg: dict) -> AZNet:
    w, h = cfg["board"]
    return AZNet(w, h, cfg["net"]["blocks"], cfg["net"]["channels"], cfg["net"].get("score_head", False))


class Evaluator:
    """Batched inference: boards -> (priors over lines, values), numpy."""

    def __init__(self, net: AZNet, device: str):
        self.net = net.to(device).eval()
        self.device = device
        self.codec = Codec(net.w, net.h)

    @torch.no_grad()
    def __call__(self, boards):
        x = torch.from_numpy(self.codec.encode_boards(boards)).to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
            logits, v, _ = self.net(x)
        logits = logits.float()
        legal = x[:, 1].flatten(1)[:, self.net.line_cell] > 0
        logits = logits.masked_fill(~legal, -1e9)
        return torch.softmax(logits, 1).cpu().numpy(), v.float().cpu().numpy()
