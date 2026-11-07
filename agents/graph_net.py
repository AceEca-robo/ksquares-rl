"""Graph network for Dots and Boxes (part 4, docs/plan_part4.md).

``GraphNet`` takes the planes of ``Codec.encode`` (Dot-2's 12) and returns
(policy, value, score) as AZNet2, so Evaluator, the C tree and the trainers
stay as they are. Nodes: boxes, lines and, with ``segments``, one node per
maximal run of link boxes (a box with two undrawn sides), the chains and
loops of sim/chains.py as they stand on the board. No node has coordinates,
so the net is equivariant to the board's symmetries by construction.

Sum aggregation throughout: parity needs counts. The long chain count and
the rule are not inputs; the net has to add them up.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from agents.az_net import N_PLANES_2, Codec

LINE_FEATS = 5     # drawn, safe, on the border, adjacent 3-sided boxes, adjacent 2-sided boxes
BOX_FEATS = 7      # free with 0..3 drawn sides, taken (one-hot), mine, opponent's
GLOBAL_FEATS = 6   # score, mover moved first, phase, dots odd, sqrt(boxes) / 10, ones
SEG_FEATS = 8      # length / 10, loop, chain of 1, 2, 3+, ground / junction / capturable ends
POOL_SCALE = 10.0  # sums stay sums, on a scale the MLPs work with


class Topology:
    """Fixed index tensors of one board size. Index n_boxes is the padding."""

    def __init__(self, rows: int, cols: int, device):
        c = Codec((cols - 1) // 2, (rows - 1) // 2)
        nb = c.n_boxes
        self.n_boxes, self.n_lines = nb, c.n_lines
        t = lambda a: torch.as_tensor(a, device=device)
        self.line_cell, self.box_cell = t(c.line_cell), t(c.box_cell)
        self.box_line = t(c.box_side_line)                  # (boxes, 4)
        self.box_nb = t(np.where(c.box_nb < 0, nb, c.box_nb))   # (boxes, 4)
        self.line_box = t(np.where(c.line_box2 < 0, nb, c.line_box2))   # (lines, 2)
        self.ground = t(c.box_nb < 0)
        self.dots_odd = float(c.dots % 2)
        self.size = float(np.sqrt(nb)) / 10


def _mlp(a: int, h: int, b: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(a, h), nn.ReLU(), nn.Linear(h, b))


def _pad(x: torch.Tensor) -> torch.Tensor:
    """A zero slot at index n_boxes along dim 1."""
    return torch.cat([x, torch.zeros_like(x[:, :1])], 1)


class Pool(nn.Module):
    """Sums per node kind -> one vector per board."""

    def __init__(self, d: int, q: int, kinds: int):
        super().__init__()
        self.q = nn.ModuleList([nn.Linear(d, q) for _ in range(kinds)])

    def forward(self, hs):
        return torch.cat([(q(h) * m).sum(1) / POOL_SCALE for q, (h, m) in zip(self.q, hs)], 1)


class GraphLayer(nn.Module):
    def __init__(self, d: int, q: int, segments: bool):
        super().__init__()
        self.segments = segments
        self.nb, self.nl, self.ng = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.m_line, self.m_box = nn.Linear(d, d), nn.Linear(d, d)
        self.f_box = _mlp(3 * d, d, d)
        self.f_line = _mlp((4 if segments else 3) * d, d, d)
        kinds = 6 if segments else 3
        self.pool = Pool(d, q, kinds)
        self.f_g = _mlp(d + kinds * q, d, d)
        if segments:
            self.ns = nn.LayerNorm(d)
            self.m_seg, self.m_up = nn.Linear(d, d), nn.Linear(d, d)
            self.f_seg = _mlp(3 * d, d, d)

    def forward(self, s: dict, kinds) -> None:
        t, hb, hl, g = s["topo"], s["box"], s["line"], s["g"]
        gb = g[:, None].expand(-1, t.n_boxes, -1)
        gl = g[:, None].expand(-1, t.n_lines, -1)
        hb = hb + self.f_box(torch.cat([self.nb(hb), self.m_line(hl)[:, t.box_line].sum(2), gb], 2))
        li = [self.nl(hl), _pad(self.m_box(hb))[:, t.line_box].sum(2), gl]
        if self.segments:
            hs = s["seg"]
            m = _pad(self.m_seg(hs) * s["seg_on"])
            li.append((m[torch.arange(len(hb), device=hb.device)[:, None, None], s["line_seg"]]
                       * s["undrawn"][:, :, None, None]).sum(2))
        hl = hl + self.f_line(torch.cat(li, 2))
        if self.segments:
            up = (self.m_up(hl) * s["undrawn"][:, :, None]).repeat_interleave(2, 1)
            d = up.shape[2]
            acc = torch.zeros(len(hb), t.n_boxes + 1, d, device=up.device, dtype=up.dtype)
            acc.scatter_add_(1, s["line_seg"].flatten(1)[:, :, None].expand(-1, -1, d), up)
            gs = g[:, None].expand(-1, t.n_boxes, -1)
            s["seg"] = hs + self.f_seg(torch.cat([self.ns(hs), acc[:, :-1], gs], 2)) * s["seg_on"]
        s["box"], s["line"] = hb, hl
        g = g + self.f_g(torch.cat([self.ng(g), self.pool(kinds(s))], 1))
        s["g"] = g


class GraphNet(nn.Module):
    def __init__(self, layers: int = 8, width: int = 93, pool: int = 16, segments: bool = True):
        super().__init__()
        self.planes = N_PLANES_2
        self.segments = segments
        d = width
        self.e_line = nn.Linear(LINE_FEATS + (2 if segments else 0), d)
        self.e_box = nn.Linear(BOX_FEATS, d)
        self.e_g = nn.Linear(GLOBAL_FEATS, d)
        if segments:
            self.e_seg = nn.Linear(SEG_FEATS, d)
        self.layers = nn.ModuleList([GraphLayer(d, pool, segments) for _ in range(layers)])
        kinds = 6 if segments else 3
        self.readout = Pool(d, pool, kinds)
        self.p_out = _mlp(2 * d, d, 1)
        self.v_fc = nn.Sequential(nn.Linear(d + kinds * pool, d), nn.ReLU(), _mlp(d, d, 1))
        self.s_fc = nn.Sequential(nn.Linear(d + kinds * pool, d), nn.ReLU(), _mlp(d, d, 1))
        self._topo = {}

    def topo(self, rows: int, cols: int) -> Topology:
        dev = self.e_g.weight.device
        k = (rows, cols, str(dev))
        if k not in self._topo:
            self._topo[k] = Topology(rows, cols, dev)
        return self._topo[k]

    def line_cells(self, rows: int, cols: int) -> torch.Tensor:
        return self.topo(rows, cols).line_cell

    @staticmethod
    @torch.no_grad()
    def find_segments(t: Topology, sides: torch.Tensor, undrawn: torch.Tensor):
        """sides (B, boxes) 0..3 free, 4 taken; undrawn (B, lines) bool ->
        label (B, boxes), the segment slot of each box (n_boxes if it is no
        link) and slot features (B, boxes, SEG_FEATS). Same chains as
        Codec.chain_hint and sim/chains.py."""
        B, nb = sides.shape
        link = sides == 2
        linkp = _pad(link)
        und = undrawn[:, t.box_line]                       # (B, boxes, 4)
        to_link = linkp[:, t.box_nb]
        conn = link[:, :, None] & und & to_link
        ids = torch.arange(nb, device=sides.device)
        lab = torch.where(link, ids, nb)
        while True:                                        # min label over connected links
            cand = torch.where(conn, _pad(lab)[:, t.box_nb], nb).amin(2)
            new = torch.minimum(lab, cand)
            if torch.equal(new, lab):
                break
            lab = new
        sp = torch.cat([sides, torch.full((B, 1), -1, device=sides.device)], 1)
        across = sp[:, t.box_nb]
        exit_ = link[:, :, None] & und & ~to_link

        def per_slot(w):                                   # sum of w (B, boxes) per segment
            a = torch.zeros(B, nb + 1, device=sides.device)
            return a.scatter_add_(1, lab, w.float())[:, :-1]

        size = per_slot(link)
        ground = per_slot((exit_ & t.ground).sum(2))
        capt = per_slot((exit_ & (across == 3)).sum(2))
        junction = per_slot((exit_ & (across <= 1) & ~t.ground).sum(2))
        on = size > 0
        loop = on & (ground + capt + junction == 0)
        chain = on & ~loop
        f = torch.stack([size / 10, loop.float(), (chain & (size == 1)).float(), (chain & (size == 2)).float(),
                         (chain & (size >= 3)).float(), ground / 2, junction / 2, capt / 2], 2)
        return lab, f

    def forward(self, x: torch.Tensor):
        t = self.topo(x.shape[2], x.shape[3])
        B = x.shape[0]
        fl = x.flatten(2)
        f32 = fl.dtype
        cl, cb = fl[:, :, t.line_cell], fl[:, :, t.box_cell]
        drawn = cl[:, 0] > 0.5
        undrawn = ~drawn
        free = cb[:, 4:8].sum(1) > 0.5
        sides = torch.where(free, cb[:, 4:8].argmax(1), 4)
        sp = torch.cat([sides, torch.full((B, 1), -1, device=x.device)], 1)
        two = (sp[:, t.line_box] == 2)
        three = (sp[:, t.line_box] == 3)
        safe = undrawn & ~two.any(2)
        border = (t.line_box == t.n_boxes).any(1)
        lf = [drawn, safe, border.expand(B, -1), three.sum(2) * undrawn / 2.0, two.sum(2) * undrawn / 2.0]
        bf = torch.cat([cb[:, 4:8].mT, (~free).to(f32)[:, :, None], cb[:, 2:4].mT], 2)
        c0 = x[:, :, 0, 0]
        gf = torch.stack([c0[:, 8], c0[:, 9], c0[:, 11], torch.full((B,), t.dots_odd, device=x.device, dtype=f32),
                          torch.full((B,), t.size, device=x.device, dtype=f32), torch.ones(B, device=x.device, dtype=f32)], 1)
        s = {"topo": t, "undrawn": undrawn}
        junction = free & (sides <= 1)
        if self.segments:
            lab, sf = self.find_segments(t, sides, undrawn)
            lseg = _pad(lab)[:, t.line_box]                # (B, lines, 2); index n_boxes: none
            lseg = lseg.masked_fill(sp[:, t.line_box] != 2, t.n_boxes)
            mid = (lseg[:, :, 0] == lseg[:, :, 1]) & (lseg[:, :, 0] < t.n_boxes) & undrawn
            end = (lseg < t.n_boxes).any(2) & undrawn & ~mid
            lf += [mid, end]
            s["line_seg"] = lseg
            s["seg_on"] = sf[:, :, 0:1].to(f32) > 0
            s["seg"] = self.e_seg(sf.to(f32)) * s["seg_on"]
            long_ = sf[:, :, 4:5].to(f32)
            loop = sf[:, :, 1:2].to(f32)
            short = sf[:, :, 2:4].sum(2, keepdim=True).to(f32)
        s["line"] = self.e_line(torch.stack([a.to(f32) for a in lf], 2))
        s["box"] = self.e_box(bf.to(f32))
        s["g"] = self.e_g(gf)
        ones = torch.ones(B, t.n_boxes, 1, device=x.device, dtype=f32)
        jm, sm = junction[:, :, None].to(f32), safe[:, :, None].to(f32)

        def kinds(st):
            k = [(st["box"], ones), (st["box"], jm), (st["line"], sm)]
            if self.segments:
                k += [(st["seg"], short), (st["seg"], long_), (st["seg"], loop)]
            return k

        for layer in self.layers:
            layer(s, kinds)
        g = s["g"]
        p = self.p_out(torch.cat([s["line"], g[:, None].expand(-1, t.n_lines, -1)], 2)).squeeze(2)
        r = torch.cat([g, self.readout(kinds(s))], 1)
        return p, torch.tanh(self.v_fc(r)).squeeze(1), self.s_fc(r).squeeze(1)
