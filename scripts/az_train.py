"""AlphaZero training loop on 5x5 (configs/az5.yaml).

One generation: self-play with the best net (worker processes, each running
many games in lock step with batched GPU inference) -> replay buffer of the
last ``buffer_size`` positions -> training of the candidate (policy CE +
value MSE + L2 via weight decay, random board symmetry per sample) ->
gating: candidate vs best, accepted at >= ``threshold`` of the games.
As in AlphaGo Zero, the trainer keeps its weights when the candidate is
rejected; only self-play stays on the best net.

Checkpoints in ``run_dir``: ``state.json`` (generation, best, history),
``trainer.pt`` (candidate + optimizer), ``buffer.npz``, ``best.pt`` and
``gen_XXXX.pt`` for every accepted net. Everything is written atomically at
the end of every phase, so the run resumes from the last finished phase.

Safe stop: create ``run_dir/STOP``; the loop saves and exits after the
current phase (a STOP file found at start-up is removed) and also stops the evaluator (scripts/az_eval.py), which it
starts in the background whenever an accepted net has no evaluation yet.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _common import ROOT, load_config

from agents.az_net import Codec, Evaluator, make_net

# --- worker processes -----------------------------------------------------------

_W = {}


def _winit(cfg):
    torch.set_num_threads(1)
    _W["cfg"] = cfg
    _W["nets"] = {}


def _evaluator(path):
    cache = _W["nets"]
    key = (path, os.path.getmtime(path))
    if key not in cache:
        cfg = _W["cfg"]
        net = make_net(cfg)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        cache.clear() if len(cache) > 4 else None
        cache[key] = Evaluator(net, cfg["device"])
    return cache[key]


def _selfplay_task(args):
    from agents.az_play import selfplay
    path, n_games, seed = args
    cfg = _W["cfg"]

    def progress(k, sec):
        print(f"[{time.strftime('%H:%M:%S')}]   worker {os.getpid()}: {k}/{n_games} games, {sec:.0f} s", flush=True)

    return selfplay(_evaluator(path), cfg, n_games, cfg["selfplay"]["games_per_worker"], seed, progress)


def _match_task(args):
    from agents.az_play import match
    path_a, path_b, n_games, seed, offset = args
    cfg = _W["cfg"]
    g = cfg["gating"]
    return match([_evaluator(path_a), _evaluator(path_b)], cfg, g["sims"], g["temp_moves"], n_games,
                 cfg["selfplay"]["games_per_worker"], seed, offset)


# --- replay buffer --------------------------------------------------------------

FIELDS = {"lines": (np.uint8, 60), "sides": (np.uint8, 25), "owner": (np.int8, 25), "diff": (np.int8, None),
          "first": (np.uint8, None), "pi": (np.float16, 60), "z": (np.int8, None), "margin": (np.int8, None)}


class Buffer:
    def __init__(self, cap: int, n_lines: int, n_boxes: int):
        self.cap, self.n, self.pos = cap, 0, 0
        shape = {"lines": n_lines, "sides": n_boxes, "owner": n_boxes, "pi": n_lines}
        self.a = {k: np.zeros((cap,) + ((shape[k],) if k in shape else ()), dtype=t)
                  for k, (t, _) in FIELDS.items()}

    def add(self, s: dict) -> None:
        m = len(s["z"])
        idx = (self.pos + np.arange(m)) % self.cap
        for k in self.a:
            self.a[k][idx] = s[k]
        self.pos = (self.pos + m) % self.cap
        self.n = min(self.cap, self.n + m)

    def sample(self, rng, k: int) -> dict:
        idx = rng.integers(0, self.n, k)
        return {f: v[idx] for f, v in self.a.items()}

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, n=self.n, pos=self.pos, **self.a)
        os.replace(tmp, path)

    def load(self, path: Path) -> None:
        d = np.load(path)
        self.n, self.pos = int(d["n"]), int(d["pos"])
        for k in self.a:
            self.a[k][:] = d[k]


# --- training -------------------------------------------------------------------

def train_steps(net, opt, buf: Buffer, codec: Codec, cfg: dict, rng, steps: int) -> dict:
    tc = cfg["train"]
    dev = cfg["device"]
    syms = codec.syms if tc["symmetries"] else codec.syms[:1]
    net.train()
    tot = {"policy": 0.0, "value": 0.0, "score": 0.0}
    sw = tc.get("score_weight", 0.0)
    for _ in range(steps):
        s = buf.sample(rng, tc["batch_size"])
        x = codec.encode(s["lines"], s["sides"], s["owner"], s["diff"].astype(np.float32), s["first"])
        B = len(x)
        x = x.reshape(B, x.shape[1], -1)
        pi = s["pi"].astype(np.float32)
        t = rng.integers(0, len(syms), B)
        for k, (cperm, lperm) in enumerate(syms):
            m = t == k
            if k and m.any():
                x[m] = x[m][:, :, cperm]
                pi[m] = pi[m][:, lperm]
        x = torch.from_numpy(x.reshape(B, -1, codec.rows, codec.cols)).to(dev, non_blocking=True)
        pi = torch.from_numpy(pi).to(dev)
        z = torch.from_numpy(s["z"].astype(np.float32)).to(dev)
        logits, v, sc = net(x)
        legal = x[:, 1].flatten(1)[:, net.line_cell] > 0
        logp = F.log_softmax(logits.masked_fill(~legal, -1e9), 1)
        lp = -(pi * logp).sum(1).mean()
        lv = F.mse_loss(v, z)
        loss = lp + lv
        if sc is not None:
            m = torch.from_numpy(s["margin"].astype(np.float32) / codec.n_boxes).to(dev)
            ls = F.mse_loss(sc, m)
            loss = loss + sw * ls
            tot["score"] += ls.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot["policy"] += lp.item()
        tot["value"] += lv.item()
    net.eval()
    return {k: v / steps for k, v in tot.items()}


# --- driver ---------------------------------------------------------------------

def save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".json":
        tmp.write_text(json.dumps(obj, indent=1))
    else:
        torch.save(obj, tmp)
    os.replace(tmp, path)


def log(run: Path, msg: str, **rec) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if rec:
        with open(run / "log.jsonl", "a") as f:
            f.write(json.dumps({"time": time.time(), **rec}) + "\n")


def split(n: int, k: int) -> list[int]:
    return [n // k + (i < n % k) for i in range(k)]


class Run:
    def __init__(self, cfg_path: str, generations: int | None):
        self.cfg_path = cfg_path
        self.cfg = cfg = load_config(cfg_path)
        self.run = ROOT / cfg["run_dir"]
        self.run.mkdir(parents=True, exist_ok=True)
        self.max_gens = generations
        w, h = cfg["board"]
        self.codec = Codec(w, h)
        torch.manual_seed(cfg["seed"])
        torch.backends.cudnn.benchmark = True
        self.net = make_net(cfg).to(cfg["device"])
        tc = cfg["train"]
        self.opt = torch.optim.SGD(self.net.parameters(), lr=tc["lr"], momentum=tc["momentum"],
                                   weight_decay=tc["weight_decay"], nesterov=True)
        self.buf = Buffer(cfg["buffer_size"], self.codec.n_lines, self.codec.n_boxes)
        self.state = {"gen": 0, "best": 0, "history": [], "phase": "selfplay"}
        self.eval_proc = None
        if (self.run / "STOP").exists():  # left over from the last safe stop
            (self.run / "STOP").unlink()
        st = self.run / "state.json"
        if st.exists():
            self.state = json.loads(st.read_text())
            ck = torch.load(self.run / "trainer.pt", map_location=cfg["device"])
            self.net.load_state_dict(ck["net"])
            self.opt.load_state_dict(ck["opt"])
            if (self.run / "buffer.npz").exists():
                self.buf.load(self.run / "buffer.npz")
            log(self.run, f"resumed: gen {self.state['gen']} phase {self.state['phase']}, best gen "
                          f"{self.state['best']}, buffer {self.buf.n}")
        else:
            if cfg.get("init_from"):
                sd = torch.load(ROOT / cfg["init_from"], map_location=cfg["device"])
                missing, unexpected = self.net.load_state_dict(sd, strict=False)
                assert not unexpected, unexpected
                log(self.run, f"initialised from {cfg['init_from']}; new (random) parameters: "
                              f"{sorted({k.split('.')[0] for k in missing})}")
            save_atomic(self.net.state_dict(), self.run / "gen_0000.pt")
            save_atomic(self.net.state_dict(), self.run / "best.pt")
            self.save()
        self.rng = np.random.default_rng([cfg["seed"], self.state["gen"], 1])

    def save(self) -> None:
        save_atomic({"net": self.net.state_dict(), "opt": self.opt.state_dict()}, self.run / "trainer.pt")
        save_atomic(self.state, self.run / "state.json")

    def stop_requested(self) -> bool:
        return (self.run / "STOP").exists()

    def maybe_eval(self) -> None:
        if self.eval_proc is not None and self.eval_proc.poll() is None:
            return
        missing = [h["gen"] for h in self.state["history"] if h["accepted"]
                   and not (self.run / "eval" / f"gen_{h['gen']:04d}.json").exists()]
        if missing:
            out = open(self.run / "eval.out", "a")
            self.eval_proc = subprocess.Popen([sys.executable, str(ROOT / "scripts/az_eval.py"), "--config",
                                               self.cfg_path], stdout=out, stderr=subprocess.STDOUT)
            log(self.run, f"evaluator started for gens {missing}")

    def loop(self) -> None:
        cfg = self.cfg
        ctx = mp.get_context("spawn")
        nw = cfg["selfplay"]["workers"]
        done_gens = 0
        with ctx.Pool(nw, initializer=_winit, initargs=(cfg,)) as pool:
            while self.max_gens is None or done_gens < self.max_gens:
                self.maybe_eval()
                if self.stop_requested():
                    break
                gen = self.state["gen"] + 1
                if self.state["phase"] == "selfplay":
                    self.selfplay(pool, gen)
                    self.state["phase"] = "train"
                    self.buf.save(self.run / "buffer.npz")
                    self.save()
                    if self.stop_requested():
                        break
                if self.state["phase"] == "train":
                    self.train(gen)
                    self.state["phase"] = "gate"
                    self.save()
                    if self.stop_requested():
                        break
                self.gate(pool, gen)
                self.state["gen"] = gen
                self.state["phase"] = "selfplay"
                self.save()
                done_gens += 1
        self.maybe_eval()
        if self.stop_requested() and self.eval_proc is not None and self.eval_proc.poll() is None:
            self.eval_proc.terminate()
            self.eval_proc.wait()
            log(self.run, "evaluator stopped")
        elif self.eval_proc is not None:
            self.eval_proc.wait()
        log(self.run, "stopped cleanly" if self.stop_requested() else "finished")

    def selfplay(self, pool, gen: int) -> None:
        sp = self.cfg["selfplay"]
        best = str(self.run / "best.pt")
        t0 = time.time()
        tasks = [(best, n, int(self.rng.integers(2**31))) for n in split(sp["games_per_gen"], sp["workers"]) if n]
        stats = []
        for samples, st in pool.imap_unordered(_selfplay_task, tasks):
            self.buf.add(samples)
            stats.append(st)
        dt = time.time() - t0
        evals = sum(s["evals"] for s in stats)
        rec = {"gen": gen, "phase": "selfplay", "games": sum(s["games"] for s in stats),
               "positions": sum(s["positions"] for s in stats), "evals": evals, "seconds": dt,
               "sims_per_s": evals / dt, "worker_sims_per_s": [s["evals"] / s["seconds"] for s in stats],
               "mean_len": float(np.mean([s["mean_len"] for s in stats])),
               "first_wins": sum(s["first_wins"] for s in stats), "buffer": self.buf.n}
        log(self.run, f"gen {gen} self-play: {rec['games']} games, {rec['positions']} positions, "
                      f"{dt:.0f} s, {rec['sims_per_s']:.0f} evals/s, first mover won {rec['first_wins']}, "
                      f"buffer {self.buf.n}", **rec)

    def train(self, gen: int) -> None:
        tc = self.cfg["train"]
        if self.buf.n < tc["min_buffer"]:
            log(self.run, f"gen {gen} train: buffer {self.buf.n} < {tc['min_buffer']}, skipped")
            return
        t0 = time.time()
        loss = train_steps(self.net, self.opt, self.buf, self.codec, self.cfg, self.rng, tc["steps_per_gen"])
        dt = time.time() - t0
        save_atomic(self.net.state_dict(), self.run / "candidate.pt")
        log(self.run, f"gen {gen} train: {tc['steps_per_gen']} steps, {dt:.0f} s, policy {loss['policy']:.3f}, "
                      f"value {loss['value']:.3f}, score {loss['score']:.4f}", gen=gen, phase="train", seconds=dt, **loss)

    def gate(self, pool, gen: int) -> None:
        g = self.cfg["gating"]
        cand = self.run / "candidate.pt"
        if not cand.exists():
            save_atomic(self.net.state_dict(), cand)
        t0 = time.time()
        tasks, off = [], 0
        # few large batches: under GPU time slicing the cost is per call
        for n in split(g["games"], g["tasks"]):
            if n:
                tasks.append((str(cand), str(self.run / "best.pt"), n, int(self.rng.integers(2**31)), off % 2))
                off += n
        res = np.sum(pool.map(_match_task, tasks), axis=0)
        wins, losses, ties, margin = (int(v) for v in res)
        n = wins + losses + ties
        score = (wins + ties / 2) / n
        ok = score >= g["threshold"]
        prev = self.state["best"]
        if ok:
            sd = torch.load(cand, map_location="cpu")
            save_atomic(sd, self.run / f"gen_{gen:04d}.pt")
            save_atomic(sd, self.run / "best.pt")
            self.state["best"] = gen
        dt = time.time() - t0
        self.state["history"].append({"gen": gen, "accepted": ok, "score": score, "wins": wins, "losses": losses,
                                      "ties": ties, "margin": margin / n, "time": time.time()})
        log(self.run, f"gen {gen} gating vs gen {prev}: {wins}-{losses} ({100 * score:.0f}%), "
                      f"margin {margin / n:+.2f}, {dt:.0f} s -> {'ACCEPTED' if ok else 'rejected'}",
            gen=gen, phase="gate", score=score, accepted=ok, vs=prev, seconds=dt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/az5.yaml"))
    ap.add_argument("--generations", type=int, default=None, help="stop after this many (default: run until STOP)")
    a = ap.parse_args()
    Run(a.config, a.generations).loop()


if __name__ == "__main__":
    main()
