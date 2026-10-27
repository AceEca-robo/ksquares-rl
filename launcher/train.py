"""The app's "Train your own Dot" without the UI: the machine check, the
run's config, the training process and its progress.

A run lives in runs/<name>/: train.yaml (configs/az5b.yaml with run_dir
runs/<name>, no init_from = from scratch, publish: false), train.log (the
trainer's and evaluator's output) and what scripts/az_train.py writes there
(state.json, log.jsonl, gen_XXXX.pt, eval/). The trainer is started in its
own session, so it outlives the app; runs/.training.json names the run in
progress, and only one runs at a time. "Stop" is the trainer's safe stop:
runs/<name>/STOP.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

from . import core

RUNS = os.path.join(core.ROOT, "runs")
LOCK = os.path.join(RUNS, ".training.json")
NAME = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
RESERVED = {"best", "az5", "az5b", "az5_smoke", "parity_probe", "strategy_probe"}


def run_dir(name):
    return os.path.join(RUNS, name)


def valid_name(name):
    return bool(NAME.match(name or "")) and name not in RESERVED


# --- the machine ------------------------------------------------------------------

def gpu():
    """{"name", "mem_gb"} of the first CUDA device, or None."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        p = torch.cuda.get_device_properties(0)
        return {"name": p.name, "mem_gb": p.total_memory / 2**30}
    except Exception:
        return None


def disk_free_gb(path=RUNS):
    while not os.path.isdir(path):
        path = os.path.dirname(path)
    return shutil.disk_usage(path).free / 2**30


def machine(cfg):
    """What the mode's first screen says: {"gpu", "disk_gb", "low_vram",
    "low_disk", "running"} (running: the name of the run in progress)."""
    tc = cfg["train"]
    g = gpu()
    free = disk_free_gb()
    return {"gpu": g, "disk_gb": free, "low_vram": bool(g and g["mem_gb"] < tc["min_vram_gb"]),
            "low_disk": free < tc["min_disk_gb"], "running": running()}


# --- the process ---------------------------------------------------------------------

def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:   # a zombie of ours counts as gone
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def running():
    """The name of the run whose trainer is alive, or None (a stale lock is
    removed)."""
    try:
        with open(LOCK) as f:
            lk = json.load(f)
    except (OSError, ValueError):
        return None
    if _alive(lk["pid"]):
        return lk["name"]
    try:
        os.remove(LOCK)
    except OSError:
        pass
    return None


def write_config(name, cfg, device):
    """runs/<name>/train.yaml from the base config; an existing one is kept
    (the run resumes with what it started with)."""
    import yaml

    d = run_dir(name)
    path = os.path.join(d, "train.yaml")
    if os.path.isfile(path):
        return path
    rc = core.load_yaml(cfg["train"]["base_config"])
    rc.update(run_dir=os.path.relpath(d, core.ROOT), init_from=None, device=device, publish=False)
    os.makedirs(d, exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write(f"# The app's \"Train your own Dot\": {cfg['train']['base_config']} from scratch.\n")
        yaml.safe_dump(rc, f, sort_keys=False)
    os.replace(path + ".tmp", path)
    return path


def start(name, rounds, cfg, device=None, argv=None):
    """Start the trainer for runs/<name> (rounds 0: until stopped) detached
    from the app; returns its pid. argv replaces the command (tests)."""
    if not valid_name(name):
        raise ValueError(f"bad run name: {name!r}")
    if running():
        raise RuntimeError(f"already training: {running()}")
    if device is None:
        device = "cuda" if gpu() else "cpu"
    conf = write_config(name, cfg, device)
    d = run_dir(name)
    try:
        os.remove(os.path.join(d, "STOP"))
    except OSError:
        pass
    if argv is None:
        argv = [sys.executable, os.path.join(core.ROOT, "scripts", "az_train.py"), "--config", conf]
        if rounds:
            argv += ["--generations", str(rounds)]
    try:
        with open(os.path.join(d, "state.json")) as f:
            first_gen = json.load(f)["gen"]
    except (OSError, ValueError, KeyError):
        first_gen = 0
    log = open(os.path.join(d, "train.log"), "a")
    log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} started from the app: {' '.join(argv)}\n")
    log.flush()
    p = subprocess.Popen(argv, cwd=core.ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    log.close()
    with open(LOCK + ".tmp", "w") as f:
        json.dump({"name": name, "pid": p.pid, "rounds": rounds, "first_gen": first_gen,
                   "started": time.time()}, f)
    os.replace(LOCK + ".tmp", LOCK)
    return p


def stop(name):
    """The trainer's safe stop: it saves and exits after the current phase."""
    open(os.path.join(run_dir(name), "STOP"), "w").close()


def kill(name):
    """Stop at once (tests, a hung trainer): the whole process group."""
    try:
        with open(LOCK) as f:
            lk = json.load(f)
        if lk["name"] == name:
            os.killpg(lk["pid"], signal.SIGTERM)
    except (OSError, ValueError, ProcessLookupError):
        pass


# --- progress -------------------------------------------------------------------------

def own_runs():
    """Names of the app's runs on disk (those with train.yaml), sorted."""
    try:
        return sorted(n for n in os.listdir(RUNS) if os.path.isfile(os.path.join(RUNS, n, "train.yaml")))
    except OSError:
        return []


def progress(name, cfg):
    """Everything the mode shows about runs/<name>, read from its files:
    {"running", "stopping", "gen", "phase", "rounds" (planned, 0 = no limit),
    "history" [{"gen", "accepted", "wins", "losses", "score", "vs",
    "seconds"}], "round_s" (median of the finished rounds), "eta_s" (to the
    planned last round, None without a limit), "since_s" (in the current
    round), "ladder" [(gen, raw % vs hard, mcts % vs hard)], "log" (last
    lines of train.log)}."""
    d = run_dir(name)
    out = {"running": running() == name, "stopping": os.path.exists(os.path.join(d, "STOP")),
           "gen": 0, "phase": None, "history": [], "round_s": None, "eta_s": None, "since_s": None,
           "ladder": [], "log": [], "rounds": 0}
    try:
        with open(LOCK) as f:
            lk = json.load(f)
        if lk["name"] == name:
            out["rounds"], started = lk["rounds"], lk["started"]
        else:
            started = None
    except (OSError, ValueError):
        started = None
    try:
        with open(os.path.join(d, "state.json")) as f:
            st = json.load(f)
        out["gen"], out["phase"] = st["gen"], st["phase"]
    except (OSError, ValueError):
        st = {"history": []}
    # per round: the three phases' seconds from log.jsonl; ends = when the gate was logged
    secs, ends, vs = {}, {}, {}
    try:
        with open(os.path.join(d, "log.jsonl")) as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if "gen" in r and "seconds" in r:
                    secs.setdefault(r["gen"], {})[r["phase"]] = r["seconds"]
                    if r["phase"] == "gate":
                        ends[r["gen"]] = r["time"]
                        vs[r["gen"]] = r.get("vs")
    except OSError:
        pass
    for h in st["history"]:
        out["history"].append({"gen": h["gen"], "accepted": h["accepted"], "wins": h["wins"], "losses": h["losses"],
                               "score": h["score"], "vs": vs.get(h["gen"]),
                               "seconds": sum(secs.get(h["gen"], {}).values()) or None})
    done = sorted(s for s in (h["seconds"] for h in out["history"]) if s)
    if done:
        out["round_s"] = done[len(done) // 2]
    last_end = max(ends.values(), default=None)
    t0 = max(x for x in (last_end, started) if x is not None) if (last_end or started) else None
    if out["running"] and t0:
        out["since_s"] = max(0.0, time.time() - t0)
    est = out["round_s"] or cfg["train"]["round_minutes_gpu"] * 60
    if out["running"] and out["rounds"]:
        left = max(0, _planned_last(name, out) - out["gen"])
        out["eta_s"] = max(0.0, left * est - (out["since_s"] or 0)) if left else 0.0
    for h in out["history"]:
        p = os.path.join(d, "eval", f"gen_{h['gen']:04d}.json")
        if h["accepted"] and os.path.isfile(p):
            try:
                with open(p) as f:
                    e = json.load(f)
                out["ladder"].append((h["gen"], _pct(e["raw"]["hard"]), _pct(e["mcts"]["hard"])))
            except (OSError, ValueError, KeyError):
                pass
    try:
        with open(os.path.join(d, "train.log"), errors="replace") as f:
            out["log"] = [ln.rstrip() for ln in f.readlines()[-6:]]
    except OSError:
        pass
    return out


def _planned_last(name, out):
    """The last round of the trainer in progress: the round it started at +
    its --generations (kept in the lock)."""
    try:
        with open(LOCK) as f:
            lk = json.load(f)
        return lk.get("first_gen", 0) + lk["rounds"]
    except (OSError, ValueError, KeyError):
        return out["gen"]


def _pct(r):
    w, l, t = r
    return 100.0 * (w + t / 2) / max(1, w + l + t)


def main(argv=None):
    """The same from a terminal: start, stop or show a run."""
    import argparse

    ap = argparse.ArgumentParser(description="Train your own Dot (as the app's mode): " + __doc__.split("\n")[0])
    ap.add_argument("name", nargs="?", help="run name (default: configs/app.yaml train.default_name)")
    ap.add_argument("--rounds", type=int, help="0: until stopped (default: train.rounds_default)")
    ap.add_argument("--stop", action="store_true", help="the safe stop of the run")
    ap.add_argument("--status", action="store_true", help="print the run's progress")
    a = ap.parse_args(argv)
    cfg = core.app_config()
    name = a.name or cfg["train"]["default_name"]
    if a.stop:
        stop(name)
        print(f"{name}: stops after the current phase")
    elif a.status:
        pr = progress(name, cfg)
        print(f"{name}: {'running' if pr['running'] else 'not running'}, rounds done {pr['gen']}, "
              f"phase {pr['phase']}")
        for h in pr["history"]:
            print(f"  round {h['gen']}: {'accepted' if h['accepted'] else 'rejected'} {h['wins']}-{h['losses']}")
        for g, raw, mcts in pr["ladder"]:
            print(f"  gen {g} vs Hard: {raw:.0f}% at once, {mcts:.0f}% with search")
    else:
        g = gpu()
        print(f"GPU: {g['name']}, {g['mem_gb']:.1f} GB" if g else
              f"no CUDA GPU: the CPU is about {cfg['train']['cpu_slowdown']}x slower")
        rounds = cfg["train"]["rounds_default"] if a.rounds is None else a.rounds
        p = start(name, rounds, cfg)
        d = os.path.relpath(run_dir(name), core.ROOT)
        print(f"started (pid {p.pid}, {rounds or 'unlimited'} rounds): {d}/, log {d}/train.log\n"
              f"progress: python -m launcher.train {name} --status; stop: python -m launcher.train {name} --stop")


if __name__ == "__main__":
    main()
