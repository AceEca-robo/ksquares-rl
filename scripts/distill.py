"""Distil Dot-2 (agents/az_net.py AZNet2) from Dot on 5x5 (configs/dot2_distill.yaml).

Positions: a replay buffer of Dot's run; targets: the teacher's policy
(softmax over legal lines), value and score head on the same position,
under a random board symmetry. Loss: cross-entropy to the teacher's policy
+ ``value_weight`` x MSE to its value + ``score_weight`` x MSE to its score.
The student's stem and body start from the teacher (``load_dot_body``);
the first ``heads_only_steps`` train only the new parameters.

Writes ``run_dir/student.pt`` (state dict, the init for configs/dot2a.yaml)
and ``run_dir/log.jsonl``; held-out positions: policy argmax agreement,
cross-entropy minus the teacher's entropy (= KL), value MSE to the teacher
and to the game result.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from _common import ROOT, load_config

from agents.az_net import N_PLANES, AZNet, Codec, load_dot_body, make_net


def batch(d, idx, codec, rng, syms):
    x = codec.encode(d["lines"][idx], d["sides"][idx], d["owner"][idx], d["diff"][idx].astype(np.float32),
                     d["first"][idx])
    if syms:
        B = len(x)
        x = x.reshape(B, x.shape[1], -1)
        t = rng.integers(0, len(codec.syms), B)
        for k, (cperm, _) in enumerate(codec.syms):
            m = t == k
            if k and m.any():
                x[m] = x[m][:, :, cperm]
        x = x.reshape(B, -1, codec.rows, codec.cols)
    return x


def targets(teacher, x, cells):
    with torch.no_grad():
        logits, v, s = teacher(x[:, :N_PLANES])
        legal = x[:, 1].flatten(1)[:, cells] > 0
        p = torch.softmax(logits.masked_fill(~legal, -1e9), 1)
    return p, v, s, legal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/dot2_distill.yaml"))
    cfg = load_config(ap.parse_args().config)
    tc = cfg["train"]
    dev = cfg["device"]
    run = ROOT / cfg["run_dir"]
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    w, h = cfg["board"]
    codec = Codec(w, h, planes=12)

    tn = cfg["teacher_net"]
    teacher = AZNet(w, h, tn["blocks"], tn["channels"], tn["score_head"])
    tsd = torch.load(ROOT / cfg["teacher"], map_location="cpu")
    teacher.load_state_dict(tsd)
    teacher.to(dev).eval()
    student = make_net(cfg)
    new = load_dot_body(student, tsd)
    student.to(dev)
    cells = student.line_cells(codec.rows, codec.cols)

    raw = np.load(ROOT / cfg["data"])
    n = int(raw["n"])
    d = {k: raw[k][:n] for k in ("lines", "sides", "owner", "diff", "first", "z")}
    perm = rng.permutation(n)
    hold, train_idx = perm[:cfg["holdout"]], perm[cfg["holdout"]:]
    print(f"{n} positions, {len(train_idx)} for training; new parameters: {sorted({k.split('.')[0] for k in new})}",
          flush=True)

    new_set = set(new)
    opt = torch.optim.AdamW(student.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    steps = tc["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps)))
    log = open(run / "log.jsonl", "w")

    def evaluate():
        student.eval()
        tot = {"agree": 0.0, "kl": 0.0, "v_teacher": 0.0, "v_result": 0.0, "teacher_v_result": 0.0}
        for i in range(0, len(hold), 1000):
            idx = hold[i:i + 1000]
            x = torch.from_numpy(batch(d, idx, codec, rng, False)).to(dev)
            p, vt, _, legal = targets(teacher, x, cells)
            with torch.no_grad():
                logits, v, _ = student(x)
                logp = F.log_softmax(logits.masked_fill(~legal, -1e9), 1)
            z = torch.from_numpy(d["z"][idx].astype(np.float32)).to(dev)
            tot["agree"] += (logp.argmax(1) == p.argmax(1)).float().sum().item()
            tot["kl"] += (p * (torch.log(p.clamp_min(1e-12)) - logp)).sum().item()
            tot["v_teacher"] += ((v - vt) ** 2).sum().item()
            tot["v_result"] += ((v - z) ** 2).sum().item()
            tot["teacher_v_result"] += ((vt - z) ** 2).sum().item()
        student.train()
        return {k: v / len(hold) for k, v in tot.items()}

    t0 = time.time()
    student.train()
    acc = {"policy": 0.0, "value": 0.0, "score": 0.0}
    for step in range(1, steps + 1):
        frozen = step <= tc["heads_only_steps"]
        for k, p in student.named_parameters():
            p.requires_grad_(not frozen or k in new_set)
        if frozen:  # BN statistics of the body stay the teacher's too
            for m in student.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
            for m in (student.g_conv, student.p_out, student.v_conv):
                m.train()
        elif step == tc["heads_only_steps"] + 1:
            student.train()
        idx = rng.choice(train_idx, tc["batch_size"])
        x = torch.from_numpy(batch(d, idx, codec, rng, tc["symmetries"])).to(dev, non_blocking=True)
        p, vt, st, legal = targets(teacher, x, cells)
        logits, v, s = student(x)
        lp = -(p * F.log_softmax(logits.masked_fill(~legal, -1e9), 1)).sum(1).mean()
        lv = F.mse_loss(v, vt)
        ls = F.mse_loss(s, st)
        loss = lp + tc["value_weight"] * lv + tc["score_weight"] * ls
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        acc["policy"] += lp.item()
        acc["value"] += lv.item()
        acc["score"] += ls.item()
        if step % tc["log_every"] == 0 or step == steps:
            k = tc["log_every"] if step % tc["log_every"] == 0 else step % tc["log_every"]
            rec = {"step": step, "seconds": time.time() - t0, **{f"train_{a}": b / k for a, b in acc.items()},
                   **evaluate()}
            acc = {a: 0.0 for a in acc}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[{time.strftime('%H:%M:%S')}] step {step}: agree {100 * rec['agree']:.1f}%, "
                  f"KL {rec['kl']:.4f}, V-mse teacher {rec['v_teacher']:.4f}, result {rec['v_result']:.3f} "
                  f"(teacher {rec['teacher_v_result']:.3f}), {rec['seconds']:.0f} s", flush=True)
            tmp = run / "student.pt.tmp"
            torch.save(student.state_dict(), tmp)
            tmp.replace(run / "student.pt")


if __name__ == "__main__":
    main()
