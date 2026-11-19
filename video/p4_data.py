"""Data of the part 4 video (docs/video_script_part4_ru.md), read from the run files and reports, never typed in.
Each loader names its source; docs/video_claims_part4.md lists the same sources per claim."""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]


def _j(p):
    return json.load(open(os.path.join(ROOT, p)))


def click_curves():
    """Scene 3: 5x5 validation % by training step. Dot-G = series 1 B s1-5 + R1 s6-10 (runs/graph_s1);
    convolution = series 2 (2c) C s1-5 (runs/graph_s2); simple graph = series 3' Ga stage 1 s1-5 (runs/graph_s3p)."""
    def acc(v):
        v = v["5x5"]
        return v["acc"] if isinstance(v, dict) else v
    out = {"gc": [], "c": [], "ga": []}
    for name in [f"B_s{i}" for i in range(1, 6)] + [f"R1_s{i}" for i in range(6, 11)]:
        out["gc"].append([(0, 50.0)] + [(e["step"], acc(e["val"])) for e in _j(f"runs/graph_s1/{name}.json")["log"]])
    for i in range(1, 6):
        out["c"].append([(0, 50.0)] + [(e["step"], acc(e["val"])) for e in _j(f"runs/graph_s2/2c_C_s{i}.json")["log"]])
        out["ga"].append([(0, 50.0)] + [(e["step"], acc(e["val"])) for e in _j(f"runs/graph_s3p/a_Ga_s{i}.json")["log"]])
    return out


def md_table(path, first_header):
    """Rows (dicts by header) of the markdown table whose header starts with first_header."""
    lines = open(os.path.join(ROOT, path), encoding="utf-8").read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("| " + first_header))
    head = [c.strip() for c in lines[i].strip("|").split("|")]
    rows = []
    for l in lines[i + 2:]:
        if not l.startswith("|"):
            break
        rows.append(dict(zip(head, [c.strip() for c in l.strip("|").split("|")])))
    return rows


def school():
    """Scene 4: 10x7 click without school (series 2 2a), with school (series 2' (a)), both sizes at the end
    without / with the 1/4 share (2' (a) / 2'' k25)."""
    def count(path, key, val="True"):
        t = open(os.path.join(ROOT, path), encoding="utf-8").read()
        return t
    t2 = open(os.path.join(ROOT, "docs/graph_s2.md"), encoding="utf-8").read()
    t2p = open(os.path.join(ROOT, "docs/graph_s2p.md"), encoding="utf-8").read()
    t2pp = open(os.path.join(ROOT, "docs/graph_s2pp.md"), encoding="utf-8").read()
    no_school = int(re.search(r"## 2a Gc: 10x7 click (\d+)/10", t2).group(1))
    a = re.search(r"Gc, 10/10 runs\. 10x7 click (\d+)/10, hold 10x7 \d+/10, hold 5x5 \d+/10, both at the end (\d+)/10", t2p)
    k25 = re.search(r"## k25: share 0\.25.*?both at the end (\d+)/10", t2pp, re.S)
    return {"no_school": no_school, "school": int(a.group(1)), "both_plain": int(a.group(2)), "both_k25": int(k25.group(1))}


def jump_cloud():
    """Scene 5: best 10x7 judge validation after the switch, series 5 seeds 6-25 (docs/graph_s5.md table 1)."""
    rows = md_table("docs/graph_s5.md", "seed | stage 1")
    return [(int(r["seed"]), float(r["best val 10x7"])) for r in rows]


def four_nets():
    """Scene 5: jumps of the 4 school graduates, 5 stage 2 seeds each (docs/graph_s6.md table 2)."""
    rows = md_table("docs/graph_s6.md", "stage 1 weights | series 5 run (not counted)")
    return [(r["stage 1 weights"], int(r["jumps"]), int(r["runs finished"])) for r in rows]


def mid_and_chart():
    """Scene 6: Gc with a jump on the chart positions by safe lines (docs/graph_s7.md table 3) and the primary
    numbers on the decisive mid positions."""
    t = open(os.path.join(ROOT, "docs/graph_s7.md"), encoding="utf-8").read()
    t3 = t[t.index("## Table 3"):]
    line = {}
    for m in re.finditer(r"^\| (\d+) \| (\d+) \| ([\d.]+) \|", t3, re.M):
        line[int(m.group(1))] = float(m.group(3))
    m = re.search(r"\| Gc with a jump \(primary\) \| 9 \| ([\d.]+) \| ([\d.]+) \|", t)
    return line, float(m.group(1)), float(m.group(2))


def ladder_rows():
    """Scenes 7-8: ladder strength with thinking (MCTS 200, Bradley-Terry on random / greedy / medium / hard /
    strong) by round, Dot-G (g5, with the 300-games values where they exist) and Dot (az5 then az5b), as in
    docs/graph_s9.md."""
    from _common import load_config
    import graph_s9 as s9
    cfg = load_config(os.path.join(ROOT, "configs/g5.yaml"))
    anchors = s9.anchors_for(cfg)
    dot_cfg = load_config(os.path.join(ROOT, cfg["series9"]["dot"]["configs"][0]))
    opps = [o for o in cfg["eval"]["opponents"] if o in dot_cfg["eval"]["opponents"]]
    G = s9.with_recheck(cfg, s9.g5_rows(cfg, anchors, opps), anchors, opps)
    D = s9.dot_rows(cfg, anchors, opps)
    return G, D, s9.first_round(G, "hard", 0.9), s9.first_round(D, "hard", 0.9)


def s10():
    """Scenes 8-9: runs/graph_s10/results.json (series 10)."""
    return _j("runs/graph_s10/results.json")


def bridge():
    """Scene 7: g5 gen 49 against the real KSquares Hard (runs/g5/bridge.json)."""
    b = _j("runs/g5/bridge.json")
    return b


def equal_time():
    return _j("runs/g5/match.json")["equal_time"]
