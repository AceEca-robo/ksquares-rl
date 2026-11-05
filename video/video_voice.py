#!/usr/bin/env python3
"""The video's voice-over: the "Голос" text of every scene of
docs/video_script_ru.md through edge-tts, one mp3 per scene (taken from
pacman-rl/video; a scene without a voice, such as "2п", is a pause).

    python3 -m venv video/.tts && video/.tts/bin/pip install edge-tts
    python video/video_voice.py            # video/voice_ru_<N>.mp3, video/voice_ru.json
    python video/video_voice.py --lang en  # docs/video_script_en.md -> video/voice_en_<N>.mp3

edge-tts sends the text to Microsoft's online speech service (no key).
voice_ru.json: per scene the mp3's duration and the scene's slot in the
script (from its "(m:ss–m:ss)" heading), so overruns show at once.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# what the voice gets instead of the script's words: the Latin names in
# Russian letters, so the Russian voice does not spell them out
PRONOUNCE = {
    "KSquares": "Кей-скверс",
    "AlphaZero": "Альфа-Зиро",
    "KDE": "Кей-Ди-И",
    "Easy": "Изи",
    "Medium": "Ми\u0301диум",
    "Hard": "Хард",
    "Dot": "Дот",                  # the network's name in the video
    "DeepMind": "Дип-Майнд",
    "install.sh": "инстол точка эс-эйч",
}
PRONOUNCE_EN = {
    "KSquares": "K-Squares",
    "KDE": "K-D-E",
    "install.sh": "install dot S-H",
}

# per language: the script, the voice, the scene heading and voice label
LANGS = {
    "ru": ("video_script_ru.md", "ru-RU-DmitryNeural", "Сцена", "**Голос:**", PRONOUNCE),
    "en": ("video_script_en.md", "en-US-GuyNeural", "Scene", "**Voice:**", PRONOUNCE_EN),
}


def scenes(path, lang="ru"):
    """[(number, title, slot seconds, voice text)] from the script; number
    is an int, or a str for an inserted scene such as "3а"."""
    _, _, heading, label, _ = LANGS[lang]
    text = open(path, encoding="utf-8").read()
    out = []
    for m in re.finditer(rf"^## {heading} (\d+[а-яa-z]?)\. (.+?) \((\d+):(\d+)–(\d+):(\d+)\)\n(.*?)(?=^## |\Z)",
                         text, re.M | re.S):
        n, title = m.group(1), m.group(2)
        n = int(n) if n.isdigit() else n
        slot = (int(m.group(5)) * 60 + int(m.group(6))) - (int(m.group(3)) * 60 + int(m.group(4)))
        body = m.group(7)
        if label not in body:
            continue                    # a pause: no voice
        v = body.index(label)
        voice = body[v + len(label):].strip()
        voice = re.sub(r"\s+", " ", voice)
        if voice[:1] + voice[-1:] in ("«»", '""'):
            voice = voice[1:-1]
        out.append((n, title, slot, voice))
    return out


def scene_order(n):
    """Sort key of a scene number: 3 < "3а" < 4."""
    m = re.match(r"(\d+)(.*)", str(n))
    return int(m.group(1)), m.group(2)


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return float(r.stdout)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--script", help="default: the lang's docs/video_script*.md")
    p.add_argument("--out", default=os.path.join(ROOT, "video"))
    p.add_argument("--voice", help="default: the lang's voice")
    p.add_argument("--lang", choices=sorted(LANGS), default="ru")
    p.add_argument("--edge-tts", default=os.path.join(ROOT, "video", ".tts", "bin", "edge-tts"))
    p.add_argument("--only", nargs="*", help="these scenes only (3, 3а, ...)")
    p.add_argument("--keep", action="store_true",
                   help="reuse the mp3s already there (only measure them)")
    args = p.parse_args()
    script, voice, _, _, pronounce = LANGS[args.lang]
    args.script = args.script or os.path.join(ROOT, "docs", script)
    args.voice = args.voice or voice
    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, f"voice_{args.lang}.json")
    summary = {}   # scene number (str) -> entry; --only keeps the other scenes
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)["scenes"]
    for n, title, slot, text in scenes(args.script, args.lang):
        if args.only and str(n) not in args.only:
            continue
        mp3 = os.path.join(args.out, f"voice_{args.lang}_{n}.mp3")
        words = len(text.split())
        for a, b in pronounce.items():
            text = text.replace(a, b)
        if os.path.exists(mp3) and os.path.getsize(mp3) == 0:
            os.remove(mp3)                  # left by a failed attempt
        if not (args.keep and os.path.exists(mp3)):
            if not tts(args, text, mp3, f"scene {n}", tries=4):
                # the service has spells where it refuses longer texts: one
                # sentence at a time, joined with a short pause
                print(f"scene {n}: sentence by sentence", flush=True)
                if not by_sentence(args, text, mp3, f"scene {n}"):
                    sys.exit(f"scene {n}: edge-tts failed (the scenes done so far are saved)")
        d = duration(mp3)
        summary[str(n)] = {"title": title, "file": os.path.basename(mp3), "seconds": round(d, 2),
                           "slot_seconds": slot, "over_slot_seconds": round(d - slot, 2),
                           "words": words}
        print(f"scene {n}: {d:6.1f} s of {slot} s  {title}", flush=True)
        save(summary_path, args.voice, summary)   # after every scene: a failure keeps them
    total = save(summary_path, args.voice, summary)
    print(f"total {total:.1f} s -> {summary_path}")


def tts(args, text, mp3, what, tries):
    """text -> mp3 through edge-tts, retried (the service now and then
    answers with no audio); False if every try failed."""
    for attempt in range(tries):
        try:
            # the service now and then never answers: edge-tts then hangs for good
            r = subprocess.run([args.edge_tts, "--voice", args.voice, "--text", text,
                                "--write-media", mp3], capture_output=True, text=True, timeout=180)
            err = (r.stderr.strip().splitlines() or ["?"])[-1]
            if r.returncode == 0 and os.path.getsize(mp3) > 0:
                return True
        except subprocess.TimeoutExpired:
            err = "no answer in 180 s"
        print(f"{what}: attempt {attempt + 1} failed: {err}", flush=True)
        if os.path.exists(mp3):
            os.remove(mp3)
        time.sleep(5 * (attempt + 1))
    return False


PAUSE = 0.35     # seconds between sentences when a scene is voiced one sentence at a time


def by_sentence(args, text, mp3, what):
    parts = []
    for i, sent in enumerate(re.split(r"(?<=[.!?…])\s+", text.strip())):
        part = f"{mp3[:-4]}.part{i:02d}.mp3"
        if os.path.exists(part) and os.path.getsize(part) > 0:
            parts.append(part)          # left by an earlier, failed run of the same text
            continue
        if not tts(args, sent, part, f"{what} sentence {i + 1}", tries=12):
            return False
        parts.append(part)
    inputs, chain = [], []
    for i, part in enumerate(parts):
        inputs += ["-i", part]
        chain.append(f"[{i}:a]aresample=24000,apad=pad_dur={PAUSE}[a{i}]")
    fc = ";".join(chain) + ";" + "".join(f"[a{i}]" for i in range(len(parts))) + \
        f"concat=n={len(parts)}:v=0:a=1,atrim=end=999[out]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", fc, "-map", "[out]",
                    "-c:a", "libmp3lame", "-b:a", "48k", "-ac", "1", mp3], check=True)
    for part in parts:
        os.remove(part)
    return True


def save(path, voice, summary):
    total = sum(v["seconds"] for v in summary.values())
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"voice": voice, "total_seconds": round(total, 2),
                   "scenes": {k: summary[k] for k in sorted(summary, key=scene_order)}},
                  f, ensure_ascii=False, indent=1)
    return total


if __name__ == "__main__":
    sys.exit(main())
