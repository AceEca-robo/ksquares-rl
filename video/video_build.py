#!/usr/bin/env python3
"""The video's draft cut: video/storyboard.yaml -> video/draft_ru.mp4.

    .venv/bin/python video/video_build.py            # draft_ru.mp4, subs_ru.srt, draft_ru.json
    .venv/bin/python video/video_build.py --lang en --storyboard video/storyboard_en_v1.yaml \
        --out video/draft_en_v1.mp4 --srt video/subs_en.srt --build video/build_en --burn

Every fragment is rendered to its own 1920x1080 25 fps clip (cache in
video/build/), the clips are concatenated, the scene voices are laid on one
track at their scene's start, and subs_ru.srt (the script's "Голос" text,
timed to the voice's pauses) goes in as a soft subtitle track (mov_text),
or with --burn into the picture (the .srt stays next to it for upload).
A scene shorter than its voice + 2 s (storyboard: min_tail) gets its last fragment stretched (a
still held longer, a video played on, looped if it runs out); every such
fix lands in the report.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from video_voice import LANGS, duration, scenes as script_scenes  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MIN_TAIL = 2.0          # the scene's picture runs at least this long after the voice
SUB_CHARS = 84          # a subtitle: at most two lines of ~42


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r


def is_still(path):
    return path.lower().endswith(".png")


def textfile(build, name, text):
    path = os.path.join(build, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def caption_filter(build, key, text, size=40, at="bottom"):
    """at: "bottom", or "top" (the storyboard's caption_at) to keep the
    bottom free for burnt-in subtitles."""
    tf = textfile(build, f"cap_{key}.txt", text)
    y = "h-th-80" if at == "bottom" else "60"
    return (f"drawtext=fontfile={FONT_BOLD}:textfile={tf}:fontsize={size}:fontcolor=white:"
            f"box=1:boxcolor=black@0.65:boxborderw=18:x=(w-tw)/2:y={y}")


def render_fragment(sb, build, key, frag, src_dur):
    """One fragment -> build/<key>.mp4 of exactly frag['seconds']."""
    w, h = sb["size"]
    fps, secs = sb["fps"], frag["seconds"]
    out = os.path.join(build, f"{key}.mp4")
    path = os.path.join(HERE, frag["file"])
    flags = ":flags=neighbor" if frag.get("pixel") else ""   # pixel art: no blur
    fit = (f"scale={w}:{h}:force_original_aspect_ratio=decrease{flags},"
           f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    vf = []
    if is_still(path):
        inp = ["-loop", "1", "-framerate", str(fps), "-t", f"{secs:.3f}", "-i", path]
        if frag.get("crop"):
            vf.append(f"crop={frag['crop']}")
        vf += [fit, "format=yuv420p"]
        d = sb["fade"]
        vf += [f"fade=in:st=0:d={d}", f"fade=out:st={secs - d:.3f}:d={d}"]
    else:
        speed = frag.get("speed", 1)
        need = secs * speed
        loop = frag.get("from", 0) + need > src_dur
        inp = (["-stream_loop", "-1"] if loop else []) + \
              ["-ss", f"{frag.get('from', 0):.3f}", "-t", f"{need:.3f}", "-i", path]
        if frag.get("crop"):
            vf.append(f"crop={frag['crop']}")
        vf += [f"setpts=(PTS-STARTPTS)/{speed}", f"fps={fps}", fit, "format=yuv420p",
               f"tpad=stop_mode=clone:stop_duration={secs:.3f}", f"trim=duration={secs:.3f}"]
    if frag.get("caption"):
        vf.append(caption_filter(build, key, frag["caption"], frag.get("caption_size", 40),
                                 sb.get("caption_at", "bottom")))
    run(["ffmpeg", "-v", "error", "-y", *inp, "-vf", ",".join(vf), "-an", "-r", str(fps),
         "-frames:v", str(round(secs * fps)),
         "-c:v", "libx264", "-preset", "medium", "-crf", str(sb["crf"]), "-pix_fmt", "yuv420p",
         "-video_track_timescale", "12800", out])
    return out


def render_image_card(sb, build, key, image, secs):
    """Title from a picture (the thumbnail): held, then a fade to black."""
    w, h = sb["size"]
    d = min(sb["fade"], secs / 4)
    out = os.path.join(build, f"{key}.mp4")
    run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", str(sb["fps"]),
         "-t", str(secs), "-i", os.path.join(HERE, image),
         "-vf", f"scale={w}:{h},setsar=1,fade=out:st={secs - d:.3f}:d={d}",
         "-c:v", "libx264", "-preset", "medium", "-crf", str(sb["crf"]), "-pix_fmt", "yuv420p",
         "-video_track_timescale", "12800", out])
    return out


def render_card(sb, build, key, lines, secs):
    """Title / end card: centred lines on black, fading in and out."""
    w, h = sb["size"]
    size, gap = 64, 96
    top = h / 2 - gap * (len(lines) - 1) / 2
    vf = []
    for i, line in enumerate(lines):
        if not line:
            continue
        tf = textfile(build, f"{key}_{i}.txt", line)
        fs = size if i == 0 else 48
        vf.append(f"drawtext=fontfile={FONT_BOLD}:textfile={tf}:fontsize={fs}:fontcolor=white:"
                  f"x=(w-tw)/2:y={top + gap * i:.0f}-th/2")
    d = min(sb["fade"], secs / 4)
    vf += [f"fade=in:st=0:d={d}", f"fade=out:st={secs - d:.3f}:d={d}"]
    out = os.path.join(build, f"{key}.mp4")
    run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"color=c=black:s={w}x{h}:r={sb['fps']}:d={secs}", "-vf", ",".join(vf),
         "-c:v", "libx264", "-preset", "medium", "-crf", str(sb["crf"]), "-pix_fmt", "yuv420p",
         "-video_track_timescale", "12800", out])
    return out


def silences(mp3):
    """[(start, end)] of the pauses in the voice."""
    r = subprocess.run(["ffmpeg", "-i", mp3, "-af", "silencedetect=noise=-35dB:d=0.18",
                        "-f", "null", "-"], capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    return list(zip(starts, ends))


def chunks(text):
    """The voice text as subtitle lines: sentences; a long one is cut into
    even pieces, each cut at the nearest , — : (else a space)."""
    out = []
    for sent in re.split(r"(?<=[.!?…])\s+", text.strip()):
        k = -(-len(sent) // SUB_CHARS)
        while k > 1:
            goal = len(sent) / k
            puncts = [m.end() for m in re.finditer(r"[,:;]\s|\s(?=—\s)", sent)]
            near = [c for c in puncts if abs(c - goal) < 22]
            cut = min(near, key=lambda c: abs(c - goal)) if near else \
                min((m.start() + 1 for m in re.finditer(r"\s", sent)), key=lambda c: abs(c - goal))
            out.append(sent[:cut].strip())
            sent = sent[cut:].strip()
            k = -(-len(sent) // SUB_CHARS)
        out.append(sent)
    return out


def time_subs(text, mp3):
    """[(start, end, line)] within the mp3: boundaries placed by length, then
    moved to the nearest pause of the voice (within 1.2 s)."""
    lines = chunks(text)
    total = duration(mp3)
    pauses = silences(mp3)
    t0 = pauses[0][1] if pauses and pauses[0][0] < 0.05 else 0.0
    t1 = pauses[-1][0] if pauses and pauses[-1][1] > total - 0.05 else total
    inner = [(a, b) for a, b in pauses if a > t0 + 0.1 and b < t1 - 0.1]
    weights = [len(s) + 8 for s in lines]          # a pause costs about as much as a word
    cum, cuts = 0, []                               # (end of a line, start of the next)
    for wt in weights[:-1]:
        cum += wt
        est = t0 + (t1 - t0) * cum / sum(weights)
        last = cuts[-1][1] if cuts else t0
        near = [p for p in inner if abs((p[0] + p[1]) / 2 - est) < 1.2 and p[0] > last + 0.6]
        if near:
            p = min(near, key=lambda p: abs((p[0] + p[1]) / 2 - est))
            inner.remove(p)
            cuts.append(p)
        else:
            e = max(est, last + 0.6)
            cuts.append((e, e))
    cuts.append((t1, t1))
    out, start = [], t0
    for line, (end, nxt) in zip(lines, cuts):
        out.append((start, min(end + 0.15, nxt), line))
        start = nxt
    return out


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--storyboard", default=os.path.join(HERE, "storyboard.yaml"))
    p.add_argument("--lang", choices=("ru", "en"), default="ru")
    p.add_argument("--script", help="default: the lang's docs/video_script*.md")
    p.add_argument("--out", default=os.path.join(HERE, "draft_ru.mp4"))
    p.add_argument("--build", default=os.path.join(HERE, "build"), help="clip cache")
    p.add_argument("--srt", default=os.path.join(HERE, "subs_ru.srt"))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--burn", action="store_true", help="subtitles burnt into the picture")
    args = p.parse_args()
    args.script = args.script or os.path.join(ROOT, "docs", LANGS[args.lang][0])
    with open(args.storyboard, encoding="utf-8") as f:
        sb = yaml.safe_load(f)
    build = os.path.abspath(args.build)
    os.makedirs(build, exist_ok=True)
    texts = {n: text for n, _, _, text in script_scenes(args.script, args.lang)}

    # plan: scene lengths, stretched where the picture is shorter than the voice
    notes, jobs, report = [], [], []
    t = sb["title"]["seconds"]
    for sc in sb["scenes"]:
        n = sc["n"]
        voice = os.path.join(HERE, sc["voice"]) if sc.get("voice") else None   # None: a pause
        vdur = duration(voice) if voice else 0.0
        frags = sc["fragments"]
        for fr in frags:
            if not is_still(fr["file"]):
                fr["src_dur"] = duration(os.path.join(HERE, fr["file"]))
        length = sum(fr["seconds"] for fr in frags)
        tail = sb.get("min_tail", MIN_TAIL)
        need = sb["voice_at"] + vdur + tail if voice else 0.0
        if length < need - 1e-6:
            last = frags[-1]
            notes.append(f"scene {n}: picture {length:.1f} s < voice + {tail:.1f} s "
                         f"({need:.1f} s): {last['file']} stretched by {need - length:.1f} s")
            last["seconds"] = round(last["seconds"] + need - length, 3)
            length = need
        for i, fr in enumerate(frags):
            if not is_still(fr["file"]):
                used = fr.get("from", 0) + fr["seconds"] * fr.get("speed", 1)
                if used > fr["src_dur"] + 1e-6:
                    notes.append(f"scene {n}: {fr['file']} runs out at {fr['src_dur']:.1f} s "
                                 f"(needs {used:.1f} s): looped")
            jobs.append((f"s{str(n).replace('/', '-')}_{i:02d}", fr))
        report.append({"scene": n, "title": sc["title"], "start": round(t, 2),
                       "seconds": round(length, 2), "voice_seconds": round(vdur, 2),
                       "voice": voice, "fragments": len(frags)})
        t += length
    total = t + sb["end"]["seconds"]

    # picture
    with ThreadPoolExecutor(args.workers) as ex:
        ti = sb["title"]
        futs = [ex.submit(render_image_card, sb, build, "title", ti["image"], ti["seconds"])
                if ti.get("image") else
                ex.submit(render_card, sb, build, "title", ti["lines"], ti["seconds"])]
        futs += [ex.submit(render_fragment, sb, build, key, fr, fr.get("src_dur", 0))
                 for key, fr in jobs]
        futs.append(ex.submit(render_card, sb, build, "end", sb["end"]["lines"],
                              sb["end"]["seconds"]))
        clips = [f.result() for f in futs]
    concat = os.path.join(build, "concat.txt")
    with open(concat, "w") as f:
        f.writelines(f"file '{c}'\n" for c in clips)
    video = os.path.join(build, "video.mp4")
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", concat,
         "-c", "copy", video])

    # subtitles
    entries = []
    voiced = [r for r in report if r["voice"]]
    for r in voiced:
        text = texts[r["scene"]]
        for a, b, line in time_subs(text, r["voice"]):
            off = r["start"] + sb["voice_at"]
            entries.append((off + a, off + b, line))
    with open(args.srt, "w", encoding="utf-8") as f:
        for i, (a, b, line) in enumerate(entries, 1):
            f.write(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{line}\n\n")

    # voice track + mux
    inputs, chains = [], []
    for i, r in enumerate(voiced):
        inputs += ["-i", r["voice"]]
        ms = int(round((r["start"] + sb["voice_at"]) * 1000))
        chains.append(f"[{i + 1}:a]aresample=48000,adelay={ms}:all=1[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(voiced)))
    fc = ";".join(chains) + f";{mix}amix=inputs={len(voiced)}:normalize=0,apad[aout]"
    iso = {"ru": "rus", "en": "eng"}[args.lang]
    if args.burn:
        srt = args.srt.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        style = ("FontName=DejaVu Sans,FontSize=14,BorderStyle=3,Outline=4,Shadow=0,"
                 "OutlineColour=&H60000000,BackColour=&H60000000,MarginV=24")
        fc += f";[0:v]subtitles='{srt}':force_style='{style}'[vout]"
        picture = ["-map", "[vout]", "-c:v", "libx264", "-preset", "medium", "-crf", str(sb["crf"]),
                   "-pix_fmt", "yuv420p"]
        subs = []
    else:
        picture = ["-map", "0:v", "-c:v", "copy"]
        subs = ["-i", args.srt]
    run(["ffmpeg", "-v", "error", "-y", "-i", video, *inputs, *subs,
         "-filter_complex", fc, *picture, "-map", "[aout]",
         *([] if args.burn else ["-map", f"{len(voiced) + 1}:s", "-c:s", "mov_text",
                                 f"-metadata:s:s:0", f"language={iso}"]),
         "-c:a", "aac", "-b:a", "192k", "-ac", "2", f"-metadata:s:a:0", f"language={iso}",
         "-t", f"{total:.3f}", "-movflags", "+faststart", args.out])

    out_dur = duration(args.out)
    for r in report:
        r.pop("voice")
    summary = {"file": os.path.relpath(args.out, ROOT), "seconds": round(out_dur, 2),
               "planned_seconds": round(total, 2), "title_seconds": sb["title"]["seconds"],
               "end_seconds": sb["end"]["seconds"], "subtitles": len(entries),
               "scenes": report, "notes": notes}
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    for r in report:
        print(f"scene {r['scene']!s:4} {r['start']:7.1f} s  {r['seconds']:5.1f} s "
              f"(voice {r['voice_seconds']:5.1f} s)  {r['title']}")
    for n in notes:
        print("note:", n)
    print(f"{args.out}: {out_dur:.1f} s ({total:.1f} planned), {len(entries)} subtitles")


if __name__ == "__main__":
    sys.exit(main())
