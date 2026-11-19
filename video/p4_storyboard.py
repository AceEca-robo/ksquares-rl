#!/usr/bin/env python3
"""Writes video/storyboard_part4_ru.yaml: every cut on the start of a sentence of the scene's voice
(video_anim.sentence_starts on video/p4/voice_ru_<N>.mp3), as in part 3.

    .venv/bin/python video/p4_storyboard.py
    .venv/bin/python video/video_build.py --storyboard video/storyboard_part4_ru.yaml \
        --script docs/video_script_part4_ru.md --out video/p4/ksquares_rl_part4_ru.mp4 \
        --srt video/p4/ksquares_rl_part4_ru.srt --build video/build_p4_ru --workers 1
"""
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p4_clips import sentences, starts  # noqa: E402

TAIL = 1.2
APP = "Dot-G, раунд 114 (красные) против Dot, раунд 84 (синие) · по 200 вариантов на ход"

# scene -> [(first words of the sentence the fragment starts on, file, extra keys)]
CUTS = {
    1: [("В прошлый раз", "p4/window_graph.mp4", {}), ("Мы построили", "p4/card_title.png", {})],
    2: [("Для Dot-G", "p4/card_sees.png", {})],
    3: [("Начали мы", "p4/card_three.png", {}), ("Первая", "p4/click.mp4", {})],
    4: [("Но это на маленькой", "p4/card_school_table1.png", {}), ("Тогда мы устроили", "p4/card_school.png", {}),
        ("После такой школы", "p4/card_school_table2.png", {}), ("Правда, выяснилось", "p4/card_school_table.png", {})],
    5: [("Уметь сказать", "p4/card_cloud0.png", {}), ("Но примерно каждая", "p4/card_cloud.png", {}),
        ("Взяли четыре сети", "p4/card_four.png", {})],
    6: [("Это было похоже", "p4/card_cloud.png", {}), ("В третьей части", "p3/card_blowouts.png", {}),
        ("Мы взяли позиции", "p4/card_map_g.png", {}), ("Результат", "p4/card_mid_bars.png", {}),
        ("И ещё одно", "p4/card_promise.png", {})],
    7: [("До этого момента", "p4/card_rounds0.png", {}), ("Встроенного противника", "p4/card_rounds.png", {}),
        ("Против настоящего", "p4/card_bridge.png", {})],
    8: [("А дальше", "p4/card_peak.png", {}), ("Поздняя выиграла", "p4/card_matches1.png", {}),
        ("Тогда мы проверили", "p4/card_matches.png", {}), ("Ошиблись не сети", "p4/card_round_robin.png", {})],
    9: [("И вот главное", "p4/app_captioned.mp4", {"speed": 2.5}),
        ("Если обе думают", "p4/card_final_match1.png", {}), ("Если обе ходят", "p4/card_final_match2.png", {}),
        ("Но у этого", "p4/card_cost.png", {}), ("И в этом матче", "p4/card_final_match.png", {})],
    10: [("Что мы узнали", "p4/card_summary1.png", {}), ("Второе", "p4/card_summary2.png", {}),
         ("Третье", "p4/card_summary3.png", {}), ("И четвёртое", "p4/card_summary.png", {}),
         ("Обе сети", "p4/app_captioned.mp4", {"from": 20}),
         ("Весь код", "p4/card_end.png", {})],
}
TITLES = {1: "Другие глаза", 2: "Как видит Dot-G", 3: "Первый урок: считать", 4: "Школа", 5: "От счёта к ходу",
          6: "Середина партии", 7: "Тогда пусть играет", 8: "Сломанная линейка", 9: "Матч", 10: "Итог"}


def main():
    scenes = []
    for n, cuts in CUTS.items():
        s, end = starts(n)
        sents = sentences(n)
        t = [0.0 if i == 0 else s[next(k for k, x in enumerate(sents) if x.startswith(w))]
             for i, (w, _, _) in enumerate(cuts)]
        assert t == sorted(t), (n, t)
        frags = []
        for i, (w, f, extra) in enumerate(cuts):
            stop = t[i + 1] if i + 1 < len(cuts) else end + TAIL
            frags.append({"file": f, "seconds": round(stop - t[i], 2), **extra})
        scenes.append({"n": n, "title": TITLES[n], "voice": f"p4/voice_ru_{n}.mp3", "fragments": frags})
    sb = {"size": [1920, 1080], "fps": 25, "crf": 18, "voice_at": 0.5, "min_tail": 0.7, "fade": 0.4,
          "caption_at": "top",
          "title": {"seconds": 3, "image": "p4/card_title.png"},
          "end": {"seconds": 6, "lines": ["github.com/AceEca-robo/ksquares-rl", "",
                                          "KSquares 21.12.3 © Matt Williams et al., GPL-2+"]},
          "scenes": scenes}
    head = ("# Russian cut of the part 4 video (docs/video_script_part4_ru.md). Written by video/p4_storyboard.py:\n"
            "# cuts on the sentence starts of video/p4/voice_ru_<N>.mp3. Materials: video/p4_cards.py, p4_clips.py,\n"
            "# p4/app_g5_vs_dot.mp4 (scripts/launcher.py --mode self, g5 gen 114 vs az5b gen 84, 200 sims, CPU).\n\n")
    with open(os.path.join(HERE, "storyboard_part4_ru.yaml"), "w", encoding="utf-8") as f:
        f.write(head + yaml.safe_dump(sb, allow_unicode=True, sort_keys=False, width=120))
    print("video/storyboard_part4_ru.yaml")


if __name__ == "__main__":
    main()
