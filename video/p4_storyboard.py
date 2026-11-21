#!/usr/bin/env python3
"""Writes video/storyboard_part4_ru.yaml: every cut on the start of a sentence of the scene's voice
(video_anim.sentence_starts on video/p4/voice_ru_<N>.mp3), as in part 3.

    .venv/bin/python video/p4_storyboard.py
    .venv/bin/python video/video_build.py --storyboard video/storyboard_part4_ru.yaml \
        --script docs/video_script_part4_ru.md --out video/p4/ksquares_rl_part4_ru.mp4 \
        --srt video/p4/ksquares_rl_part4_ru.srt --build video/build_p4_ru --workers 1

English cut (docs/video_script_part4_en.md, video/p4/voice_en_<N>.mp3, materials *_en):
    P4_LANG=en .venv/bin/python video/p4_storyboard.py     # video/storyboard_part4_en.yaml
    .venv/bin/python video/video_build.py --lang en --storyboard video/storyboard_part4_en.yaml \
        --script docs/video_script_part4_en.md --out video/p4/ksquares_rl_part4_en.mp4 \
        --srt video/p4/ksquares_rl_part4_en.srt --build video/build_p4_en --workers 1
"""
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p4_clips import LANG, sentences, starts, suffix  # noqa: E402

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
        ("Тогда мы провели", "p4/card_promise.png", {}), ("Когда сеть училась", "p4/card_s12b.png", {}),
        ("Когда к ответам", "p4/card_s12b2.png", {})],
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
# the English cut: the same fragments, cut on the starts of the same sentences of the English voice
CUTS_EN = {
    1: [("Last time", "p4/window_graph.mp4", {}), ("We built one", "p4/card_title.png", {})],
    2: [("For Dot-G", "p4/card_sees.png", {})],
    3: [("We started", "p4/card_three.png", {}), ("The first", "p4/click.mp4", {})],
    4: [("But that's on the small", "p4/card_school_table1.png", {}), ("So we set up", "p4/card_school.png", {}),
        ("After this school", "p4/card_school_table2.png", {}), ("But it turned out", "p4/card_school_table.png", {})],
    5: [("Knowing who", "p4/card_cloud0.png", {}), ("But roughly every fourth", "p4/card_cloud.png", {}),
        ("We took four networks", "p4/card_four.png", {})],
    6: [("This looked like", "p4/card_cloud.png", {}), ("In the third part", "p3/card_blowouts.png", {}),
        ("We took the middle-game", "p4/card_map_g.png", {}), ("The result", "p4/card_mid_bars.png", {}),
        ("Then we ran", "p4/card_promise.png", {}), ("When the whole network", "p4/card_s12b.png", {}),
        ("When Dot-2's own games", "p4/card_s12b2.png", {})],
    7: [("Up to this point", "p4/card_rounds0.png", {}), ("She started beating", "p4/card_rounds.png", {}),
        ("Against the real", "p4/card_bridge.png", {})],
    8: [("And then something", "p4/card_peak.png", {}), ("The late one won", "p4/card_matches1.png", {}),
        ("Then we checked", "p4/card_matches.png", {}), ("It wasn't the networks", "p4/card_round_robin.png", {})],
    9: [("And now the main", "p4/app_captioned.mp4", {"speed": 2.5}),
        ("If both think", "p4/card_final_match1.png", {}), ("If both move", "p4/card_final_match2.png", {}),
        ("But this comes", "p4/card_cost.png", {}), ("And in this match", "p4/card_final_match.png", {})],
    10: [("What we learned", "p4/card_summary1.png", {}), ("Second", "p4/card_summary2.png", {}),
         ("Third", "p4/card_summary3.png", {}), ("And fourth", "p4/card_summary.png", {}),
         ("Both networks", "p4/app_captioned.mp4", {"from": 20}),
         ("All the code", "p4/card_end.png", {})],
}
TITLES_EN = {1: "Other eyes", 2: "How Dot-G sees", 3: "First lesson: counting", 4: "School",
             5: "From counting to the move", 6: "The middle of the game", 7: "Then let her play",
             8: "The broken ruler", 9: "The match", 10: "Summary"}
TITLES = {1: "Другие глаза", 2: "Как видит Dot-G", 3: "Первый урок: считать", 4: "Школа", 5: "От счёта к ходу",
          6: "Середина партии", 7: "Тогда пусть играет", 8: "Сломанная линейка", 9: "Матч", 10: "Итог"}


def localized(f):
    """p4/card_x.png -> p4/card_x_en.png in the English cut."""
    root, ext = os.path.splitext(f)
    return root + suffix() + ext


def main():
    cuts_all, titles = (CUTS, TITLES) if LANG == "ru" else (CUTS_EN, TITLES_EN)
    scenes = []
    for n, cuts in cuts_all.items():
        s, end = starts(n)
        sents = sentences(n)
        t = [0.0 if i == 0 else s[next(k for k, x in enumerate(sents) if x.startswith(w))]
             for i, (w, _, _) in enumerate(cuts)]
        assert t == sorted(t), (n, t)
        frags = []
        for i, (w, f, extra) in enumerate(cuts):
            stop = t[i + 1] if i + 1 < len(cuts) else end + TAIL
            frags.append({"file": localized(f), "seconds": round(stop - t[i], 2), **extra})
        scenes.append({"n": n, "title": titles[n], "voice": f"p4/voice_{LANG}_{n}.mp3", "fragments": frags})
    sb = {"size": [1920, 1080], "fps": 25, "crf": 18, "voice_at": 0.5, "min_tail": 0.7, "fade": 0.4,
          "caption_at": "top",
          "title": {"seconds": 3, "image": localized("p4/card_title.png")},
          "end": {"seconds": 6, "lines": ["github.com/AceEca-robo/ksquares-rl", "",
                                          "KSquares 21.12.3 © Matt Williams et al., GPL-2+"]},
          "scenes": scenes}
    lang_name = {"ru": "Russian", "en": "English"}[LANG]
    head = (f"# {lang_name} cut of the part 4 video (docs/video_script_part4_{LANG}.md). Written by video/p4_storyboard.py:\n"
            f"# cuts on the sentence starts of video/p4/voice_{LANG}_<N>.mp3. Materials: video/p4_cards.py, p4_clips.py,\n"
            f"# p4/app_g5_vs_dot{suffix()}.mp4 (scripts/launcher.py --mode self, g5 gen 114 vs az5b gen 84, 200 sims, CPU).\n\n")
    name = f"storyboard_part4_{LANG}.yaml"
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
        f.write(head + yaml.safe_dump(sb, allow_unicode=True, sort_keys=False, width=120))
    print(f"video/{name}")


if __name__ == "__main__":
    main()
