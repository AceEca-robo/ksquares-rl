"""Language of the part 4 materials: P4_LANG=en selects the English cut (and sets P3_LANG for the part 3 code the
part 4 cards reuse: the map, the window / graph clip). L(ru, en) picks the screen text; file names get suffix()."""
import os

LANG = os.environ.get("P4_LANG", "ru")
os.environ.setdefault("P3_LANG", LANG)
assert os.environ["P3_LANG"] == LANG, "P3_LANG and P4_LANG differ"


def L(ru, en):
    return ru if LANG == "ru" else en


def suffix():
    return "" if LANG == "ru" else f"_{LANG}"
