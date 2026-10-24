"""The desktop entry install.sh writes, with its translations from i18n.py:

    python -m launcher.desktop <exec> > ~/.local/share/applications/ksquares-rl.desktop
"""

import sys

from .i18n import DEFAULT, LANGS, t


def entry(exec_path):
    if " " in exec_path:
        exec_path = f'"{exec_path}"'
    lines = ["[Desktop Entry]", "Type=Application", "Version=1.0"]
    for key, field in (("desktop_name", "Name"), ("desktop_comment", "Comment")):
        lines.append(f"{field}={t(DEFAULT, key)}")
        lines += [f"{field}[{lang}]={t(lang, key)}" for lang in LANGS if lang != DEFAULT]
    lines += [f"Exec={exec_path}", "Icon=ksquares-rl", "Terminal=false",
              "Categories=Game;BoardGame;", "Keywords=dots;boxes;ksquares;alphazero;neural;network;RL;",
              "StartupWMClass=ksquares-rl", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.write(entry(sys.argv[1]))
