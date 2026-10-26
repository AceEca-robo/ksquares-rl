#!/usr/bin/env bash
# Install ksquares-rl as a desktop app for the current user, from this
# checkout: checks the system packages, builds KSquares with the RL bridge
# (build/), creates .venv with the package, and puts the app
# (scripts/launcher.py: one window with the menu and the three modes) into
# the applications menu (~/.local/share/applications/ksquares-rl.desktop,
# icon, ~/.local/bin/ksquares-rl). Nothing needs root; missing packages are
# listed with the apt command that installs them. Safe to run again.
#
#   ./install.sh          # torch as pip resolves it (with CUDA on Linux)
#   ./install.sh --cpu    # torch's CPU-only build: a much smaller download
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
BIN="$HOME/.local/bin"
CPU=0
for a in "$@"; do
    case "$a" in
        --cpu) CPU=1 ;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $a (see --help)" >&2; exit 2 ;;
    esac
done
step() { printf '\n== %s\n' "$*"; }

step "checking system packages"
missing=()
need_cmd() { command -v "$1" >/dev/null 2>&1 || missing+=("$2"); }
need_file() { [ -e "$1" ] || missing+=("$2"); }
need_cmd g++ build-essential
need_cmd cmake cmake
need_file /usr/share/ECM/cmake/ECMConfig.cmake extra-cmake-modules
need_cmd msgfmt gettext
need_file /usr/include/x86_64-linux-gnu/qt5/QtCore/QtCore qtbase5-dev
for kf in Config ConfigWidgets CoreAddons Crash DBusAddons DocTools I18n WidgetsAddons XmlGui Completion; do
    pkg="libkf5$(echo "$kf" | tr '[:upper:]' '[:lower:]')-dev"
    need_file "/usr/lib/x86_64-linux-gnu/cmake/KF5$kf" "$pkg"
done
need_file /usr/lib/x86_64-linux-gnu/cmake/KF5KDEGames libkf5kdegames-dev
need_cmd Xvfb xvfb          # KSquares runs on a private Xvfb in the app
need_cmd xdotool xdotool    # finds its window there
need_cmd xwininfo x11-utils # and its position
need_cmd ffmpeg ffmpeg      # --record
if ! command -v python3 >/dev/null 2>&1; then
    missing+=(python3 python3-venv)
elif ! python3 -c 'import sys; assert sys.version_info >= (3, 10)' 2>/dev/null; then
    echo "Python 3.10 or newer is needed, python3 is $(python3 -V 2>&1)" >&2; exit 1
elif ! python3 -c 'import ensurepip, venv' 2>/dev/null; then
    missing+=(python3-venv)
fi
if [ ${#missing[@]} -gt 0 ]; then
    mapfile -t missing < <(printf '%s\n' "${missing[@]}" | sort -u)
    echo "Missing: ${missing[*]}"
    echo "Install them and run ./install.sh again:"
    echo "  sudo apt install ${missing[*]}"
    exit 1
fi
echo "all there"

step "building KSquares with the RL bridge (build/)"
log="$(mktemp)"
if ! (cmake -S "$ROOT/game" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo && \
      cmake --build "$ROOT/build" -j) >"$log" 2>&1; then
    tail -n 30 "$log"; echo "the build failed, full log: $log" >&2; exit 1
fi
rm -f "$log"
echo "build/src/ksquares"

step "Python environment (.venv)"
[ -x "$ROOT/.venv/bin/python" ] || python3 -m venv "$ROOT/.venv"
PY="$ROOT/.venv/bin/python"
"$PY" -m pip install --quiet --upgrade pip
if [ "$CPU" = 1 ]; then
    torch="$(sed -n 's/^ *az = \[\"\(torch==[^"]*\)".*/\1/p' "$ROOT/pyproject.toml")"
    "$PY" -m pip install --quiet "$torch" --index-url https://download.pytorch.org/whl/cpu
fi
"$PY" -m pip install --quiet -e "$ROOT[az,app]"
echo "$("$PY" -c 'import torch; print("torch", torch.__version__, "CUDA" if torch.cuda.is_available() else "CPU only")')"

step "desktop entry"
mkdir -p "$DATA/applications" "$DATA/icons/hicolor/scalable/apps" "$BIN"
install -m 644 "$ROOT/launcher/ksquares-rl.svg" "$DATA/icons/hicolor/scalable/apps/ksquares-rl.svg"
cat >"$BIN/ksquares-rl" <<SH
#!/bin/sh
# ksquares-rl launcher, written by $ROOT/install.sh
exec "$PY" "$ROOT/scripts/launcher.py" "\$@"
SH
chmod 755 "$BIN/ksquares-rl"
(cd "$ROOT" && "$PY" -m launcher.desktop "$BIN/ksquares-rl") >"$DATA/applications/ksquares-rl.desktop"
command -v desktop-file-validate >/dev/null && desktop-file-validate "$DATA/applications/ksquares-rl.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database "$DATA/applications" 2>/dev/null || true
command -v gtk-update-icon-cache >/dev/null && \
    gtk-update-icon-cache -q -t "$DATA/icons/hicolor" 2>/dev/null || true
echo "$DATA/applications/ksquares-rl.desktop -> $BIN/ksquares-rl"

if (cd "$ROOT/scripts" && "$PY" update_best.py) | grep -q "'gen'"; then
    ck="the network is $(readlink "$ROOT/runs/best/best.pt")"
else
    ck="the app offers to download the network (az5b gen 54, 2.2 MB, from the v0.1 release) on its first start"
fi
printf '\nDone. Start "KSquares RL" from the applications menu or run ksquares-rl;\n%s.\n' "$ck"
