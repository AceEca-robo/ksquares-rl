#!/usr/bin/env bash
# Install ksquares-rl from this checkout (Ubuntu 22.04 / 24.04):
#   1. system packages through apt (the KSquares build dependencies, Xvfb,
#      xdotool, ffmpeg, Python venv, fonts), only the missing ones, with sudo
#      (or directly as root);
#   2. KSquares with the RL bridge, built into build/;
#   3. .venv with requirements.txt and torch: the CUDA build when nvidia-smi
#      is there (cu130 or cu126, by the driver's CUDA version), else the
#      CPU-only build, with a warning;
#   4. the app in the applications menu (~/.local/share/applications,
#      ~/.local/bin/ksquares-rl);
#   5. a self-check (scripts/selfcheck.py): KSquares, Xvfb, a game over the
#      bridge, torch and the GPU, fonts; a short report.
# Safe to run again: done steps are quick.
#
#   ./install.sh            # everything
#   ./install.sh --cpu      # torch's CPU-only build even with a GPU (a much smaller download)
#   ./install.sh --no-apt   # do not install system packages, only say which are missing
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
BIN="$HOME/.local/bin"
TORCH="torch==2.14.1"   # as pyproject.toml's "az" extra
CPU=0
APT=1
for a in "$@"; do
    case "$a" in
        --cpu) CPU=1 ;;
        --no-apt) APT=0 ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $a (see --help)" >&2; exit 2 ;;
    esac
done
step() { printf '\n== %s\n' "$*"; }
warn() { printf '\nWARNING: %s\n' "$*" >&2; }
t_start=$(date +%s)

step "system packages"
# the build dependencies of KSquares (as `apt-get build-dep ksquares`, without
# needing deb-src), then what the app and the recording use
PKGS=(build-essential cmake extra-cmake-modules gettext git
      qtbase5-dev libkf5config-dev libkf5configwidgets-dev libkf5coreaddons-dev
      libkf5crash-dev libkf5dbusaddons-dev libkf5doctools-dev libkf5i18n-dev
      libkf5kdegames-dev libkf5widgetsaddons-dev libkf5xmlgui-dev libkf5completion-dev
      xvfb xdotool x11-utils ffmpeg
      python3 python3-venv
      fonts-dejavu-core fontconfig ca-certificates dbus)
missing=()
for p in "${PKGS[@]}"; do
    # python3-venv: any package that makes `python3 -m venv` work will do
    if [ "$p" = python3-venv ] && python3 -c 'import ensurepip, venv' 2>/dev/null; then continue; fi
    dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed" || missing+=("$p")
done
if [ ${#missing[@]} -eq 0 ]; then
    echo "all there"
elif [ "$APT" = 0 ]; then
    echo "Missing: ${missing[*]}"
    echo "Install them and run ./install.sh again:"
    echo "  sudo apt-get install ${missing[*]}"
    exit 1
else
    if [ "$(id -u)" = 0 ]; then SUDO=(); elif command -v sudo >/dev/null; then SUDO=(sudo); else
        echo "Missing: ${missing[*]}; not root and no sudo: install them as root and run again" >&2; exit 1
    fi
    echo "installing: ${missing[*]}"
    export DEBIAN_FRONTEND=noninteractive
    "${SUDO[@]}" apt-get update -q
    "${SUDO[@]}" apt-get install -y -q --no-install-recommends "${missing[@]}"
fi
if ! python3 -c 'import sys; assert sys.version_info >= (3, 10)' 2>/dev/null; then
    echo "Python 3.10 or newer is needed, python3 is $(python3 -V 2>&1)" >&2; exit 1
fi

step "building KSquares with the RL bridge (build/)"
log="$(mktemp)"
if ! (cmake -S "$ROOT/game" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo && \
      cmake --build "$ROOT/build" -j"$(nproc)") >"$log" 2>&1; then
    tail -n 30 "$log"; echo "the build failed, full log: $log" >&2; exit 1
fi
rm -f "$log"
echo "build/src/ksquares"

step "Python environment (.venv)"
[ -x "$ROOT/.venv/bin/python" ] || python3 -m venv "$ROOT/.venv"
PY="$ROOT/.venv/bin/python"
"$PY" -m pip install --quiet --upgrade pip
# torch: the build that fits the machine
index=cpu
if [ "$CPU" = 1 ]; then
    echo "torch: the CPU-only build (--cpu)"
elif ! command -v nvidia-smi >/dev/null; then
    warn "no nvidia-smi: torch's CPU-only build. The app plays fine without a GPU;
training your own Dot needs an NVIDIA GPU with its driver (on the CPU about 10x slower)."
else
    cuda="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1 \2/p' | head -1)"
    read -r major minor <<<"${cuda:-0 0}"
    if [ "$major" -ge 13 ]; then index=cu130
    elif [ "$major" -eq 12 ] && [ "$minor" -ge 6 ]; then index=cu126
    else
        warn "the NVIDIA driver supports CUDA ${major}.${minor}, torch needs 12.6 or newer:
torch's CPU-only build. Update the driver and run ./install.sh again for the GPU."
    fi
    echo "torch: the $index build (the driver supports CUDA ${major}.${minor})"
fi
have="$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
want="${TORCH#torch==}+$index"
if [ "$have" != "$want" ]; then
    "$PY" -m pip install --quiet "$TORCH" --index-url "https://download.pytorch.org/whl/$index"
fi
"$PY" -m pip install --quiet -r "$ROOT/requirements.txt"
"$PY" -m pip install --quiet --no-deps -e "$ROOT"
echo "$("$PY" -c 'import torch; print("torch", torch.__version__)')"

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
if command -v desktop-file-validate >/dev/null; then desktop-file-validate "$DATA/applications/ksquares-rl.desktop"; fi
command -v update-desktop-database >/dev/null && update-desktop-database "$DATA/applications" 2>/dev/null || true
command -v gtk-update-icon-cache >/dev/null && \
    gtk-update-icon-cache -q -t "$DATA/icons/hicolor" 2>/dev/null || true
echo "$DATA/applications/ksquares-rl.desktop -> $BIN/ksquares-rl"

if (cd "$ROOT/scripts" && "$PY" update_best.py) | grep -q "'gen'"; then
    ck="the network is $(readlink "$ROOT/runs/best/best.pt")"
else
    ck="the app offers to download the networks (az5b gen 54 for 5x5, dot2c gen 16 and dot2s gen 39 for 10x7, 6.4 MB, from the v0.1 and v0.2 releases) on its first start"
fi

step "self-check"
ok=0
"$PY" "$ROOT/scripts/selfcheck.py" || ok=$?
t=$(( $(date +%s) - t_start ))
printf '\nInstall took %d min %d s; %s used by build/ and .venv.\n' $((t / 60)) $((t % 60)) \
    "$(du -sch "$ROOT/build" "$ROOT/.venv" 2>/dev/null | tail -1 | cut -f1)"
if [ "$ok" != 0 ]; then
    echo "Some checks failed (see above)." >&2; exit 1
fi
printf 'Done. Start "KSquares RL" from the applications menu or run ksquares-rl;\n%s.\n' "$ck"
case ":$PATH:" in *":$BIN:"*) ;; *) echo "($BIN is not on PATH: open a new terminal or run $BIN/ksquares-rl)";; esac
