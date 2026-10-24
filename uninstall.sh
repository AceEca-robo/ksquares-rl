#!/usr/bin/env bash
# Remove what install.sh put outside this checkout: the desktop entry, the
# icon and ~/.local/bin/ksquares-rl. The settings and the log stay unless
# --purge is given. .venv, build/ and runs/ stay in the checkout.
#
#   ./uninstall.sh [--purge]
set -euo pipefail

DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
PURGE=0
for a in "$@"; do
    case "$a" in
        --purge) PURGE=1 ;;
        -h|--help) sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $a (see --help)" >&2; exit 2 ;;
    esac
done

rm -fv "$DATA/applications/ksquares-rl.desktop" \
       "$DATA/icons/hicolor/scalable/apps/ksquares-rl.svg" \
       "$HOME/.local/bin/ksquares-rl"
command -v update-desktop-database >/dev/null && update-desktop-database "$DATA/applications" 2>/dev/null || true
command -v gtk-update-icon-cache >/dev/null && \
    gtk-update-icon-cache -q -t "$DATA/icons/hicolor" 2>/dev/null || true
if [ "$PURGE" = 1 ]; then
    rm -rfv "${XDG_CONFIG_HOME:-$HOME/.config}/ksquares-rl" "${XDG_CACHE_HOME:-$HOME/.cache}/ksquares-rl"
fi
echo "ksquares-rl removed from the applications menu"
