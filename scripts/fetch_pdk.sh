#!/usr/bin/env bash
# Fetch an open PDK's ngspice models into pdks/<name>/.
#
# No PDK content is vendored into this repository. Both supported PDKs are
# Apache-2.0 licensed and freely redistributable, but they are large and they
# belong to their foundries, not to this benchmark.
#
#   scripts/fetch_pdk.sh sky130       SkyWater 130nm
#   scripts/fetch_pdk.sh ihp-sg13g2   IHP SG13G2 130nm BiCMOS (smaller, faster)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PDK_DIR="$REPO_ROOT/pdks"
TARGET="${1:-}"

die() { echo "error: $*" >&2; exit 1; }

command -v git >/dev/null || die "git is required"

case "$TARGET" in
  sky130)
    DEST="$PDK_DIR/sky130"
    if [ -d "$DEST/.git" ]; then
      echo "sky130 already fetched at $DEST"
      exit 0
    fi
    echo "Fetching SkyWater 130nm primitive models (shallow clone, ~200MB)..."
    git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/google/skywater-pdk-libs-sky130_fd_pr.git "$DEST"
    git -C "$DEST" sparse-checkout set models cells
    echo
    echo "Fetched to $DEST"
    echo "NOTE: sky130 ships models as a tree of .spice includes. If"
    echo "bias.pdk.SKY130's .lib path does not resolve, point it at the"
    echo "top-level corner file inside $DEST/models/."
    ;;

  ihp|ihp-sg13g2)
    DEST="$PDK_DIR/ihp-sg13g2"
    if [ -d "$DEST/.git" ]; then
      echo "ihp-sg13g2 already fetched at $DEST"
      exit 0
    fi
    echo "Fetching IHP SG13G2 ngspice models (shallow clone)..."
    git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/IHP-GmbH/IHP-Open-PDK.git "$DEST.tmp"
    git -C "$DEST.tmp" sparse-checkout set \
      ihp-sg13g2/libs.tech/ngspice
    mkdir -p "$DEST"
    cp -R "$DEST.tmp/ihp-sg13g2/libs.tech/ngspice/." "$DEST/"
    rm -rf "$DEST.tmp"
    echo
    echo "Fetched to $DEST"
    ;;

  "")
    die "usage: $0 {sky130|ihp-sg13g2}"
    ;;

  *)
    die "unknown PDK '$TARGET'; expected sky130 or ihp-sg13g2"
    ;;
esac

echo
echo "Verify with:  .venv/bin/bias list"
