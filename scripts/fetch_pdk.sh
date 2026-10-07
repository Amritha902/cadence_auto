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
  sky130|sky130-minimal)
    DEST="$PDK_DIR/sky130"
    if [ -d "$DEST/.git" ]; then
      echo "sky130 already fetched at $DEST"
      exit 0
    fi
    # bias instantiates exactly two devices, so the default is to fetch only
    # those. The full primitive library is ~770MB; the two cells it actually
    # uses are under 2MB, which is the difference between a deployable image
    # and an undeployable one.
    if [ "$TARGET" = "sky130" ]; then
      SPARSE="cells/nfet_01v8 cells/pfet_01v8"
      echo "Fetching the two sky130 primitives bias uses (~2MB)..."
      echo "For the whole primitive library instead: $0 sky130-full"
    fi
    git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/google/skywater-pdk-libs-sky130_fd_pr.git "$DEST"
    git -C "$DEST" sparse-checkout set $SPARSE
    echo
    echo "Fetched to $DEST (~780MB)"
    echo
    echo "bias includes only nfet_01v8 and pfet_01v8 rather than the whole"
    echo "'tt' library section. That section also pulls in the 5V and ESD"
    echo "models, several of which are written with a bare 'include' that"
    echo "ngspice parses as a current source and dies on -- which is why the"
    echo "usual advice is to build sky130 through open_pdks first. Including"
    echo "only the primitives in use avoids that and parses faster."
    echo
    echo "pdks/sky130_nominal.spice (tracked in this repo) defines the"
    echo "statistical slope parameters the tt models reference but nothing"
    echo "here defines. Zero is the nominal corner."
    ;;

  sky130-full)
    DEST="$PDK_DIR/sky130"
    if [ -d "$DEST/.git" ]; then
      echo "sky130 already fetched at $DEST"
      exit 0
    fi
    echo "Fetching the full SkyWater primitive library (~770MB)..."
    git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/google/skywater-pdk-libs-sky130_fd_pr.git "$DEST"
    git -C "$DEST" sparse-checkout set models cells
    echo "Fetched to $DEST"
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
    echo
    echo "WARNING: these are PSP 103.6 models, which ngspice can only load as"
    echo "a compiled OSDI shared object. The upstream repository ships no"
    echo "osdi/ directory, so you must build psp103.osdi with OpenVAF for"
    echo "your platform before this PDK will simulate. sky130 is BSIM4 and"
    echo "needs no compiled models -- prefer it unless you specifically need"
    echo "SG13G2."
    ;;

  "")
    die "usage: $0 {sky130|sky130-full|ihp-sg13g2}"
    ;;

  *)
    die "unknown PDK '$TARGET'; expected sky130, sky130-full or ihp-sg13g2"
    ;;
esac

echo
echo "Verify with:  .venv/bin/bias list"
