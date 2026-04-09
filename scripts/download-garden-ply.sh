#!/usr/bin/env bash
# Download a standard INRIA-format 3D Gaussian Splatting .ply for tsplat.
#
# Default: Mip-NeRF 360 "garden" from the official pretrained bundle
# (graphdeco-inria / Kerbl et al., SIGGRAPH 2023; bundle updated on INRIA with
# viewer/training improvements — same PLY layout tsplat expects).
#
# The garden model lives inside models.zip (~14 GiB). There is no official
# single-file garden URL; this script downloads the zip once, extracts one PLY,
# then removes the zip unless KEEP_ZIP=1.
#
# Quick smoke test (different scene, ~254 MiB):  --quick
#
# Usage:
#   ./scripts/download-garden-ply.sh
#   MODELS_URL=https://huggingface.co/camenduru/gaussian-splatting/resolve/main/models.zip ./scripts/download-garden-ply.sh
#   ./scripts/download-garden-ply.sh --quick

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/data/3dgs}"
mkdir -p "$OUT_DIR"

MODELS_URL_DEFAULT="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip"
MODELS_URL="${MODELS_URL:-$MODELS_URL_DEFAULT}"

QUICK_URL="https://huggingface.co/camenduru/gaussian-splatting/resolve/main/train/point_cloud/iteration_30000/point_cloud.ply"

usage() {
  sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage 0
fi

if [[ "${1:-}" == "--quick" ]]; then
  DEST="$OUT_DIR/tandt_train_30k.ply"
  echo "Downloading Tanks & Temples 'train' scene (INRIA PLY, ~254 MiB) -> $DEST"
  echo "(This is not the garden scene; use default mode for garden.)"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 --continue-at - -o "$DEST" "$QUICK_URL"
  else
    wget -O "$DEST" "$QUICK_URL"
  fi
  echo "Done. Example: cargo run --release -- $DEST --dump-stats"
  exit 0
fi

if [[ -n "${1:-}" ]]; then
  echo "Unknown option: $1" >&2
  usage 1
fi

command -v unzip >/dev/null 2>&1 || {
  echo "unzip is required to extract garden from models.zip" >&2
  exit 1
}

ZIP_PATH="${ZIP_PATH:-$OUT_DIR/models.zip}"
IN_ZIP="garden/point_cloud/iteration_30000/point_cloud.ply"
DEST="$OUT_DIR/garden.ply"

if [[ -f "$DEST" ]]; then
  echo "Already present: $DEST"
  echo "Remove it or set OUT_DIR to re-download."
  exit 0
fi

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Downloading pretrained models bundle (large, ~14 GiB)..."
  echo "URL: $MODELS_URL"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 5 --continue-at - -o "$ZIP_PATH" "$MODELS_URL"
  else
    wget -O "$ZIP_PATH" "$MODELS_URL"
  fi
else
  echo "Using existing zip: $ZIP_PATH"
fi

TMP_EXTRACT="${TMPDIR:-/tmp}/tsplat-garden-extract.$$"
mkdir -p "$TMP_EXTRACT"
cleanup() { rm -rf "$TMP_EXTRACT"; }
trap cleanup EXIT

echo "Extracting $IN_ZIP ..."
unzip -q "$ZIP_PATH" "$IN_ZIP" -d "$TMP_EXTRACT"
mv -f "$TMP_EXTRACT/$IN_ZIP" "$DEST"

if [[ "${KEEP_ZIP:-0}" != "1" ]]; then
  rm -f "$ZIP_PATH"
  echo "Removed $ZIP_PATH (set KEEP_ZIP=1 to keep it)."
else
  echo "Kept zip at $ZIP_PATH"
fi

echo "Done: $DEST"
echo "Example: cargo run --release -- $DEST --dump-stats"
