#!/usr/bin/env bash
set -euo pipefail

# Download released pretrained assets (checkpoints + generators) and unpack into repo root.
#
# Default source is the Dropbox link in README; override with:
#   ASSETS_URL="..." bash scripts/download_assets.sh

cd "$(dirname "$0")/.."

ASSETS_URL_DEFAULT="https://www.dropbox.com/scl/fi/syteb6jjm8a1y672lx4ol/RUAGO_checkpoints.zip?rlkey=z9wg6u3p2otkk9eylxkqldilw&st=fwoym6ng&dl=0"
ASSETS_URL="${ASSETS_URL:-$ASSETS_URL_DEFAULT}"

# Dropbox tip: dl=1 triggers direct download.
ASSETS_URL="${ASSETS_URL/dl=0/dl=1}"

OUT_ZIP="RUAGO_assets.zip"

echo "[download_assets] downloading: ${ASSETS_URL}"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail -o "${OUT_ZIP}" "${ASSETS_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${OUT_ZIP}" "${ASSETS_URL}"
else
  echo "[download_assets] ERROR: need curl or wget" >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "[download_assets] ERROR: unzip not found. Please install unzip." >&2
  exit 1
fi

echo "[download_assets] validating zip..."
# ZIP files start with "PK" (0x50 0x4b). This avoids accidentally unzipping an HTML error page.
if ! head -c 2 "${OUT_ZIP}" | grep -q "PK"; then
  echo "[download_assets] ERROR: downloaded file does not look like a ZIP (missing PK header)." >&2
  echo "[download_assets] Hint: the link may require permission or Dropbox returned a non-file response." >&2
  exit 1
fi

# Integrity test before extracting.
unzip -t "${OUT_ZIP}" >/dev/null

echo "[download_assets] unpacking ${OUT_ZIP}"
unzip -o "${OUT_ZIP}" -d .

echo "[download_assets] verifying expected files..."
test -f "generators/gan_coco.pkl" || echo "[download_assets] WARN: generators/gan_coco.pkl not found"
test -f "generators/gan_tiny.pkl" || echo "[download_assets] WARN: generators/gan_tiny.pkl not found"

echo "[download_assets] done"


