#!/usr/bin/env bash
set -euo pipefail

# Usage: download_history_range.sh SYMBOL INTERVAL START_DATE END_DATE PART
if [ $# -ne 5 ]; then
  echo "Usage: $0 SYMBOL INTERVAL START_DATE END_DATE PART"
  exit 1
fi

SYMBOL="$1"        # z.B. BTCUSDT
INTERVAL="$2"      # z.B. 15m, 1h, 4h, 5m, 1m
START="$3"         # YYYY-MM-DD (inklusive)
END="$4"           # YYYY-MM-DD (exklusive, d.h. bis gestern +1)
PART="$5"          # part1 | part2

# Quelle der Daily ZIPs
BASE_URL="https://data.binance.vision/data/futures/um/daily/klines"

# Zielverzeichnis
TARGET_DIR="historical/${SYMBOL}/${INTERVAL}/${PART}"
mkdir -p "$TARGET_DIR"

cur="$START"
# Korrekte String-Vergleichs-Syntax für Datum
while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
  filename="${SYMBOL}-${INTERVAL}-${cur}.csv"
  target_file="${TARGET_DIR}/${filename}"

  if [ -f "$target_file" ]; then
    echo "✅ Skipping existing $filename"
  else
    echo "⬇️  Downloading $filename"
    zip_url="${BASE_URL}/${SYMBOL}/${INTERVAL}/${SYMBOL}-${INTERVAL}-${cur}.zip"
    tmp_zip="/tmp/${SYMBOL}-${INTERVAL}-${cur}.zip"

    # ZIP herunterladen
    if curl -sSf "$zip_url" -o "$tmp_zip"; then
      # direkt aus dem ZIP in die CSV streamen
      unzip -p "$tmp_zip" "${SYMBOL}-${INTERVAL}-${cur}.csv" > "$target_file"
      rm "$tmp_zip"
      echo "   ✔️ Saved to $target_file"
    else
      echo "   ⚠ Warning: ZIP not found for $cur"
      rm -f "$tmp_zip"
    fi
  fi

  # Nächster Tag
  cur=$(date -I -d "$cur +1 day")
done
