#!/usr/bin/env bash
set -euo pipefail

# args: SYMBOL INTERVAL START_DATE END_DATE OUT_DIR
if [ $# -ne 5 ]; then
  echo "Usage: $0 SYMBOL INTERVAL START_DATE END_DATE OUT_DIR"
  exit 1
fi

SYMBOL="$1"
INTERVAL="$2"
START_DATE="$3"
END_DATE="$4"
OUT_DIR="$5"

mkdir -p "$OUT_DIR"

current="$START_DATE"
# wir laufen bis inkl. END_DATE
stop=$(date -I -d "$END_DATE + 1 day")

while [[ "$current" != "$stop" ]]; do
  file="${OUT_DIR}/${SYMBOL}-${INTERVAL}-${current}.zip"
  url="https://data.binance.vision/data/futures/um/daily/klines/${SYMBOL}/${INTERVAL}/${SYMBOL}-${INTERVAL}-${current}.zip"

  if [[ ! -f "$file" ]]; then
    http_code=$(curl -sSL -w "%{http_code}" -o "$file" "$url" || true)
    if [[ "$http_code" -ne 200 ]]; then
      rm -f "$file"
      echo "⚠️ $SYMBOL $INTERVAL $current → HTTP $http_code, skipping"
    else
      echo "✅ saved $file"
    fi
  else
    echo "↪️ already exists $file"
  fi

  current=$(date -I -d "$current + 1 day")
done
