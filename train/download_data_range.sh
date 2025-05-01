#!/usr/bin/env bash
set -euo pipefail

# args: SYMBOL INTERVAL START_DATE END_DATE OUT_DIR
SYMBOL="$1"
INTERVAL="$2"
START_DATE="$3"
END_DATE="$4"
OUT_DIR="$5"

# sicherstellen, dass Ausgabeverzeichnis existiert
mkdir -p "$OUT_DIR"

current="$START_DATE"
# Endetag +1, damit die Schleife inkl. END_DATE läuft
stop=$(date -I -d "$END_DATE + 1 day")

while [[ "$current" != "$stop" ]]; do
  file="${OUT_DIR}/${SYMBOL}-${INTERVAL}-${current}.zip"
  url="https://data.binance.vision/data/futures/um/daily/klines/${SYMBOL}/${INTERVAL}/${SYMBOL}-${INTERVAL}-${current}.zip"

  if [[ ! -f "$file" ]]; then
    # mit curl herunterladen, HTTP-Status prüfen
    http_code=$(curl -sSL -w "%{http_code}" -o "$file" "$url" || true)
    if [[ "$http_code" -ne 200 ]]; then
      rm -f "$file"
      echo "⚠️ $SYMBOL $INTERVAL $current → HTTP $http_code, übersprungen"
    else
      echo "✅ saved $file"
    fi
  else
    echo "↪️ already exists $file"
  fi

  # nächster Tag
  current=$(date -I -d "$current + 1 day")
done
