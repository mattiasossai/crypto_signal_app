#!/usr/bin/env bash
set -euo pipefail

# args: SYMBOL INTERVAL START_DATE END_DATE
if [ $# -ne 4 ]; then
  echo "Usage: $0 SYMBOL INTERVAL START_DATE END_DATE"
  exit 1
fi

SYMBOL="$1"      # z.B. BTCUSDT
INTERVAL="$2"    # z.B. 15m,1h,4h
START="$3"       # YYYY-MM-DD
END="$4"         # YYYY-MM-DD (inclusive)

# Zielordner
TARGET="historical/${SYMBOL}/${INTERVAL}"
mkdir -p "$TARGET"

# Helper: YYYY-MM-DD → Binance-Zeitstempel in ms
to_ms(){ date -d "$1" +%s000; }

cur="$START"
while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END +1 day")" ]]; do
  nxt=$(date -I -d "$cur +1 day")
  s=$(to_ms "$cur")
  e=$(to_ms "$nxt")

  echo "↓ Downloading ${SYMBOL}-${INTERVAL} @ ${cur}"
  # URL per Binance ZIP-Endpoint (Vision)
  # z.B. https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/15m/BTCUSDT-15m-2025-05-07.zip
  ZIP_URL="https://data.binance.vision/data/futures/um/daily/klines/${SYMBOL}/${INTERVAL}/${SYMBOL}-${INTERVAL}-${cur}.zip"

  # Versuch: runterladen & entpacken
  if curl -sSf "$ZIP_URL" -o tmp.zip; then
    unzip -p tmp.zip "${SYMBOL}-${INTERVAL}-${cur}.csv" > "${TARGET}/${SYMBOL}-${INTERVAL}-${cur}.csv"
    rm tmp.zip
  else
    echo "⚠️ Keine Datei für ${cur} (404), überspringe"
  fi

  cur="$nxt"
done
