#!/usr/bin/env bash
set -euo pipefail

# args: SYMBOL INTERVAL START_DATE END_DATE PART_NAME
if [ $# -ne 5 ]; then
  echo "Usage: $0 SYMBOL INTERVAL START_DATE END_DATE PART_NAME"
  exit 1
fi

SYMBOL="$1"        # z.B. BTCUSDT
INTERVAL="$2"      # 1m | 5m | 15m | 1h | 4h
START="$3"         # YYYY-MM-DD
END="$4"           # YYYY-MM-DD (exklusiv)
PART="$5"          # part1 | part2

TARGET="historical/${SYMBOL}/${INTERVAL}/${PART}"
rm -rf "$TARGET"
mkdir -p "$TARGET"

# Hilfsfunktion für Millisekunden
to_ms(){ date -d "$1" +%s000; }

# Tägliche Schleife
cur="$START"
while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
  echo "→ Download ${SYMBOL}-${INTERVAL} @ $cur"
  # Beispiel über Binance Vision (ZIP → CSV):
  URL="https://data.binance.vision/data/futures/um/daily/klines/${SYMBOL}/${INTERVAL}/${SYMBOL}-${INTERVAL}-${cur}.zip"

  if curl --fail -s "$URL" --output - | funzip > "${TARGET}/${SYMBOL}-${INTERVAL}-$cur.csv"; then
    :
  else
    echo "⚠️ Nicht gefunden oder Fehler: $URL"
  fi

  # nächster Tag
  cur=$(date -I -d "$cur +1 day")
done
