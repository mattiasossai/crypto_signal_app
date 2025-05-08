#!/usr/bin/env bash
# historical/download_history_range.sh

set -euo pipefail

# args: SYMBOL INTERVAL START_DATE END_DATE
if [ $# -ne 4 ]; then
  echo "Usage: $0 SYMBOL INTERVAL START_DATE END_DATE"
  exit 1
fi

SYMBOL="$1"        # z.B. BTCUSDT
INTERVAL="$2"      # z.B. 1h, 4h, 1d
START="$3"         # YYYY-MM-DD (inklusive)
END="$4"           # YYYY-MM-DD (exklusiv)
TARGET="historical/${SYMBOL}/${INTERVAL}"

mkdir -p "$TARGET"

cur="$START"
while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
  FILE="${TARGET}/${SYMBOL}-${INTERVAL}-${cur}.csv"
  if [ -f "$FILE" ]; then
    echo "✔️ Skipping existing $FILE"
  else
    echo "→ Download ${SYMBOL}-${INTERVAL} @ $cur"
    URL="https://data.binance.vision/data/futures/um/daily/klines/${SYMBOL}/${INTERVAL}/${SYMBOL}-${INTERVAL}-${cur}.zip"
    if curl --fail -s "$URL" --output - | funzip > "$FILE"; then
      echo "   ✅ Saved $FILE"
    else
      echo "⚠️ Fehler oder nicht gefunden: $URL"
      rm -f "$FILE"
    fi
  fi
  # nächsten Tag
  cur=$(date -I -d "$cur +1 day")
done
