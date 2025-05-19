#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 4 ]; then
  echo "Usage: $0 SYMBOL INTERVAL START_DATE END_DATE"
  exit 1
fi

SYMBOL=$1       # z.B. BTCUSDT
INTERVAL=$2     # 1m | 5m | 15m | 1h | 4h
START=$3        # YYYY-MM-DD
END=$4          # YYYY-MM-DD

TARGET="historical/historical-${SYMBOL}-${INTERVAL}"
mkdir -p "$TARGET"

# Hilfsfunktion: Datum zu Unix-Sekunden
to_sec(){ date -d "$1" +%s; }

cur="$START"
end_sec=$(to_sec "$END")
while [ "$(to_sec "$cur")" -le "$end_sec" ]; do
  FILE="${TARGET}/${SYMBOL}-${INTERVAL}-${cur}.csv"

  if [ -f "$FILE" ]; then
    echo "✔️ Skipping existing $FILE"
  else
    echo "→ Download ${SYMBOL}-${INTERVAL} @ $cur"
    URL="https://data.binance.vision/data/futures/um/daily/klines/${SYMBOL}/${INTERVAL}/${SYMBOL}-${INTERVAL}-${cur}.zip"
    if curl --fail -s "$URL" | funzip > "$FILE"; then
      echo " ✅ Saved $FILE"
    else
      echo "⚠️ Not found or error: $URL"
      rm -f "$FILE"
    fi
  fi

  cur=$(date -I -d "$cur + 1 day")
done
