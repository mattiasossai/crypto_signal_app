#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 4 ]; then
  echo "Usage: $0 SYMBOL INTERVAL START_DATE END_DATE"
  exit 1
fi

SYMBOL=$1
INTERVAL=$2
START=$3
END=$4

# Zielverzeichnis exakt wie in deinem Repo
TARGET="historical/historical-${SYMBOL}-${INTERVAL}"
mkdir -p "$TARGET"

# Datum → Sekunden
to_sec(){ date -d "$1" +%s; }

cur="$START"
end_sec=$(to_sec "$END")
while [ "$(to_sec "$cur")" -le "$end_sec" ]; do
  OUT="${TARGET}/${SYMBOL}-${INTERVAL}-${cur}.csv"

  if [ -f "$OUT" ]; then
    echo "✔️ Skipping existing $OUT"
  else
    echo "→ Download ${SYMBOL}-${INTERVAL} @ $cur"
    URL="https://data.binance.vision/data/futures/um/daily/klines/${SYMBOL}/${INTERVAL}/${SYMBOL}-${INTERVAL}-${cur}.zip"
    if curl --fail -s "$URL" | funzip > "$OUT"; then
      echo " ✅ Saved $OUT"
    else
      echo "⚠️ Not found or error: $URL"
      rm -f "$OUT"
    fi
  fi

  cur=$(date -I -d "$cur + 1 day")
done
