#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 3 ]; then
  echo "Usage: $0 <label> <start_date> <end_date>"
  exit 1
fi

LABEL=$1
START=$2      # im Format YYYY-MM-DD
END=$3        # im Format YYYY-MM-DD

SYMBOLS=(BTCUSDT ETHUSDT BNBUSDT XRPUSDT SOLUSDT ENAUSDT)
INTERVALS=(1m 5m 15m 1h 4h)

current=$START
while [[ "$current" < "$END" || "$current" == "$END" ]]; do
  for sym in "${SYMBOLS[@]}"; do
    for iv in "${INTERVALS[@]}"; do
      url="https://data.binance.vision/data/futures/um/daily/klines/${sym}/${iv}/${sym}-${iv}-${current}.zip"
      target_dir="historical/${LABEL}/${sym}/${iv}"
      mkdir -p "$target_dir"
      echo "[$LABEL] Downloading $sym $iv $current …"
      if [[ ! -f "$target_dir/$(basename $url)" ]]; then
        curl -sSf "$url" -o "$target_dir/$(basename $url)" \
          || echo "⚠️  Missing: $url"
      fi
    done
  done
  current=$(date -I -d "$current + 1 day")
done

echo "✅ [$LABEL] Download complete."
