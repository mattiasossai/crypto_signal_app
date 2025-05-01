#!/usr/bin/env bash
set -e
START="$1"
END="$2"
OUT_DIR="$3"
mkdir -p "$OUT_DIR"

for symbol in BTCUSDT ETHUSDT BNBUSDT XRPUSDT SOLUSDT ENAUSDT; do
  for metric in open_interest funding_rates liquidity; do
    # Beispielaufruf Deines Workers:
    curl -s \
      "$WORKER_URL/metrics?symbol=$symbol&metric=$metric&start=$START&end=$END" \
      -o "$OUT_DIR/${symbol}_${metric}_${START}_${END}.json"
  done
done
