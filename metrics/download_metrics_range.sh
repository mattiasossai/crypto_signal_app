#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 SYMBOL METRIC START_DATE END_DATE OUT_DIR"
  exit 1
fi

SYMBOL="$1"
METRIC="$2"
START="$3"
END="$4"
OUT_DIR="$5"

# WORKER_URL muss via ENV gesetzt sein (siehe Workflow)
WORKER_URL="${WORKER_URL:?Bitte setze WORKER_URL als ENV!}"

to_ms(){ date -d "$1" +%s000; }

cur="$START"
stop=$(date -I -d "$END + 1 day")

case "$METRIC" in
  open_interest)
    while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
      nxt=$(date -I -d "$cur +1 day")
      s=$(to_ms "$cur") e=$(to_ms "$nxt")
      url="${WORKER_URL}/open-interest?symbol=${SYMBOL}&period=1d&startTime=${s}&endTime=${e}"
      out="${OUT_DIR}/${SYMBOL}_${cur}.json"
      curl -sS "$url" > "$out"
      echo "✅ $METRIC $SYMBOL $cur"
      cur="$nxt"
    done
    ;;
  funding_rate)
    while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
      for h in 0 8 16; do
        s=$(date -d "$cur +${h} hour" +%s000)
        e=$(date -d "$cur +$((h+8)) hour" +%s000)
        url="${WORKER_URL}/funding-rate?symbol=${SYMBOL}&startTime=${s}&endTime=${e}"
        out="${OUT_DIR}/${SYMBOL}_${cur}_${h}.json"
        curl -sS "$url" > "$out"
        echo "✅ $METRIC $SYMBOL $cur+$h"
      done
      cur=$(date -I -d "$cur +1 day")
    done
    ;;
  liquidity)
    while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
      url="${WORKER_URL}/liquidity?symbol=${SYMBOL}"
      out="${OUT_DIR}/${SYMBOL}_${cur}.json"
      curl -sS "$url" > "$out"
      echo "✅ $METRIC $SYMBOL $cur"
      cur=$(date -I -d "$cur +1 day")
    done
    ;;
  *)
    echo "Unknown metric: $METRIC"
    exit 1
    ;;
esac
