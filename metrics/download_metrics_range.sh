#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 4 ]; then
  echo "Usage: $0 METRIC START_DATE END_DATE PART_NAME"
  exit 1
fi

METRIC="$1"
START="$2"
END="$3"
PART="$4"

WORKER_URL="${WORKER_URL:?Bitte setze WORKER_URL als ENV!}"
SYMBOLS=(BTCUSDT ETHUSDT BNBUSDT XRPUSDT SOLUSDT ENAUSDT)

to_ms(){ date -d "$1" +%s000; }

TARGET_DIR="metrics/${PART}/${METRIC}"
mkdir -p "$TARGET_DIR"
# Alte JSONs überschreiben: lösche alles
rm -f "$TARGET_DIR"/*.json

case "$METRIC" in
  open_interest)
    cur="$START"
    while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
      nxt=$(date -I -d "$cur +1 day")
      s=$(to_ms "$cur") e=$(to_ms "$nxt")
      for sym in "${SYMBOLS[@]}"; do
        curl -s "${WORKER_URL}/open_interest?symbol=${sym}&period=1d&startTime=${s}&endTime=${e}" \
          > "${TARGET_DIR}/${sym}_${cur}.json"
        sleep 0.1
      done
      cur="$nxt"
    done
    ;;
  funding_rate)
    cur="$START"
    while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
      for h in 0 8 16; do
        s=$(date -d "$cur +${h} hour" +%s000)
        e=$(date -d "$cur +$((h+8)) hour" +%s000)
        for sym in "${SYMBOLS[@]}"; do
          curl -s "${WORKER_URL}/funding_rate?symbol=${sym}&startTime=${s}&endTime=${e}" \
            > "${TARGET_DIR}/${sym}_${cur}_${h}.json"
          sleep 0.1
        done
      done
      cur=$(date -I -d "$cur +1 day")
    done
    ;;
  liquidity)
    cur="$START"
    while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
      for sym in "${SYMBOLS[@]}"; do
        curl -s "${WORKER_URL}/liquidity?symbol=${sym}" \
          > "${TARGET_DIR}/${sym}_${cur}.json"
        sleep 0.1
      done
      cur=$(date -I -d "$cur +1 day")
    done
    ;;
  *)
    echo "Unknown metric: $METRIC"
    exit 1
    ;;
esac
