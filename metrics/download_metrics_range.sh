#!/usr/bin/env bash
set -euo pipefail

# args: METRIC START_DATE END_DATE PART_NAME
if [ $# -ne 4 ]; then
  echo "Usage: $0 METRIC START_DATE END_DATE PART_NAME"
  exit 1
fi

METRIC="$1"
START="$2"
END="$3"
PART="$4"

: "${WORKER_URL:?Please set WORKER_URL in env!}"

TARGET="metrics/${PART}/${METRIC}"
# ← hier löschen wir ALLE alten Dateien, damit neu überschrieben wird
rm -rf "$TARGET"
mkdir -p "$TARGET"

SYMBOLS=(BTCUSDT ETHUSDT BNBUSDT XRPUSDT SOLUSDT ENAUSDT)
to_ms(){ date -d "$1" +%s000; }

if [[ "$METRIC" == "open_interest" ]]; then
  cur="$START"
  while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
    nxt=$(date -I -d "$cur +1 day")
    s=$(to_ms "$cur") e=$(to_ms "$nxt")
    for sym in "${SYMBOLS[@]}"; do
      curl -s "${WORKER_URL}/open-interest?symbol=${sym}&period=1d&startTime=${s}&endTime=${e}" \
        > "${TARGET}/${sym}_${cur}.json"
      sleep 0.1
    done
    cur="$nxt"
  done

elif [[ "$METRIC" == "funding_rate" ]]; then
  cur="$START"
  while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
    for h in 0 8 16; do
      s=$(date -d "$cur +${h} hour" +%s000)
      e=$(date -d "$cur +$((h+8)) hour" +%s000)
      for sym in "${SYMBOLS[@]}"; do
        curl -s "${WORKER_URL}/funding-rate?symbol=${sym}&startTime=${s}&endTime=${e}" \
          > "${TARGET}/${sym}_${cur}_${h}.json"
        sleep 0.1
      done
    done
    cur=$(date -I -d "$cur +1 day")
  done

elif [[ "$METRIC" == "liquidity" ]]; then
  cur="$START"
  while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
    for sym in "${SYMBOLS[@]}"; do
      curl -s "${WORKER_URL}/liquidity?symbol=${sym}" \
        > "${TARGET}/${sym}_${cur}.json"
      sleep 0.1
    done
    cur=$(date -I -d "$cur +1 day")
  done

else
  echo "Unknown metric: $METRIC"
  exit 1
fi
