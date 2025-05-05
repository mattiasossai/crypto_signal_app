#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 3 ]; then
  echo "Usage: $0 START_DATE END_DATE PART_NAME"
  exit 1
fi

# ★ WORKER_URL muss per ENV gesetzt sein
WORKER_URL="${WORKER_URL:?Bitte setze WORKER_URL als ENV!}"
START="$1"
END="$2"
PART="$3"

SYMBOLS=(BTCUSDT ETHUSDT BNBUSDT XRPUSDT SOLUSDT ENAUSDT)

to_ms(){ date -d "$1" +%s000; }

TARGET="metrics/${PART}"
mkdir -p "${TARGET}/open_interest" "${TARGET}/funding_rate" "${TARGET}/liquidity"

# 1) Open Interest (1d)
cur="$START"
while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
  nxt=$(date -I -d "$cur +1 day")
  s=$(to_ms "$cur") e=$(to_ms "$nxt")
  for sym in "${SYMBOLS[@]}"; do
    curl -s "${WORKER_URL}/open-interest?symbol=${sym}&period=1d&startTime=${s}&endTime=${e}" \
      > "${TARGET}/open_interest/${sym}_${cur}.json"
  done
  cur="$nxt"
done

# 2) Funding Rates (8h)
cur="$START"
while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
  for h in 0 8 16; do
    s=$(date -d "$cur +${h} hour" +%s000)
    e=$(date -d "$cur +$((h+8)) hour" +%s000)
    for sym in "${SYMBOLS[@]}"; do
      curl -s "${WORKER_URL}/funding-rate?symbol=${sym}&startTime=${s}&endTime=${e}" \
        > "${TARGET}/funding_rate/${sym}_${cur}_${h}.json"
    done
  done
  cur=$(date -I -d "$cur +1 day")
done

# 3) Liquidity (1d snapshot)
cur="$START"
while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
  for sym in "${SYMBOLS[@]}"; do
    curl -s "${WORKER_URL}/liquidity?symbol=${sym}" \
      > "${TARGET}/liquidity/${sym}_${cur}.json"
  done
  cur=$(date -I -d "$cur +1 day")
done
