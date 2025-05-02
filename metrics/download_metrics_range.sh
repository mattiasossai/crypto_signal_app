#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 3 ]; then
  echo "Usage: $0 START_DATE END_DATE PART_NAME"
  exit 1
fi

# Cloudflare-Worker Proxy URL via GitHub Secret CF_PROXY_URL
WORKER_URL="${WORKER_URL:?Bitte setze WORKER_URL als ENV-Variable!}"
START="$1"
END="$2"
PART="$3"

SYMBOLS=(BTCUSDT ETHUSDT BNBUSDT XRPUSDT SOLUSDT ENAUSDT)

to_ms(){ date -d "$1" +%s000; }

TARGET="metrics/${PART}"
mkdir -p "${TARGET}/open_interest" "${TARGET}/funding_rate" "${TARGET}/liquidity"

# 1) Open Interest (1d)
cur="$START"
while [[ "$cur" < "$END" ]]; do
  nxt=$(date -I -d "$cur +1 day")
  s=$(to_ms "$cur") e=$(to_ms "$nxt")
  for sym in "${SYMBOLS[@]}"; do
    if curl -s --fail \
        "${WORKER_URL}/open-interest?symbol=${sym}&period=1d&startTime=${s}&endTime=${e}" \
        > "${TARGET}/open_interest/${sym}_${cur}.json"; then
      echo "✅ OI $sym $cur"
    else
      echo "⏩ OI $sym $cur – no data"
      rm -f "${TARGET}/open_interest/${sym}_${cur}.json"
    fi
  done
  cur="$nxt"
done

# 2) Funding Rates (8h)
cur="$START"
while [[ "$cur" < "$END" ]]; do
  for h in 0 8 16; do
    s=$(date -d "$cur +${h} hour" +%s000)
    e=$(date -d "$cur +$((h+8)) hour" +%s000)
    for sym in "${SYMBOLS[@]}"; do
      if curl -s --fail \
          "${WORKER_URL}/funding-rate?symbol=${sym}&startTime=${s}&endTime=${e}" \
          > "${TARGET}/funding_rate/${sym}_${cur}_${h}.json"; then
        echo "✅ FR $sym $cur+$h"
      else
        echo "⏩ FR $sym $cur+$h – no data"
        rm -f "${TARGET}/funding_rate/${sym}_${cur}_${h}.json"
      fi
    done
  done
  cur=$(date -I -d "$cur +1 day")
done

# 3) Liquidity (1d snapshot)
cur="$START"
while [[ "$cur" < "$END" ]]; do
  for sym in "${SYMBOLS[@]}"; do
    if curl -s --fail \
        "${WORKER_URL}/liquidity?symbol=${sym}" \
        > "${TARGET}/liquidity/${sym}_${cur}.json"; then
      echo "✅ LQ $sym $cur"
    else
      echo "⏩ LQ $sym $cur – no data"
      rm -f "${TARGET}/liquidity/${sym}_${cur}.json"
    fi
  done
  cur=$(date -I -d "$cur +1 day")
done
