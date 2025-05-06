#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 4 ]; then
  echo "Usage: $0 METRIC START_DATE END_DATE PART"
  exit 1
fi

METRIC="$1"         # open_interest | funding_rate | liquidity
START="$2"          # z.B. 2020-01-01
END="$3"            # z.B. 2022-01-01 oder gestern
PART="$4"           # part1 | part2

WORKER_URL="${WORKER_URL:?Bitte setze WORKER_URL in ENV!}"

SYMBOLS=(BTCUSDT ETHUSDT BNBUSDT XRPUSDT SOLUSDT ENAUSDT)
TARGET="metrics/${PART}/${METRIC}"
mkdir -p "$TARGET"

# convert date to ms
to_ms(){ date -d "$1" +%s000; }

case "$METRIC" in
  open_interest)
    # 1d snapshots
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
    ;;

  funding_rate)
    # 8h snapshots
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
    ;;

  liquidity)
    # tägliche Snapshot
    cur="$START"
    while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
      for sym in "${SYMBOLS[@]}"; do
        curl -s "${WORKER_URL}/liquidity?symbol=${sym}" \
          > "${TARGET}/${sym}_${cur}.json"
        sleep 0.1
      done
      cur=$(date -I -d "$cur +1 day")
    done
    ;;
  *)
    echo "Unbekannte METRIC: $METRIC"
    exit 1
    ;;
esac
