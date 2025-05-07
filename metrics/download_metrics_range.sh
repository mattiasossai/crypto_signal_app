#!/usr/bin/env bash
set -euo pipefail

# args: METRIC START_DATE END_DATE PART_NAME
if [ $# -ne 4 ]; then
  echo "Usage: $0 METRIC START_DATE END_DATE PART_NAME"
  exit 1
fi

METRIC="$1"    # open_interest | funding_rate | liquidity
START="$2"     # YYYY-MM-DD
END="$3"       # YYYY-MM-DD (exclusive)
PART="$4"      # part1 | part2

: "${WORKER_URL:?Please set WORKER_URL in env!}"

TARGET="metrics/${PART}/${METRIC}"
rm -rf "$TARGET"
mkdir -p "$TARGET"

SYMBOLS=(BTCUSDT ETHUSDT BNBUSDT XRPUSDT SOLUSDT ENAUSDT)

to_ms(){ date -d "$1" +%s000; }

if [[ "$METRIC" == "open_interest" ]]; then
  # bleibt wie gehabt, täglich 1d-Windows, Splitting pro Tag
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
  # Neu: 1 Request pro Tag + limit=1000
  cur="$START"
  while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
    nxt=$(date -I -d "$cur +1 day")
    s=$(to_ms "$cur") e=$(to_ms "$nxt")
    for sym in "${SYMBOLS[@]}"; do
      curl -s "${WORKER_URL}/funding-rate?symbol=${sym}&startTime=${s}&endTime=${e}&limit=1000" \
        > "${TARGET}/${sym}_${cur}.json"
      sleep 0.1
    done
    cur="$nxt"
  done

elif [[ "$METRIC" == "liquidity" ]]; then
  # Neu: 1 Snapshot pro Tag → Kennzahlen berechnen
  cur="$START"
  while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
    for sym in "${SYMBOLS[@]}"; do
      raw=$(curl -s "${WORKER_URL}/liquidity?symbol=${sym}&limit=100")
      # mid-price & spread
      bid0=$(jq '(.bids[0][0] | tonumber)'  <<<"$raw")
      ask0=$(jq '(.asks[0][0] | tonumber)'  <<<"$raw")
      mid=$(jq -n --arg b "$bid0" --arg a "$ask0" '((($b|tonumber)+($a|tonumber))/2)')
      spread=$(jq -n --arg b "$bid0" --arg a "$ask0" '(($a|tonumber)-($b|tonumber))')
      # depths
      bid_depth=$(jq '[ .bids[][1]|tonumber ] | add' <<<"$raw")
      ask_depth=$(jq '[ .asks[][1]|tonumber ] | add' <<<"$raw")

      jq -n \
        --arg symbol "$sym" \
        --arg date   "$cur" \
        --argjson mid         "$mid" \
        --argjson spread      "$spread" \
        --argjson bid_depth   "$bid_depth" \
        --argjson ask_depth   "$ask_depth" \
        '{
          symbol: $symbol,
          date:   $date,
          mid,
          spread,
          bid_depth,
          ask_depth
        }' > "${TARGET}/${sym}_${cur}.json"

      sleep 0.1
    done
    cur=$(date -I -d "$cur +1 day")
  done

else
  echo "Unknown metric: $METRIC"
  exit 1
fi
