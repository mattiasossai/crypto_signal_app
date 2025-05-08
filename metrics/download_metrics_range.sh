#!/usr/bin/env bash
set -euo pipefail

# args: METRIC START_DATE END_DATE PART_NAME [SYMBOL]
if [ $# -lt 4 ] || [ $# -gt 5 ]; then
  echo "Usage: $0 METRIC START_DATE END_DATE PART_NAME [SYMBOL]"
  exit 1
fi

METRIC="$1"
START="$2"
END="$3"
PART="$4"
if [ $# -eq 5 ]; then
  SYMBOLS=("$5")
else
  SYMBOLS=(BTCUSDT ETHUSDT BNBUSDT XRPUSDT SOLUSDT ENAUSDT)
fi

: "${PROXY_URL:?Please set PROXY_URL in env!}"

TARGET_DIR="metrics/${PART}/${METRIC}"
rm -rf "${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"

# 10-stellige Sekunden +000 → Binance-kompatibel
to_sec(){ date -d "$1" +%s; }
to_ms(){ printf '%s000' "$(to_sec "$1")"; }

# URL-encode helper
urlencode() {
  local s="$1" enc="" i c o
  for ((i=0; i<${#s}; i++)); do
    c=${s:i:1}
    case "$c" in
      [a-zA-Z0-9.~_-]) o="$c" ;;
      *) printf -v o '%%%02X' "'$c" ;;
    esac
    enc+="$o"
  done
  echo "$enc"
}

if [[ "$METRIC" == "open_interest" ]]; then
  cur="$START"
  while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
    nxt=$(date -I -d "$cur +1 day")
    s=$(to_ms "$cur")  e=$(to_ms "$nxt")

    for sym in "${SYMBOLS[@]}"; do
      BIN_URL="https://fapi.binance.com/futures/data/openInterestHist?symbol=${sym}&period=1d&startTime=${s}&endTime=${e}&limit=1000"
      EURL=$(urlencode "$BIN_URL")
      echo "→ Download OI ${sym} @ ${cur}"
      curl -sSf "${PROXY_URL}/proxy?url=${EURL}" \
        > "${TARGET_DIR}/${sym}_${cur}.json" \
      || echo '{"error":"oi-error"}' > "${TARGET_DIR}/${sym}_${cur}.json"
      sleep 0.05
    done
    cur="$nxt"
  done

elif [[ "$METRIC" == "funding_rate" ]]; then
  cur="$START"
  while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
    nxt=$(date -I -d "$cur +1 day")
    s=$(to_ms "$cur")  e=$(to_ms "$nxt")

    for sym in "${SYMBOLS[@]}"; do
      BIN_URL="https://fapi.binance.com/fapi/v1/fundingRate?symbol=${sym}&startTime=${s}&endTime=${e}&limit=1000"
      EURL=$(urlencode "$BIN_URL")
      echo "→ Download FR ${sym} @ ${cur}"
      raw=$(curl -sSf "${PROXY_URL}/proxy?url=${EURL}" || echo '[]')
      # Wenn leeres Array oder kein Eintrag: setze Rate auf 0
      if [[ "$raw" == "[]" ]]; then
        echo '[{"fundingRate":"0","fundingTime":0}]' > tmp.json
        raw=$(<tmp.json)
        rm tmp.json
      fi
      echo "$raw" > "${TARGET_DIR}/${sym}_${cur}.json"
      sleep 0.05
    done
    cur="$nxt"
  done

elif [[ "$METRIC" == "liquidity" ]]; then
  cur="$START"
  while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
    for sym in "${SYMBOLS[@]}"; do
      # Futures-Orderbook endpoint
      BIN_URL="https://fapi.binance.com/fapi/v1/depth?symbol=${sym}&limit=100"
      EURL=$(urlencode "$BIN_URL")
      echo "→ Download Liquidity ${sym} @ ${cur}"

      raw=$(curl -sSf "${PROXY_URL}/proxy?url=${EURL}" || echo '{"bids":[],"asks":[]}')

      # Berechnung, wenn bids/asks da sind
      bid0=$(echo "$raw" | jq 'if .bids|length>0 then .bids[0][0]|tonumber else null end')
      ask0=$(echo "$raw" | jq 'if .asks|length>0 then .asks[0][0]|tonumber else null end')
      mid=$(jq -n --arg b "$bid0" --arg a "$ask0" 'if $b and $a then ((($b|tonumber)+($a|tonumber))/2) else null end')
      spread=$(jq -n --arg b "$bid0" --arg a "$ask0" 'if $b and $a then (($a|tonumber)-($b|tonumber)) else null end')
      bid_depth=$(echo "$raw" | jq '[.bids[][1]|tonumber] | add // null')
      ask_depth=$(echo "$raw" | jq '[.asks[][1]|tonumber] | add // null')

      jq -n \
        --arg symbol "$sym" \
        --arg date   "$cur" \
        --argjson mid       "$mid" \
        --argjson spread    "$spread" \
        --argjson bid_depth "$bid_depth" \
        --argjson ask_depth "$ask_depth" \
        '{
          symbol,
          date,
          mid,
          spread,
          bid_depth,
          ask_depth
        }' > "${TARGET_DIR}/${sym}_${cur}.json"

      sleep 0.05
    done
    cur=$(date -I -d "$cur +1 day")
  done

else
  echo "Unknown metric: $METRIC"
  exit 1
fi
