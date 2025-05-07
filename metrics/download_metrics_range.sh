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
  SYMBOLS=(BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT ENAUSDT)
fi

: "${PROXY_URL:?Please set PROXY_URL in env!}"

TARGET="metrics/${PART}/${METRIC}/${SYMBOLS[0]}"
rm -rf "$TARGET"
mkdir -p "$TARGET"

to_ms() { date -d "$1" +%s000; }

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

if [[ "$METRIC" == "open_interest" || "$METRIC" == "funding_rate" ]]; then
  cur="$START"
  while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
    nxt=$(date -I -d "$cur +1 day")
    s=$(to_ms "$cur") e=$(to_ms "$nxt")

    for sym in "${SYMBOLS[@]}"; do
      if [[ "$METRIC" == "open_interest" ]]; then
        BIN_URL="https://fapi.binance.com/futures/data/openInterestHist?symbol=${sym}&period=1d&startTime=${s}&endTime=${e}&limit=500"
      else
        BIN_URL="https://fapi.binance.com/fapi/v1/fundingRate?symbol=${sym}&startTime=${s}&endTime=${e}&limit=1000"
      fi

      EURL=$(urlencode "$BIN_URL")
      echo "→ Downloading ${METRIC} ${sym} @ ${cur}"
      if curl -sSf "${PROXY_URL}/proxy?url=${EURL}" > "${TARGET}/${sym}_${cur}.json"; then
        :
      else
        echo "⚠️ API-Error für ${sym} @ ${cur}, schreibe leere Datei"
        echo '{}' > "${TARGET}/${sym}_${cur}.json"
      fi

      sleep 0.05
    done

    cur="$nxt"
  done

elif [[ "$METRIC" == "liquidity" ]]; then
  cur="$START"
  while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
    for sym in "${SYMBOLS[@]}"; do
      BIN_URL="https://api.binance.com/api/v3/depth?symbol=${sym}&limit=100"
      EURL=$(urlencode "$BIN_URL")
      echo "→ Downloading liquidity ${sym} @ ${cur}"

      # Rohdaten ziehen oder leeres Array-Objekt
      raw=$(curl -sSf "${PROXY_URL}/proxy?url=${EURL}") || raw='{"bids":[],"asks":[]}'

      # valid JSON mit bids/asks?
      if ! echo "$raw" | jq -e 'has("bids") and has("asks")' >/dev/null; then
        echo "⚠️ Ungültige Liquidity-Antwort für ${sym} @ ${cur}, überspringe"
        continue
      fi

      # Default-Arrays, falls null
      bids=$(echo "$raw" | jq '.bids // []')
      asks=$(echo "$raw" | jq '.asks // []')

      # erster Bid/Ask
      bid0=$(echo "$bids"   | jq 'if length>0 then .[0][0] | tonumber else 0 end')
      ask0=$(echo "$asks"   | jq 'if length>0 then .[0][0] | tonumber else 0 end')

      # weitere Kennzahlen
      mid=$(jq -n --arg b "$bid0" --arg a "$ask0" '((($b|tonumber)+($a|tonumber))/2)')
      spread=$(jq -n --arg b "$bid0" --arg a "$ask0" '(($a|tonumber)-($b|tonumber))')
      bid_depth=$(echo "$bids" | jq '[ .[][1] | tonumber ] | add')
      ask_depth=$(echo "$asks" | jq '[ .[][1] | tonumber ] | add')

      # schreiben
      jq -n \
        --arg symbol "$sym" \
        --arg date   "$cur" \
        --argjson mid       "$mid" \
        --argjson spread    "$spread" \
        --argjson bid_depth "$bid_depth" \
        --argjson ask_depth "$ask_depth" \
        '{
          symbol:    $symbol,
          date:      $date,
          mid,
          spread,
          bid_depth,
          ask_depth
        }' > "${TARGET}/${sym}_${cur}.json"

      sleep 0.05
    done

    cur=$(date -I -d "$cur +1 day")
  done

else
  echo "Unknown metric: $METRIC"
  exit 1
fi
