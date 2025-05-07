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

to_ms(){ date -d "$1" +%s000; }

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
      curl -sSf "${PROXY_URL}/proxy?url=${EURL}" \
        > "${TARGET}/${sym}_${cur}.json" \
        || { echo "⚠️ Error ${sym} ${cur}, writing empty"; echo '{}' > "${TARGET}/${sym}_${cur}.json"; }

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

      raw=$(curl -sSf "${PROXY_URL}/proxy?url=${EURL}") || raw='{"bids":[],"asks":[]}'

      # null/leer prüfen
      if [[ -z "$raw" ]] || [[ "$raw" == "null" ]]; then
        echo "⚠️ Liquidity empty/blocked for ${sym} @ ${cur}"
        continue
      fi

      # sichere Arrays
      bids=$(jq -e '.bids // []' <<<"$raw")
      asks=$(jq -e '.asks // []' <<<"$raw")

      bid0=$(jq 'if ($bids|length)>0 then $bids[0][0]|tonumber else 0 end' --argjson bids "$bids" <<<"$raw")
      ask0=$(jq 'if ($asks|length)>0 then $asks[0][0]|tonumber else 0 end' --argjson asks "$asks" <<<"$raw")

      mid=$(jq -n --arg b "$bid0" --arg a "$ask0" '((($b|tonumber)+($a|tonumber))/2)')
      spread=$(jq -n --arg b "$bid0" --arg a "$ask0" '(($a|tonumber)-($b|tonumber))')
      bid_depth=$(jq '[ .bids[][1]|tonumber ] | add' <<<"$raw")
      ask_depth=$(jq '[ .asks[][1]|tonumber ] | add' <<<"$raw")

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
