#!/usr/bin/env bash
set -euo pipefail

# args: METRIC START_DATE END_DATE PART_NAME [SYMBOL]
if [ $# -lt 4 ] || [ $# -gt 5 ]; then
  echo "Usage: $0 METRIC START_DATE END_DATE PART_NAME [SYMBOL]"
  exit 1
fi

METRIC="$1"    # open_interest | funding_rate | liquidity
START="$2"     # YYYY-MM-DD
END="$3"       # YYYY-MM-DD (exclusive)
PART="$4"      # part1 | part2

# Optional: nur ein Symbol, wenn 5. Argument gesetzt
if [ $# -eq 5 ]; then
  SYMBOLS=("$5")
else
  SYMBOLS=(BTCUSDT ETHUSDT BNBUSDT XRPUSDT SOLUSDT ENAUSDT)
fi

: "${PROXY_URL:?Please set PROXY_URL in env!}"

# Zielverzeichnis für diesen Job (je ein Symbol)
TARGET="metrics/${PART}/${METRIC}/${SYMBOLS[0]}"
rm -rf "$TARGET"
mkdir -p "$TARGET"

# Hilfsfunktion: YYYY-MM-DD → ms
to_ms(){ date -d "$1" +%s000; }

# Bash-URL-Encoder
urlencode() {
  local s="$1" enc="" i c o
  for (( i=0; i<${#s}; i++ )); do
    c=${s:i:1}
    case "$c" in
      [a-zA-Z0-9.~_-]) o="$c" ;;
      *) printf -v o '%%%02X' "'$c" ;;
    esac
    enc+="$o"
  done
  echo "$enc"
}

case "$METRIC" in
  open_interest|funding_rate)
    cur="$START"
    while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
      nxt=$(date -I -d "$cur +1 day")
      s=$(to_ms "$cur") e=$(to_ms "$nxt")

      for sym in "${SYMBOLS[@]}"; do
        if [[ "$METRIC" == "open_interest" ]]; then
          # Limit nun 500 statt 1000, um code:-1130 zu vermeiden
          BIN_URL="https://fapi.binance.com/futures/data/openInterestHist?symbol=${sym}&period=1d&startTime=${s}&endTime=${e}&limit=500"
        else
          BIN_URL="https://fapi.binance.com/fapi/v1/fundingRate?symbol=${sym}&startTime=${s}&endTime=${e}&limit=1000"
        fi

        EURL=$(urlencode "$BIN_URL")
        echo "→ Downloading ${METRIC} ${sym} @ ${cur}"
        curl -sSf "${PROXY_URL}/proxy?url=${EURL}" \
          > "metrics/${PART}/${METRIC}/${sym}_${cur}.json" \
          || { echo "⚠️ Fehler bei ${sym} ${cur}, schreibe leere Datei"; echo '{}' > "metrics/${PART}/${METRIC}/${sym}_${cur}.json"; }

        # reduziertes Sleep
        sleep 0.05
      done

      cur="$nxt"
    done
    ;;

  liquidity)
    cur="$START"
    while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
      for sym in "${SYMBOLS[@]}"; do
        BIN_URL="https://api.binance.com/api/v3/depth?symbol=${sym}&limit=100"
        EURL=$(urlencode "$BIN_URL")
        echo "→ Downloading liquidity ${sym} @ ${cur}"

        raw=$(curl -sSf "${PROXY_URL}/proxy?url=${EURL}") \
          || { echo "⚠️ Fehler bei ${sym} ${cur}, fallback leer"; raw='{"bids":[],"asks":[]}' ; }

        bid0=$(jq '(.bids[0][0] // 0) | tonumber'  <<<"$raw")
        ask0=$(jq '(.asks[0][0] // 0) | tonumber'  <<<"$raw")
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
          }' > "metrics/${PART}/${METRIC}/${sym}_${cur}.json"

        sleep 0.05
      done

      cur=$(date -I -d "$cur +1 day")
    done
    ;;
  *)
    echo "Unknown metric: $METRIC"
    exit 1
    ;;
esac
