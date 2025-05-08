#!/usr/bin/env bash
set -euo pipefail

# args: METRIC START_DATE END_DATE PART_NAME [SYMBOL]
if [ $# -lt 4 ] || [ $# -gt 5 ]; then
  echo "Usage: $0 METRIC START_DATE END_DATE PART_NAME [SYMBOL]"
  exit 1
fi

METRIC="$1"         # open_interest | funding_rate | liquidity
START="$2"          # YYYY-MM-DD
END="$3"            # YYYY-MM-DD (exclusive for OI/FR, inclusive for liquidity loop)
PART="$4"           # part1 | part2
if [ $# -eq 5 ]; then
  SYMBOLS=("$5")
else
  SYMBOLS=(BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT ENAUSDT)
fi

: "${PROXY_URL:?Please set PROXY_URL in env!}"

TARGET_DIR="metrics/${PART}/${METRIC}"
rm -rf "${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"

# ───────────────────────────────────────
# Hilfsfunktionen
# ───────────────────────────────────────

# 10-stellige Unix-Sekunden
to_sec(){ date -d "$1" +%s; }
# Millisekunden im Binance-Format
to_ms(){ printf '%s000' "$(to_sec "$1")"; }

# URL-Encoding
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

# ───────────────────────────────────────
# 1) Open Interest
# ───────────────────────────────────────
if [[ "$METRIC" == "open_interest" ]]; then
  cur="$START"
  while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
    nxt=$(date -I -d "$cur +1 day")
    s=$(to_ms "$cur")  e=$(to_ms "$nxt")

    for sym in "${SYMBOLS[@]}"; do
      BIN_URL="https://fapi.binance.com/futures/data/openInterestHist?symbol=${sym}&period=1d&startTime=${s}&endTime=${e}&limit=1000"
      EURL=$(urlencode "$BIN_URL")
      echo "→ Download OpenInterest ${sym} @ ${cur}"
      curl -sSf "${PROXY_URL}/proxy?url=${EURL}" \
        > "${TARGET_DIR}/${sym}_${cur}.json" \
      || echo '{"error":"oi-error"}' > "${TARGET_DIR}/${sym}_${cur}.json"
      sleep 0.05
    done

    cur="$nxt"
  done

# ───────────────────────────────────────
# 2) Funding Rate
# ───────────────────────────────────────
elif [[ "$METRIC" == "funding_rate" ]]; then
  cur="$START"
  while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END")" ]]; do
    nxt=$(date -I -d "$cur +1 day")
    s=$(to_ms "$cur")  e=$(to_ms "$nxt")

    for sym in "${SYMBOLS[@]}"; do
      BIN_URL="https://fapi.binance.com/fapi/v1/fundingRate?symbol=${sym}&startTime=${s}&endTime=${e}&limit=1000"
      EURL=$(urlencode "$BIN_URL")
      echo "→ Download FundingRate ${sym} @ ${cur}"
      raw=$(curl -sSf "${PROXY_URL}/proxy?url=${EURL}" || echo '[]')
      # Wenn kein Eintrag, ersetze durch Rate=0
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

# ───────────────────────────────────────
# 3) Liquidity
# ───────────────────────────────────────
elif [[ "$METRIC" == "liquidity" ]]; then
  cur="$START"
  while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END +1 day")" ]]; do

    for sym in "${SYMBOLS[@]}"; do
      BIN_URL="https://fapi.binance.com/fapi/v1/depth?symbol=${sym}&limit=100"
      EURL=$(urlencode "$BIN_URL")
      echo "→ Download Liquidity ${sym} @ ${cur}"

      raw=$(curl -sSf "${PROXY_URL}/proxy?url=${EURL}" || echo '{"bids":[],"asks":[]}')

      # Ein einziger jq-Filter rechnet nur, wenn Werte da sind
      echo "$raw" | jq --arg symbol "$sym" --arg date "$cur" '
        {
          symbol: $symbol,
          date: $date,
          mid:      (if (.bids|length>0 and .asks|length>0) then ((.bids[0][0]|tonumber + .asks[0][0]|tonumber)/2) else null end),
          spread:   (if (.bids|length>0 and .asks|length>0) then ((.asks[0][0]|tonumber - .bids[0][0]|tonumber)) else null end),
          bid_depth:(if (.bids|length>0) then ([.bids[][1]|tonumber] | add) else null end),
          ask_depth:(if (.asks|length>0) then ([.asks[][1]|tonumber] | add) else null end)
        }
      ' > "${TARGET_DIR}/${sym}_${cur}.json"

      sleep 0.05
    done

    cur=$(date -I -d "$cur +1 day")
  done

else
  echo "Unknown metric: $METRIC"
  exit 1
fi
