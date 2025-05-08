#!/usr/bin/env bash
set -euo pipefail

# args: METRIC START_DATE END_DATE PART SYMBOL
if [ $# -ne 5 ]; then
  echo "Usage: $0 METRIC START_DATE END_DATE PART_NAME SYMBOL"
  exit 1
fi

METRIC="$1"         # open_interest | funding_rate | liquidity
START="$2"          # YYYY-MM-DD (inclusive)
END="$3"            # YYYY-MM-DD (inclusive)
PART="$4"           # part1 | part2
SYMBOL="$5"         # e.g. BNBUSDT

: "${PROXY_URL:?Please set PROXY_URL!}"

TARGET="metrics/${PART}/${METRIC}/${SYMBOL}"
mkdir -p "$TARGET"

# helper: date → ms
to_ms(){ date -d "$1" +%s000; }

# loop from START to inclusive END
cur="$START"
end_ts=$(date -d "$END +1 day" +%s)
while [ "$(date -d "$cur" +%s)" -lt "$end_ts" ]; do
  FILE="${TARGET}/${SYMBOL}_${cur}.json"
  if [ -f "$FILE" ]; then
    echo "✔️ Skipping existing $FILE"
  else
    echo "→ Downloading $METRIC | $PART | $SYMBOL @ $cur"

    # 1) build real Binance URL
    case "$METRIC" in
      open_interest)
        next=$(date -I -d "$cur +1 day")
        binance_url="https://fapi.binance.com/futures/data/openInterestHist?symbol=${SYMBOL}&period=1d&startTime=$(to_ms "$cur")&endTime=$(to_ms "$next")&limit=1000"
        ;;
      funding_rate)
        next=$(date -I -d "$cur +1 day")
        binance_url="https://fapi.binance.com/fapi/v1/fundingRate?symbol=${SYMBOL}&startTime=$(to_ms "$cur")&endTime=$(to_ms "$next")&limit=1000"
        ;;
      liquidity)
        binance_url="https://fapi.binance.com/fapi/v1/depth?symbol=${SYMBOL}&limit=500"
        ;;
      *)
        echo "❌ Unknown metric: $METRIC"
        exit 2
        ;;
    esac

    # 2) proxy-tunnel via /proxy endpoint
    echo "   Proxy→ ${PROXY_URL}/proxy?url=…"
    if curl -sSf -G "${PROXY_URL}/proxy" \
             --data-urlencode "url=${binance_url}" \
             -o "$FILE.tmp"; then

      # 3) post-process per metric
      if [ "$METRIC" = "funding_rate" ]; then
        # impute empty arrays
        if jq -e 'type=="array" and length==0' "$FILE.tmp" > /dev/null; then
          echo '[{"fundingRate":"0","fundingTime":0}]' > "$FILE"
          echo "   🚑 Imputed empty funding_rate for $cur"
        else
          mv "$FILE.tmp" "$FILE"
          echo "   ✅ Saved $FILE"
        fi

      elif [ "$METRIC" = "liquidity" ]; then
        # ensure we have bids & asks
        if ! jq -e 'type=="object" and has("bids") and has("asks")' "$FILE.tmp" > /dev/null; then
          raw='{"bids":[],"asks":[]}'
        else
          raw=$(<"$FILE.tmp")
        fi
        # flatten to summary JSON
        echo "$raw" | jq --arg symbol "$SYMBOL" --arg date "$cur" '{
          symbol:   $symbol,
          date:     $date,
          mid:      (if .bids|length>0 and .asks|length>0 then ((.bids[0][0]|tonumber + .asks[0][0]|tonumber)/2) else null end),
          spread:   (if .bids|length>0 and .asks|length>0 then ((.asks[0][0]|tonumber - .bids[0][0]|tonumber)) else null end),
          bid_depth:(if .bids|length>0 then ([.bids[][1]|tonumber] | add) else null end),
          ask_depth:(if .asks|length>0 then ([.asks[][1]|tonumber] | add) else null end)
        }' > "$FILE"
        echo "   ✅ Saved processed liquidity JSON"

      else
        # open_interest
        mv "$FILE.tmp" "$FILE"
        echo "   ✅ Saved $FILE"
      fi

    else
      echo "⚠️ Proxy-Download failed: $binance_url"
      rm -f "$FILE.tmp"
    fi
  fi

  # next day
  cur=$(date -I -d "$cur +1 day")
done
