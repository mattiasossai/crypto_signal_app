#!/usr/bin/env bash
set -euo pipefail

# args: METRIC START_DATE END_DATE PART SYMBOL
if [ $# -ne 5 ]; then
  echo "Usage: $0 METRIC START_DATE END_DATE PART_NAME SYMBOL"
  exit 1
fi

METRIC="$1"         # open_interest | funding_rate
START="$2"          # YYYY-MM-DD (inclusive)
END="$3"            # YYYY-MM-DD (inclusive)
PART="$4"           # part1 | part2
SYMBOL="$5"         # e.g. BNBUSDT

: "${PROXY_URL:?Please set PROXY_URL!}"

TARGET="metrics/${PART}/${METRIC}/${SYMBOL}"
mkdir -p "$TARGET"

# helper: date → ms
to_ms(){ date -d "$1" +%s000; }

cur="$START"
end_ts=$(date -d "$END +1 day" +%s)
while [ "$(date -d "$cur" +%s)" -lt "$end_ts" ]; do
  FILE="${TARGET}/${SYMBOL}_${cur}.json"
  if [ -f "$FILE" ]; then
    echo "✔️ Skipping existing $FILE"
  else
    echo "→ Downloading $METRIC | $PART | $SYMBOL @ $cur"

    next=$(date -I -d "$cur +1 day")
    case "$METRIC" in
      open_interest)
        binance_url="https://fapi.binance.com/futures/data/openInterestHist?symbol=${SYMBOL}&period=1d&startTime=$(to_ms "$cur")&endTime=$(to_ms "$next")"
        ;;
      funding_rate)
        binance_url="https://fapi.binance.com/fapi/v1/fundingRate?symbol=${SYMBOL}&startTime=$(to_ms "$cur")&endTime=$(to_ms "$next")&limit=1000"
        ;;
      *)
        echo "❌ Unknown metric: $METRIC"
        exit 2
        ;;
    esac

    # Tunnel über Proxy
    if curl -sSf -G "${PROXY_URL}/proxy" \
             --data-urlencode "url=${binance_url}" \
             -o "$FILE.tmp"; then

      if [ "$METRIC" = "funding_rate" ]; then
        # Impute leere Arrays
        if jq -e 'type=="array" and length==0' "$FILE.tmp" > /dev/null; then
          echo '[{"fundingRate":"0","fundingTime":0}]' > "$FILE"
          echo "   🚑 Imputed empty funding_rate for $cur"
        else
          mv "$FILE.tmp" "$FILE"
          echo "   ✅ Saved $FILE"
        fi
      else
        # open_interest → direkt übernehmen
        mv "$FILE.tmp" "$FILE"
        echo "   ✅ Saved $FILE"
      fi

    else
      echo "⚠️ Proxy-Download failed: $binance_url"
      rm -f "$FILE.tmp"
    fi
  fi

  cur=$(date -I -d "$cur +1 day")
done
