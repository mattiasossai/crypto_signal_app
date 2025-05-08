#!/usr/bin/env bash
set -euo pipefail

# args: METRIC START_DATE END_DATE PART_NAME SYMBOL
if [ $# -ne 5 ]; then
  echo "Usage: $0 METRIC START_DATE END_DATE PART_NAME SYMBOL"
  exit 1
fi

METRIC="$1"        # open_interest | funding_rate | liquidity
START="$2"         # YYYY-MM-DD (inklusive)
END="$3"           # YYYY-MM-DD (inclusive)
PART="$4"          # part1 | part2
SYMBOL="$5"        # z.B. SOLUSDT

# Proxy-URL aus Workflow-Env
: "${PROXY_URL:?Please set PROXY_URL!}"

TARGET="metrics/${PART}/${METRIC}/${SYMBOL}"
mkdir -p "$TARGET"

# Hilfsfunktion: Datum → ms-Timestamp
to_ms() {
  date -d "$1" +%s000
}

# Download-Loop
cur="$START"
while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END +1 day")" ]]; do
  FILE="${TARGET}/${SYMBOL}_${cur}.json"
  if [ -f "$FILE" ]; then
    echo "✔️ Skipping existing $FILE"
  else
    echo "→ Downloading $METRIC | $PART | $SYMBOL @ $cur"
    case "$METRIC" in
      open_interest)
        URL="${PROXY_URL}/fapi/v1/openInterestHist?symbol=${SYMBOL}&period=1d&startTime=$(to_ms "$cur")"
        ;;
      funding_rate)
        URL="${PROXY_URL}/fapi/v1/fundingRate?symbol=${SYMBOL}&startTime=$(to_ms "$cur")&limit=1000"
        ;;
      liquidity)
        URL="${PROXY_URL}/fapi/v1/depth?symbol=${SYMBOL}&limit=500"
        ;;
      *)
        echo "❌ Unknown metric: $METRIC"
        exit 2
        ;;
    esac

    # Download + Post-Processing
    if curl --fail -s "$URL" -o "$FILE.tmp"; then
      if [[ "$METRIC" == "funding_rate" ]] && grep -q '^\s*\[\s*\]' "$FILE.tmp"; then
        echo '[{"fundingRate":"0","fundingTime":0}]' > "$FILE"
        echo "   🚑 Imputed empty funding_rate for $cur"
      elif [[ "$METRIC" == "liquidity" ]]; then
        # Stelle sicher, dass wir bids/asks haben
        raw=$(jq -e 'type=="object" and has("bids") and has("asks")' "$FILE.tmp" >/dev/null 2>&1 && cat "$FILE.tmp" || echo '{"bids":[],"asks":[]}')
        echo "$raw" | jq --arg symbol "$SYMBOL" --arg date "$cur" '
          { symbol:$symbol,
            date:$date,
            mid:(if .bids|length>0 and .asks|length>0 then ((.bids[0][0]|tonumber + .asks[0][0]|tonumber)/2) else null end),
            spread:(if .bids|length>0 and .asks|length>0 then ((.asks[0][0]|tonumber - .bids[0][0]|tonumber)) else null end),
            bid_depth:(if .bids|length>0 then ([.bids[][1]|tonumber] | add) else null end),
            ask_depth:(if .asks|length>0 then ([.asks[][1]|tonumber] | add) else null end)
          }' > "$FILE"
        echo "   ✅ Saved processed liquidity JSON"
      else
        mv "$FILE.tmp" "$FILE"
        echo "   ✅ Saved $FILE"
      fi
    else
      echo "⚠️ Download failed: $URL"
      rm -f "$FILE.tmp"
    fi
  fi

  # nächster Tag
  cur=$(date -I -d "$cur + 1 day")
done
