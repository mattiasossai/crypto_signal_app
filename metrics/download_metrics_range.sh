#!/usr/bin/env bash
set -euo pipefail

# args: METRIC START_DATE END_DATE PART_NAME SYMBOL
if [ $# -ne 5 ]; then
  echo "Usage: $0 METRIC START_DATE END_DATE PART_NAME SYMBOL"
  exit 1
fi

METRIC="$1"        # open_interest | funding_rate | liquidity
START="$2"         # YYYY-MM-DD (inklusive)
END="$3"           # YYYY-MM-DD (inklusive)
PART="$4"          # part1 | part2
SYMBOL="$5"        # z.B. SOLUSDT

: "${PROXY_URL:?Please set PROXY_URL!}"

# Root-Verzeichnis im Repo
TARGET="metrics/${PART}/${METRIC}/${SYMBOL}"
mkdir -p "$TARGET"

# Helper: Datum → ms-Timestamp
to_ms() { date -d "$1" +%s000; }

# Basis-URL für Binance Futures
BASE_API="https://fapi.binance.com"

cur="$START"
# inclusive END, daher +1 day in Vergleich
while [[ "$(date -I -d "$cur")" < "$(date -I -d "$END + 1 day")" ]]; do
  FILE="${TARGET}/${SYMBOL}_${cur}.json"
  if [ -f "$FILE" ]; then
    echo "✔️ Skipping existing $FILE"
  else
    echo "→ Downloading $METRIC | $PART | $SYMBOL @ $cur"

    # Wähle den tatsächlichen Binance-URL
    case "$METRIC" in
      open_interest)
        NEXT=$(date -I -d "$cur + 1 day")
        TARGET_URL="${BASE_API}/futures/data/openInterestHist?symbol=${SYMBOL}&period=1d&startTime=$(to_ms "$cur")&endTime=$(to_ms "$NEXT")&limit=1000"
        ;;
      funding_rate)
        NEXT=$(date -I -d "$cur + 1 day")
        TARGET_URL="${BASE_API}/fapi/v1/fundingRate?symbol=${SYMBOL}&startTime=$(to_ms "$cur")&endTime=$(to_ms "$NEXT")&limit=1000"
        ;;
      liquidity)
        TARGET_URL="${BASE_API}/fapi/v1/depth?symbol=${SYMBOL}&limit=500"
        ;;
      *)
        echo "❌ Unknown metric: $METRIC"
        exit 2
        ;;
    esac

    # Download via Proxy → Query-Param url=… URL-encoded
    if curl -sSf -G "${PROXY_URL}/proxy" \
          --data-urlencode "url=${TARGET_URL}" \
          -o "$FILE.tmp"; then

      # Post-Processing
      if [[ "$METRIC" == "funding_rate" ]]; then
        # Leere Arrays impute
        if jq -e 'type=="array" and length==0' "$FILE.tmp" > /dev/null; then
          echo '[{"fundingRate":"0","fundingTime":0}]' > "$FILE"
          echo "   🚑 Imputed empty funding_rate for $cur"
        else
          mv "$FILE.tmp" "$FILE"
          echo "   ✅ Saved $FILE"
        fi

      elif [[ "$METRIC" == "liquidity" ]]; then
        # Sicherstellen, dass bids/asks da sind
        if ! jq -e 'type=="object" and has("bids") and has("asks")' "$FILE.tmp" > /dev/null; then
          raw='{"bids":[],"asks":[]}'
        else
          raw=$(cat "$FILE.tmp")
        fi
        # Transformiere zu flachem Objekt
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
        # open_interest oder andere → direkt übernehmen
        mv "$FILE.tmp" "$FILE"
        echo "   ✅ Saved $FILE"
      fi

    else
      echo "⚠️ Download failed: $TARGET_URL"
      rm -f "$FILE.tmp"
    fi
  fi

  # nächster Tag
  cur=$(date -I -d "$cur + 1 day")
done
