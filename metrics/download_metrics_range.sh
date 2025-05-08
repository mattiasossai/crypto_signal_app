#!/usr/bin/env bash
set -euo pipefail

# args: METRIC START_DATE END_DATE PART SYMBOL
if [ $# -ne 5 ]; then
  echo "Usage: $0 METRIC START_DATE END_DATE PART SYMBOL"
  exit 1
fi

METRIC="$1"   # open_interest | funding_rate
START="$2"    # YYYY-MM-DD inclusive
END="$3"      # YYYY-MM-DD inclusive
PART="$4"     # part1 | part2
SYMBOL="$5"   # e.g. BTCUSDT

: "${PROXY_URL:?Please set PROXY_URL!}"

TARGET="metrics/${PART}/${METRIC}/${SYMBOL}"
mkdir -p "$TARGET"

to_ms(){ date -d "$1" +%s000; }

if [ "$METRIC" == "open_interest" ]; then
  # ─── Open Interest via data.binance.vision ────────────────────────────────────
  echo "🔍 Using archive from data.binance.vision for open_interest"
  cur="$START"
  end_ts=$(date -d "$END +1 day" +%s)
  while [ "$(date -d "$cur" +%s)" -lt "$end_ts" ]; do
    # Monatsweiser Download
    s=$(date -d "$cur" +%Y%m%d)
    next_month=$(date -d "$cur +1 month" +%Y-%m-01)
    e=$(date -d "$next_month -1 day" +%Y%m%d)

    zipfile="${SYMBOL}-openInterest-1d-${s}-${e}.zip"
    url="https://data.binance.vision/data/futures/um/daily/openInterest/${zipfile}"
    echo "→ Fetching $url"
    curl -sSf "$url" -o "$TARGET/$zipfile"

    # Entpacken & nach Datum splitten
    unzip -p "$TARGET/$zipfile" | python3 - "$SYMBOL" "$TARGET" << 'PYCODE'
import sys, json
import pandas as pd

symbol, target = sys.argv[1], sys.argv[2]
df = pd.read_csv(sys.stdin)
df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
for date, grp in df.groupby('date'):
    recs = grp.to_dict('records')
    fname = f"{target}/{symbol}_{date}.json"
    with open(fname, "w") as f:
        json.dump(recs, f)
PYCODE

    rm "$TARGET/$zipfile"
    # nächster Monat
    cur=$(date -d "$cur +1 month" +%Y-%m-01)
  done

else
  # ─── Funding Rate via Proxy ─────────────────────────────────────────────────
  cur="$START"
  end_ts=$(date -d "$END +1 day" +%s)
  while [ "$(date -d "$cur" +%s)" -lt "$end_ts" ]; do
    FILE="${TARGET}/${SYMBOL}_${cur}.json"
    if [ -f "$FILE" ]; then
      echo "✔️ Skipping existing $FILE"
    else
      echo "→ Downloading funding_rate @ $cur"
      next=$(date -I -d "$cur +1 day")
      binance_url="https://fapi.binance.com/fapi/v1/fundingRate?symbol=${SYMBOL}&startTime=$(to_ms "$cur")&endTime=$(to_ms "$next")&limit=1000"
      # Proxy-Tunnel
      if curl -sSf -G "${PROXY_URL}/proxy" \
               --data-urlencode "url=${binance_url}" \
               -o "$FILE.tmp"; then
        # Leere Arrays imputieren
        if jq -e 'type=="array" and length==0' "$FILE.tmp" > /dev/null; then
          echo '[{"fundingRate":"0","fundingTime":0}]' > "$FILE"
          echo "   🚑 Imputed empty funding_rate for $cur"
        else
          mv "$FILE.tmp" "$FILE"
          echo "   ✅ Saved $FILE"
        fi
      else
        echo "⚠️ Proxy-Download failed: $binance_url"
        rm -f "$FILE.tmp"
      fi
    fi
    # nächster Tag
    cur=$(date -I -d "$cur +1 day")
  done
fi
