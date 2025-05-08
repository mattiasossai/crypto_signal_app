#!/usr/bin/env bash
set -euo pipefail

# Usage: $0 SYMBOL INTERVAL START_DATE END_DATE PART
if [ $# -ne 5 ]; then
  echo "Usage: $0 SYMBOL INTERVAL START_DATE END_DATE PART"
  exit 1
fi

SYMBOL="$1"        # z.B. BTCUSDT
INTERVAL="$2"      # z.B. 5m,15m,1h,4h
START="$3"         # YYYY-MM-DD (inklusiv)
END="$4"           # YYYY-MM-DD (inklusiv)
PART="$5"          # part2 (oder part1, falls Du das nochmal brauchst)

BASE_DIR="historical/${SYMBOL}/${INTERVAL}/${PART}"
mkdir -p "${BASE_DIR}"

cur="$START"
while [[ "$(date -I -d "$cur")" <= "$(date -I -d "$END")" ]]; do
  outfile="${BASE_DIR}/${SYMBOL}-${INTERVAL}-${cur}.csv"

  if [[ -f "$outfile" ]]; then
    echo "→ SKIP existing ${outfile}"
  else
    echo "→ DOWNLOAD ${SYMBOL} ${INTERVAL} @ ${cur}"
    # 1) URL für den Binanz-Vision-Daily-Download
    ZIP_URL="https://data.binance.vision/data/futures/um/daily/klines/${SYMBOL}/${INTERVAL}/${SYMBOL}-${INTERVAL}-${cur}.zip"
    # 2) zip holen, entpacken, CSV verschieben
    tmpdir=$(mktemp -d)
    if curl -sfL "$ZIP_URL" -o "${tmpdir}/dl.zip"; then
      unzip -q "${tmpdir}/dl.zip" -d "$tmpdir"
      mv "${tmpdir}/${SYMBOL}-${INTERVAL}-${cur}.csv" "$outfile"
      echo "   ✔ saved to ${outfile}"
    else
      echo "   ⚠ no data for ${cur}, skipping"
    fi
    rm -rf "$tmpdir"
  fi

  # nächster Tag
  cur=$(date -I -d "$cur +1 day")
done
