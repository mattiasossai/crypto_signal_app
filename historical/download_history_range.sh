#!/usr/bin/env bash
set -euo pipefail

# args: SYMBOL INTERVAL
if [ $# -ne 2 ]; then
  echo "Usage: $0 SYMBOL INTERVAL"
  exit 1
fi

SYMBOL="$1"      # z.B. BTCUSDT
INTERVAL="$2"    # z.B. 5m, 1h, 1d

# Zeitfenster: letzter 30 Tage bis gestern
START=$(date -I -d "30 days ago")
END=$(date -I -d "yesterday")

BASE_URL="https://data.binance.vision/data/futures/um/daily/klines/${SYMBOL}/${INTERVAL}"
TARGET_DIR="historical/${SYMBOL}/${INTERVAL}"
mkdir -p "${TARGET_DIR}"

cur="${START}"
while [[ "${cur}" < "${END}" ]]; do
  FILENAME="${SYMBOL}-${INTERVAL}-${cur}.csv"
  FILEPATH="${TARGET_DIR}/${FILENAME}"

  if [[ -f "${FILEPATH}" ]]; then
    echo "→ Skipping existing ${FILEPATH}"
  else
    ZIPNAME="${SYMBOL}-${INTERVAL}-${cur}.zip"
    URL="${BASE_URL}/${ZIPNAME}"
    echo "→ Downloading ${ZIPNAME} → ${FILENAME}"

    # temporär ins /tmp-Verzeichnis laden und entpacken
    mkdir -p /tmp/cli_hist && cd /tmp/cli_hist
    if curl -sSfL "${URL}" -o "${ZIPNAME}"; then
      unzip -p "${ZIPNAME}" > "${GITHUB_WORKSPACE}/${FILEPATH}"
      echo "   ✓ Saved ${FILEPATH}"
    else
      echo "   ⚠️ Datei nicht gefunden: ${URL}"
    fi
    cd - >/dev/null
    rm -rf /tmp/cli_hist
  fi

  # nächsten Tag
  cur=$(date -I -d "${cur} +1 day")
done
