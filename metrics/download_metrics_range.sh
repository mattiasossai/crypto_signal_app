#!/usr/bin/env bash
set -euo pipefail

# Erwartet: WORKER_URL im Environment
if [ -z "${WORKER_URL:-}" ]; then
  echo "❌ Error: WORKER_URL is not set"
  exit 1
fi

symbol="$1"       # z.B. BTCUSDT
metric="$2"       # open_interest | funding_rates | liquidity
start_date="$3"   # YYYY-MM-DD
end_date="$4"     # YYYY-MM-DD (exklusive)
out_dir="$5"      # Zielordner

current="$start_date"
while [[ "$current" < "$end_date" ]]; do
  url="${WORKER_URL}/metrics/${symbol}/${metric}?date=${current}"
  dest="${out_dir}/${symbol}-${metric}-${current}.json"

  if curl --silent --fail "$url" -o "$dest"; then
    echo "✅ saved $dest"
  else
    echo "⏩ no data for $current — skipped"
    rm -f "$dest"
  fi

  # nächsten Tag
  current=$(date -I -d "$current + 1 day")
done
