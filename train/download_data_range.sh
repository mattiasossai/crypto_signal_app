#!/usr/bin/env bash
# Usage: download_data_range.sh <symbol> <interval> <start> <end> <outdir>
symbol=$1; interval=$2; start=$3; end=$4; outdir=$5
mkdir -p "$outdir/$symbol"
current=$start
while [[ "$current" < "$end" || "$current" == "$end" ]]; do
  url="https://data.binance.vision/data/futures/um/daily/klines/${symbol}/${interval}/${symbol}-${interval}-${current}.zip"
  out="$outdir/$symbol/${symbol}_${interval}_${current}.zip"
  echo "→ $url"
  curl --fail --silent "$url" -o "$out" || rm -f "$out"
  current=$(date -I -d "$current + 1 day")
done
