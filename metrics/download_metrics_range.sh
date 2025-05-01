#!/usr/bin/env bash
#
# usage: download_metrics_range.sh SYMBOL METRIC START_DATE END_DATE OUTDIR
#
# Beispiel:
#   download_metrics_range.sh BTCUSDT funding_rate 2021-05-01 2023-05-01 ./metrics/BTCUSDT/part1
#

SYMBOL=$1
METRIC=$2       # funding_rate | open_interest | liquidity
START=$3        # z.B. 2021-05-01
END=$4          # z.B. 2023-05-01
OUTDIR=$5

# Unterverzeichnis nach Metric
mkdir -p "$OUTDIR/$METRIC"

CURRENT="$START"
while [[ "$CURRENT" < "$END" ]]; do
  case "$METRIC" in
    funding_rate)
      # je Tag 3 Aufrufe: 0h, 8h, 16h
      for H in 0 8 16; do
        DATE_STR="${CURRENT}_${H}"
        URL="https://data.binance.vision/data/futures/um/daily/funding_rate/${SYMBOL}/${SYMBOL}_${CURRENT}_${H}.json"
        curl -sf "$URL" -o "$OUTDIR/$METRIC/${SYMBOL}_${CURRENT}_${H}.json"
      done
      ;;
    open_interest)
      URL="https://data.binance.vision/data/futures/um/daily/open_interest/${SYMBOL}/${SYMBOL}_${CURRENT}.json"
      curl -sf "$URL" -o "$OUTDIR/$METRIC/${SYMBOL}_${CURRENT}.json"
      ;;
    liquidity)
      URL="https://data.binance.vision/data/futures/um/daily/liquidity/${SYMBOL}/${SYMBOL}_${CURRENT}.json"
      curl -sf "$URL" -o "$OUTDIR/$METRIC/${SYMBOL}_${CURRENT}.json"
      ;;
    *)
      echo "Unknown metric: $METRIC" >&2
      exit 1
      ;;
  esac

  # nächster Tag
  CURRENT=$(date -I -d "$CURRENT + 1 day")
done
