#!/usr/bin/env python3
import os
import glob
import argparse
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

EXPECTED_COLS = [
    "agg_trade_id", "price", "quantity",
    "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker"
]

def process_one_file(fn: str) -> dict | None:
    # Header-Erkennung
    with open(fn, "r") as f:
        tokens = f.readline().strip().split(",")
    header = tokens == EXPECTED_COLS

    df = pd.read_csv(
        fn,
        header=0 if header else None,
        names=EXPECTED_COLS if not header else None,
        usecols=["quantity", "transact_time", "is_buyer_maker"],
        dtype={"quantity": float, "transact_time": int, "is_buyer_maker": bool},
    )

    # Timestamp und Tagesfenster
    df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    day_str = os.path.basename(fn).split("-aggTrades-")[1].replace(".csv", "")
    sd = pd.to_datetime(day_str).tz_localize("UTC")
    ed = sd + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    df_day = df[sd:ed]

    if df_day.empty:
        # kein einziger Trade an diesem Tag
        return {
            "date": day_str,
            "has_data": False
        }

    total   = df_day["quantity"].sum()
    buy     = df_day.loc[~df_day["is_buyer_maker"], "quantity"].sum()
    sell    = df_day.loc[ df_day["is_buyer_maker"],  "quantity"].sum()

    # robuste Segment-Summen: min_count=1 → NaN, wenn gar kein Trade
    seg_00_08 = df_day.between_time("00:00", "07:59")["quantity"].sum(min_count=1)
    seg_08_16 = df_day.between_time("08:00", "15:59")["quantity"].sum(min_count=1)
    seg_16_24 = df_day.between_time("16:00", "23:59")["quantity"].sum(min_count=1)

    return {
        "date": day_str,
        "has_data":    True,
        "total_volume":     total,
        "buy_volume":       buy,
        "sell_volume":      sell,
        "max_vol_1h":       df_day["quantity"].resample("1h").sum().max(),
        "max_vol_4h":       df_day["quantity"].resample("4h").sum().max(),
        "avg_trades_per_min": df_day["quantity"].resample("1min").count().mean(),
        "imbalance":        (buy - sell) / total if total > 0 else 0,
        "seg_00_08":        seg_00_08,
        "seg_08_16":        seg_08_16,
        "seg_16_24":        seg_16_24,
        "has_00_08":        not pd.isna(seg_00_08),
        "has_08_16":        not pd.isna(seg_08_16),
        "has_16_24":        not pd.isna(seg_16_24),
    }

def main(input_dir, output_file, start_date, end_date):
    logging.info("→ Scanning CSVs in '%s'", input_dir)
    all_csv = sorted(glob.glob(os.path.join(input_dir, "*.csv")))

    # 1-Tag-Overlap einbeziehen für Inception-Logik
    sd = pd.to_datetime(start_date) - pd.Timedelta(days=1)
    ed = pd.to_datetime(end_date)
    files = [
        fn for fn in all_csv
        if sd <= pd.to_datetime(os.path.basename(fn).split("-aggTrades-")[1][:10]) <= ed
    ]

    if not files:
        logging.error("❌ No CSV files found in range; exiting.")
        return

    rows = []
    for fn in files:
        r = process_one_file(fn)
        if r is not None:
            rows.append(r)

    # DataFrame aufbauen
    df_feats = pd.DataFrame(rows).set_index("date").sort_index()

    # Überlappungs-Starttag entfernen
    df_feats = df_feats.loc[df_feats.index >= start_date]

    # Reindex auf jeden Kalendertag → fehlende Tage als NaN
    idx = pd.date_range(
        start=pd.to_datetime(start_date).tz_localize("UTC"),
        end=pd.to_datetime(end_date).tz_localize("UTC"),
        freq="D",
        tz="UTC"
    )
    df_feats.index = pd.to_datetime(df_feats.index).tz_localize("UTC")
    df_feats = df_feats.reindex(idx)  # fill_value=None → NaN

    df_feats.index.name = "date"

    # Ergebnis schreiben
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_feats.to_parquet(output_file, compression="snappy")
    logging.info("✅ Wrote features to %s (%d days)", output_file, len(df_feats))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",   required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--start-date",  required=True)
    p.add_argument("--end-date",    required=True)
    args = p.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date)
