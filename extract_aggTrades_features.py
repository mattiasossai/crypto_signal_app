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

# Symbol-spezifisches Inception-Datum (erster Handelstag)
INCEPTION = {
    "BTCUSDT": "2019-12-31",
    "ETHUSDT": "2019-12-31",
    "BNBUSDT": "2020-02-10",
    "SOLUSDT": "2020-09-14",
    "XRPUSDT": "2020-01-06",
    "ENAUSDT": "2024-04-02",
}

def process_one_file(fn: str) -> dict | None:
    # Header-Erkennung
    with open(fn, "r") as f:
        tokens = f.readline().strip().split(",")
    header = tokens[:7] == [
        "agg_trade_id", "price", "quantity",
        "first_trade_id", "last_trade_id",
        "transact_time", "is_buyer_maker"
    ]

    df = pd.read_csv(
        fn,
        header=0 if header else None,
        names=tokens if header else [
            "agg_trade_id","price","quantity",
            "first_trade_id","last_trade_id",
            "transact_time","is_buyer_maker"
        ],
        usecols=["quantity", "transact_time", "is_buyer_maker"],
        dtype={"quantity": float, "transact_time": int, "is_buyer_maker": bool},
    )

    df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    day = os.path.basename(fn).split("-aggTrades-")[1][:10]
    sd = pd.to_datetime(day).tz_localize("UTC")
    ed = sd + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    df_day = df[sd:ed]

    if df_day.empty:
        logging.warning("Empty data for %s", day)
        return None

    total   = df_day["quantity"].sum()
    buy     = df_day.loc[~df_day["is_buyer_maker"], "quantity"].sum()
    sell    = df_day.loc[ df_day["is_buyer_maker"], "quantity"].sum()

    return {
        "date": day,
        "total_volume": total,
        "buy_volume":    buy,
        "sell_volume":   sell,
        "max_vol_1h":    df_day["quantity"].resample("1h").sum().max(),
        "max_vol_4h":    df_day["quantity"].resample("4h").sum().max(),
        "avg_trades_per_min": df_day["quantity"].resample("1min").count().mean(),
        "imbalance":     (buy - sell) / total if total > 0 else 0,
    }

def main(input_dir, output_file, start_date, end_date):
    logging.info("→ Scanning CSVs in '%s'", input_dir)
    sym    = os.path.basename(input_dir)
    user_sd = pd.to_datetime(start_date).tz_localize("UTC")
    user_ed = pd.to_datetime(end_date).tz_localize("UTC")
    inc     = pd.to_datetime(INCEPTION[sym]).tz_localize("UTC")
    real_sd = max(user_sd, inc)

    # 1-Day-Overlap für die Berechnung
    sd = real_sd - pd.Timedelta(days=1)
    ed = user_ed

    all_csv = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    files   = [
        fn for fn in all_csv
        if sd <= pd.to_datetime(os.path.basename(fn).split("-aggTrades-")[1][:10]).tz_localize("UTC") <= ed
    ]

    if not files:
        logging.error("❌ No CSVs in range → exiting.")
        return

    rows = [r for fn in files if (r := process_one_file(fn))]

    if not rows:
        logging.error("❌ All days empty → exiting.")
        return

    df = pd.DataFrame(rows).set_index("date").sort_index()

    # Overlap-Tag verwerfen
    df = df.loc[df.index >= real_sd.strftime("%Y-%m-%d")]

    # auf volle Kalender-Tage reindexen (fehlende → NaN)
    idx = pd.date_range(real_sd.normalize(), user_ed.normalize(), freq="D", tz="UTC")
    df.index = pd.to_datetime(df.index).tz_localize("UTC")
    df = df.reindex(idx)
    df.index.name = "date"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_parquet(output_file, compression="snappy")
    logging.info("✅ Wrote %d days to %s", len(df), output_file)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",   required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--start-date",  required=True)
    p.add_argument("--end-date",    required=True)
    args = p.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date)
