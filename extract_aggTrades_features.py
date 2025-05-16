#!/usr/bin/env python3
import os
import glob
import argparse
import logging
import pandas as pd

from scipy.stats import skew, kurtosis

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

    # Zeitindex
    df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    # Tagesfenster extrahieren
    day_str = os.path.basename(fn).split("-aggTrades-")[1].replace(".csv", "")
    sd = pd.to_datetime(day_str).tz_localize("UTC")
    ed = sd + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    df_day = df[sd:ed]

    if df_day.empty:
        logging.warning("Empty data for %s", day_str)
        return None

    total   = df_day["quantity"].sum()
    buy     = df_day.loc[~df_day["is_buyer_maker"], "quantity"].sum()
    sell    = df_day.loc[ df_day["is_buyer_maker"],  "quantity"].sum()

    # klassische Tages-Features
    max1h   = df_day["quantity"].resample("1h").sum().max()
    max4h   = df_day["quantity"].resample("4h").sum().max()
    avg_trd = df_day["quantity"].resample("1min").count().mean()
    imbalance = (buy - sell) / total if total > 0 else 0

    # —── Neue Features: Verteilungs-Momente
    q = df_day["quantity"]
    qty_mean    = q.mean()
    qty_var     = q.var(ddof=0)
    qty_skew    = skew(q, bias=False)
    qty_kurt    = kurtosis(q, bias=False)

    # —── Neue Features: Intraday-Segmente
    seg1 = df_day.between_time("00:00", "07:59")["quantity"].sum()
    seg2 = df_day.between_time("08:00", "15:59")["quantity"].sum()
    seg3 = df_day.between_time("16:00", "23:59")["quantity"].sum()

    return {
        "date": day_str,
        "total_volume": total,
        "buy_volume": buy,
        "sell_volume": sell,
        "max_vol_1h": max1h,
        "max_vol_4h": max4h,
        "avg_trades_per_min": avg_trd,
        "imbalance": imbalance,
        # Verteilungs-Momente
        "qty_mean": qty_mean,
        "qty_var": qty_var,
        "qty_skew": qty_skew,
        "qty_kurt": qty_kurt,
        # Intraday-Segmente
        "vol_00_08": seg1,
        "vol_08_16": seg2,
        "vol_16_24": seg3,
    }

def main(input_dir, output_file, start_date, end_date):
    logging.info("→ Scanning CSVs in '%s'", input_dir)
    all_csv = sorted(glob.glob(os.path.join(input_dir, "*.csv")))

    # Überlappungstag mitnehmen
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
    if not rows:
        logging.error("❌ No data after processing; exiting.")
        return

    df_feats = pd.DataFrame(rows).set_index("date").sort_index()
    # initialen Overlap-Tag wegschneiden
    df_feats = df_feats.loc[df_feats.index >= start_date]

    # schon vorhandene Rolling-Features
    df_feats["vol_7d_ma"]   = df_feats["total_volume"].rolling(7,  min_periods=7).mean()
    df_feats["vol_7d_std"]  = df_feats["total_volume"].rolling(7,  min_periods=7).std()
    df_feats["imb_14d_ma"]  = df_feats["imbalance"].rolling(14, min_periods=14).mean()
    df_feats["imb_14d_std"] = df_feats["imbalance"].rolling(14, min_periods=14).std()
    df_feats["vol_7d_z"]    = (df_feats["total_volume"] - df_feats["vol_7d_ma"]) / df_feats["vol_7d_std"]
    df_feats["vol_7d_mom"]  = df_feats["total_volume"].pct_change(periods=7)
    df_feats = df_feats.dropna(subset=["vol_7d_std", "imb_14d_std"])

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
