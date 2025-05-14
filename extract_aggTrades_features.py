#!/usr/bin/env python3
"""
extract_aggTrades_features.py

Lädt alle CSVs eines Symbols (inkl. Overlap-Tag) und berechnet pro Tag:
  - total_volume
  - buy_volume, sell_volume
  - imbalance (geclippt auf [-0.9,0.9])
  - max_vol_1h, max_vol_4h
  - avg_trades_per_min (Median Trades/Min)

Schreibt Snappy-komprimiertes Parquet für ML.
"""

import argparse
import glob
import logging
import os
import pandas as pd

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def compute_agg_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("timestamp").sort_index()

    # buy vs. sell
    df["buy_volume"]  = df["quantity"].where(~df["isBuyerMaker"], 0.0)
    df["sell_volume"] = df["quantity"].where( df["isBuyerMaker"], 0.0)

    # Tages-Resample
    daily = df.resample("1D").agg(
        total_volume = ("quantity", "sum"),
        buy_volume   = ("buy_volume", "sum"),
        sell_volume  = ("sell_volume", "sum"),
    )

    # imbalance & clip
    daily["imbalance"] = (
        (daily["buy_volume"] - daily["sell_volume"])
        / daily["total_volume"]
    ).clip(-0.9, 0.9)

    # max_vol_1h / max_vol_4h
    vol1h = df["quantity"].resample("1H").sum()
    vol4h = df["quantity"].resample("4H").sum()
    daily["max_vol_1h"] = vol1h.resample("1D").max()
    daily["max_vol_4h"] = vol4h.resample("1D").max()

    # avg_trades_per_min (Median)
    trades_per_min = df["quantity"].resample("1T").count()
    daily["avg_trades_per_min"] = trades_per_min.resample("1D").median()

    return daily.reset_index()

def main(input_dir: str, output_file: str, start_date: str, end_date: str):
    logging.info(f"→ Reading CSVs from {input_dir}")
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not files:
        logging.warning("No CSVs found; skipping.")
        return

    df_list = []
    for fn in files:
        try:
            tmp = pd.read_csv(
                fn,
                header=None,
                usecols=[4,2,5],  # timestamp(ms), quantity, isBuyerMaker
                names=["timestamp","quantity","isBuyerMaker"],
                dtype={"quantity":"float64","isBuyerMaker":"int8"},
            )
            # bool cast
            tmp["isBuyerMaker"] = tmp["isBuyerMaker"] == 1
            # to datetime64[ns] without tz
            tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], unit="ms").dt.tz_localize(None)
            df_list.append(tmp)
        except Exception as e:
            logging.error(f"Error reading {fn}: {e}")

    if not df_list:
        logging.error("No valid data after reading; exit.")
        return

    df = pd.concat(df_list, ignore_index=True).drop_duplicates()
    logging.info(f"Combined DataFrame shape: {df.shape}")
    logging.info(f"Dtype of timestamp: {df['timestamp'].dtype}")
    logging.info(f"Sample timestamps:\n{df['timestamp'].head()}")

    # Filter including overlap
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    logging.info(f"Filtering between {sd - pd.Timedelta(days=1)} and {ed}")
    mask = (df["timestamp"] >= (sd - pd.Timedelta(days=1))) & (df["timestamp"] <= ed)
    df = df.loc[mask]
    logging.info(f"After filtering: {df.shape}")
    if df.empty:
        logging.warning(f"No trades between {start_date} and {end_date}.")
        return

    # Compute features
    features = compute_agg_features(df)
    before = features.shape[0]
    # drop overlap day
    features = features[features["timestamp"].dt.normalize() > sd.normalize()]
    after = features.shape[0]
    logging.info(f"Rows before/after dropping overlap: {before}/{after}")

    # write parquet
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    features.to_parquet(output_file, index=False, compression="snappy")
    logging.info(f"[OK] Wrote {after} rows to {output_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Extract aggTrades features per symbol with 1-day overlap"
    )
    p.add_argument("--input-dir",   required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--start-date",  required=True)
    p.add_argument("--end-date",    required=True)
    args = p.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date)
