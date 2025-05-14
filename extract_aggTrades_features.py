#!/usr/bin/env python3
import os
import glob
import argparse
import logging
from datetime import datetime
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def read_aggtrade_csv(fn: str) -> pd.DataFrame:
    """
    Liest eine AggTrades-CSV und harmonisiert Spalten:
     - erkennt und entfernt eine Kopfzeile, falls vorhanden
     - benennt transact_time -> timestamp (in ms)
    """
    with open(fn, "r") as f:
        first = f.readline().strip().split(",")
    # erwartete Spalten:
    cols = ["agg_trade_id","price","quantity","first_trade_id","last_trade_id","transact_time","is_buyer_maker"]
    header = (first == cols)
    df = pd.read_csv(
        fn,
        names=cols if not header else None,
        header=0 if header else None,
        usecols=["quantity","transact_time","is_buyer_maker"],
        dtype={"quantity":float, "transact_time":int, "is_buyer_maker":bool},
    )
    return df

def extract_features(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Berechnet pro Tag:
      • total_volume, buy_volume, sell_volume
      • max_vol_1h, max_vol_4h
      • avg_trades_per_min
      • imbalance = (buy−sell)/total
    Filtert auf [start, end].
    """
    # timestamp index
    df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    # Tagesfilter inkl. letzter Millisekunde
    sd = pd.to_datetime(start)
    ed = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    df = df[sd:ed]
    if df.empty:
        logging.warning("⚠️ No trades between %s and %s.", start, end)
        return pd.DataFrame()

    # Resample täglich
    daily = df.resample("1D").agg(
        total_volume=("quantity", "sum"),
        buy_volume=("quantity", lambda x: x[~df.loc[x.index, "is_buyer_maker"]].sum()),
        sell_volume=("quantity", lambda x: x[df.loc[x.index, "is_buyer_maker"]].sum()),
        max_vol_1h=("quantity", lambda x: x.resample("1H").sum().max()),
        max_vol_4h=("quantity", lambda x: x.resample("4H").sum().max()),
        avg_trades_per_min=("quantity", lambda x: x.resample("1T").count().mean()),
    )
    daily["imbalance"] = (daily["buy_volume"] - daily["sell_volume"]) / daily["total_volume"]
    return daily

def main(input_dir: str, output_file: str, start_date: str, end_date: str):
    logging.info("→ Reading CSVs from '%s'", input_dir)
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not files:
        logging.error("❌ No CSV files found in %s", input_dir)
        return

    df_list = []
    for fn in files:
        try:
            df_list.append(read_aggtrade_csv(fn))
        except Exception as e:
            logging.error("Error reading %s: %s", fn, e)
    if not df_list:
        logging.error("❌ No valid data; exiting.")
        return

    df = pd.concat(df_list, ignore_index=True)
    logging.info("Combined DataFrame shape: %s", df.shape)

    feats = extract_features(df, start_date, end_date)
    if feats.empty:
        logging.info("✅ Finished (no features).")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    feats.to_parquet(output_file, index=True, compression="snappy")
    logging.info("✅ Wrote features to %s", output_file)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract aggTrades features")
    p.add_argument("--input-dir",   required=True, help="Ordner mit CSVs")
    p.add_argument("--output-file", required=True, help="Ziel-Parquet-Datei")
    p.add_argument("--start-date",  required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date",    required=True, help="YYYY-MM-DD")
    args = p.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date)
