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
    "agg_trade_id","price","quantity",
    "first_trade_id","last_trade_id",
    "transact_time","is_buyer_maker"
]

def read_aggtrade_csv(fn: str) -> pd.DataFrame:
    # Prüfen, ob Header vorhanden ist
    with open(fn, "r") as f:
        first = f.readline().strip().split(",")
    header = (first == EXPECTED_COLS)

    df = pd.read_csv(
        fn,
        header=0 if header else None,
        names=EXPECTED_COLS if not header else None,
        usecols=["quantity","transact_time","is_buyer_maker"],
        dtype={"quantity": float, "transact_time": int, "is_buyer_maker": bool},
    )
    return df

def extract_features(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    # zeitstempel erzeugen und als Index (UTC-aware)
    df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    # Slice-Grenzen tz-aware machen
    sd = pd.to_datetime(start).tz_localize("UTC")
    ed = pd.to_datetime(end).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    df = df[sd:ed]

    if df.empty:
        logging.warning("⚠️ No trades between %s and %s", start, end)
        return pd.DataFrame()

    # 1-Tages‐Summe Gesamt, Buy, Sell
    total = df["quantity"].resample("1D").sum()
    buy   = df.loc[~df["is_buyer_maker"], "quantity"].resample("1D").sum()
    sell  = df.loc[ df["is_buyer_maker"], "quantity"].resample("1D").sum()

    # 1h/4h‐Maxima auf Tageslevel
    max1h = df["quantity"].resample("1H").sum().resample("1D").max()
    max4h = df["quantity"].resample("4H").sum().resample("1D").max()

    # avg trades per minute pro Tag
    avg_trades = df["quantity"].resample("1T").count().resample("1D").mean()

    # DataFrame zusammenbauen
    feats = pd.DataFrame({
        "total_volume":        total,
        "buy_volume":          buy,
        "sell_volume":         sell,
        "max_vol_1h":          max1h,
        "max_vol_4h":          max4h,
        "avg_trades_per_min":  avg_trades,
    })
    feats["imbalance"] = (feats["buy_volume"] - feats["sell_volume"]) / feats["total_volume"]

    return feats

def main(input_dir: str, output_file: str, start_date: str, end_date: str):
    logging.info("→ Reading CSVs from '%s'", input_dir)
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not files:
        logging.error("❌ No CSV files found in %s", input_dir)
        return

    dfs = []
    for fn in files:
        try:
            dfs.append(read_aggtrade_csv(fn))
        except Exception as e:
            logging.error("Error reading %s: %s", fn, e)

    if not dfs:
        logging.error("❌ No valid data; exiting.")
        return

    df = pd.concat(dfs, ignore_index=True)
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
