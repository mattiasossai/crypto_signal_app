#!/usr/bin/env python3
import os, glob, argparse, logging
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
    # Prüfen, ob Header vorhanden
    with open(fn, "r") as f:
        first = f.readline().strip().split(",")
    header = (first == EXPECTED_COLS)

    return pd.read_csv(
        fn,
        header=0 if header else None,
        names=EXPECTED_COLS if not header else None,
        usecols=["quantity","transact_time","is_buyer_maker"],
        dtype={"quantity": float, "transact_time": int, "is_buyer_maker": bool},
    )

def extract_features(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    # 1) Timestamp & Index
    df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)   # ← WICHTIG!

    # 2) Auf den gewünschten Zeitraum slicen
    sd = pd.to_datetime(start).tz_localize("UTC")
    ed = pd.to_datetime(end).tz_localize("UTC") \
         + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    df = df[sd:ed]

    if df.empty:
        logging.warning("⚠️ No trades between %s and %s", start, end)
        return pd.DataFrame()

    # 3) Tages-Aggregationen
    total   = df["quantity"].resample("1D").sum()
    buy     = df.loc[~df["is_buyer_maker"], "quantity"].resample("1D").sum()
    sell    = df.loc[ df["is_buyer_maker"], "quantity"].resample("1D").sum()
    max1h   = df["quantity"].resample("1h").sum().resample("1D").max()
    max4h   = df["quantity"].resample("4h").sum().resample("1D").max()
    avg_trd = df["quantity"].resample("1T").count().resample("1D").mean()

    feats = pd.DataFrame({
        "total_volume":       total,
        "buy_volume":         buy,
        "sell_volume":        sell,
        "max_vol_1h":         max1h,
        "max_vol_4h":         max4h,
        "avg_trades_per_min": avg_trd,
    })
    feats["imbalance"] = (feats["buy_volume"] - feats["sell_volume"]) / feats["total_volume"]
    return feats

def main(input_dir, output_file, start_date, end_date):
    logging.info("→ Reading CSVs from '%s'", input_dir)
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
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
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    feats.to_parquet(output_file, index=True, compression="snappy")
    logging.info("✅ Wrote features to %s", output_file)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",   required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--start-date",  required=True)
    p.add_argument("--end-date",    required=True)
    args = p.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date)
