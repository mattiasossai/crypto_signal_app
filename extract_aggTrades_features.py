#!/usr/bin/env python3
import argparse
import os
import glob
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def extract_features(df: pd.DataFrame) -> pd.Series:
    """
    aggregiert pro Tag:
      total_volume, buy_volume, sell_volume,
      max_vol_1h, max_vol_4h, avg_trades_per_min, imbalance
    """
    # daily resample
    daily = df.resample("1D").agg(
        total_volume=("quantity", "sum"),
        buy_volume=("quantity", lambda x: x[~df.loc[x.index, "is_buyer_maker"]].sum()),
        sell_volume=("quantity", lambda x: x[df.loc[x.index, "is_buyer_maker"]].sum()),
        max_vol_1h=("quantity", lambda x: x.resample("1H").sum().max()),
        max_vol_4h=("quantity", lambda x: x.resample("4H").sum().max()),
        avg_trades_per_min=("quantity", lambda x: x.resample("1T").count().mean()),
    )
    daily["imbalance"] = (daily["buy_volume"] - daily["sell_volume"]) / daily["total_volume"].replace(0, pd.NA)
    return daily

def main(input_dir: str, output_file: str, start_date: str, end_date: str):
    # 1) CSVs sammeln
    pattern = os.path.join(input_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        logging.error(f"No files found in {input_dir}")
        return

    dfs = []
    for fn in files:
        # 2) Prüfen, ob erste Zeile Header ist
        with open(fn, "r") as f:
            first = f.readline().strip().split(",")[0]
        skip = 1 if first == "agg_trade_id" else 0

        try:
            tmp = pd.read_csv(
                fn,
                header=None,
                skiprows=skip,
                names=[
                    "agg_trade_id", "price", "quantity",
                    "first_trade_id", "last_trade_id",
                    "transact_time", "is_buyer_maker"
                ],
                dtype={
                    "agg_trade_id":   "Int64",
                    "price":          "float64",
                    "quantity":       "float64",
                    "first_trade_id": "Int64",
                    "last_trade_id":  "Int64",
                    "transact_time":  "Int64",
                    "is_buyer_maker": "boolean"
                },
                usecols=["quantity","transact_time","is_buyer_maker"],
                na_values=["", "NA"],
                engine="c"
            )
        except Exception as e:
            logging.error(f"Error reading {fn}: {e}")
            continue

        # 3) Timestamp erzeugen und index setzen
        tmp["timestamp"] = pd.to_datetime(tmp["transact_time"], unit="ms", utc=True)
        tmp = tmp.set_index("timestamp").sort_index()
        dfs.append(tmp)

    if not dfs:
        logging.warning("No valid data after reading; exiting.")
        return

    df = pd.concat(dfs, axis=0)

    # 4) Auf den gewünschten Zeitraum filtern
    sd = pd.to_datetime(start_date).tz_localize("UTC")
    ed = pd.to_datetime(end_date).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    df = df.loc[(df.index >= sd) & (df.index <= ed)]
    if df.empty:
        logging.warning(f"No trades between {start_date} and {end_date}.")
        return

    # 5) Feature-Engineering
    features = extract_features(df)

    # 6) Schreiben
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    features.to_parquet(output_file, index=True)
    logging.info(f"Wrote features to {output_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract aggTrades Features")
    p.add_argument("--input-dir",  required=True, help="Ordner mit CSVs")
    p.add_argument("--output-file", required=True, help="Ziel Parquet-Datei")
    p.add_argument("--start-date",  required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date",    required=True, help="YYYY-MM-DD")
    args = p.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date)
