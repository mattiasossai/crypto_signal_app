#!/usr/bin/env python3
"""
extract_aggTrades_features.py

Lädt alle CSVs eines Symbols (mit einem Overlap-Tag) und berechnet pro Tag:
  - total_volume
  - buy_volume, sell_volume
  - imbalance (clipped auf [-0.9, +0.9])
  - max_vol_1h, max_vol_4h
  - avg_trades_per_min (Median statt Mean, um Ausreißer zu dämpfen)

Schreibt Parquet-Output (Snappy-komprimiert) für ML.
"""

import argparse
import glob
import logging
import os
import pandas as pd

# ─── Logging konfigurieren ─────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def compute_agg_features(df: pd.DataFrame) -> pd.DataFrame:
    # Timestamp in UTC-Datetime umwandeln
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")

    # buy vs. sell split
    df["buy_volume"]  = df["quantity"].where(~df["isBuyerMaker"], 0.0)
    df["sell_volume"] = df["quantity"].where( df["isBuyerMaker"], 0.0)

    # Tages-Aggregate
    daily = df.resample("1D").agg(
        total_volume = ("quantity", "sum"),
        buy_volume   = ("buy_volume",  "sum"),
        sell_volume  = ("sell_volume", "sum"),
    )

    # Imbalance und Clipping
    daily["imbalance"] = (
        (daily["buy_volume"] - daily["sell_volume"])
        / daily["total_volume"]
    ).clip(-0.9, 0.9)

    # max_vol_1h & 4h
    vol1h = df["quantity"].resample("1H").sum()
    vol4h = df["quantity"].resample("4H").sum()
    daily["max_vol_1h"] = vol1h.resample("1D").max()
    daily["max_vol_4h"] = vol4h.resample("1D").max()

    # avg_trades_per_min: Median Trades/Min
    trades_per_min = df["quantity"].resample("1T").count()
    daily["avg_trades_per_min"] = trades_per_min.resample("1D").median()

    return daily.reset_index()

def main(input_dir: str, output_file: str, start_date: str, end_date: str):
    logging.info(f"→ Reading CSVs from {input_dir!r}")
    pattern = os.path.join(input_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        logging.warning(f"Keine CSVs gefunden in {input_dir!r}, skip.")
        return

    # alle CSVs einlesen (ohne dtype auf timestamp festzunageln!)
    df_list = []
    for fn in files:
        try:
            tmp = pd.read_csv(
                fn,
                header=None,
                usecols=[4, 2, 5],          # 4=timestamp(ms), 2=quantity, 5=isBuyerMaker
                names=["timestamp","quantity","isBuyerMaker"],
                dtype={"quantity":"float64","isBuyerMaker":"bool"},
            )
            df_list.append(tmp)
        except Exception as e:
            logging.error(f"Error reading {fn}: {e}")

    if not df_list:
        logging.error("Kein Datensatz nach dem Einlesen, exit.")
        return

    df = pd.concat(df_list, ignore_index=True)
    df = df.drop_duplicates()

    # Filter auf Zeitfenster (inkl. Overlap-Tag)
    start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
    # end_date inclusive bis 23:59:59.999
    end_ts   = int((pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)).timestamp() * 1000)
    mask = df["timestamp"].between(start_ts, end_ts)
    df = df.loc[mask]
    if df.empty:
        logging.warning(f"No trades between {start_date} and {end_date}.")
        return

    logging.info(f"Combined DataFrame shape: {df.shape}")

    # Features berechnen
    features = compute_agg_features(df)
    # Drop overlap-Tag (alles ≤ start_date)
    before = features.shape[0]
    features = features[features["timestamp"].dt.normalize() > pd.to_datetime(start_date)]
    after  = features.shape[0]
    logging.info(f"Features before dropping overlap: {before}, after: {after}")

    # Verzeichnisse anlegen & Parquet schreiben
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    features.to_parquet(
        output_file,
        index=False,
        compression="snappy",
        engine="pyarrow"
    )
    logging.info(f"[OK] Wrote {features.shape[0]} rows to {output_file!r}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Extract aggTrades Features per symbol (inkl. 1-Tag Overlap)"
    )
    p.add_argument("--input-dir",   required=True, help="CSV-Ordner des Symbols")
    p.add_argument("--output-file", required=True, help="Ziel-Parquet")
    p.add_argument("--start-date",  required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date",    required=True, help="YYYY-MM-DD")
    args = p.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date)
