#!/usr/bin/env python3
# extract_aggTrades_features.py

import os
import glob
import argparse
import logging
import pandas as pd

# ─── Konfiguration ────────────────────────────────────────────────────────────────
# Welche Spalten wir aus den Raw-CSVs brauchen:
COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]

# ─── Helfer zum Einlesen einer einzigen CSV ────────────────────────────────────────
def read_one_csv(fn: str) -> pd.DataFrame:
    """
    Liest eine einzelne aggTrades-CSV ein und überspringt
    automatisch die Header-Zeile, falls vorhanden.
    """
    # Prüfen, ob die erste Zeile bereits Header ist:
    with open(fn, "r") as f:
        first = f.readline().strip().split(",")
    has_header = first == COLUMNS

    try:
        df = pd.read_csv(
            fn,
            header=0 if has_header else None,
            names=COLUMNS,
            usecols=COLUMNS,
            dtype={
                "agg_trade_id": "Int64",
                "price": float,
                "quantity": float,
                "first_trade_id": "Int64",
                "last_trade_id": "Int64",
                "transact_time": "Int64",
                "is_buyer_maker": bool,
            },
            skiprows=1 if has_header else 0,
        )
        return df
    except Exception as e:
        logging.error(f"Error reading {fn}: {e}")
        return pd.DataFrame(columns=COLUMNS)

# ─── Hauptfunktion ───────────────────────────────────────────────────────────────
def main(input_dir: str, output_file: str, start_date: str, end_date: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info(f"→ Reading CSVs from '{input_dir}'")

    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not files:
        logging.error("No CSV files found.")
        return

    # Einlesen & Zusammenführen
    df_list = [read_one_csv(fn) for fn in files]
    df = pd.concat(df_list, ignore_index=True)
    if df.empty:
        logging.warning("No valid data after reading; exiting.")
        return

    # Timestamp konvertieren
    df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[["price", "quantity", "is_buyer_maker"]]

    # Filter auf den gewünschten Zeitraum (inkl. Overlap-Tag)
    start_ts = pd.to_datetime(start_date, utc=True)
    end_ts   = pd.to_datetime(end_date,   utc=True) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    mask = (df.index >= start_ts - pd.Timedelta(days=1)) & (df.index <= end_ts)
    df = df.loc[mask]
    if df.empty:
        logging.warning(f"No trades between {start_date} and {end_date}.")
        return

    # Aggregationen pro Tag
    daily = df.resample("1D").agg(
        total_volume=("quantity", "sum"),
        buy_volume=("quantity", lambda x: x[df.loc[x.index, "is_buyer_maker"]==False].sum()),
        sell_volume=("quantity", lambda x: x[df.loc[x.index, "is_buyer_maker"]==True].sum()),
        max_vol_1h=("quantity", lambda x: x.resample("1H").sum().max()),
        max_vol_4h=("quantity", lambda x: x.resample("4H").sum().max()),
        avg_trades_per_min=("quantity", lambda x: x.resample("1min").count().mean()),
    )
    daily["imbalance"] = (daily["buy_volume"] - daily["sell_volume"]) / daily["total_volume"]

    # Output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    daily.to_parquet(output_file, index=True, compression="snappy")
    logging.info(f"[INFO] Wrote features to {output_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract aggTrades features")
    p.add_argument("--input-dir",   required=True, help="Ordner mit entpackten CSVs")
    p.add_argument("--output-file", required=True, help="Ziel-Parquet-Datei")
    p.add_argument("--start-date",  required=True, help="Startdatum YYYY-MM-DD")
    p.add_argument("--end-date",    required=True, help="Enddatum   YYYY-MM-DD")
    args = p.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date)
