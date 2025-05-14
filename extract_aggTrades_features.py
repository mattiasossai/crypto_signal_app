#!/usr/bin/env python3
import argparse
import os
import glob
import logging
import pandas as pd

def main(input_dir: str, output_file: str, start_date: str, end_date: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 1) Alle CSV-Dateien einlesen
    pattern = os.path.join(input_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        logging.error(f"No CSV files found in {input_dir}")
        return

    df_list = []
    for fn in files:
        try:
            tmp = pd.read_csv(
                fn,
                names=["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"],
                header=0,                # Kopfzeile manuell einlesen
                dtype={
                    "agg_trade_id": int,
                    "price": float,
                    "quantity": float,
                    "first_trade_id": int,
                    "last_trade_id": int,
                    "transact_time": int,
                    "is_buyer_maker": bool
                },
                usecols=["quantity", "transact_time", "is_buyer_maker"],
                na_values=["", "NA"],
                engine="c"
            )
        except Exception as e:
            logging.error(f"Error reading {fn}: {e}")
            continue

        # Timestamp-Spalte
        tmp["timestamp"] = pd.to_datetime(tmp["transact_time"], unit="ms", utc=True)
        tmp = tmp.set_index("timestamp").sort_index()
        df_list.append(tmp[["quantity", "is_buyer_maker"]])

    if not df_list:
        logging.error("No valid data after reading; exit.")
        return

    df = pd.concat(df_list)
    logging.info(f"Combined DataFrame shape: {df.shape}")

    # 2) In Epoch-Filter
    sd = pd.to_datetime(start_date, utc=True)
    ed = pd.to_datetime(end_date,   utc=True) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    df = df.loc[(df.index >= sd) & (df.index <= ed)]
    if df.empty:
        logging.warning(f"No trades between {start_date} and {end_date}.")
        return
    logging.info(f"After filtering: {df.shape}")

    # 3) Buy/Sell-Quantities als eigene Spalten
    df["buy_quantity"]  = df["quantity"].where(~df["is_buyer_maker"], 0.0)
    df["sell_quantity"] = df["quantity"].where( df["is_buyer_maker"], 0.0)

    # 4) Tages-Aggregation
    daily = df.resample("1D").agg(
        total_volume      = ("quantity",      "sum"),
        buy_volume        = ("buy_quantity",  "sum"),
        sell_volume       = ("sell_quantity", "sum"),
        max_vol_1h        = ("quantity",      lambda x: x.resample("1h").sum().max()),
        max_vol_4h        = ("quantity",      lambda x: x.resample("4h").sum().max()),
        avg_trades_per_min= ("quantity",      lambda x: x.resample("1min").count().mean()),
    )
    # 5) Imbalance
    daily["imbalance"] = (daily["buy_volume"] - daily["sell_volume"]) / daily["total_volume"]

    # 6) Abspeichern
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    daily.to_parquet(output_file, index=True, compression="snappy")
    logging.info(f"Wrote features to {output_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract aggTrades features per day")
    p.add_argument("--input-dir",   required=True, help="Folder with CSVs")
    p.add_argument("--output-file", required=True, help="Path for output Parquet")
    p.add_argument("--start-date",  required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date",    required=True, help="YYYY-MM-DD")
    args = p.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date)
