#!/usr/bin/env python3
"""
extract_aggTrades_features.py

Lädt alle CSVs (inkl. 1-Tag-Overlap) eines Symbols und
berechnet täglich:
 - total_volume, buy_volume, sell_volume
 - imbalance (guard zero division)
 - max_vol_1h, max_vol_4h
 - avg_trades_per_min

Debug-Logging, korrekte usecols und fehlerfreie End-Timestamp-Berechnung.
"""

import argparse
import logging
import os
import glob
import pandas as pd
import numpy as np

# ----------------------------------------
# Logging Setup
# ----------------------------------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def cap_outliers(df, cols, low=0.01, high=0.99):
    for c in cols:
        lo, hi = df[c].quantile(low), df[c].quantile(high)
        df[c] = df[c].clip(lo, hi)
    return df

def compute_agg_features(df, normalize=False):
    # 1) Timestamp → UTC-Datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp').sort_index()
    logging.info(f"  → Data spans from {df.index.min()} to {df.index.max()} ({len(df)} rows)")

    # 2) Buyer/Seller Volume
    df['buy_volume']  = df['quantity'].mask(df['isBuyerMaker'], 0.0)
    df['sell_volume'] = df['quantity'].mask(~df['isBuyerMaker'], 0.0)

    # 3) Tages-Resample
    daily = df.resample('1D').agg(
        total_volume    = ('quantity','sum'),
        buy_volume      = ('buy_volume','sum'),
        sell_volume     = ('sell_volume','sum'),
        max_vol_1h      = ('quantity', lambda x: x.resample('1h').sum().max()),
        max_vol_4h      = ('quantity', lambda x: x.resample('4h').sum().max()),
        avg_trades_per_min = ('quantity', lambda x: x.resample('min').count().mean())
    )
    logging.info(f"  → After daily resample: {daily.shape[0]} days from {daily.index.min().date()} to {daily.index.max().date()}")
    logging.info(f"  → Sample:\n{daily.head(2)}")

    # 4) Imbalance (guard zero division)
    mask = daily['total_volume'] > 0
    daily['imbalance'] = np.where(
        mask,
        (daily['buy_volume'] - daily['sell_volume']) / daily['total_volume'],
        0.0
    )

    # 5) Clip Outliers
    cols = ['total_volume','buy_volume','sell_volume','max_vol_1h','max_vol_4h','avg_trades_per_min']
    daily = cap_outliers(daily, cols)

    # 6) Optional Z-Score-Normierung
    if normalize:
        daily[cols] = daily[cols].apply(lambda s: (s - s.mean())/s.std(ddof=0))

    return daily.reset_index().rename(columns={'timestamp':'date'})

def main(input_dir, output_file, start_date, end_date, normalize=False):
    logging.info(f"--> Reading CSVs from {input_dir}")
    files = sorted(glob.glob(os.path.join(input_dir,'*.csv')))
    logging.info(f"Found {len(files)} files: {files[:1]} … {files[-1:]}")
    if not files:
        logging.warning("No CSVs found; exiting.")
        return

    df_list = []
    for f in files:
        try:
            tmp = pd.read_csv(
                f,
                header=None,
                usecols=[4,2,5],                         # 4=timestamp,2=quantity,5=isBuyerMaker
                names=['timestamp','quantity','isBuyerMaker'],
                dtype={'timestamp':'Int64','quantity':'float64','isBuyerMaker':'boolean'}
            )
            df_list.append(tmp)
        except Exception as e:
            logging.error(f"Error reading {f}: {e}")

    if not df_list:
        logging.warning("No valid data to process; exiting.")
        return

    df_all = pd.concat(df_list, ignore_index=True).drop_duplicates()
    logging.info(f"Combined DF: {df_all.shape}")

    # Filter bis einschließlich End-Date (Overlap erhalten)
    end_ts = int((
        pd.to_datetime(end_date).tz_localize('UTC')
        + pd.Timedelta(days=1)
        - pd.Timedelta(milliseconds=1)
    ).timestamp() * 1000)
    df_all = df_all[df_all['timestamp'] <= end_ts]
    logging.info(f"After ≤ end_date ({end_date}): {df_all.shape}")

    # Features berechnen
    feats = compute_agg_features(df_all, normalize)
    logging.info(f"Features before dropping overlap: {feats.shape}")

    # Drop Overlap-Tag (< start_date)
    feats['date'] = pd.to_datetime(feats['date']).dt.date
    sd = pd.to_datetime(start_date).date()
    feats = feats[feats['date'] >= sd].drop(columns=['date'])
    logging.info(f"Features after dropping < {start_date}: {feats.shape}")
    logging.info(f"Sample output:\n{feats.head(2)}")

    # Parquet schreiben
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    feats.to_parquet(output_file, index=False, compression='snappy')
    logging.info(f"Wrote features to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract aggTrades Features per symbol")
    parser.add_argument('--input-dir',   required=True, help="Ordner mit Roh-CSVs")
    parser.add_argument('--output-file', required=True, help="Ziel-Parquet-Datei")
    parser.add_argument('--start-date',  required=True, help="YYYY-MM-DD")
    parser.add_argument('--end-date',    required=True, help="YYYY-MM-DD")
    parser.add_argument('--normalize',   action='store_true', help="Z-Score-Normierung aktivieren")
    args = parser.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date, args.normalize)
