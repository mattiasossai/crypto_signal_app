#!/usr/bin/env python3
"""
extract_aggTrades_features.py

Lädt alle CSVs eines Symbols (inkl. 1-Tag Overlap) und berechnet pro Tag:
 - total_volume, buy_volume, sell_volume
 - imbalance (mit Zero-Division-Guard)
 - max_vol_1h, max_vol_4h
 - avg_trades_per_min

Behalte für die erste Tages-Aggregation Trades ab Overlap-Tag,
droppe diesen Tag vor dem Parquet-Export.
"""

import argparse
import logging
import os
import glob
import pandas as pd
import numpy as np

# Logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def cap_outliers(df, cols, lower_q=0.01, upper_q=0.99):
    for c in cols:
        lo, hi = df[c].quantile(lower_q), df[c].quantile(upper_q)
        df[c] = df[c].clip(lo, hi)
    return df

def compute_agg_features(df, normalize=False):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp').sort_index()

    df['buy_volume']  = df['quantity'].mask(df['isBuyerMaker'], 0.0)
    df['sell_volume'] = df['quantity'].mask(~df['isBuyerMaker'], 0.0)

    daily = df.resample('1D').agg(
        total_volume=('quantity','sum'),
        buy_volume=('buy_volume','sum'),
        sell_volume=('sell_volume','sum'),
        max_vol_1h=('quantity', lambda x: x.resample('1h').sum().max()),
        max_vol_4h=('quantity', lambda x: x.resample('4h').sum().max()),
        avg_trades_per_min=('quantity', lambda x: x.resample('min').count().mean())
    )

    mask = daily['total_volume'] > 0
    daily['imbalance'] = np.where(mask,
        (daily['buy_volume'] - daily['sell_volume']) / daily['total_volume'], 0.0)

    cols = ['total_volume','buy_volume','sell_volume','max_vol_1h','max_vol_4h','avg_trades_per_min']
    daily = cap_outliers(daily, cols)

    if normalize:
        daily[cols] = daily[cols].apply(lambda x: (x - x.mean()) / x.std(ddof=0))

    daily = daily.reset_index().rename(columns={'timestamp':'date'})
    return daily

def main(input_dir, output_file, start_date, end_date, normalize=False):
    logging.info(f"--> Reading CSVs from {input_dir}")
    if not os.path.isdir(input_dir):
        logging.error(f"Input dir {input_dir} not found.")
        return

    files = sorted(glob.glob(os.path.join(input_dir,'*.csv')))
    logging.info(f"Found {len(files)} CSV files.")
    if files:
        logging.info(f"First: {files[0]}\nLast:  {files[-1]}")
    else:
        logging.warning("No CSVs found, skipping.")
        return

    df_list = []
    for f in files:
        try:
            tmp = pd.read_csv(
                f,
                header=None,
                usecols=[5,2,6],
                names=['timestamp','quantity','isBuyerMaker'],
                dtype={'quantity':float,'isBuyerMaker':bool}
            )
            df_list.append(tmp)
        except Exception as e:
            logging.error(f"Error reading {f}: {e}")

    if not df_list:
        logging.warning("No valid data to process.")
        return

    df_all = pd.concat(df_list, ignore_index=True).drop_duplicates()
    logging.info(f"Combined shape: {df_all.shape}")

    end_ts = int((pd.to_datetime(end_date).tz_localize('UTC')
                 + pd.Timedelta(days=1)
                 - pd.Timedelta(milliseconds=1)).timestamp()*1000)
    df_all = df_all[df_all['timestamp'] <= end_ts]
    logging.info(f"After end_date filter ({end_date}): {df_all.shape}")

    features = compute_agg_features(df_all, normalize)
    logging.info(f"Features before dropping overlap: {features.shape}")

    features['date'] = pd.to_datetime(features['date']).dt.date
    sd = pd.to_datetime(start_date).date()
    features = features[features['date'] >= sd].drop(columns=['date'])
    logging.info(f"Features after dropping overlap <= {start_date}: {features.shape}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    features.to_parquet(output_file, index=False, compression='snappy')
    logging.info(f"Wrote features to {output_file}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir',   required=True)
    p.add_argument('--output-file', required=True)
    p.add_argument('--start-date',  required=True)
    p.add_argument('--end-date',    required=True)
    p.add_argument('--normalize',   action='store_true')
    args = p.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date, args.normalize)
