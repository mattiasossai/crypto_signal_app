#!/usr/bin/env python3
"""
extract_aggTrades_features.py
...

Debug-Logging hinzugefügt, um Resample-Fehler zu finden.
"""

import argparse, logging, os, glob
import pandas as pd, numpy as np

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

    # 2) Compute buyer/seller volumes
    df['buy_volume']  = df['quantity'].mask(df['isBuyerMaker'], 0.0)
    df['sell_volume'] = df['quantity'].mask(~df['isBuyerMaker'], 0.0)

    # 3) Daily resample
    daily = df.resample('1D').agg(
        total_volume=('quantity','sum'),
        buy_volume=('buy_volume','sum'),
        sell_volume=('sell_volume','sum'),
        max_vol_1h=('quantity', lambda x: x.resample('1h').sum().max()),
        max_vol_4h=('quantity', lambda x: x.resample('4h').sum().max()),
        avg_trades_per_min=('quantity', lambda x: x.resample('min').count().mean())
    )
    logging.info(f"  → After daily resample: {daily.shape[0]} days, index from {daily.index.min().date()} to {daily.index.max().date()}")
    logging.info(f"  → Sample daily:\n{daily.head(3)}")

    # 4) Imbalance (guard zero-div)
    mask = daily['total_volume'] > 0
    daily['imbalance'] = np.where(
        mask,
        (daily['buy_volume'] - daily['sell_volume']) / daily['total_volume'],
        0.0
    )

    # 5) Outlier-Clipping
    cols = ['total_volume','buy_volume','sell_volume','max_vol_1h','max_vol_4h','avg_trades_per_min']
    daily = cap_outliers(daily, cols)

    if normalize:
        daily[cols] = daily[cols].apply(lambda s: (s - s.mean())/s.std(ddof=0))

    daily = daily.reset_index().rename(columns={'timestamp':'date'})
    return daily

def main(input_dir, output_file, start_date, end_date, normalize=False):
    logging.info(f"--> Reading CSVs from {input_dir}")
    files = sorted(glob.glob(os.path.join(input_dir,'*.csv')))
    logging.info(f"Found {len(files)} files: {files[:1]} ... {files[-1:]}")
    if not files:
        logging.warning("No CSVs found, exiting.")
        return

    df_list=[]
    for f in files:
        try:
            tmp = pd.read_csv(f, header=None, usecols=[5,2,6],
                              names=['timestamp','quantity','isBuyerMaker'],
                              dtype={'quantity':float,'isBuyerMaker':bool})
            df_list.append(tmp)
        except Exception as e:
            logging.error(f"Error reading {f}: {e}")

    if not df_list:
        logging.warning("No valid data after read, exiting.")
        return

    df_all = pd.concat(df_list, ignore_index=True).drop_duplicates()
    logging.info(f"Combined DF shape: {df_all.shape}")

    # Filter by end_date only (keep overlap)
    end_ts = int((pd.to_datetime(end_date).tz_localize('UTC')
                 + pd.Timedelta(days=1)
                 - pd.Timedelta(milliseconds=1)).timestamp()*1000)
    df_all = df_all[df_all['timestamp'] <= end_ts]
    logging.info(f"After ≤ end_date ({end_date}): {df_all.shape}")

    # Compute features
    features = compute_agg_features(df_all, normalize)
    logging.info(f"Features before dropping overlap: {features.shape}")

    # Drop overlap day < start_date
    features['date'] = pd.to_datetime(features['date']).dt.date
    sd = pd.to_datetime(start_date).date()
    features = features[features['date'] >= sd].drop(columns=['date'])
    logging.info(f"Features after dropping < {start_date}: {features.shape}")
    logging.info(f"Sample features:\n{features.head(3)}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    features.to_parquet(output_file, index=False, compression='snappy')
    logging.info(f"Wrote {output_file}")

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',   required=True)
    p.add_argument('--output-file', required=True)
    p.add_argument('--start-date',  required=True)
    p.add_argument('--end-date',    required=True)
    p.add_argument('--normalize',   action='store_true')
    args=p.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date, args.normalize)
