#!/usr/bin/env python3
"""
extract_aggTrades_features.py

Wie gehabt,
aber mit korrigiertem dtype in read_csv.
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
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp').sort_index()
    logging.info(f"  → Data spans {df.index.min()} … {df.index.max()} ({len(df)})")

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
    daily['imbalance'] = np.where(
        mask,
        (daily['buy_volume'] - daily['sell_volume'])/daily['total_volume'],
        0.0
    )
    cols = ['total_volume','buy_volume','sell_volume','max_vol_1h','max_vol_4h','avg_trades_per_min']
    daily = cap_outliers(daily, cols)
    if normalize:
        daily[cols] = daily[cols].apply(lambda s: (s-s.mean())/s.std(ddof=0))
    return daily.reset_index().rename(columns={'timestamp':'date'})

def main(input_dir, output_file, start_date, end_date, normalize=False):
    logging.info(f"Reading CSVs from {input_dir}")
    files = sorted(glob.glob(os.path.join(input_dir,'*.csv')))
    logging.info(f"Found {len(files)} files: {files[:1]} … {files[-1:]}")
    if not files: return logging.warning("No CSVs found.")

    df_list=[]
    for f in files:
        try:
            tmp = pd.read_csv(
                f,
                header=None,
                usecols=[5,2,6],
                names=['timestamp','quantity','isBuyerMaker'],
                dtype={5:'Int64', 2:'float64', 6:'boolean'}
            )
            df_list.append(tmp)
        except Exception as e:
            logging.error(f"Error reading {f}: {e}")

    if not df_list:
        return logging.warning("No valid data to process.")

    df_all = pd.concat(df_list, ignore_index=True).drop_duplicates()
    logging.info(f"Combined DF: {df_all.shape}")

    end_ts = int((pd.to_datetime(end_date).tz_localize('UTC')
                 + pd.Timedelta(days=1) - pd.Timedelta(ms=1)
                ).timestamp()*1000)
    df_all = df_all[df_all['timestamp'] <= end_ts]
    logging.info(f"After ≤ end_date {end_date}: {df_all.shape}")

    feats = compute_agg_features(df_all, normalize)
    logging.info(f"Before drop overlap: {feats.shape}")

    feats['date'] = pd.to_datetime(feats['date']).dt.date
    sd = pd.to_datetime(start_date).date()
    feats = feats[feats['date'] >= sd].drop(columns=['date'])
    logging.info(f"After drop < {start_date}: {feats.shape}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    feats.to_parquet(output_file, index=False, compression='snappy')
    logging.info(f"Wrote {output_file}")

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir',   required=True)
    p.add_argument('--output-file', required=True)
    p.add_argument('--start-date',  required=True)
    p.add_argument('--end-date',    required=True)
    p.add_argument('--normalize',   action='store_true')
    args = p.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date, args.normalize)
