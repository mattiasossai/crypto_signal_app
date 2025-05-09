#!/usr/bin/env python3
import argparse
import os
import glob
import pandas as pd

def compute_book_features(df: pd.DataFrame) -> pd.DataFrame:
    # Timestamp in datetime
    df['timestamp'] = pd.to_datetime(df['updateTime'], unit='ms')
    df.set_index('timestamp', inplace=True)
    # Spread & Mid-Price
    df['spread'] = df['askPrice'] - df['bidPrice']
    df['mid_price'] = (df['askPrice'] + df['bidPrice']) / 2
    # Tages-Statistiken
    daily = df.resample('1D').agg({
        'spread': ['mean','std','max'],
        'mid_price': ['first','last']
    })
    daily.columns = [
        'spread_mean','spread_std','spread_max',
        'mid_first','mid_last'
    ]
    # Relative Spread und Volatilität
    daily['rel_spread'] = daily['spread_mean'] / daily['mid_last']
    # Rolling-Volatilitäten (1h, 4h) auf Spread
    roll1h = df['spread'].rolling('1H').std().resample('1D').max().rename('spread_vol_1h')
    roll4h = df['spread'].rolling('4H').std().resample('1D').max().rename('spread_vol_4h')
    daily = daily.join(roll1h).join(roll4h)
    return daily.reset_index()

def main(input_dir: str, output_file: str):
    pattern = os.path.join(input_dir, '*/*.csv')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Keine Dateien gefunden in {pattern}")
    df_list = [pd.read_csv(f, usecols=['updateTime','bidPrice','askPrice']) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    features = compute_book_features(df)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    features.to_parquet(output_file, index=False)
    print(f"Features geschrieben nach {output_file}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Extract bookTicker Features")
    p.add_argument('--input-dir',  required=True, help="Ordner mit entpackten CSVs")
    p.add_argument('--output-file', required=True, help="Ziel-Parquet-Datei")
    args = p.parse_args()
    main(args.input_dir, args.output_file)
