#!/usr/bin/env python3
"""
extract_aggTrades_features.py
… unveränderte Dokumentation …
"""

import argparse
import os
import glob
import pandas as pd

def compute_agg_features(df: pd.DataFrame) -> pd.DataFrame:
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    df['buy_volume']  = df['quantity'].where(~df['isBuyerMaker'], 0.0)
    df['sell_volume'] = df['quantity'].where(df['isBuyerMaker'],  0.0)

    daily = df.resample('1D').agg({
        'quantity':    'sum',
        'buy_volume':  'sum',
        'sell_volume': 'sum'
    }).rename(columns={'quantity': 'total_volume'})

    daily['imbalance'] = (
        (daily['buy_volume'] - daily['sell_volume']) / daily['total_volume']
    )

    vol_1h = df['quantity'].resample('1H').sum()
    vol_4h = df['quantity'].resample('4H').sum()
    daily['max_vol_1h'] = vol_1h.resample('1D').max()
    daily['max_vol_4h'] = vol_4h.resample('1D').max()

    trades_per_min = df['quantity'].resample('1T').count()
    daily['avg_trades_per_min'] = trades_per_min.resample('1D').mean()

    return daily.reset_index()

def main(input_dir: str, output_file: str, start_date: str, end_date: str):
    pattern = os.path.join(input_dir, '*.csv')
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[WARN] Keine CSVs in {input_dir} – überspringe.")
        return

    # Nur die drei Spalten per Index lesen (7-col CSVs)
    df_list = []
    for f in files:
        tmp = pd.read_csv(
            f,
            header=None,
            usecols=[5, 2, 6],                # timestamp, quantity, isBuyerMaker
            dtype={5: 'Int64', 2: 'float', 6: 'bool'},
        )
        tmp.columns = ['timestamp', 'quantity', 'isBuyerMaker']
        df_list.append(tmp)

    df = pd.concat(df_list, ignore_index=True)
    df = df.drop_duplicates()

    # Zeitfenster filtern
    start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_ts = int((pd.to_datetime(end_date) + pd.Timedelta(days=1)
                  - pd.Timedelta(milliseconds=1)).timestamp() * 1000)
    df = df[df['timestamp'].between(start_ts, end_ts)]
    if df.empty:
        print(f"[WARN] Keine Trades im Zeitraum {start_date}–{end_date}.")
        return

    # Features berechnen & schreiben
    features = compute_agg_features(df)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    features.to_parquet(output_file, index=False)
    print(f"[OK] Wrote features to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract aggTrades Features per symbol"
    )
    parser.add_argument('--input-dir',   required=True, help="CSV-Ordner")
    parser.add_argument('--output-file', required=True, help="Ziel-Parquet")
    parser.add_argument('--start-date',  required=True, help="YYYY-MM-DD")
    parser.add_argument('--end-date',    required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(
        input_dir   = args.input_dir,
        output_file = args.output_file,
        start_date  = args.start_date,
        end_date    = args.end_date
    )
