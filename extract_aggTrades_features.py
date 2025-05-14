#!/usr/bin/env python3
import os, glob, argparse
import pandas as pd
import numpy as np

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    # Index = UTC-Timestamp
    df['timestamp'] = pd.to_datetime(df['transact_time'], unit='ms', utc=True)
    df = df.set_index('timestamp').sort_index()

    # Filter nur Trades im gewählten Fenster
    # (start/end kommen schon im Aufrufpunkt gefiltert)

    # Buy/Sell Mengen
    df['buy_qty']  = np.where(df['is_buyer_maker']==False, df['quantity'], 0.0)
    df['sell_qty'] = np.where(df['is_buyer_maker']==True,  df['quantity'], 0.0)

    # Tages-Resample
    daily = pd.DataFrame({
        'total_volume': df['quantity'].resample('1D').sum(),
        'buy_volume':   df['buy_qty'].resample('1D').sum(),
        'sell_volume':  df['sell_qty'].resample('1D').sum(),
    })

    # Imbalance
    daily['imbalance'] = (
        (daily['buy_volume'] - daily['sell_volume'])
        / daily['total_volume']
    ).fillna(0.0)

    # 1h/4h Peaks
    hourly = df['quantity'].resample('1H').sum().resample('1D').max()
    fourh  = df['quantity'].resample('4H').sum().resample('1D').max()
    daily['max_vol_1h'] = hourly
    daily['max_vol_4h'] = fourh

    # Avg Trades pro Minute
    tpm = df['quantity'].resample('1T').count().resample('1D').mean()
    daily['avg_trades_per_min'] = tpm

    return daily

def main(input_dir, output_file, start_date, end_date):
    files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
    if not files:
        raise FileNotFoundError(f"No CSVs in {input_dir}")

    df_list = []
    for fn in files:
        try:
            tmp = pd.read_csv(
                fn,
                header=0,                        # erste Zeile = Header
                usecols=['quantity','transact_time','is_buyer_maker'],
                dtype={'quantity': float,
                       'is_buyer_maker': bool}
            )
            df_list.append(tmp)
        except Exception as e:
            print(f"[ERROR] Reading {fn}: {e}")

    if not df_list:
        print("No valid data; exiting.")
        return

    df = pd.concat(df_list, ignore_index=True)

    # Filter nach Zeitfenster (inkl. Overlap-Tag)
    df['timestamp'] = pd.to_datetime(df['transact_time'], unit='ms', utc=True)
    mask = (df['timestamp'] >= pd.to_datetime(start_date))
    mask &= (df['timestamp'] <  pd.to_datetime(end_date) + pd.Timedelta(days=1))
    df = df.loc[mask]

    if df.empty:
        print(f"[WARNING] No trades between {start_date} and {end_date}.")
        return

    features = extract_features(df)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    features.to_parquet(output_file, index=True)
    print(f"[INFO] Wrote features to {output_file}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Extract aggTrades Features")
    p.add_argument('--input-dir',   required=True)
    p.add_argument('--output-file', required=True)
    p.add_argument('--start-date',  required=True)
    p.add_argument('--end-date',    required=True)
    args = p.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date)
