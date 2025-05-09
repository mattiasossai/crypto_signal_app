#!/usr/bin/env python3
import argparse
import os
import glob
import pandas as pd

def compute_agg_features(df: pd.DataFrame) -> pd.DataFrame:
    # Timestamp in datetime umwandeln
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    # Markiere Buyers vs Sellers (isBuyerMaker == False → taker buy)
    df['buy_volume'] = df['quantity'].where(~df['isBuyerMaker'], 0.0)
    df['sell_volume'] = df['quantity'].where(df['isBuyerMaker'], 0.0)
    # Resample auf Tagesbasis
    daily = df.resample('1D').agg({
        'quantity': 'sum',
        'buy_volume': 'sum',
        'sell_volume': 'sum'
    }).rename(columns={'quantity': 'total_volume'})
    # Imbalance
    daily['imbalance'] = (daily['buy_volume'] - daily['sell_volume']) / daily['total_volume']
    # Rolling-Fenster (1h & 4h) für Volumen-Spikes
    df_hourly = df['quantity'].resample('1H').sum().rename('vol_1h')
    df_4h    = df['quantity'].resample('4H').sum().rename('vol_4h')
    daily['max_vol_1h'] = df_hourly.resample('1D').max()
    daily['max_vol_4h'] = df_4h.resample('1D').max()
    # Trades pro Minute (Average)
    trades_per_min = df['quantity'].resample('1T').count().rename('trades_per_min')
    daily['avg_trades_per_min'] = trades_per_min.resample('1D').mean()
    return daily.reset_index()

def main(input_dir: str, output_file: str):
    # Alle CSVs zusammenladen
    pattern = os.path.join(input_dir, '*/*.csv')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Keine Dateien gefunden in {pattern}")
    df_list = [pd.read_csv(f, usecols=['timestamp','price','quantity','isBuyerMaker']) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    features = compute_agg_features(df)
    # Kompakt speichern
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    features.to_parquet(output_file, index=False)
    print(f"Features geschrieben nach {output_file}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Extract aggTrades Features")
    p.add_argument('--input-dir',  required=True, help="Ordner mit entpackten CSVs")
    p.add_argument('--output-file', required=True, help="Ziel-Parquet-Datei")
    args = p.parse_args()
    main(args.input_dir, args.output_file)
