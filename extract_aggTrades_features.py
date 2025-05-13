#!/usr/bin/env python3
"""
extract_aggTrades_features.py

… (Dokumentation unverändert) …

Neu:
 - Bei komplett fehlenden CSVs wird **leise** übersprungen (kein Fehler).
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

    # Wenn gar keine CSVs da sind, skip ohne Fehler
    if not files:
        print(f"[WARN] Keine CSVs in {input_dir} – überspringe Symbol.")
        return

    # CSVs ohne Header einlesen
    cols = [
        'aggTradeId','price','quantity','firstTradeId',
        'lastTradeId','timestamp','isBuyerMaker','isBestMatch'
    ]
    df = pd.concat([
        pd.read_csv(
            f,
            header=None,
            names=cols,
            usecols=['timestamp','quantity','isBuyerMaker'],
            dtype={'timestamp': 'Int64','quantity':'float','isBuyerMaker':'bool'}
        )
        for f in files
    ], ignore_index=True)

    # Duplikate (Overlap) entfernen
    df = df.drop_duplicates()

    # Zeitfenster filtern (Millisekunden)
    start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_ts = int((pd.to_datetime(end_date) + pd.Timedelta(days=1)
                  - pd.Timedelta(milliseconds=1)).timestamp() * 1000)
    df = df[df['timestamp'].between(start_ts, end_ts)]

    if df.empty:
        print(f"[WARN] Keine Trades im Zeitraum {start_date}–{end_date} für {input_dir}.")
        return

    # Feature-Berechnung & Parquet-Export
    features = compute_agg_features(df)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    features.to_parquet(output_file, index=False)
    print(f"[OK] Wrote features to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract aggTrades Features per symbol")
    parser.add_argument('--input-dir',   required=True, help="Folder with symbol CSVs")
    parser.add_argument('--output-file', required=True, help="Parquet output path")
    parser.add_argument('--start-date',  required=True, help="YYYY-MM-DD")
    parser.add_argument('--end-date',    required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date)
