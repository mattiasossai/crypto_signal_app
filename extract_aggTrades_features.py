#!/usr/bin/env python3
"""
extract_aggTrades_features.py

Lädt alle entpackten aggTrades-CSV-Dateien aus einem Verzeichnis und berechnet pro Tag:
 - total_volume
 - buy_volume, sell_volume
 - imbalance = (buy_volume - sell_volume) / total_volume
 - max volumens in 1h und 4h pro Tag
 - durchschnittliche Trades pro Minute

Schreibt das Ergebnis als Parquet für das ML-Training.
"""
import argparse
import os
import glob
import pandas as pd

def compute_agg_features(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Zeitindex setzen
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # 2) Taker-Volumina
    df['buy_volume']  = df['quantity'].where(~df['isBuyerMaker'], 0.0)
    df['sell_volume'] = df['quantity'].where(df['isBuyerMaker'],  0.0)

    # 3) Tages-Resampling
    daily = df.resample('1D').agg({
        'quantity':    'sum',   # total_volume
        'buy_volume':  'sum',
        'sell_volume': 'sum'
    }).rename(columns={'quantity': 'total_volume'})

    # 4) Imbalance
    daily['imbalance'] = (
        (daily['buy_volume'] - daily['sell_volume'])
        / daily['total_volume']
    )

    # 5) Rolling-Volumen-Spitzen
    vol_1h = df['quantity'].resample('1H').sum()
    vol_4h = df['quantity'].resample('4H').sum()
    daily['max_vol_1h'] = vol_1h.resample('1D').max()
    daily['max_vol_4h'] = vol_4h.resample('1D').max()

    # 6) Trades pro Minute (Takte pro Minute zählen, dann Tagesmittel)
    trades_per_min = df['quantity'].resample('1T').count()
    daily['avg_trades_per_min'] = trades_per_min.resample('1D').mean()

    return daily.reset_index()

def main(input_dir: str, output_file: str):
    pattern = os.path.join(input_dir, '*/*.csv')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Keine CSV-Dateien gefunden in {pattern}")

    df_list = []
    for f in files:
        df_list.append(
            pd.read_csv(
                f,
                usecols=['timestamp','quantity','isBuyerMaker'],
                dtype={'timestamp': 'Int64','quantity':'float','isBuyerMaker':'bool'}
            )
        )
    df = pd.concat(df_list, ignore_index=True)
    features = compute_agg_features(df)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    features.to_parquet(output_file, index=False)
    print(f"[OK] Features geschrieben nach {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract aggTrades Features für ML"
    )
    parser.add_argument(
        '--input-dir',  required=True,
        help="Pfad zu entpackten aggTrades-CSVs"
    )
    parser.add_argument(
        '--output-file', required=True,
        help="Ziel-Parquet-Datei, z.B. historical_tech/aggTrades/features/agg_YYYY-MM-DD.parquet"
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_file)
