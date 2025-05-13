#!/usr/bin/env python3
"""
extract_aggTrades_features.py

Lädt alle CSVs eines Symbols und berechnet pro Tag:
 - total_volume
 - buy_volume, sell_volume
 - imbalance (sicher vor Division durch Null)
 - max_vol_1h, max_vol_4h
 - avg_trades_per_min

Schreibt Parquet-Output für ML mit optionaler Normalisierung und Outlier-Capping.
"""

import argparse
import logging
import os
import glob
import pandas as pd
import numpy as np

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def cap_outliers(df: pd.DataFrame, columns, lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    """
    Winsorize Features in 'columns' durch Clipping an den Quantilen.
    """
    for col in columns:
        lo = df[col].quantile(lower_q)
        hi = df[col].quantile(upper_q)
        df[col] = df[col].clip(lo, hi)
    return df

def compute_agg_features(df: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    # Timestamp in UTC und sortieren
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)

    # Buy/Sell-Volumen
    df['buy_volume']  = df['quantity'].mask(df['isBuyerMaker'], 0.0)
    df['sell_volume'] = df['quantity'].mask(~df['isBuyerMaker'], 0.0)

    # Resampling und Aggregation
    daily = df.resample('1D').agg(
        total_volume=('quantity', 'sum'),
        buy_volume=('buy_volume', 'sum'),
        sell_volume=('sell_volume', 'sum'),
        max_vol_1h=('quantity', lambda x: x.resample('1H').sum().max()),
        max_vol_4h=('quantity', lambda x: x.resample('4H').sum().max()),
        avg_trades_per_min=('quantity', lambda x: x.resample('1T').count().mean())
    )

    # Absicherung gegen Division durch Null
    mask = daily['total_volume'] > 0
    daily['imbalance'] = np.where(
        mask,
        (daily['buy_volume'] - daily['sell_volume']) / daily['total_volume'],
        0.0
    )

    # Outlier-Capping (Winsorization)
    numeric_cols = [
        'total_volume', 'buy_volume', 'sell_volume',
        'max_vol_1h', 'max_vol_4h', 'avg_trades_per_min'
    ]
    daily = cap_outliers(daily, numeric_cols)

    # Optionale Z-Score-Normalisierung
    if normalize:
        daily[numeric_cols] = daily[numeric_cols].apply(
            lambda x: (x - x.mean()) / x.std(ddof=0)
        )

    return daily.reset_index()

def main(input_dir: str, output_file: str, start_date: str, end_date: str, normalize: bool):
    pattern = os.path.join(input_dir, '*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        logging.warning(f"Keine CSVs in {input_dir} – überspringe.")
        return

    # CSV einlesen mit Typen und Fehlerbehandlung
    df_list = []
    for f in files:
        try:
            tmp = pd.read_csv(
                f,
                header=None,
                usecols=[5, 2, 6],
                names=['timestamp', 'quantity', 'isBuyerMaker'],
                dtype={'timestamp': 'Int64', 'quantity': float, 'isBuyerMaker': bool}
            )
            df_list.append(tmp)
        except Exception as e:
            logging.error(f"Fehler beim Lesen {f}: {e}")

    if not df_list:
        logging.warning("Keine valide Daten zum Verarbeiten.")
        return

    df_all = pd.concat(df_list, ignore_index=True).drop_duplicates()

    # Zeitfilter in UTC Millisekunden
    start_ts = int(pd.to_datetime(start_date).tz_localize('UTC').timestamp() * 1000)
    end_ts = int((pd.to_datetime(end_date).tz_localize('UTC') + pd.Timedelta(days=1)
                  - pd.Timedelta(milliseconds=1)).timestamp() * 1000)
    df_all = df_all[df_all['timestamp'].between(start_ts, end_ts)]

    if df_all.empty:
        logging.warning(f"Keine Trades im Zeitraum {start_date}–{end_date}.")
        return

    # Features berechnen
    features = compute_agg_features(df_all, normalize)

    # Speichern als Parquet mit Kompression
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    features.to_parquet(output_file, index=False, compression='snappy')
    logging.info(f"Features geschrieben nach {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract aggTrades Features per symbol")
    parser.add_argument('--input-dir',   required=True, help="CSV-Ordner des Symbols")
    parser.add_argument('--output-file', required=True, help="Ziel-Parquet")
    parser.add_argument('--start-date',  required=True, help="YYYY-MM-DD")
    parser.add_argument('--end-date',    required=True, help="YYYY-MM-DD")
    parser.add_argument('--normalize',   action='store_true', help="Z-Score-Normalisierung der Features")
    args = parser.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date, args.normalize)
