#!/usr/bin/env python3
"""
extract_aggTrades_features.py

Lädt alle CSVs eines Symbols (inkl. 1-Tag Overlap) und berechnet pro Tag:
 - total_volume
 - buy_volume, sell_volume
 - imbalance (sicher vor Division durch Null)
 - max_vol_1h, max_vol_4h
 - avg_trades_per_min

Behalte für die erste Tages-Aggregation bewusst Trades ab Overlap-Tag,
wie im Workflow geladen, dann drope das Overlap-Datum vor dem Schreiben.
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
    # Timestamps in UTC
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)

    # Buy/Sell-Volumen markieren
    df['buy_volume']  = df['quantity'].mask(df['isBuyerMaker'], 0.0)
    df['sell_volume'] = df['quantity'].mask(~df['isBuyerMaker'], 0.0)

    # Resample und Aggregation
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

    # Timestamp als Spalte zurückgeben
    daily = daily.reset_index().rename(columns={'timestamp': 'date'})
    return daily


def main(input_dir: str, output_file: str, start_date: str, end_date: str, normalize: bool):
    # 1) Lade alle CSVs (Overlap-Tag inklusive)
    files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
    if not files:
        logging.warning(f"Keine CSVs in {input_dir} – überspringe.")
        return

    df_list = []
    for f in files:
        try:
            tmp = pd.read_csv(
                f,
                header=None,
                usecols=[5, 2, 6],
                names=['timestamp', 'quantity', 'isBuyerMaker'],
                dtype={'quantity': float, 'isBuyerMaker': bool}
            )
            df_list.append(tmp)
        except Exception as e:
            logging.error(f"Fehler beim Lesen {f}: {e}")

    if not df_list:
        logging.warning("Keine valide Daten zum Verarbeiten.")
        return

    df_all = pd.concat(df_list, ignore_index=True).drop_duplicates()

    # 2) Filter nur nach end_date (Overlap-Daten bleiben erhalten)
    end_ts = int((pd.to_datetime(end_date).tz_localize('UTC') + 
                  pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)).timestamp() * 1000)
    df_all = df_all[df_all['timestamp'] <= end_ts]

    if df_all.empty:
        logging.warning(f"Keine Trades bis {end_date}.")
        return

    # 3) Compute Features über [Overlap-Tag .. end_date]
    features = compute_agg_features(df_all, normalize)

    # 4) Drop Overlap-Datum: behalte nur ab start_date
    features['date'] = pd.to_datetime(features['date']).dt.date
    sd = pd.to_datetime(start_date).date()
    features = features[features['date'] >= sd].drop(columns=['date'])

    # 5) Speichern als Parquet mit Kompression
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    features.to_parquet(output_file, index=False, compression='snappy')
    logging.info(f"Features geschrieben nach {output_file}")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Extract aggTrades Features per symbol")
    p.add_argument('--input-dir',   required=True, help="CSV-Ordner des Symbols")
    p.add_argument('--output-file', required=True, help="Ziel-Parquet")
    p.add_argument('--start-date',  required=True, help="YYYY-MM-DD")
    p.add_argument('--end-date',    required=True, help="YYYY-MM-DD")
    p.add_argument('--normalize',   action='store_true', help="Z-Score-Normalisierung der Features")
    args = p.parse_args()
    main(
        input_dir   = args.input_dir,
        output_file = args.output_file,
        start_date  = args.start_date,
        end_date    = args.end_date,
        normalize   = args.normalize
    )
