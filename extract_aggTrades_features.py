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

Neu:
 - CLI-Flags --start-date und --end-date zum Beschneiden des Zeitfensters.
 - Entfernt exakte Duplikate vor der Berechnung.
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

def main(input_dir: str, output_file: str, start_date: str, end_date: str):
    # CSV-Dateien sammeln
    pattern = os.path.join(input_dir, '*/*.csv')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Keine CSV-Dateien gefunden in {pattern}")

    # 1) Alle CSVs einlesen
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

    # 2) Exakte Duplikate entfernen
    df = df.drop_duplicates()

    # 3) Zeitfenster filtern (auf Basis der originalen ms-Timestamps)
    #    start_date/end_date sind inklusiv im Format YYYY-MM-DD
    start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
    # Ende des end_date: eine Sekunde vor Mitternacht des Folgetages
    end_ts = int((pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)).timestamp() * 1000)
    df = df[df['timestamp'].between(start_ts, end_ts)]
    if df.empty:
        raise ValueError(f"Keine Trades im Zeitraum {start_date} bis {end_date}.")

    # 4) Features berechnen
    features = compute_agg_features(df)

    # 5) Parquet schreiben
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
        help="Ziel-Parquet-Datei, z.B. historical_tech/aggTrades/features/agg_2020-01-01_to_2020-06-30.parquet"
    )
    parser.add_argument(
        '--start-date', required=True,
        help="Startdatum (YYYY-MM-DD), inklusiv"
    )
    parser.add_argument(
        '--end-date', required=True,
        help="Enddatum (YYYY-MM-DD), inklusiv"
    )
    args = parser.parse_args()

    main(
        input_dir   = args.input_dir,
        output_file = args.output_file,
        start_date  = args.start_date,
        end_date    = args.end_date
    )
