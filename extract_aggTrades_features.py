#!/usr/bin/env python3
"""
extract_aggTrades_features.py

Lädt alle entpackten aggTrades-CSV-Dateien eines Symbols aus einem Verzeichnis
und berechnet pro Tag:
 - total_volume
 - buy_volume, sell_volume
 - imbalance = (buy_volume - sell_volume) / total_volume
 - max_vol_1h, max_vol_4h pro Tag
 - avg_trades_per_min

Schreibt das Ergebnis als Parquet für das ML-Training.

Anpassungen:
 - Liest die CSVs **ohne** Header und weist feste Spaltennamen zu,
   gemäß der Binance-Public-Data-Doku:
     aggTradeId, price, quantity, firstTradeId, lastTradeId,
     timestamp, isBuyerMaker, isBestMatch :contentReference[oaicite:0]{index=0}
 - Wählt nur die benötigten `timestamp`, `quantity`, `isBuyerMaker`.
 - Droppt Duplikate (Overlap-Tag).
 - Filtern auf CLI-Flags `--start-date` / `--end-date`.
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
        raise FileNotFoundError(f"No CSV files found in {pattern}")

    # 1) CSVs ohne Header einlesen, feste Spaltennamen vergeben
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

    # 2) Duplikate entfernen (Overlap-Tag)
    df = df.drop_duplicates()

    # 3) Zeitfenster in ms filtern (inkl. Ende bis 23:59:59.999)
    start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_ts = int((pd.to_datetime(end_date) + pd.Timedelta(days=1)
                  - pd.Timedelta(milliseconds=1)).timestamp() * 1000)
    df = df[df['timestamp'].between(start_ts, end_ts)]
    if df.empty:
        raise RuntimeError(f"No trades in period {start_date} to {end_date}")

    # 4) Feature-Berechnung
    features = compute_agg_features(df)

    # 5) Parquet schreiben
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    features.to_parquet(output_file, index=False)
    print(f"[OK] Wrote features to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract aggTrades Features per symbol"
    )
    parser.add_argument(
        '--input-dir',   required=True,
        help="Folder with CSVs for one symbol"
    )
    parser.add_argument(
        '--output-file', required=True,
        help="Parquet output path"
    )
    parser.add_argument(
        '--start-date',  required=True,
        help="Startdatum YYYY-MM-DD"
    )
    parser.add_argument(
        '--end-date',    required=True,
        help="Enddatum YYYY-MM-DD"
    )
    args = parser.parse_args()
    main(
        input_dir   = args.input_dir,
        output_file = args.output_file,
        start_date  = args.start_date,
        end_date    = args.end_date
    )
