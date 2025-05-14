#!/usr/bin/env python3
import os, glob, argparse, logging
import pandas as pd

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

# fixe Spaltennamen in der exakten Reihenfolge, wie sie bei Binance liegen
EXPECTED_COLS = [
    'agg_trade_id',
    'price',
    'quantity',
    'first_trade_id',
    'last_trade_id',
    'transact_time',
    'is_buyer_maker'
]

def load_and_concat(input_dir: str) -> pd.DataFrame:
    df_list = []
    for fn in sorted(glob.glob(os.path.join(input_dir, '*.csv'))):
        try:
            # 1) immer ohne Header einlesen
            df = pd.read_csv(fn, header=None, names=EXPECTED_COLS, dtype=str)
            # 2) falls die erste Zeile genau unser Header ist, wegwerfen
            if list(df.iloc[0]) == EXPECTED_COLS:
                df = df.iloc[1:]
            # 3) saubere Typen erzwingen, fehlerhafte Zeilen droppen
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
            df['transact_time'] = pd.to_numeric(df['transact_time'], errors='coerce')
            df['is_buyer_maker'] = df['is_buyer_maker'].map({'true': True, 'false': False})
            df = df.dropna(subset=['quantity', 'transact_time', 'is_buyer_maker'])
            df_list.append(df[['quantity', 'transact_time', 'is_buyer_maker']])
        except Exception as e:
            logging.error(f"Error reading {fn}: {e}")
    if not df_list:
        logging.warning("No valid data; exiting.")
        exit(0)
    combined = pd.concat(df_list, ignore_index=True)
    logging.info(f"Combined DataFrame shape: {combined.shape}")
    return combined

def extract_features(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    # Timestamp aus ms
    df['timestamp'] = pd.to_datetime(df['transact_time'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    # auf den gewünschten Zeitraum filtern (inklusive Ende)
    df = df[start_date : pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(ms=1)]
    if df.empty:
        logging.warning("No trades in window.")
        return pd.DataFrame()

    # Resample pro Tag
    daily = df.resample('1D').agg(
        total_volume=('quantity', 'sum'),
        buy_volume=('quantity', lambda x: x[~df.loc[x.index, 'is_buyer_maker']].sum()),
        sell_volume=('quantity', lambda x: x[df.loc[x.index, 'is_buyer_maker']].sum()),
        max_vol_1h=('quantity', lambda x: x.resample('1H').sum().max()),
        max_vol_4h=('quantity', lambda x: x.resample('4H').sum().max()),
        avg_trades_per_min=('quantity', lambda x: x.resample('1T').count().mean())
    )
    # Imbalance
    daily['imbalance'] = (daily.buy_volume - daily.sell_volume) / daily.total_volume.clip(lower=1e-9)
    return daily

def main(input_dir, output_file, start_date, end_date):
    logging.info(f"→ Reading CSVs from '{input_dir}'")
    df = load_and_concat(input_dir)
    feats = extract_features(df, start_date, end_date)
    if feats.empty:
        logging.warning("No features to write; exiting.")
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    feats.to_parquet(output_file, index=True)
    logging.info(f"Wrote features to {output_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir',  required=True)
    p.add_argument('--output-file', required=True)
    p.add_argument('--start-date',  required=True)
    p.add_argument('--end-date',    required=True)
    args = p.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date)
