#!/usr/bin/env python3
import os, glob, argparse, logging
import pandas as pd

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

def detect_cols(df):
    cols = df.columns
    # find quantity
    q = next((c for c in cols if 'quantity' in c.lower()), None)
    # find timestamp
    t = next((c for c in cols if 'time' in c.lower() and c != q), None)
    # find is_buyer_maker
    m = next((c for c in cols if 'buyer' in c.lower() and 'maker' in c.lower()), None)
    if not all([q, t, m]):
        raise ValueError(f"Could not detect cols in {cols.tolist()}")
    return t, q, m

def load_and_concat(input_dir):
    all_dfs = []
    for fn in sorted(glob.glob(os.path.join(input_dir, '*.csv'))):
        try:
            df = pd.read_csv(fn)
            ts_col, qty_col, mk_col = detect_cols(df)
            sub = df[[ts_col, qty_col, mk_col]].rename(columns={
                ts_col: 'ts', qty_col: 'quantity', mk_col: 'is_buyer_maker'
            })
            all_dfs.append(sub)
        except Exception as e:
            logging.error(f"Error reading {fn}: {e}")
    if not all_dfs:
        logging.warning("No valid data; exiting.")
        exit(0)
    return pd.concat(all_dfs, ignore_index=True)

def extract_features(df, start_date, end_date):
    # parse timestamp
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['is_buyer_maker'] = df['is_buyer_maker'].astype(bool)
    df = df.set_index('timestamp')
    # filter window
    df = df[start_date: end_date]
    if df.empty:
        logging.warning("No trades between window.")
        return pd.DataFrame()

    # daily aggregations
    ag = df.resample('1D').agg(
        total_volume=('quantity', 'sum'),
        buy_volume=('quantity', lambda x: x[~df.loc[x.index, 'is_buyer_maker']].sum()),
        sell_volume=('quantity', lambda x: x[df.loc[x.index, 'is_buyer_maker']].sum()),
        max_vol_1h=('quantity', lambda x: x.resample('1H').sum().max()),
        max_vol_4h=('quantity', lambda x: x.resample('4H').sum().max()),
        avg_trades_per_min=('quantity', lambda x: x.resample('1T').count().mean())
    )
    ag['imbalance'] = (ag.buy_volume - ag.sell_volume) / ag.total_volume.clip(lower=1e-9)
    return ag

def main(input_dir, output_file, start_date, end_date):
    logging.info(f"→ Reading CSVs from '{input_dir}'")
    df = load_and_concat(input_dir)
    logging.info(f"Combined DF: {df.shape}")
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
