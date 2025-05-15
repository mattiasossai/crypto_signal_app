#!/usr/bin/env python3
import os
import glob
import argparse
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

EXPECTED_COLS = [
    "agg_trade_id", "price", "quantity",
    "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker"
]

def process_one_file(fn: str) -> dict | None:
    # Header-Erkennung wie vorher
    with open(fn, "r") as f:
        tokens = f.readline().strip().split(",")
    header = tokens == EXPECTED_COLS

    df = pd.read_csv(
        fn,
        header=0 if header else None,
        names=EXPECTED_COLS if not header else None,
        usecols=["quantity", "transact_time", "is_buyer_maker"],
        dtype={"quantity": float, "transact_time": int, "is_buyer_maker": bool},
    )

    # Zeitindex setzen
    df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    # Tagesfenster ermitteln
    day_str = os.path.basename(fn).split("-aggTrades-")[1].replace(".csv", "")
    sd = pd.to_datetime(day_str).tz_localize("UTC")
    ed = sd + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    df_day = df[sd:ed]

    if df_day.empty:
        logging.warning("Empty data for %s", day_str)
        return None

    # Tages-Aggregationen
    total   = df_day["quantity"].sum()
    buy     = df_day.loc[~df_day["is_buyer_maker"], "quantity"].sum()
    sell    = df_day.loc[ df_day["is_buyer_maker"],  "quantity"].sum()
    max1h   = df_day["quantity"].resample("1h").sum().max()
    max4h   = df_day["quantity"].resample("4h").sum().max()
    avg_trd = df_day["quantity"].resample("1min").count().mean()
    imbalance = (buy - sell) / total if total > 0 else 0

    return {
        "date": day_str,
        "total_volume": total,
        "buy_volume": buy,
        "sell_volume": sell,
        "max_vol_1h": max1h,
        "max_vol_4h": max4h,
        "avg_trades_per_min": avg_trd,
        "imbalance": imbalance,
    }

def main(input_dir, output_file, start_date, end_date):
    logging.info("→ Scanning CSVs in '%s'", input_dir)
    all_csv = sorted(glob.glob(os.path.join(input_dir, "*.csv")))

    # Einschluss des Overlap-Tages für rollende Fenster
    sd = pd.to_datetime(start_date) - pd.Timedelta(days=1)
    ed = pd.to_datetime(end_date)
    files = [
        fn for fn in all_csv
        if sd <= pd.to_datetime(os.path.basename(fn).split("-aggTrades-")[1][:10]) <= ed
    ]

    if not files:
        logging.error("❌ No CSV files found in range; exiting.")
        return

    rows = []
    for fn in files:
        r = process_one_file(fn)
        if r is not None:
            rows.append(r)

    if not rows:
        logging.error("❌ No data after processing; exiting.")
        return

    # DataFrame erstellen und sortieren
    df_feats = pd.DataFrame(rows).set_index("date").sort_index()

    # Drop Overlap-Tag: nur ab echtem Start
    df_feats = df_feats.loc[df_feats.index >= start_date]

    # ————— Rolling-Window-Features ergänzen —————
    # Konvertiere Index in DatetimeIndex
    df_feats.index = pd.to_datetime(df_feats.index)

    # Definiere Rolling-Fenster
    rolling_windows = {
        '1d': '1D',
        '3d': '3D',
        '7d': '7D',
    }

    for name, win in rolling_windows.items():
        df_feats[f'total_vol_roll_{name}']    = df_feats['total_volume'].rolling(win).mean()
        df_feats[f'imbalance_std_roll_{name}'] = df_feats['imbalance'].rolling(win).std()
        # Weitere Fenster-Kennzahlen können hier ergänzt werden
    # NaN-Werte der ersten Window-Tage entfernen
    df_feats = df_feats.dropna()
    # ————————————————————————————————

    # Zu Parquet schreiben
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_feats.to_parquet(output_file, compression="snappy")
    logging.info("✅ Wrote features to %s (%d days)", output_file, len(df_feats))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",   required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--start-date",  required=True)
    parser.add_argument("--end-date",    required=True)
    args = parser.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date)
