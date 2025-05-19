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

# Symbol-spezifisches Inception-Datum (erster Handelstag)
INCEPTION = {
    "BTCUSDT": "2019-12-31",
    "ETHUSDT": "2019-12-31",
    "BNBUSDT": "2020-02-10",
    "SOLUSDT": "2020-09-14",
    "XRPUSDT": "2020-01-06",
    "ENAUSDT": "2024-04-02",
}

EXPECTED_COLS = [
    "agg_trade_id", "price", "quantity",
    "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker"
]

def process_one_file(fn: str) -> dict | None:
    # Header-Erkennung
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

    # in UTC timestamp
    df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    # Tagesfenster aus dem Dateinamen
    day_str = os.path.basename(fn).split("-aggTrades-")[1][:10]
    sd = pd.to_datetime(day_str).tz_localize("UTC")
    ed = sd + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    df_day = df[sd:ed]

    if df_day.empty:
        logging.warning("Empty data for %s", day_str)
        return None

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
    symbol = os.path.basename(input_dir)

    # parse user-dates
    user_sd = pd.to_datetime(start_date).tz_localize("UTC")
    user_ed = pd.to_datetime(end_date).tz_localize("UTC")

    # inception-date parsieren
    inc = pd.to_datetime(INCEPTION.get(symbol, start_date)).tz_localize("UTC")

    # real_start = max(user_start, inception)
    real_start = max(user_sd, inc)

    # 1-Tag-Overlap für Features, aber erst ab real_start
    sd = real_start - pd.Timedelta(days=1)
    ed = user_ed

    # alle CSVs
    all_csv = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    files = [
        fn for fn in all_csv
        if sd <= pd.to_datetime(os.path.basename(fn).split("-aggTrades-")[1][:10]).tz_localize("UTC") <= ed
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

    # DataFrame mit Datum-String-Index
    df_feats = pd.DataFrame(rows).set_index("date").sort_index()

    # Overlap-Tag (erste Zeile) verwerfen
    df_feats = df_feats.loc[df_feats.index >= real_start.strftime("%Y-%m-%d")]

    # Robustes Lücken-Handling:
    # Index erst ab real_start bis end_date
    full_idx = pd.date_range(
        start=real_start.normalize(),
        end=user_ed.normalize(),
        freq="D",
        tz="UTC"
    )
    df_feats.index = pd.to_datetime(df_feats.index).tz_localize("UTC")
    df_feats = df_feats.reindex(full_idx)  # fehlende Tage → NaN
    df_feats.index.name = "date"

    # Parquet schreiben
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_feats.to_parquet(output_file, compression="snappy")
    logging.info("✅ Wrote features to %s (%d days)", output_file, len(df_feats))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",   required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--start-date",  required=True)
    p.add_argument("--end-date",    required=True)
    args = p.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date)
