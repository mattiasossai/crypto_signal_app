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
    with open(fn, "r") as f:
        tokens = f.readline().strip().split(",")
    header = tokens == EXPECTED_COLS

    df = pd.read_csv(
        fn,
        header=0 if header else None,
        names=EXPECTED_COLS if not header else None,
        usecols=["quantity","transact_time","is_buyer_maker"],
        dtype={"quantity": float, "transact_time": int, "is_buyer_maker": bool},
    )
    df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    day = os.path.basename(fn).split("-aggTrades-")[1][:10]
    sd = pd.to_datetime(day).tz_localize("UTC")
    ed = sd + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    df_day = df[sd:ed]

    if df_day.empty:
        logging.warning("Empty data for %s", day)
        return None

    total   = df_day["quantity"].sum()
    buy     = df_day.loc[~df_day["is_buyer_maker"], "quantity"].sum()
    sell    = df_day.loc[ df_day["is_buyer_maker"], "quantity"].sum()
    max1h   = df_day["quantity"].resample("1h").sum().max()
    max4h   = df_day["quantity"].resample("4h").sum().max()
    avg_trd = df_day["quantity"].resample("1min").count().mean()
    imbalance = (buy - sell) / total if total > 0 else 0

    return {
        "date": day,
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
    symbol  = os.path.basename(input_dir)
    user_sd = pd.to_datetime(start_date).tz_localize("UTC")
    user_ed = pd.to_datetime(end_date).tz_localize("UTC")
    inc     = pd.to_datetime(INCEPTION.get(symbol, start_date)).tz_localize("UTC")
    new_sd  = max(user_sd, inc).normalize()

    # Suche letztes vorhandenes File im output dir
    out_dir    = os.path.dirname(output_file)
    pattern    = os.path.join(out_dir, f"{symbol}-features-*.parquet")
    old_files  = glob.glob(pattern)
    if old_files:
        # lade jenes mit dem höchsten End-Datum
        best = max(old_files)
        df_old = pd.read_parquet(best)
        last  = df_old.index.max()
        new_sd = (pd.to_datetime(last) + pd.Timedelta(days=1)).normalize()
        logging.info("→ Append mode: resume at %s", new_sd.date())
    else:
        df_old = None
        logging.info("→ Fresh mode: start at %s", new_sd.date())

    if new_sd > user_ed:
        logging.info("ℹ️ Nothing to append.")
        return

    # finde alle CSV-Dateien ab new_sd bis end_date
    all_csv = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    files   = [
        fn for fn in all_csv
        if new_sd <= pd.to_datetime(os.path.basename(fn).split("-aggTrades-")[1][:10]).tz_localize("UTC") <= user_ed
    ]

    rows = []
    for fn in files:
        r = process_one_file(fn)
        if r: rows.append(r)

    df_new = pd.DataFrame(rows).set_index("date").sort_index()
    df_new.index = pd.to_datetime(df_new.index).tz_localize("UTC")

    # Merge alt + neu
    df = pd.concat([df_old, df_new]) if df_old is not None else df_new

    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(output_file, compression="snappy")
    logging.info("✅ Wrote %d days to %s", len(df_new), output_file)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",   required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--start-date",  required=True)
    p.add_argument("--end-date",    required=True)
    args = p.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date)
