#!/usr/bin/env python3
import glob, os
import pandas as pd

def merge_symbol(symbol_dir: str):
    symbol = os.path.basename(symbol_dir)
    pattern = os.path.join(symbol_dir, f"{symbol}-features-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files for {symbol}, skipping.")
        return

    # 1) Einlesen und zusammenführen
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # 2) Datum auf datetime konvertieren und sortieren
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["date"]).sort_values("date")

    # 3) Neue Dateiname von erstem bis letztem Datum
    start = df["date"].min().strftime("%Y-%m-%d")
    end   = df["date"].max().strftime("%Y-%m-%d")
    out_file = os.path.join(symbol_dir, f"{symbol}-features-{start}_to_{end}.parquet")

    # 4) Schreiben
    df.to_parquet(out_file)
    print(f"Merged {len(files)} → {out_file}")

if __name__ == "__main__":
    base = "features/aggTrades"
    for symbol_dir in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.isdir(symbol_dir):
            merge_symbol(symbol_dir)
