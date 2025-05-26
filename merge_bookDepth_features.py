#!/usr/bin/env python3
import glob
import os
import pandas as pd

def get_time_series(df: pd.DataFrame) -> pd.Series:
    # 1) Spalte 'date'
    if "date" in df.columns:
        return df["date"]
    # 2) Index-Name 'date'
    if df.index.name == "date":
        return pd.Series(df.index.values, index=df.index)
    # 3) Spalte 'timestamp'
    if "timestamp" in df.columns:
        return df["timestamp"]
    # 4) First integer column >1e12
    int_cols = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
    for c in int_cols:
        col = df[c].dropna()
        if not col.empty and col.gt(1e12).all():
            return col
    raise KeyError("No date-like column or index")

def merge_symbol(symbol_dir: str):
    symbol = os.path.basename(symbol_dir)
    pattern = os.path.join(symbol_dir, f"{symbol}-features-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files for {symbol}, skipping.")
        return

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        raw_ts = get_time_series(df)
        # konvertiere Raw-Timestamps in datetime
        ts = pd.to_datetime(raw_ts, unit="ms", utc=True)
        df = df.copy()
        df["__merge_date"] = ts
        frames.append(df)

    big = pd.concat(frames, ignore_index=True)
    big = big.drop_duplicates(subset="__merge_date").sort_values("__merge_date")

    start = big["__merge_date"].min().strftime("%Y-%m-%d")
    end   = big["__merge_date"].max().strftime("%Y-%m-%d")
    out_file = os.path.join(symbol_dir, f"{symbol}-features-{start}_to_{end}.parquet")

    # Umbenennen & Speichern
    big = big.rename(columns={"__merge_date": "date"})
    big.to_parquet(out_file)
    print(f"Merged {len(files)} → {out_file}")

if __name__ == "__main__":
    base = "features/bookDepth"
    for symbol_dir in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.isdir(symbol_dir):
            merge_symbol(symbol_dir)
