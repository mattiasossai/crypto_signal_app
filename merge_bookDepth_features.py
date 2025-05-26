#!/usr/bin/env python3
import glob
import os
import pandas as pd

def detect_timeseries(df: pd.DataFrame) -> pd.Series:
    # 1) Direkte Spaltenabfrage
    if "date" in df.columns:
        return df["date"]
    if "timestamp" in df.columns:
        return df["timestamp"]
    # 2) Integer-Spalten mit großen Werten (ms-Timestamps > 1e12)
    int_cols = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
    for c in int_cols:
        # prüfen, ob alle Werte > 1e12
        col = df[c].dropna()
        if not col.empty and col.gt(1e12).all():
            return col
    # 3) Ist der Index schon datetime?
    if pd.api.types.is_datetime64_any_dtype(df.index.dtype):
        return pd.Series(df.index.values, index=df.index)
    raise KeyError("No date-like column found")

def merge_symbol(symbol_dir: str):
    symbol = os.path.basename(symbol_dir)
    pattern = os.path.join(symbol_dir, f"{symbol}-features-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files for {symbol}, skipping.")
        return

    merged = []
    for f in files:
        df = pd.read_parquet(f)
        ts = detect_timeseries(df)
        # konvertiere in datetime
        ts = pd.to_datetime(ts, unit="ms", utc=True)
        df = df.copy()
        df["__merge_date"] = ts
        merged.append(df)

    big = pd.concat(merged, ignore_index=True)
    big = big.drop_duplicates(subset="__merge_date").sort_values("__merge_date")

    # Erstelle neuen Dateinamen
    start = big["__merge_date"].min().strftime("%Y-%m-%d")
    end   = big["__merge_date"].max().strftime("%Y-%m-%d")
    out_file = os.path.join(symbol_dir, f"{symbol}-features-{start}_to_{end}.parquet")

    # rename & speichern
    big = big.rename(columns={"__merge_date": "date"})
    big.to_parquet(out_file)
    print(f"Merged {len(files)} → {out_file}")

if __name__ == "__main__":
    base = "features/bookDepth"
    for symbol_dir in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.isdir(symbol_dir):
            merge_symbol(symbol_dir)
