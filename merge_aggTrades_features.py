#!/usr/bin/env python3
import glob
import os
import pandas as pd

def merge_symbol(symbol_dir: str):
    symbol = os.path.basename(symbol_dir)
    pattern = os.path.join(symbol_dir, f"{symbol}-features-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files for {symbol}, skipping.")
        return

    dfs = []
    for f in files:
        df = pd.read_parquet(f)

        # 1) Finden, wo der Zeitstempel ist
        if   "date" in df.columns:
            raw = df["date"]
        elif "timestamp" in df.columns:
            raw = df["timestamp"]
        elif df.index.name in ("date","timestamp"):
            raw = df.index.to_series()
        else:
            # suche erstes Integer-Feld
            ints = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
            if not ints:
                raise KeyError(f"No date-like column in {f}")
            raw = df[ints[0]]

        # 2) In Datetime umwandeln
        df["__merge_date"] = pd.to_datetime(raw, unit="ms", utc=True)
        dfs.append(df)

    # 3) Konkat und Sort
    big = pd.concat(dfs, ignore_index=True)
    big = big.drop_duplicates(subset="__merge_date").sort_values("__merge_date")

    # 4) New filename
    start = big["__merge_date"].min().strftime("%Y-%m-%d")
    end   = big["__merge_date"].max().strftime("%Y-%m-%d")
    out_file = os.path.join(symbol_dir, f"{symbol}-features-{start}_to_{end}.parquet")

    # 5) __merge_date → date, alte Spalte/Index entfernen
    big = big.rename(columns={"__merge_date": "date"})
    # falls date als Index nötig ist, könnt ihr hier noch big = big.set_index("date")
    big.to_parquet(out_file)
    print(f"Merged {len(files)} files → {out_file}")

if __name__ == "__main__":
    base = "features/aggTrades"
    for symbol_dir in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.isdir(symbol_dir):
            merge_symbol(symbol_dir)
