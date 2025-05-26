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

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        # 1) Flexible Erkennung der Zeit-Spalte
        if   "date"      in df.columns: ts_col = "date"
        elif "timestamp" in df.columns: ts_col = "timestamp"
        else:
            # nimm erstes int-Feld
            int_cols = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
            if not int_cols:
                raise KeyError(f"No date-like column in {f}")
            ts_col = int_cols[0]
        # 2) In datetime konvertieren
        df["__merge_date"] = pd.to_datetime(df[ts_col], unit="ms", utc=True)
        dfs.append(df)

    # 3) Concat & Sort
    big = pd.concat(dfs, ignore_index=True)
    big = big.drop_duplicates(subset="__merge_date").sort_values("__merge_date")

    # 4) Neuer Date-String & File
    start = big["__merge_date"].min().strftime("%Y-%m-%d")
    end   = big["__merge_date"].max().strftime("%Y-%m-%d")
    out_file = os.path.join(symbol_dir, f"{symbol}-features-{start}_to_{end}.parquet")

    # 5) __merge_date umbenennen & speichern
    big = big.rename(columns={"__merge_date": "date"}).drop(columns=[ts_col], errors="ignore")
    big.to_parquet(out_file)
    print(f"Merged {len(files)} ➞ {out_file}")

if __name__ == "__main__":
    base = "features/bookDepth"
    for symbol_dir in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.isdir(symbol_dir):
            merge_symbol(symbol_dir)
