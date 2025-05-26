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

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        # 1) Falls 'date' als Index existiert: als Spalte zurückholen
        if df.index.name == "date":
            df = df.reset_index()
        # 2) Spalten-Duplikate entfernen (z.B. doppelte 'date')
        df = df.loc[:, ~df.columns.duplicated()]
        # 3) 'date' in datetime konvertieren (ms → UTC)
        if "date" not in df.columns:
            raise KeyError(f"No 'date' column in {f}")
        df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
        frames.append(df)

    # 4) Alles zusammenführen, Duplikate nach date raus, sortieren
    big = pd.concat(frames, ignore_index=True)
    big = big.drop_duplicates(subset="date").sort_values("date")

    # 5) New filename
    start = big["date"].min().strftime("%Y-%m-%d")
    end   = big["date"].max().strftime("%Y-%m-%d")
    out_file = os.path.join(symbol_dir,
                f"{symbol}-features-{start}_to_{end}.parquet")

    # 6) Speichern
    big.to_parquet(out_file)
    print(f"Merged {len(files)} files → {out_file}")

if __name__ == "__main__":
    base = "features/bookDepth"
    for symbol_dir in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.isdir(symbol_dir):
            merge_symbol(symbol_dir)
