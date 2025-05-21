#!/usr/bin/env python3
import sys
import pandas as pd

"""
append_bookDepth.py

1) Pfad zum alten Parquet
2) Pfad zum neuen tmp_<SYMBOL>.parquet
3) Pfad zum Out-Parquet

Fügt beide DataFrames zusammen, sortiert und schreibt mit Snappy-Kompression.
"""
old_path, new_path, out_path = sys.argv[1:]
df_old = pd.read_parquet(old_path) if old_path else None
df_new = pd.read_parquet(new_path)

if df_old is not None:
    df = pd.concat([df_old, df_new]).sort_index()
else:
    df = df_new

df.to_parquet(out_path, compression="snappy")
