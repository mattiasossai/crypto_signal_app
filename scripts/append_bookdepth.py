#!/usr/bin/env python3
import sys
import pandas as pd

"""
append_bookdepth.py

Argumente:
  1) path zum alten Parquet
  2) path zum neuen tmp_<SYMBOL>.parquet
  3) path zum Out-Parquet

Fügt beide DataFrames zusammen, sortiert nach Index (date) 
und schreibt das Ergebnis mit Snappy-Kompression.
"""
old_path, new_path, out_path = sys.argv[1:]
old = pd.read_parquet(old_path)
new = pd.read_parquet(new_path)
df  = pd.concat([old, new]).sort_index()
df.to_parquet(out_path, compression="snappy")
