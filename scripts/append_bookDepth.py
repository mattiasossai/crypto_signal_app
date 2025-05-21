#!/usr/bin/env python3
import sys
import pandas as pd

"""
append_bookdepth.py

Argumente:
  1) Path zum alten Parquet
  2) Path zum neuen tmp_<SYMBOL>.parquet
  3) Path zum Out-Parquet
"""
old_path, new_path, out_path = sys.argv[1:]
df_old = pd.read_parquet(old_path)
df_new = pd.read_parquet(new_path)
df = pd.concat([df_old, df_new]).sort_index()
df.to_parquet(out_path, compression="snappy")
