#!/usr/bin/env python3
import sys
import pandas as pd

old_path, new_path, out_path = sys.argv[1:]
old = pd.read_parquet(old_path)
new = pd.read_parquet(new_path)
df = pd.concat([old, new]).sort_index()
df.to_parquet(out_path, compression="snappy")
