#!/usr/bin/env python3
import sys
import pandas as pd

# läd tmp-Parquet, exit 0 wenn Zeilen >0, sonst exit 1
df = pd.read_parquet(sys.argv[1])
sys.exit(0 if len(df) > 0 else 1)
