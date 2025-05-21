#!/usr/bin/env python3
import sys
import os
import pandas as pd

# argv[1] = Pfad zur tmp_<SYMBOL>.parquet
tmp = sys.argv[1]
if os.path.exists(tmp):
    df = pd.read_parquet(tmp)
    sys.exit(0 if len(df) > 0 else 1)
else:
    sys.exit(1)
