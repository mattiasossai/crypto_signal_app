#!/usr/bin/env python3
import sys, os
import pandas as pd

tmp = sys.argv[1]
if os.path.exists(tmp) and len(pd.read_parquet(tmp)) > 0:
    sys.exit(0)
else:
    sys.exit(1)
