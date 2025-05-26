#!/usr/bin/env python3
import glob, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def compute_bookdepth(df: pd.DataFrame) -> pd.DataFrame:
    # (1) Debug: Columns at entry
    print("    ▶ Columns in DF at start:", df.columns.tolist())

    # 0) Drop alter Index-Overhead
    if "__index_level_0__" in df.columns:
        df = df.drop(columns="__index_level_0__")

    # 1) Datum → daily datetime index
    df["date"] = (
        pd.to_datetime(df["date"], unit="ms", utc=True)
          .dt.floor("D")
    )
    df = df.set_index("date").sort_index()

    # 2) Rollierende Imbalances + Flags
    for w in (7, 14, 21):
        for base in ("notional_imbalance", "depth_imbalance"):
            if base not in df.columns:
                print(f"    ⚠️ Missing base column: {base}")
                continue
            col    = f"{base}_roll_{w}d"
            roll   = df[base].rolling(window=w, min_periods=w).mean()
            df[col]        = roll.fillna(0)
            df[f"has_{col}"] = roll.notna()

    # 3) Microstructure
    if "total_notional" not in df.columns or "total_depth" not in df.columns:
        print("    ⚠️ Missing total_notional or total_depth → skipping microstructure")
    else:
        df["mid_price"] = (df.total_notional / df.total_depth).replace([np.inf,-np.inf], np.nan)
        X = df.total_notional.diff().dropna().values.reshape(-1,1)
        y = df.mid_price.diff().abs().dropna().values
        df["kyle_lambda"] = LinearRegression().fit(X, y).coef_[0] if len(X) >= 2 else 0
        df["ret"]    = df.mid_price.pct_change(fill_method=None).abs().fillna(0)
        df["amihud"] = (df.ret / df.total_notional.replace(0, np.nan)).mean()
        df["vpin"]   = df.notional_imbalance.abs().rolling(window=50, min_periods=1).mean()
        # Liquidity slope
        if "rel_depth_1pct" in df.columns and "spread_pct" in df.columns:
            X2 = df.rel_depth_1pct.values.reshape(-1,1)
            y2 = df.spread_pct.values
            mask = (~np.isnan(X2.flatten())) & (~np.isnan(y2))
            df["liq_slope"] = (
                LinearRegression().fit(X2[mask], y2[mask]).coef_[0]
                if mask.sum() >= 2 else 0
            )
        else:
            print("    ⚠️ Missing rel_depth_1pct or spread_pct → skipping liq_slope")

    # 4) Debug: Columns at exit
    print("    ▶ Columns in DF at end:  ", df.columns.tolist())

    return df.fillna(0).reset_index()

def update_file(fn: str):
    print(f"▶ Updating {fn}")
    df  = pd.read_parquet(fn)
    df2 = compute_bookdepth(df)
    df2.to_parquet(fn)
    print(f"✅ Done {fn}")

if __name__ == "__main__":
    pattern = "features/bookDepth/*/*_to_*.parquet"
    files   = sorted(glob.glob(pattern))
    if not files:
        print("‼ No merged files found with pattern:", pattern)
    for fn in files:
        update_file(fn)
