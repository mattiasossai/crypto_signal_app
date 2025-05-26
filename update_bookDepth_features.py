#!/usr/bin/env python3
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def compute_bookdepth(df: pd.DataFrame) -> pd.DataFrame:
# Alten Index-Überrest entfernen, falls er da ist:
if "__index_level_0__" in df.columns:
    df = df.drop(columns="__index_level_0__")
    
    # 1) Datum → daily index
    df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True).dt.floor("D")
    df = df.set_index("date").sort_index()

    # 2) Rollierende Imbalances + Flags
    for w in (7, 14, 21):
        col = f"notional_imbalance_roll_{w}d"
        roll = df.notional_imbalance.rolling(window=w, min_periods=w).mean()
        df[col] = roll.fillna(0)
        df[f"has_{col}"] = roll.notna()

        col2 = f"depth_imbalance_roll_{w}d"
        roll2 = df.depth_imbalance.rolling(window=w, min_periods=w).mean()
        df[col2] = roll2.fillna(0)
        df[f"has_{col2}"] = roll2.notna()

    # 3) Kyle Lambda (Regression ΔNotional → |ΔPreis|)
    # mid_price = total_notional / total_depth
    df["mid_price"] = (df.total_notional / df.total_depth).replace([np.inf, -np.inf], np.nan)
    X = df.total_notional.diff().dropna().values.reshape(-1,1)
    y = df.mid_price.diff().abs().dropna().values
    λ = LinearRegression().fit(X, y).coef_[0] if len(X) else 0
    df["kyle_lambda"] = λ

    # 4) Amihud Illiquidity
    df["ret"] = df.mid_price.pct_change().abs().fillna(0)
    df["amihud"] = (df.ret / df.total_notional.replace(0, np.nan)).mean()
    
    # 5) VPIN (vereinfachte Version)
    df["vpin"] = (
      df.notional_imbalance.abs()
        .rolling(window=50, min_periods=1)
        .mean()
    )

    # 6) Liquidity-Slope (rel_depth_1pct → spread_pct)
    X2 = df.rel_depth_1pct.values.reshape(-1,1)
    y2 = df.spread_pct.values
    slope = LinearRegression().fit(X2, y2).coef_[0] if len(X2) else 0
    df["liq_slope"] = slope

    # 7) NaN → 0
    df.fillna(0, inplace=True)
    return df.reset_index()

def update_file(fn: str):
    df = pd.read_parquet(fn)
    df2 = compute_bookdepth(df)
    df2.to_parquet(fn)
    print(f"✅ Updated {fn} with bookDepth-features.")

if __name__ == "__main__":
    for fn in glob.glob("features/bookDepth/*/*.parquet"):
        update_file(fn)
