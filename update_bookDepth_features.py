#!/usr/bin/env python3
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def compute_bookdepth(df: pd.DataFrame) -> pd.DataFrame:
    # 0) Alten Index-Überrest entfernen
    if "__index_level_0__" in df.columns:
        df = df.drop(columns="__index_level_0__")

    # 1) Unix-ms → daily Datetime-Index
    df["date"] = (
        pd.to_datetime(df["date"], unit="ms", utc=True)
          .dt.floor("D")
    )
    df = df.set_index("date").sort_index()

    # 2) Rollierende Imbalances + Flags (7/14/21 Tage)
    for w in (7, 14, 21):
        # Notional-Imbalance
        col = f"notional_imbalance_roll_{w}d"
        roll = df.notional_imbalance.rolling(window=w, min_periods=w).mean()
        df[col]            = roll.fillna(0)
        df[f"has_{col}"]   = roll.notna()
        # Depth-Imbalance
        col2 = f"depth_imbalance_roll_{w}d"
        roll2 = df.depth_imbalance.rolling(window=w, min_periods=w).mean()
        df[col2]           = roll2.fillna(0)
        df[f"has_{col2}"]  = roll2.notna()

    # 3) Mikrostruktur-Features
    # 3a) Kyle Lambda: |ΔMidPrice| ~ ΔNotional
    df["mid_price"] = (df.total_notional / df.total_depth).replace([np.inf, -np.inf], np.nan)
    X = df.total_notional.diff().dropna().values.reshape(-1,1)
    y = df.mid_price.diff().abs().dropna().values
    df["kyle_lambda"] = LinearRegression().fit(X, y).coef_[0] if len(X) >= 2 else 0

    # 3b) Amihud Illiquidity
    df["ret"]    = df.mid_price.pct_change(fill_method=None).abs().fillna(0)
    df["amihud"] = (df.ret / df.total_notional.replace(0, np.nan)).mean()

    # 3c) VPIN (rolling mean of abs imbalance)
    df["vpin"] = df.notional_imbalance.abs().rolling(window=50, min_periods=1).mean()

    # 3d) Liquidity Slope: rel_depth_1pct → spread_pct
    X2 = df.rel_depth_1pct.values.reshape(-1,1)
    y2 = df.spread_pct.values
    mask = (~np.isnan(X2.flatten())) & (~np.isnan(y2))
    df["liq_slope"] = (
        LinearRegression().fit(X2[mask], y2[mask]).coef_[0]
        if mask.sum() >= 2 else
        0
    )

    # 4) NaNs auffüllen und zurück in Spalten-Format
    return df.fillna(0).reset_index()

def update_file(fn: str):
    df = pd.read_parquet(fn)
    df2 = compute_bookdepth(df)
    df2.to_parquet(fn)
    print(f"✅ Updated {fn}")

if __name__ == "__main__":
    for fn in glob.glob("features/bookDepth/*/*.parquet"):
        update_file(fn)
