#!/usr/bin/env python3
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def compute_bookdepth(df: pd.DataFrame) -> pd.DataFrame:
    # 0) Alten Index-Überrest entfernen
    if "__index_level_0__" in df.columns:
        df = df.drop(columns="__index_level_0__")

    # 1) Unix-ms → täglicher Datetime-Index
    df["date"] = (
        pd.to_datetime(df["date"], unit="ms", utc=True)
          .dt.floor("D")
    )
    df = df.set_index("date").sort_index()

    # 2) Rollierende Imbalances + Flags (7 / 14 / 21 Tage)
    for w in (7, 14, 21):
        for base in ("notional_imbalance", "depth_imbalance"):
            col       = f"{base}_roll_{w}d"
            roll      = df[base].rolling(window=w, min_periods=w).mean()
            df[col]   = roll.fillna(0)
            df[f"has_{col}"] = roll.notna()

    # 3) VPIN (50-Sample Rolling Mean)
    df["vpin"] = df.notional_imbalance.abs().rolling(window=50, min_periods=1).mean()

    # 4) Dynamische Microstructure-Features (30-Tage-Rolling)
    window = 30
    # Hilfsspalten für Microstructure
    df["mid_price"] = (df.total_notional / df.total_depth).replace([np.inf, -np.inf], np.nan)
    df["ret"]       = df.mid_price.pct_change(fill_method=None).abs().fillna(0)

    # 4a) Rolling Kyle Lambda
    kl = []
    for i in range(len(df)):
        if i < window:
            kl.append(0)
        else:
            sub = df.iloc[i-window+1 : i+1]
            X = sub.total_notional.diff().dropna().values.reshape(-1,1)
            y = sub.mid_price.diff().abs().dropna().values
            kl.append(LinearRegression().fit(X, y).coef_[0] if len(X) >= 2 else 0)
    df[f"kyle_lambda_roll_{window}d"] = kl
    df[f"has_kyle_lambda_roll_{window}d"] = [i >= window-1 for i in range(len(df))]

    # 4b) Rolling Amihud Illiquidity
    ai = df.ret / df.total_notional.replace(0, np.nan)
    roll_ai = ai.rolling(window=window, min_periods=window).mean().fillna(0)
    df[f"amihud_roll_{window}d"]    = roll_ai
    df[f"has_amihud_roll_{window}d"] = roll_ai.notna()

    # 4c) Rolling Liquidity Slope
    ls = []
    for i in range(len(df)):
        if i < window:
            ls.append(0)
        else:
            sub = df.iloc[i-window+1 : i+1]
            X2 = sub.rel_depth_1pct.values.reshape(-1,1)
            y2 = sub.spread_pct.values
            mask = (~np.isnan(X2.flatten())) & (~np.isnan(y2))
            ls.append(LinearRegression().fit(X2[mask], y2[mask]).coef_[0] if mask.sum() >= 2 else 0)
    df[f"liq_slope_roll_{window}d"]    = ls
    df[f"has_liq_slope_roll_{window}d"] = [i >= window-1 for i in range(len(df))]

    # 5) Alte konstante und Hilfsspalten entfernen
    drop_cols = ["kyle_lambda", "amihud", "liq_slope", "mid_price", "ret"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # 6) NaNs auffüllen und Index zurücksetzen
    df = df.fillna(0).reset_index()
    return df

def update_file(fn: str):
    df = pd.read_parquet(fn)
    df2 = compute_bookdepth(df)
    df2.to_parquet(fn)
    print(f"✅ Updated {fn}")

if __name__ == "__main__":
    for fn in glob.glob("features/bookDepth/*/*_to_*.parquet"):
        update_file(fn)
