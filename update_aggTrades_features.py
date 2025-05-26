#!/usr/bin/env python3
import glob
import numpy as np
import pandas as pd

def compute_orderflow(df: pd.DataFrame) -> pd.Series:
    # 1) Timestamp sortieren
    if not np.issubdtype(df.timestamp.dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.sort_values("timestamp")

    # 2) Signed volume & CVD
    df["signed_vol"] = df.quantity * np.where(df.is_buyer_maker, 1, -1)
    df["cvd"] = df.signed_vol.cumsum()

    out = {}

    # 3) Rolling CVD-Stats
    windows = {"15s": "15s", "1h": "1h", "4h": "4h"}
    for label, freq in windows.items():
        roll = df.cvd.rolling(freq, on="timestamp")
        for stat in ("max", "mean", "std"):
            col = f"cvd_{label}_{stat}"
            val = getattr(roll, stat)().iloc[-1]
            out[col] = 0 if pd.isna(val) else val

    # 4) Tick-Imbalance
    ticks = df.is_buyer_maker.value_counts()
    b = ticks.get(True, 0)
    s = ticks.get(False, 0)
    out["tick_imbalance_pct"] = (b - s) / (b + s) if (b + s) else 0

    # 5) Sweep-Rate (Buyer-Spikes > µ+3σ in 1min-Fenstern)
    buy = df[df.is_buyer_maker]
    sums = buy.quantity.rolling("1min", on="timestamp").sum()
    mu, sigma = sums.mean(), sums.std()
    spikes = int((sums > mu + 3*sigma).sum())
    total_windows = int(((df.timestamp.max() - df.timestamp.min())
                         .total_seconds() // 60) + 1)
    out["sweep_rate"] = spikes / total_windows if total_windows else 0

    # 6) Iceberg-Detection (Heuristik)
    iceberg = 0
    mean_q = df.quantity.mean()
    seq = 0
    prev_ts = None
    for qty, ts in zip(df.quantity, df.timestamp):
        if prev_ts and (ts - prev_ts).total_seconds() <= 10 and qty < mean_q:
            seq += 1
        else:
            if seq >= 5:
                iceberg += 1
            seq = 1
        prev_ts = ts
    out["iceberg_count"] = iceberg

    return pd.Series(out)

def update_file(fn: str):
    df = pd.read_parquet(fn)
    new_feats = compute_orderflow(df)
    # 7) Anhängen und Fillna
    for c,v in new_feats.items():
        df[c] = v
    df.fillna(0, inplace=True)
    df.to_parquet(fn)
    print(f"✅ Updated {fn} with {len(new_feats)} aggTrades-features.")

if __name__ == "__main__":
    for fn in glob.glob("features/aggTrades/*/*.parquet"):
        update_file(fn)
