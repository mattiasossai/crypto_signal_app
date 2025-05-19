#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Symbol-spezifische Inception-Daten
INCEPTION = {
    "BTCUSDT": "2023-01-01",
    "ETHUSDT": "2023-01-01",
    "BNBUSDT": "2023-01-01",
    "SOLUSDT": "2023-01-01",
    "XRPUSDT": "2023-01-06",
    "ENAUSDT": "2024-04-02",
}

def fp_contains_header(fp: str) -> bool:
    with open(fp, "r") as f:
        first = f.readline().split(",")[0]
    return not first.isdigit()

def extract_features(input_dir: str, start_date: str, end_date: str) -> pd.DataFrame:
    symbol = os.path.basename(input_dir)
    user_sd = pd.to_datetime(start_date).tz_localize("UTC")
    inception = pd.to_datetime(INCEPTION.get(symbol, start_date)).tz_localize("UTC")
    real_start = max(user_sd, inception)
    user_ed = pd.to_datetime(end_date).tz_localize("UTC")

    # build daily index
    dates = pd.date_range(
        start=real_start.normalize(),
        end=user_ed.normalize(),
        freq="D",
        tz="UTC"
    )

    rows = []
    for day in dates:
        day_str = day.strftime("%Y-%m-%d")
        fp = os.path.join(input_dir, f"{symbol}-bookDepth-{day_str}.csv")
        if os.path.exists(fp):
            df = pd.read_csv(
                fp,
                header=0 if fp_contains_header(fp) else None,
                names=["timestamp", "percentage", "depth", "notional"],
            )
            # Timestamp robust
            if np.issubdtype(df["timestamp"].dtype, np.number):
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)
            # slice exact day
            sd = day
            ed = day + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
            full = df[sd:ed]
        else:
            full = pd.DataFrame(columns=["percentage","depth","notional"])

        total_notional = full["notional"].sum()
        total_depth    = full["depth"].sum()

        mask1 = full["percentage"].abs() <= 1.0
        mask10 = full["percentage"].abs() <= 10.0
        notional_1pct  = full.loc[mask1,  "notional"].sum()
        depth_1pct     = full.loc[mask1,     "depth"].sum()
        notional_10pct = full.loc[mask10, "notional"].sum()
        depth_10pct    = full.loc[mask10,    "depth"].sum()

        # relative
        rel_not1 = notional_1pct / total_notional if total_notional else 0
        rel_dep1 = depth_1pct    / total_depth    if total_depth    else 0

        # spread
        pos_min = full.loc[full["percentage"] > 0, "percentage"].min() or 0
        neg_max = full.loc[full["percentage"] < 0, "percentage"].max() or 0
        spread_pct = pos_min + abs(neg_max)

        # imbalance
        bid_not = full.loc[full["percentage"] < 0, "notional"].sum()
        ask_not = full.loc[full["percentage"] > 0, "notional"].sum()
        not_imb = (bid_not - ask_not) / total_notional if total_notional else 0

        bid_dep = full.loc[full["percentage"] < 0, "depth"].sum()
        ask_dep = full.loc[full["percentage"] > 0, "depth"].sum()
        dep_imb = (bid_dep - ask_dep) / total_depth if total_depth else 0

        # distribution moments
        n = full["notional"]
        d = full["depth"]
        moments = {
            "not_mean":  n.mean(),
            "not_var":   n.var(ddof=0),
            "not_skew":  skew(n, bias=False),
            "not_kurt":  kurtosis(n, bias=False),
            "dep_mean":  d.mean(),
            "dep_var":   d.var(ddof=0),
            "dep_skew":  skew(d, bias=False),
            "dep_kurt":  kurtosis(d, bias=False),
        }

        # intraday segments
        seg1 = full.between_time("00:00","07:59")[["notional","depth"]].sum()
        seg2 = full.between_time("08:00","15:59")[["notional","depth"]].sum()
        seg3 = full.between_time("16:00","23:59")[["notional","depth"]].sum()

        rows.append({
            "date": day,
            "total_notional":    total_notional,
            "total_depth":       total_depth,
            "notional_1pct":     notional_1pct,
            "depth_1pct":        depth_1pct,
            "rel_notional_1pct": rel_not1,
            "rel_depth_1pct":    rel_dep1,
            "notional_10pct":    notional_10pct,
            "depth_10pct":       depth_10pct,
            "spread_pct":        spread_pct,
            "notional_imbalance": not_imb,
            "depth_imbalance":    dep_imb,
            **moments,
            "notional_00_08":   seg1["notional"],
            "depth_00_08":      seg1["depth"],
            "notional_08_16":   seg2["notional"],
            "depth_08_16":      seg2["depth"],
            "notional_16_24":   seg3["notional"],
            "depth_16_24":      seg3["depth"],
        })

    df_out = pd.DataFrame(rows).set_index("date")
    # schon reindexed – keine weiteren Lücken mehr
    df_out.index = df_out.index.tz_convert("UTC")
    df_out.index.name = "date"
    return df_out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",   required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--start-date",  required=True)
    p.add_argument("--end-date",    required=True)
    args = p.parse_args()

    out = extract_features(
        args.input_dir,
        args.start_date,
        args.end_date
    )
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    out.to_parquet(args.output_file, compression="snappy")
    print(f"✅ Wrote features to {args.output_file}")

if __name__ == "__main__":
    main()
