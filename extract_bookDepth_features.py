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
    symbol   = os.path.basename(input_dir)
    user_sd  = pd.to_datetime(start_date).tz_localize("UTC")
    inception= pd.to_datetime(INCEPTION.get(symbol, start_date)).tz_localize("UTC")
    real_sd  = max(user_sd, inception)
    user_ed  = pd.to_datetime(end_date).tz_localize("UTC")

    # 1) täglichen Kalender bauen (inkl. Inception)
    dates = pd.date_range(
        start=real_sd.normalize(),
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
            # Timestamp robust parsen
            if np.issubdtype(df["timestamp"].dtype, np.number):
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)
            # exakt auf den Tag beschneiden
            sd = day
            ed = day + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
            full = df[sd:ed]
            has_data = True
        else:
            # Datei fehlt → leeres DataFrame
            full = pd.DataFrame(columns=["percentage","depth","notional"])
            has_data = False

        # grundlegende Summen
        total_notional = full["notional"].sum()
        total_depth    = full["depth"].sum()

        # feste %-Bins
        mask1   = full["percentage"].abs() <= 1.0
        mask10  = full["percentage"].abs() <= 10.0
        n1      = full.loc[mask1,  "notional"].sum()
        d1      = full.loc[mask1,     "depth"].sum()
        n10     = full.loc[mask10, "notional"].sum()
        d10     = full.loc[mask10,    "depth"].sum()

        # relative Kennzahlen
        rel_n1 = n1  / total_notional if total_notional else np.nan
        rel_d1 = d1  / total_depth    if total_depth    else np.nan

        # Spread in %-Punkten
        pos_min   = full.loc[full["percentage"] > 0, "percentage"].min() or 0
        neg_max   = full.loc[full["percentage"] < 0, "percentage"].max() or 0
        spread_pct= pos_min + abs(neg_max)

        # Imbalance absolut & relativ
        bid_n     = full.loc[full["percentage"] < 0, "notional"].sum()
        ask_n     = full.loc[full["percentage"] > 0, "notional"].sum()
        not_imb   = (bid_n - ask_n) / total_notional if total_notional else np.nan

        bid_d     = full.loc[full["percentage"] < 0, "depth"].sum()
        ask_d     = full.loc[full["percentage"] > 0, "depth"].sum()
        dep_imb   = (bid_d - ask_d) / total_depth    if total_depth    else np.nan

        # Verteilungs-Momente
        n_series = full["notional"]
        d_series = full["depth"]
        moments = {
            "not_mean":  n_series.mean(),
            "not_var":   n_series.var(ddof=0),
            "not_skew":  skew(n_series, bias=False),
            "not_kurt":  kurtosis(n_series, bias=False),
            "dep_mean":  d_series.mean(),
            "dep_var":   d_series.var(ddof=0),
            "dep_skew":  skew(d_series, bias=False),
            "dep_kurt":  kurtosis(d_series, bias=False),
        }

        # Intraday-Segmente (mit min_count=0 → 0 wenn leer)
        seg1 = full.between_time("00:00","07:59")[["notional","depth"]].sum(min_count=1)
        seg2 = full.between_time("08:00","15:59")[["notional","depth"]].sum(min_count=1)
        seg3 = full.between_time("16:00","23:59")[["notional","depth"]].sum(min_count=1)

        rows.append({
            "date": day,
            "has_data":          has_data,
            "total_notional":    total_notional,
            "total_depth":       total_depth,
            "notional_1pct":     n1,
            "depth_1pct":        d1,
            "rel_notional_1pct": rel_n1,
            "rel_depth_1pct":    rel_d1,
            "notional_10pct":    n10,
            "depth_10pct":       d10,
            "spread_pct":        spread_pct,
            "notional_imbalance":not_imb,
            "depth_imbalance":   dep_imb,
            **moments,
            "notional_00_08":    seg1["notional"],
            "depth_00_08":       seg1["depth"],
            "notional_08_16":    seg2["notional"],
            "depth_08_16":       seg2["depth"],
            "notional_16_24":    seg3["notional"],
            "depth_16_24":       seg3["depth"],
            "has_00_08":         not pd.isna(seg1["notional"]),
            "has_08_16":         not pd.isna(seg2["notional"]),
            "has_16_24":         not pd.isna(seg3["notional"]),
        })

    df_out = pd.DataFrame(rows).set_index("date")
    # Index ist bereits vollständiger Kalendertag, tz-aware
    df_out.index.name = "date"
    return df_out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",   required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--start-date",  required=True)
    p.add_argument("--end-date",    required=True)
    args = p.parse_args()

    df = extract_features(
        args.input_dir,
        args.start_date,
        args.end_date
    )
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df.to_parquet(args.output_file, compression="snappy")
    print(f"✅ Wrote features to {args.output_file}")

if __name__ == "__main__":
    main()
