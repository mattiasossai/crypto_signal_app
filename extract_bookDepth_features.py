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
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    sym = os.path.basename(input_dir)
    inc_date = pd.to_datetime(INCEPTION.get(sym, start_date))

    dfs = []
    for fp in files:
        day_str = os.path.basename(fp).replace(".csv", "")
        day = pd.to_datetime(day_str, errors="coerce")
        if pd.isna(day) or day < inc_date:
            continue

        df = pd.read_csv(
            fp,
            header=0 if fp_contains_header(fp) else None,
            names=["timestamp", "percentage", "depth", "notional"],
        )

        # robustes Timestamp-Parsen
        if np.issubdtype(df["timestamp"].dtype, np.number):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        df.set_index("timestamp", inplace=True)
        dfs.append(df)

    if not dfs:
        raise SystemExit("⚠️ Keine bookDepth-Daten gefunden in diesem Zeitraum.")

    full = pd.concat(dfs).sort_index()

    # Fenster auf [start_date, end_date] schneiden
    sd = pd.to_datetime(start_date).tz_localize("UTC")
    ed = pd.to_datetime(end_date).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    full = full[sd:ed]

    # 1) globale Summen
    total_notional = full["notional"].sum()
    total_depth    = full["depth"].sum()

    # 2) feste %-Bins
    mask1   = full["percentage"].abs() <= 1.0
    mask10  = full["percentage"].abs() <= 10.0
    not_1pct = full.loc[mask1, "notional"].sum()
    dep_1pct = full.loc[mask1, "depth"].sum()
    not_10pct = full.loc[mask10, "notional"].sum()
    dep_10pct = full.loc[mask10, "depth"].sum()

    # 3) dynamische Quantil-Bins (Quartile)
    qs = full["percentage"].quantile([0.25, 0.5, 0.75])
    q1, q2, q3 = qs.iloc[0], qs.iloc[1], qs.iloc[2]
    not_q1 = full.loc[full["percentage"] <= q1, "notional"].sum()
    not_q2 = full.loc[(full["percentage"] > q1) & (full["percentage"] <= q2), "notional"].sum()
    not_q3 = full.loc[(full["percentage"] > q2) & (full["percentage"] <= q3), "notional"].sum()
    not_q4 = full.loc[full["percentage"] > q3, "notional"].sum()
    dep_q1 = full.loc[full["percentage"] <= q1, "depth"].sum()
    dep_q2 = full.loc[(full["percentage"] > q1) & (full["percentage"] <= q2), "depth"].sum()
    dep_q3 = full.loc[(full["percentage"] > q2) & (full["percentage"] <= q3), "depth"].sum()
    dep_q4 = full.loc[full["percentage"] > q3, "depth"].sum()

    # 4) Spread in %-Punkten
    pos_min = full.loc[full["percentage"] > 0, "percentage"].min() or 0
    neg_max = full.loc[full["percentage"] < 0, "percentage"].max() or 0
    spread_pct = pos_min + abs(neg_max)

    # 5) Imbalances (absolut & relativ)
    bid_not = full.loc[full["percentage"] < 0, "notional"].sum()
    ask_not = full.loc[full["percentage"] > 0, "notional"].sum()
    bid_dep = full.loc[full["percentage"] < 0, "depth"].sum()
    ask_dep = full.loc[full["percentage"] > 0, "depth"].sum()
    not_imb  = (bid_not - ask_not) / total_notional if total_notional else 0
    dep_imb  = (bid_dep - ask_dep) / total_depth    if total_depth    else 0
    rel_not1 = not_1pct / total_notional if total_notional else 0
    rel_dep1 = dep_1pct / total_depth    if total_depth    else 0

    # 6) Verteilungs-Momente
    n = full["notional"]
    d = full["depth"]
    moments = {
        "not_mean":   n.mean(),
        "not_var":    n.var(ddof=0),
        "not_skew":   skew(n, bias=False),
        "not_kurt":   kurtosis(n, bias=False),
        "dep_mean":   d.mean(),
        "dep_var":    d.var(ddof=0),
        "dep_skew":   skew(d, bias=False),
        "dep_kurt":   kurtosis(d, bias=False),
    }

    # 7) Intraday-Segmente
    seg1 = full.between_time("00:00","07:59")[["notional","depth"]].sum()
    seg2 = full.between_time("08:00","15:59")[["notional","depth"]].sum()
    seg3 = full.between_time("16:00","23:59")[["notional","depth"]].sum()

    # Aufbau des Ergebnis-DataFrames
    out = {
        "total_notional":   total_notional,
        "total_depth":      total_depth,
        "notional_1pct":    not_1pct,
        "depth_1pct":       dep_1pct,
        "rel_notional_1pct":rel_not1,
        "rel_depth_1pct":   rel_dep1,
        "notional_10pct":   not_10pct,
        "depth_10pct":      dep_10pct,
        "notional_q1":      not_q1,
        "notional_q2":      not_q2,
        "notional_q3":      not_q3,
        "notional_q4":      not_q4,
        "depth_q1":         dep_q1,
        "depth_q2":         dep_q2,
        "depth_q3":         dep_q3,
        "depth_q4":         dep_q4,
        "spread_pct":       spread_pct,
        "notional_imbalance":   not_imb,
        "depth_imbalance":      dep_imb,
        **moments,
        "notional_00_08":   seg1["notional"],
        "depth_00_08":      seg1["depth"],
        "notional_08_16":   seg2["notional"],
        "depth_08_16":      seg2["depth"],
        "notional_16_24":   seg3["notional"],
        "depth_16_24":      seg3["depth"],
    }

    df_out = pd.DataFrame([out])
    df_out.index = [sd.normalize()]
    df_out.index.name = "date"
    return df_out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",   required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--start-date",  required=True)
    p.add_argument("--end-date",    required=True)
    args = p.parse_args()

    df = extract_features(args.input_dir, args.start_date, args.end_date)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df.to_parquet(args.output_file, compression="snappy")
    print(f"✅ Wrote features to {args.output_file}")

if __name__ == "__main__":
    main()
