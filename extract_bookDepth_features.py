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
    """True, wenn die erste Spalte kein Integer ist → Header vorhanden."""
    with open(fp, "r") as f:
        first = f.readline().split(",")[0]
    return not first.isdigit()

def extract_features(input_dir: str, start_date: str, end_date: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    sym = os.path.basename(input_dir)
    inc = pd.to_datetime(INCEPTION.get(sym, start_date))
    dfs = []

    # 1) Einlesen aller Tages-CSVs (ab Inception)
    for fp in files:
        day_str = os.path.basename(fp).replace(".csv", "")
        day = pd.to_datetime(day_str, errors="coerce")
        if pd.isna(day) or day < inc:
            continue

        df = pd.read_csv(
            fp,
            header=0 if fp_contains_header(fp) else None,
            names=["timestamp", "percentage", "depth", "notional"],
        )

        # robustes Timestamp-Parsing
        if np.issubdtype(df["timestamp"].dtype, np.number):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        df.set_index("timestamp", inplace=True)
        dfs.append(df)

    if not dfs:
        raise SystemExit("⚠️ Keine bookDepth-Daten gefunden in diesem Zeitraum.")

    full = pd.concat(dfs).sort_index()

    # 2) Window-Schnitt [start, end]
    sd = pd.to_datetime(start_date).tz_localize("UTC")
    ed = pd.to_datetime(end_date).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    full = full[sd:ed]

    # 3) Globale Summen
    total_notional = full["notional"].sum()
    total_depth    = full["depth"].sum()

    # 4) Feste %-Bins: 0.1%, 1%, 5%, 10%
    p = full["percentage"].abs()
    bins = {
        "0.1": p <= 0.1,
        "1":   p <= 1.0,
        "5":   p <= 5.0,
        "10":  p <= 10.0
    }
    feats = {}
    for k, mask in bins.items():
        feats[f"notional_{k}pct"] = full.loc[mask, "notional"].sum()
        feats[f"depth_{k}pct"]    = full.loc[mask, "depth"].sum()

    # 5) Dynamische Quartil-Bins
    qs = full["percentage"].quantile([0.25,0.5,0.75])
    q1,q2,q3 = qs.iloc[0], qs.iloc[1], qs.iloc[2]
    for i,(low,high) in enumerate([(None,q1),(q1,q2),(q2,q3),(q3,None)], start=1):
        sel = full
        if low is not None:  sel = sel[sel["percentage"]> low]
        if high is not None: sel = sel[sel["percentage"]<=high]
        feats[f"notional_q{i}"] = sel["notional"].sum()
        feats[f"depth_q{i}"]    = sel["depth"].sum()

    # 6) Spread in %-Punkten
    pos_min = full.loc[full["percentage"]>0, "percentage"].min() or 0
    neg_max = full.loc[full["percentage"]<0, "percentage"].max() or 0
    feats["spread_pct"] = pos_min + abs(neg_max)

    # 7) Imbalances absolut & relativ
    bid_not = full.loc[full["percentage"]<0, "notional"].sum()
    ask_not = full.loc[full["percentage"]>0, "notional"].sum()
    bid_dep = full.loc[full["percentage"]<0, "depth"].sum()
    ask_dep = full.loc[full["percentage"]>0, "depth"].sum()
    feats["notional_imbalance"] = (bid_not - ask_not)/total_notional if total_notional else 0
    feats["depth_imbalance"]    = (bid_dep - ask_dep)/total_depth    if total_depth    else 0
    feats["rel_notional_1pct"]  = feats["notional_1pct"]/total_notional if total_notional else 0
    feats["rel_depth_1pct"]     = feats["depth_1pct"]/total_depth     if total_depth    else 0

    # 8) Verteilungs-Momente
    n = full["notional"];  d = full["depth"]
    feats.update({
        "not_mean":  n.mean(),
        "not_var":   n.var(ddof=0),
        "not_skew":  skew(n, bias=False),
        "not_kurt":  kurtosis(n, bias=False),
        "dep_mean":  d.mean(),
        "dep_var":   d.var(ddof=0),
        "dep_skew":  skew(d, bias=False),
        "dep_kurt":  kurtosis(d, bias=False),
    })

    # 9) Intraday-Segmente
    seg1 = full.between_time("00:00","07:59")[["notional","depth"]].sum()
    seg2 = full.between_time("08:00","15:59")[["notional","depth"]].sum()
    seg3 = full.between_time("16:00","23:59")[["notional","depth"]].sum()
    feats.update({
        "notional_00_08": seg1["notional"],
        "depth_00_08":    seg1["depth"],
        "notional_08_16": seg2["notional"],
        "depth_08_16":    seg2["depth"],
        "notional_16_24": seg3["notional"],
        "depth_16_24":    seg3["depth"],
    })

    # 10) Tages-Index auf Mitternacht
    df_out = pd.DataFrame([feats], index=[sd.normalize()])
    df_out.index.name = "date"

    # 11) Lücken-Handling: sicherstellen, dass jeder Kalendertag im Fenster vertreten ist
    all_days = pd.date_range(start=sd.normalize(), end=ed.normalize(), freq="D", tz="UTC")
    df_out = df_out.reindex(all_days, fill_value=0)

    return df_out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",   required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--start-date",  required=True)
    p.add_argument("--end-date",    required=True)
    args = p.parse_args()

    out = extract_features(args.input_dir, args.start_date, args.end_date)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    out.to_parquet(args.output_file, compression="snappy")
    print(f"✅ Wrote features to {args.output_file}")

if __name__ == "__main__":
    main()
