#!/usr/bin/env python3
import os
import glob
import argparse
import logging
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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
    symbol     = os.path.basename(input_dir)
    user_sd    = pd.to_datetime(start_date).tz_localize("UTC")
    inception  = pd.to_datetime(INCEPTION.get(symbol, start_date)).tz_localize("UTC")
    real_start = max(user_sd, inception)
    user_ed    = pd.to_datetime(end_date).tz_localize("UTC")

    # Tages-Index von real_start bis user_ed
    days = pd.date_range(
        real_start.normalize(),
        user_ed.normalize(),
        freq="D", tz="UTC"
    )

    rows = []
    for day in days:
        day_str     = day.strftime("%Y-%m-%d")
        fp          = os.path.join(input_dir, f"{symbol}-bookDepth-{day_str}.csv")
        file_exists = os.path.exists(fp)

        # CSV einlesen oder leeren DataFrame anlegen
        if file_exists:
            df = pd.read_csv(
                fp,
                header=0 if fp_contains_header(fp) else None,
                names=["timestamp","percentage","depth","notional"],
            )
            if np.issubdtype(df["timestamp"].dtype, np.number):
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)

            sd       = day
            ed       = day + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
            full     = df[sd:ed]
            has_data = not full.empty
        else:
            full     = pd.DataFrame(
                columns=["percentage","depth","notional"],
                index=pd.DatetimeIndex([], tz="UTC")
            )
            has_data = False

        # Summen & Bins
        total_notional = full["notional"].sum()
        total_depth    = full["depth"].sum()
        mask1          = full["percentage"].abs() <= 1.0
        mask10         = full["percentage"].abs() <= 10.0
        not_1          = full.loc[mask1,  "notional"].sum()
        dep_1          = full.loc[mask1,     "depth"].sum()
        not_10         = full.loc[mask10, "notional"].sum()
        dep_10         = full.loc[mask10,    "depth"].sum()

        # Relativkennzahlen
        rel_not1 = not_1  / total_notional if total_notional else np.nan
        rel_dep1 = dep_1  / total_depth    if total_depth    else np.nan

        # Spread
        pos_min    = full.loc[full["percentage"]>0, "percentage"].min() or 0
        neg_max    = full.loc[full["percentage"]<0, "percentage"].max() or 0
        spread_pct = pos_min + abs(neg_max)

        # Imbalance
        bid_not = full.loc[full["percentage"]<0, "notional"].sum()
        ask_not = full.loc[full["percentage"]>0, "notional"].sum()
        not_imb = (bid_not-ask_not)/total_notional if total_notional else np.nan

        bid_dep = full.loc[full["percentage"]<0, "depth"].sum()
        ask_dep = full.loc[full["percentage"]>0, "depth"].sum()
        dep_imb = (bid_dep-ask_dep)/total_depth    if total_depth    else np.nan

        # Verteilungs-Momente
        n = full["notional"]; d = full["depth"]
        moments = {
            "not_mean":  n.mean(),
            "not_var":   n.var(ddof=0),
            "not_skew":  skew(n, bias=False) if len(n)>1 else np.nan,
            "not_kurt":  kurtosis(n, bias=False) if len(n)>1 else np.nan,
            "dep_mean":  d.mean(),
            "dep_var":   d.var(ddof=0),
            "dep_skew":  skew(d, bias=False) if len(d)>1 else np.nan,
            "dep_kurt":  kurtosis(d, bias=False) if len(d)>1 else np.nan,
        }

        # Intraday-Segmente
        seg1 = full.between_time("00:00","07:59")[["notional","depth"]]
        seg2 = full.between_time("08:00","15:59")[["notional","depth"]]
        seg3 = full.between_time("16:00","23:59")[["notional","depth"]]
        seg1s = seg1.sum(min_count=1)
        seg2s = seg2.sum(min_count=1)
        seg3s = seg3.sum(min_count=1)

        # Segment-Flags
        has_00_08 = not seg1.empty
        has_08_16 = not seg2.empty
        has_16_24 = not seg3.empty

        # Gesamt-Flags
        has_notional = total_notional > 0
        has_depth    = total_depth    > 0

        # Wenn gar keine Daten im Tag, alles numerische auf NaN
        if not has_data:
            total_notional = total_depth = np.nan
            not_1 = dep_1 = not_10 = dep_10 = np.nan
            rel_not1 = rel_dep1 = np.nan
            spread_pct = not_imb = dep_imb = np.nan
            for k in moments: moments[k] = np.nan
            seg1s[:] = np.nan; seg2s[:] = np.nan; seg3s[:] = np.nan

        rows.append({
            "date":                day,
            "file_exists":         file_exists,
            "has_data":            has_data,
            "has_notional":        has_notional,
            "has_depth":           has_depth,
            "total_notional":      total_notional,
            "total_depth":         total_depth,
            "notional_1pct":       not_1,
            "depth_1pct":          dep_1,
            "rel_notional_1pct":   rel_not1,
            "rel_depth_1pct":      rel_dep1,
            "notional_10pct":      not_10,
            "depth_10pct":         dep_10,
            "spread_pct":          spread_pct,
            "notional_imbalance":  not_imb,
            "depth_imbalance":     dep_imb,
            **moments,
            "notional_00_08":      seg1s["notional"],
            "depth_00_08":         seg1s["depth"],
            "notional_08_16":      seg2s["notional"],
            "depth_08_16":         seg2s["depth"],
            "notional_16_24":      seg3s["notional"],
            "depth_16_24":         seg3s["depth"],
            "has_00_08":           has_00_08,
            "has_08_16":           has_08_16,
            "has_16_24":           has_16_24,
        })

    df_out = pd.DataFrame(rows).set_index("date")
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
