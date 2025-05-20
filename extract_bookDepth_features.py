#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Symbol‐spezifische Inception‐Daten
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

def extract_for_days(input_dir: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Erzeuge Tages‐Features für alle Tage von start..end inkl."""
    symbol = os.path.basename(input_dir)
    days = pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")
    rows = []
    for day in days:
        day_str = day.strftime("%Y-%m-%d")
        fp = os.path.join(input_dir, f"{symbol}-bookDepth-{day_str}.csv")
        # CSV einlesen oder Empty-DF
        if os.path.exists(fp):
            df = pd.read_csv(fp,
                header=0 if fp_contains_header(fp) else None,
                names=["timestamp","percentage","depth","notional"])
            if np.issubdtype(df["timestamp"].dtype, np.number):
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df.set_index("timestamp", inplace=True)
            full = df[day : day + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)]
            has_data = not full.empty
        else:
            full = pd.DataFrame(columns=["percentage","depth","notional"],
                                index=pd.DatetimeIndex([], tz="UTC"))
            has_data = False

        # 1) globale Summen & Bins
        total_notional = full["notional"].sum()
        total_depth    = full["depth"].sum()
        mask1   = full["percentage"].abs() <= 1.0
        mask10  = full["percentage"].abs() <= 10.0
        n1 = full.loc[mask1, "notional"].sum()
        d1 = full.loc[mask1, "depth"].sum()
        n10= full.loc[mask10,"notional"].sum()
        d10= full.loc[mask10,"depth"].sum()

        # 2) Relativkennzahlen
        rel_n1 = n1/total_notional if total_notional else np.nan
        rel_d1 = d1/total_depth    if total_depth    else np.nan

        # 3) Spread
        p_min = full.loc[full["percentage"]>0,"percentage"].min() or 0
        n_max = full.loc[full["percentage"]<0,"percentage"].max() or 0
        spread_pct = p_min + abs(n_max)

        # 4) Imbalances
        bid_n = full.loc[full["percentage"]<0,"notional"].sum()
        ask_n = full.loc[full["percentage"]>0,"notional"].sum()
        imb_n = (bid_n-ask_n)/total_notional if total_notional else np.nan
        bid_d = full.loc[full["percentage"]<0,"depth"].sum()
        ask_d = full.loc[full["percentage"]>0,"depth"].sum()
        imb_d = (bid_d-ask_d)/total_depth    if total_depth    else np.nan

        # 5) Verteilungs‐Momente
        n = full["notional"]; d = full["depth"]
        moments = {
            "not_mean": n.mean(),
            "not_var":  n.var(ddof=0),
            "not_skew": skew(n, bias=False) if len(n)>1 else np.nan,
            "not_kurt": kurtosis(n, bias=False) if len(n)>1 else np.nan,
            "dep_mean": d.mean(),
            "dep_var":  d.var(ddof=0),
            "dep_skew": skew(d, bias=False) if len(d)>1 else np.nan,
            "dep_kurt": kurtosis(d, bias=False) if len(d)>1 else np.nan,
        }

        # 6) Intraday‐Segmente
        seg1 = full.between_time("00:00","07:59")[["notional","depth"]].sum(min_count=1)
        seg2 = full.between_time("08:00","15:59")[["notional","depth"]].sum(min_count=1)
        seg3 = full.between_time("16:00","23:59")[["notional","depth"]].sum(min_count=1)

        rows.append({
            "date": day,
            "file_exists": os.path.exists(fp),
            "has_data":    has_data,
            "has_notional": total_notional>0,
            "has_depth":    total_depth>0,
            "total_notional":     total_notional,
            "total_depth":        total_depth,
            "notional_1pct":      n1,
            "depth_1pct":         d1,
            "rel_notional_1pct":  rel_n1,
            "rel_depth_1pct":     rel_d1,
            "notional_10pct":     n10,
            "depth_10pct":        d10,
            "spread_pct":         spread_pct,
            "notional_imbalance": imb_n,
            "depth_imbalance":    imb_d,
            **moments,
            "notional_00_08": seg1["notional"], "depth_00_08": seg1["depth"],
            "notional_08_16": seg2["notional"], "depth_08_16": seg2["depth"],
            "notional_16_24": seg3["notional"], "depth_16_24": seg3["depth"],
            "has_00_08": not pd.isna(seg1["notional"]),
            "has_08_16": not pd.isna(seg2["notional"]),
            "has_16_24": not pd.isna(seg3["notional"]),
        })

    df = pd.DataFrame(rows).set_index("date")
    df.index.name = "date"
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",   required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--start-date",  required=True)
    p.add_argument("--end-date",    required=True)
    args = p.parse_args()

    out_dir = os.path.dirname(args.output_file)
    symbol  = os.path.basename(args.input_dir)
    user_sd = pd.to_datetime(args.start_date).tz_localize("UTC")
    user_ed = pd.to_datetime(args.end_date).tz_localize("UTC")

    # --- Resume/Append Logic ---
    pattern = os.path.join(out_dir, f"{symbol}-features-*.parquet")
    old_files = glob.glob(pattern)
    if old_files:
        # wähle Datei mit höchstem Datum
        best, best_date = None, pd.Timestamp.min
        for f in old_files:
            tmp = pd.read_parquet(f)
            m = tmp.index.max()
            if m>best_date:
                best_date, best = m, f
        df_old    = pd.read_parquet(best)
        new_start = (best_date + pd.Timedelta(days=1)).normalize()
        print(f"→ Append mode, resume at {new_start.date()}")
    else:
        df_old    = None
        inc = pd.to_datetime(INCEPTION[symbol]).tz_localize("UTC")
        new_start = max(user_sd, inc).normalize()
        print(f"→ Fresh mode, start at {new_start.date()}")

    if new_start > user_ed:
        print("ℹ️ Nothing new to append."); return

    df_new = extract_for_days(args.input_dir, new_start, user_ed)

    # merge und schreiben
    df = pd.concat([df_old, df_new]).sort_index() if df_old is not None else df_new
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(args.output_file, compression="snappy")
    print(f"✅ Wrote combined features to {args.output_file} ({len(df)} days)")

if __name__=="__main__":
    main()
