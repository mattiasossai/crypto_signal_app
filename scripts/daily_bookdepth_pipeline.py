#!/usr/bin/env python3
import os
import glob
import shutil
import datetime
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression

# Symbol‐spezifische Inception‐Daten (tz-aware Strings)
INCEPTION = {
    "BTCUSDT": "2023-01-01",
    "ETHUSDT": "2023-01-01",
    "BNBUSDT": "2023-01-01",
    "SOLUSDT": "2023-01-01",
    "XRPUSDT": "2023-01-06",
    "ENAUSDT": "2024-04-02",
}

def download_and_unzip(symbol: str, days, raw_dir: str):
    os.makedirs(raw_dir, exist_ok=True)
    for day in days:
        ds = day.strftime("%Y-%m-%d")
        zip_name = f"{symbol}-bookDepth-{ds}.zip"
        zip_path = os.path.join(raw_dir, zip_name)
        url = f"https://data.binance.vision/data/futures/um/daily/bookDepth/{symbol}/{zip_name}"
        print(f"→ FETCH {symbol} {ds}: {url}")
        res = subprocess.run(["curl", "-sSf", url, "-o", zip_path], capture_output=True)
        if res.returncode == 0:
            subprocess.run(["unzip", "-q", "-o", zip_path, "-d", raw_dir], check=True)
            os.remove(zip_path)
        else:
            print(f"   ⚠️  {symbol} {ds}: ZIP nicht gefunden")

def extract_raw_for_days(symbol: str, raw_dir: str, start, end) -> pd.DataFrame:
    # Tage als tz-aware Range
    days = pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")
    rows = []
    for day in days:
        ds = day.strftime("%Y-%m-%d")
        csv_fp = os.path.join(raw_dir, f"{symbol}-bookDepth-{ds}.csv")

        if os.path.exists(csv_fp):
            df_raw = pd.read_csv(csv_fp)
            print(f"   • {symbol} {ds}: eingelesen {len(df_raw)} Zeilen")
            if str(df_raw.columns[0]).isdigit():
                df_raw.columns = ["timestamp","percentage","depth","notional"]
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"],
                                                 unit="ms", utc=True, errors="coerce")
            df_raw.set_index("timestamp", inplace=True)
            df_raw.sort_index(inplace=True)

            # --- HALBOFFENES INTERVALL statt label-slice ---
            next_day = day + pd.Timedelta(days=1)
            mask = (df_raw.index >= day) & (df_raw.index < next_day)
            sl = df_raw.loc[mask]
            print(f"      → Slice: {len(sl)} Zeilen")
            has_data = not sl.empty
        else:
            print(f"   • {symbol} {ds}: keine CSV")
            sl = pd.DataFrame(columns=["percentage","depth","notional"],
                              index=pd.DatetimeIndex([], tz="UTC"))
            has_data = False

        # --- alles wie gehabt: Aggregationen, Momente, Segmente ---
        tot_not = sl["notional"].sum()
        tot_dep = sl["depth"].sum()
        mask1   = sl["percentage"].abs() <= 1.0
        mask10  = sl["percentage"].abs() <= 10.0
        n1,   d1   = sl.loc[mask1,"notional"].sum(), sl.loc[mask1,"depth"].sum()
        n10,  d10  = sl.loc[mask10,"notional"].sum(), sl.loc[mask10,"depth"].sum()
        rel_n1 = n1/tot_not if tot_not else np.nan
        rel_d1 = d1/tot_dep if tot_dep else np.nan

        p_min    = sl.loc[sl["percentage"]>0,"percentage"].min() or 0
        n_max    = sl.loc[sl["percentage"]<0,"percentage"].max() or 0
        spread_pct = p_min + abs(n_max)

        bid_n = sl.loc[sl["percentage"]<0,"notional"].sum()
        ask_n = sl.loc[sl["percentage"]>0,"notional"].sum()
        imb_n  = (bid_n-ask_n)/tot_not if tot_not else np.nan
        bid_d = sl.loc[sl["percentage"]<0,"depth"].sum()
        ask_d = sl.loc[sl["percentage"]>0,"depth"].sum()
        imb_d  = (bid_d-ask_d)/tot_dep if tot_dep else np.nan

        n, d = sl["notional"], sl["depth"]
        moments = {
            "not_mean": n.mean(),
            "not_var":  n.var(ddof=0),
            "not_skew": skew(n, bias=False)   if len(n)>1 else np.nan,
            "not_kurt": kurtosis(n, bias=False) if len(n)>1 else np.nan,
            "dep_mean": d.mean(),
            "dep_var":  d.var(ddof=0),
            "dep_skew": skew(d, bias=False)   if len(d)>1 else np.nan,
            "dep_kurt": kurtosis(d, bias=False) if len(d)>1 else np.nan,
        }

        seg1 = sl.between_time("00:00","07:59")[["notional","depth"]].sum(min_count=1)
        seg2 = sl.between_time("08:00","15:59")[["notional","depth"]].sum(min_count=1)
        seg3 = sl.between_time("16:00","23:59")[["notional","depth"]].sum(min_count=1)

        rows.append({
            "date":               day,
            "file_exists":        has_data,
            "has_notional":       tot_not>0,
            "has_depth":          tot_dep>0,
            "total_notional":     tot_not,
            "total_depth":        tot_dep,
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
            "notional_00_08":     seg1["notional"],
            "depth_00_08":        seg1["depth"],
            "notional_08_16":     seg2["notional"],
            "depth_08_16":        seg2["depth"],
            "notional_16_24":     seg3["notional"],
            "depth_16_24":        seg3["depth"],
            "has_00_08":          not pd.isna(seg1["notional"]),
            "has_08_16":          not pd.isna(seg2["notional"]),
            "has_16_24":          not pd.isna(seg3["notional"]),
        })

    df = pd.DataFrame(rows).set_index("date")
    df.index.name = "date"
    return df

def add_rolling_micro(df: pd.DataFrame) -> pd.DataFrame:
    # … rollierende und microstructure-Features wie gehabt …
    # (unverändert)

    return df.drop(columns=["mid_price","ret"], errors="ignore")

def main():
    yesterday = datetime.datetime.utcnow().date() - datetime.timedelta(days=1)
    y_ts = pd.to_datetime(yesterday).tz_localize("UTC")
    y_str = y_ts.strftime("%Y-%m-%d")

    base = "features/bookDepth"
    os.makedirs(base, exist_ok=True)

    for sym, inc in INCEPTION.items():
        inc_dt = pd.to_datetime(inc, utc=True).floor("D")
        outd   = os.path.join(base, sym)
        os.makedirs(outd, exist_ok=True)

        # Resume
        pattern = os.path.join(outd, f"{sym}-features-*.parquet")
        files   = glob.glob(pattern)
        if files:
            latest = max(files, key=lambda f: pd.read_parquet(f).index.max())
            df_old = pd.read_parquet(latest)
            start  = (df_old.index.max() + pd.Timedelta(days=1)).normalize()
        else:
            df_old = pd.DataFrame()
            start  = inc_dt

        # ensure tz-awareness
        if start.tzinfo is None:
            start = start.tz_localize("UTC")

        if start > y_ts:
            print(f"ℹ️ {sym}: bis {y_str} aktuell → skip")
            continue

        days   = pd.date_range(start, y_ts, freq="D", tz="UTC")
        rawdir = os.path.join("raw/bookDepth", sym)
        print(f"→ {sym}: Downloading {len(days)} days…")
        download_and_unzip(sym, days, rawdir)

        df_new = extract_raw_for_days(sym, rawdir, start, y_ts)
        shutil.rmtree(rawdir)

        df_all = pd.concat([df_old, df_new]).sort_index()
        df_upd = add_rolling_micro(df_all)

        out_file = latest if files else os.path.join(outd, f"{sym}-features-{start.date()}_to_{y_str}.parquet")
        df_upd.to_parquet(out_file, compression="snappy")
        print(f"✅ {sym}: aktualisiert bis {y_str}")

if __name__ == "__main__":
    main()
