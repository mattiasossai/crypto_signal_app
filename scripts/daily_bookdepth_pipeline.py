#!/usr/bin/env python3
import argparse
import os
import glob
import shutil
import datetime
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression

# 1) Inception‐Dates (tz‐naive strings; wir lokalisieren später)
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
        res = subprocess.run(["curl","-sSf",url,"-o",zip_path], capture_output=True)
        if res.returncode == 0:
            subprocess.run(["unzip","-q","-o",zip_path,"-d",raw_dir], check=True)
            os.remove(zip_path)
        else:
            print(f"   ⚠️  {symbol} {ds}: ZIP nicht gefunden")

def extract_raw_for_days(symbol: str, raw_dir: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    days = pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")
    rows = []
    for day in days:
        ds = day.strftime("%Y-%m-%d")
        csv_fp = os.path.join(raw_dir, f"{symbol}-bookDepth-{ds}.csv")
        if os.path.exists(csv_fp):
            df = pd.read_csv(csv_fp)
            if str(df.columns[0]).isdigit():
                df.columns = ["timestamp","percentage","depth","notional"]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            next_day = day + pd.Timedelta(days=1)
            sl = df.loc[(df.index >= day) & (df.index < next_day)]
            has_data = not sl.empty
        else:
            sl = pd.DataFrame(columns=["percentage","depth","notional"],
                              index=pd.DatetimeIndex([], tz="UTC"))
            has_data = False

        # --- Basis‐Aggregationen ---
        tot_not = sl["notional"].sum()
        tot_dep = sl["depth"].sum()
        m1 = sl["percentage"].abs() <= 1.0
        m10= sl["percentage"].abs() <= 10.0
        n1, d1   = sl.loc[m1,"notional"].sum(), sl.loc[m1,"depth"].sum()
        n10,d10  = sl.loc[m10,"notional"].sum(), sl.loc[m10,"depth"].sum()
        rel_n1 = n1/tot_not if tot_not else np.nan
        rel_d1 = d1/tot_dep if tot_dep else np.nan
        p_min = sl.loc[sl["percentage"]>0,"percentage"].min() or 0
        n_max = sl.loc[sl["percentage"]<0,"percentage"].max() or 0
        spread_pct = p_min + abs(n_max)
        bid_n = sl.loc[sl["percentage"]<0,"notional"].sum()
        ask_n = sl.loc[sl["percentage"]>0,"notional"].sum()
        imb_n = (bid_n-ask_n)/tot_not if tot_not else np.nan
        bid_d = sl.loc[sl["percentage"]<0,"depth"].sum()
        ask_d = sl.loc[sl["percentage"]>0,"depth"].sum()
        imb_d = (bid_d-ask_d)/tot_dep if tot_dep else np.nan

        n, d = sl["notional"], sl["depth"]
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
    # 7/14/21d Imbalances
    for w in (7,14,21):
        for base in ("notional_imbalance","depth_imbalance"):
            col = f"{base}_roll_{w}d"
            roll = df[base].rolling(window=w, min_periods=w).mean()
            df[col]          = roll.fillna(0)
            df[f"has_{col}"] = roll.notna()

    # VPIN
    df["vpin"] = df.notional_imbalance.abs().rolling(window=50, min_periods=1).mean()

    # 30d Microstructure
    w = 30
    df["mid_price"] = (df.total_notional/df.total_depth).replace([np.inf,-np.inf],np.nan)
    df["ret"]       = df.mid_price.pct_change().abs().fillna(0)

    # Kyle‐Lambda
    kl = []
    for i in range(len(df)):
        if i < w:
            kl.append(0)
        else:
            sub = df.iloc[i-w+1:i+1]
            X = sub.total_notional.diff().values.reshape(-1,1)
            y = sub.mid_price.diff().abs().values
            m = (~np.isnan(X.flatten())) & (~np.isnan(y))
            kl.append(LinearRegression().fit(X[m],y[m]).coef_[0] if m.sum()>=2 else 0)
    df[f"kyle_lambda_roll_{w}d"]     = kl
    df[f"has_kyle_lambda_roll_{w}d"] = [i>=w-1 for i in range(len(df))]

    # Amihud
    ai      = df.ret / df.total_notional.replace(0,np.nan)
    roll_ai = ai.rolling(window=w, min_periods=w).mean().fillna(0)
    df[f"amihud_roll_{w}d"]    = roll_ai
    df[f"has_amihud_roll_{w}d"] = roll_ai.notna()

    # Liquidity‐Slope
    ls = []
    for i in range(len(df)):
        if i < w:
            ls.append(0)
        else:
            sub = df.iloc[i-w+1:i+1]
            X = sub.rel_depth_1pct.values.reshape(-1,1)
            y = sub.spread_pct.values
            m = (~np.isnan(X.flatten())) & (~np.isnan(y))
            ls.append(LinearRegression().fit(X[m],y[m]).coef_[0] if m.sum()>=2 else 0)
    df[f"liq_slope_roll_{w}d"]     = ls
    df[f"has_liq_slope_roll_{w}d"] = [i>=w-1 for i in range(len(df))]

    return df.drop(columns=["mid_price","ret"], errors="ignore")

def process_symbol(symbol: str, start_date: str, end_date: str):
    # a) parse dates
    if start_date and end_date:
        sd = pd.to_datetime(start_date).tz_localize("UTC")
        ed = pd.to_datetime(end_date).tz_localize("UTC")
    else:
        y = datetime.datetime.utcnow().date() - datetime.timedelta(days=1)
        sd = None
        ed = pd.to_datetime(y).tz_localize("UTC")

    # b) inception/resume logic
    out_dir = f"features/bookDepth/{symbol}"
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, f"{symbol}-features-*.parquet")
    files = glob.glob(pattern)

    if sd is None:
        # daily => resume or inception
        if files:
            latest = max(files, key=lambda f: pd.read_parquet(f).index.max())
            df_old = pd.read_parquet(latest)
            # ────────────────────────────────────────────────
            # 1) Stelle sicher, dass der Index datetime+UTC ist
            if "date" in df_old.columns:
                df_old["date"] = pd.to_datetime(df_old["date"], utc=True)
                df_old = df_old.set_index("date").sort_index()
                else:
                    df_old.index = pd.to_datetime(df_old.index, utc=True)
                    df_old = df_old.sort_index()
                # 2) Jetzt geht Timestamp + Timedelta wieder
                sd = (df_old.index.max() + pd.Timedelta(days=1)).normalize()
            # ────────────────────────────────────────────────
            out_file = latest
        else:
            df_old = pd.DataFrame()
            sd = pd.to_datetime(INCEPTION[symbol]).tz_localize("UTC")
            out_file = os.path.join(out_dir, f"{symbol}-features-{sd.date()}_to_{ed.date()}.parquet")
    else:
        # historical fresh
        df_old = pd.DataFrame()
        out_file = os.path.join(out_dir, f"{symbol}-features-{sd.date()}_to_{ed.date()}.parquet")

    if sd.tzinfo is None: sd = sd.tz_localize("UTC")
    if sd > ed:
        print(f"ℹ️ {symbol}: nichts zu tun ({sd.date()} > {ed.date()})")
        return

    # c) download, extract, rolling, write
    days = pd.date_range(sd.normalize(), ed.normalize(), freq="D", tz="UTC")
    rawdir = f"raw/bookDepth/{symbol}"
    print(f"→ {symbol}: Downloading {len(days)} days…")
    download_and_unzip(symbol, days, rawdir)

    df_new = extract_raw_for_days(symbol, rawdir, sd, ed)
    shutil.rmtree(rawdir)

    df_all = pd.concat([df_old, df_new]).sort_index()
    df_upd = add_rolling_micro(df_all)

    df_upd.to_parquet(out_file, compression="snappy")
    print(f"✅ {symbol}: writ­ten {len(df_upd)} days to {out_file}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",      required=True)
    p.add_argument("--start-date",  default=None)
    p.add_argument("--end-date",    default=None)
    args = p.parse_args()
    process_symbol(args.symbol, args.start_date, args.end_date)

if __name__ == "__main__":
    main()
