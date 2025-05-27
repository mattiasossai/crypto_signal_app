#!/usr/bin/env python3
import os
import glob
import argparse
import datetime
import shutil
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression

# Symbol-specific inception dates
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
            subprocess.run(["unzip","-q","-o",zip_path,"-d",raw_dir], check=True)
            os.remove(zip_path)
        else:
            print(f"   ⚠️  {symbol} {ds}: ZIP nicht gefunden")

def extract_raw_for_days(symbol: str, raw_dir: str, start, end) -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D", tz="UTC")
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
            # exakt wie im alten Skript
            sl = df[day : day + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)]
            has_data = not sl.empty
        else:
            sl = pd.DataFrame(columns=["percentage","depth","notional"],
                               index=pd.DatetimeIndex([], tz="UTC"))
            has_data = False

        # --- Basis-Features (Summen, Bins, Imbalance, Momente, Segmente) ---
        tot_not, tot_dep = sl["notional"].sum(), sl["depth"].sum()
        mask1, mask10 = sl["percentage"].abs()<=1, sl["percentage"].abs()<=10
        n1, d1 = sl.loc[mask1,"notional"].sum(), sl.loc[mask1,"depth"].sum()
        n10, d10 = sl.loc[mask10,"notional"].sum(), sl.loc[mask10,"depth"].sum()
        rel_n1 = n1/tot_not if tot_not else np.nan
        rel_d1 = d1/tot_dep if tot_dep else np.nan
        p_min = sl.loc[sl["percentage"]>0,"percentage"].min() or 0
        n_max = sl.loc[sl["percentage"]<0,"percentage"].max() or 0
        spread_pct = p_min + abs(n_max)
        ask_n = sl.loc[sl["percentage"]>0,"notional"].sum()
        bid_n = sl.loc[sl["percentage"]<0,"notional"].sum()
        imb_n = (bid_n-ask_n)/tot_not if tot_not else np.nan
        ask_d = sl.loc[sl["percentage"]>0,"depth"].sum()
        bid_d = sl.loc[sl["percentage"]<0,"depth"].sum()
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
            "date": day,
            "file_exists": has_data,
            "has_data": has_data,
            "has_notional": tot_not>0,
            "has_depth": tot_dep>0,
            "total_notional": tot_not,
            "total_depth": tot_dep,
            "notional_1pct": n1,
            "depth_1pct": d1,
            "rel_notional_1pct": rel_n1,
            "rel_depth_1pct": rel_d1,
            "notional_10pct": n10,
            "depth_10pct": d10,
            "spread_pct": spread_pct,
            "notional_imbalance": imb_n,
            "depth_imbalance": imb_d,
            **moments,
            "notional_00_08": seg1["notional"],
            "depth_00_08": seg1["depth"],
            "notional_08_16": seg2["notional"],
            "depth_08_16": seg2["depth"],
            "notional_16_24": seg3["notional"],
            "depth_16_24": seg3["depth"],
            "has_00_08": pd.notna(seg1["notional"]),
            "has_08_16": pd.notna(seg2["notional"]),
            "has_16_24": pd.notna(seg3["notional"]),
        })
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df

def add_rolling_micro(df: pd.DataFrame) -> pd.DataFrame:
    # 7/14/21d Imbalances
    for w in (7,14,21):
        for base in ("notional_imbalance","depth_imbalance"):
            col = f"{base}_roll_{w}d"
            roll = df[base].rolling(window=w, min_periods=w).mean()
            df[col] = roll.fillna(0)
            df[f"has_{col}"] = roll.notna()
    # VPIN
    df["vpin"] = df.notional_imbalance.abs().rolling(window=50,min_periods=1).mean()
    # 30d micro
    w = 30
    df["mid_price"] = (df.total_notional/df.total_depth).replace([np.inf,-np.inf],np.nan)
    df["ret"]       = df.mid_price.pct_change().abs().fillna(0)
    # Kyle
    kl=[]
    for i in range(len(df)):
        if i<w: kl.append(0)
        else:
            sub = df.iloc[i-w+1:i+1]
            X = sub.total_notional.diff().values.reshape(-1,1)
            y = sub.mid_price.diff().abs().values
            m = (~np.isnan(X.flatten())) & (~np.isnan(y))
            kl.append(LinearRegression().fit(X[m],y[m]).coef_[0] if m.sum()>=2 else 0)
    df[f"kyle_lambda_roll_{w}d"] = kl
    df[f"has_kyle_lambda_roll_{w}d"] = [i>=w-1 for i in range(len(df))]
    # Amihud
    ai = df.ret / df.total_notional.replace(0,np.nan)
    roll_ai = ai.rolling(window=w,min_periods=w).mean().fillna(0)
    df[f"amihud_roll_{w}d"]    = roll_ai
    df[f"has_amihud_roll_{w}d"] = roll_ai.notna()
    # Liquidity Slope
    ls=[]
    for i in range(len(df)):
        if i<w: ls.append(0)
        else:
            sub = df.iloc[i-w+1:i+1]
            X = sub.rel_depth_1pct.values.reshape(-1,1)
            y = sub.spread_pct.values
            m = (~np.isnan(X.flatten())) & (~np.isnan(y))
            ls.append(LinearRegression().fit(X[m],y[m]).coef_[0] if m.sum()>=2 else 0)
    df[f"liq_slope_roll_{w}d"]     = ls
    df[f"has_liq_slope_roll_{w}d"] = [i>=w-1 for i in range(len(df))]
    return df.drop(columns=["mid_price","ret"], errors="ignore")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-date",  required=False,
                   help="YYYY-MM-DD; default=yesterday")
    p.add_argument("--end-date",    required=False,
                   help="YYYY-MM-DD; default=yesterday")
    args = p.parse_args()

    # determine start/end
    if args.start_date and args.end_date:
        sd = pd.to_datetime(args.start_date).tz_localize("UTC")
        ed = pd.to_datetime(args.end_date).tz_localize("UTC")
    else:
        y = datetime.datetime.utcnow().date() - datetime.timedelta(days=1)
        sd = ed = pd.to_datetime(y).tz_localize("UTC")

    base = "features/bookDepth"
    for sym, inc in INCEPTION.items():
        inc_dt = pd.to_datetime(inc).tz_localize("UTC")
        out_dir = os.path.join(base, sym); os.makedirs(out_dir, exist_ok=True)
        # resume
        pattern = os.path.join(out_dir, f"{sym}-features-*.parquet")
        files = glob.glob(pattern)
        if files:
            latest = max(files, key=lambda f: pd.read_parquet(f).index.max())
            df_old = pd.read_parquet(latest)
            df_old.index = pd.to_datetime(df_old.index).tz_localize("UTC")
            start = max(sd, (df_old.index.max()+pd.Timedelta(days=1))).normalize()
            out_file = latest
        else:
            df_old = pd.DataFrame()
            start = max(sd, inc_dt).normalize()
            out_file = os.path.join(out_dir,
                           f"{sym}-features-{start.date()}_to_{ed.date()}.parquet")

        if start > ed:
            print(f"ℹ️ {sym}: nothing new ({start.date()} > {ed.date()})")
            continue

        # download & extract
        days = pd.date_range(start, ed, freq="D", tz="UTC")
        raw = os.path.join("raw/bookDepth", sym)
        download_and_unzip(sym, days, raw)
        df_new = extract_raw_for_days(sym, raw, start, ed)
        shutil.rmtree(raw)

        df_all = pd.concat([df_old, df_new]).sort_index()
        df_upd = add_rolling_micro(df_all)
        df_upd.to_parquet(out_file, compression="snappy")
        print(f"✅ {sym}: updated to {ed.date()}")

if __name__ == "__main__":
    main()
