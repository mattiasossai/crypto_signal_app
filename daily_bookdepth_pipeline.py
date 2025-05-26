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

# 1) Inception-Dates
INCEPTION = {
    "BTCUSDT": "2023-01-01",
    "ETHUSDT": "2023-01-01",
    "BNBUSDT": "2023-01-01",
    "SOLUSDT": "2023-01-01",
    "XRPUSDT": "2023-01-06",
    "ENAUSDT": "2024-04-02",
}

# 2) Download & Entpacken für eine Liste von Tagen
def download_and_unzip(symbol: str, days, raw_dir: str):
    os.makedirs(raw_dir, exist_ok=True)
    for day in days:
        ds = day.strftime("%Y-%m-%d")
        zip_name = f"{symbol}-bookDepth-{ds}.zip"
        zip_path = os.path.join(raw_dir, zip_name)
        url = (
            f"https://data.binance.vision/data/futures/um/daily/"
            f"bookDepth/{symbol}/{zip_name}"
        )
        res = subprocess.run(
            ["curl", "-sSf", url, "-o", zip_path],
            capture_output=True
        )
        if res.returncode == 0:
            subprocess.run(
                ["unzip", "-q", "-o", zip_path, "-d", raw_dir],
                check=True
            )
            os.remove(zip_path)
        else:
            print(f"⚠️  {symbol} {ds}: ZIP nicht gefunden → skip")

# 3) Basis-Features aus Roh-CSV lesen
def extract_raw_for_days(symbol: str, raw_dir: str, start, end) -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D")
    rows = []
    for day in days:
        ds = day.strftime("%Y-%m-%d")
        csv_fp = os.path.join(raw_dir, f"{symbol}-bookDepth-{ds}.csv")
        if os.path.exists(csv_fp):
            df_raw = pd.read_csv(csv_fp)
            if str(df_raw.columns[0]).isdigit():
                df_raw.columns = ["timestamp","percentage","depth","notional"]
            df_raw["timestamp"] = pd.to_datetime(
                df_raw["timestamp"], unit="ms", utc=True, errors="coerce"
            )
            df_raw.set_index("timestamp", inplace=True)
            # ← hier: boolean mask statt label-slice
            day_start = day
            day_end   = day + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
            sl = df_raw[(df_raw.index >= day_start) & (df_raw.index <= day_end)]
            has_data = not sl.empty
        else:
            sl = pd.DataFrame(
                columns=["percentage","depth","notional"],
                index=pd.DatetimeIndex([], tz="UTC"),
            )
            has_data = False

        # Summen & Bins
        tot_not = sl["notional"].sum()
        tot_dep = sl["depth"].sum()
        mask1  = sl["percentage"].abs() <= 1.0
        mask10 = sl["percentage"].abs() <= 10.0
        n1 = sl.loc[mask1, "notional"].sum()
        d1 = sl.loc[mask1, "depth"].sum()
        n10 = sl.loc[mask10, "notional"].sum()
        d10 = sl.loc[mask10, "depth"].sum()
        rel_n1 = n1 / tot_not if tot_not else np.nan
        rel_d1 = d1 / tot_dep if tot_dep else np.nan

        # Spread
        p_min = sl.loc[sl["percentage"]>0,"percentage"].min() or 0
        n_max = sl.loc[sl["percentage"]<0,"percentage"].max() or 0
        spread_pct = p_min + abs(n_max)

        # Imbalances
        ask_n = sl.loc[sl["percentage"]>0,"notional"].sum()
        bid_n = sl.loc[sl["percentage"]<0,"notional"].sum()
        imb_n = (bid_n - ask_n) / tot_not if tot_not else np.nan
        ask_d = sl.loc[sl["percentage"]>0,"depth"].sum()
        bid_d = sl.loc[sl["percentage"]<0,"depth"].sum()
        imb_d = (bid_d - ask_d) / tot_dep if tot_dep else np.nan

        # Momente
        n = sl["notional"]; d = sl["depth"]
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

        # Intraday-Segmente
        seg1 = sl.between_time("00:00","07:59")[["notional","depth"]].sum(min_count=1)
        seg2 = sl.between_time("08:00","15:59")[["notional","depth"]].sum(min_count=1)
        seg3 = sl.between_time("16:00","23:59")[["notional","depth"]].sum(min_count=1)

        rows.append({
            "date":               day,
            "file_exists":        has_data,
            "has_data":           has_data,
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
            "has_00_08":          pd.notna(seg1["notional"]),
            "has_08_16":          pd.notna(seg2["notional"]),
            "has_16_24":          pd.notna(seg3["notional"]),
        })
    df = pd.DataFrame(rows).set_index("date").sort_index()
    df.index.name = "date"
    return df

# 4) Rolling & Microstructure (30d)
def add_rolling_micro(df: pd.DataFrame) -> pd.DataFrame:
    # roll 7/14/21d, VPIN, then 30d micro
    for w in (7,14,21):
        for base in ("notional_imbalance","depth_imbalance"):
            col = f"{base}_roll_{w}d"
            roll = df[base].rolling(window=w, min_periods=w).mean()
            df[col]          = roll.fillna(0)
            df[f"has_{col}"] = roll.notna()

    df["vpin"] = df.notional_imbalance.abs().rolling(window=50,min_periods=1).mean()

    # die 30-Tage Micro-Features
    w = 30
    df["mid_price"] = (df.total_notional/df.total_depth).replace([np.inf,-np.inf], np.nan)
    df["ret"]       = df.mid_price.pct_change().abs().fillna(0)

    # Kyle Lambda
    kl = []
    for i in range(len(df)):
        if i < w:
            kl.append(0)
        else:
            sub = df.iloc[i-w+1:i+1]
            dn, dp = sub.total_notional.diff().values, sub.mid_price.diff().abs().values
            mask = (~np.isnan(dn)) & (~np.isnan(dp))
            kl.append(
              LinearRegression().fit(dn[mask].reshape(-1,1), dp[mask]).coef_[0]
              if mask.sum()>=2 else 0
            )
    df[f"kyle_lambda_roll_{w}d"]     = kl
    df[f"has_kyle_lambda_roll_{w}d"] = [i>=w-1 for i in range(len(df))]

    # Amihud
    ai      = df.ret / df.total_notional.replace(0,np.nan)
    roll_ai = ai.rolling(window=w,min_periods=w).mean().fillna(0)
    df[f"amihud_roll_{w}d"]    = roll_ai
    df[f"has_amihud_roll_{w}d"] = roll_ai.notna()

    # Liquidity Slope
    ls = []
    for i in range(len(df)):
        if i < w:
            ls.append(0)
        else:
            sub = df.iloc[i-w+1:i+1]
            rd, sp = sub.rel_depth_1pct.values.reshape(-1,1), sub.spread_pct.values
            mask = (~np.isnan(rd.flatten())) & (~np.isnan(sp))
            ls.append(
              LinearRegression().fit(rd[mask], sp[mask]).coef_[0]
              if mask.sum()>=2 else 0
            )
    df[f"liq_slope_roll_{w}d"]     = ls
    df[f"has_liq_slope_roll_{w}d"] = [i>=w-1 for i in range(len(df))]

    return df.drop(columns=["mid_price","ret"], errors="ignore")

# 5) Main: Resume, Download, Extract, Merge, Cleanup
def main():
    # yesterday tz-naive → tz-aware UTC
    yesterday = datetime.datetime.utcnow().date() - datetime.timedelta(days=1)
    yesterday_ts = pd.to_datetime(yesterday).tz_localize("UTC")
    today_str    = yesterday_ts.strftime("%Y-%m-%d")

    base_feat = "features/bookDepth"
    os.makedirs(base_feat, exist_ok=True)

    for symbol, inc in INCEPTION.items():
        inc_date  = pd.to_datetime(inc, utc=True).floor("D")
        out_dir   = os.path.join(base_feat, symbol)
        os.makedirs(out_dir, exist_ok=True)

        # Resume
        pattern  = os.path.join(out_dir, f"{symbol}-features-*.parquet")
        existing = glob.glob(pattern)
        if existing:
            latest = max(existing, key=lambda f: pd.read_parquet(f).index.max())
            df_old = pd.read_parquet(latest)
            if "date" in df_old.columns:
                df_old["date"] = pd.to_datetime(df_old["date"], utc=True)
                df_old = df_old.set_index("date").sort_index()
            start   = (df_old.index.max() + pd.Timedelta(days=1)).normalize()
            out_file = latest
        else:
            df_old   = pd.DataFrame()
            start    = inc_date
            out_file = os.path.join(
                out_dir,
                f"{symbol}-features-{start.date()}_to_{today_str}.parquet"
            )

        if start > yesterday_ts:
            print(f"ℹ️ {symbol}: schon aktuell bis {today_str}, skip")
            continue

        # Download
        days   = pd.date_range(start, yesterday_ts, freq="D")
        raw_dir = os.path.join("raw/bookDepth", symbol)
        print(f"→ {symbol}: Downloading {len(days)} zips…")
        download_and_unzip(symbol, days, raw_dir)

        # Extract
        df_new = extract_raw_for_days(symbol, raw_dir, start, yesterday_ts)
        shutil.rmtree(raw_dir)

        # Merge + Rolling
        df_all = pd.concat([df_old, df_new]).sort_index()
        df_upd = add_rolling_micro(df_all)

        # Write
        df_upd.to_parquet(out_file, compression="snappy")
        print(f"✅ {symbol}: aktualisiert bis {today_str}")

if __name__ == "__main__":
    main()
