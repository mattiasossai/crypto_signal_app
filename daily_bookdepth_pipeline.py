#!/usr/bin/env python3
import os
import glob
import argparse
import logging
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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
    first = open(fp, "r").readline().split(",")[0]
    return not first.isdigit()

def extract_for_days(input_dir: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    symbol = os.path.basename(input_dir)
    days   = pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")
    logging.info("→ %s: Generating features for %d days from %s to %s",
                 symbol, len(days), start.date(), end.date())
    rows   = []

    for day in days:
        day_str = day.strftime("%Y-%m-%d")
        fp = os.path.join(input_dir, f"{symbol}-bookDepth-{day_str}.csv")

        if os.path.exists(fp):
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

            # Original‐Label‐Slice
            full = df[day : day + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)]
            has_data = not full.empty
            logging.info("   • %s %s: read %d rows → %d in slice",
                         symbol, day_str, len(df), len(full))
        else:
            full = pd.DataFrame(
                columns=["percentage","depth","notional"],
                index=pd.DatetimeIndex([], tz="UTC")
            )
            has_data = False
            logging.info("   • %s %s: no file → empty slice", symbol, day_str)

        # 1) globale Summen & Bins
        tot_not = full["notional"].sum()
        tot_dep = full["depth"].sum()
        mask1   = full["percentage"].abs() <= 1.0
        mask10  = full["percentage"].abs() <= 10.0
        n1, d1  = full.loc[mask1, "notional"].sum(), full.loc[mask1, "depth"].sum()
        n10,d10 = full.loc[mask10,"notional"].sum(), full.loc[mask10,"depth"].sum()
        rel_n1 = n1/tot_not if tot_not else np.nan
        rel_d1 = d1/tot_dep if tot_dep else np.nan

        # 2) Spread
        p_min = full.loc[full["percentage"]>0,"percentage"].min() or 0
        n_max = full.loc[full["percentage"]<0,"percentage"].max() or 0
        spread_pct = p_min + abs(n_max)

        # 3) Imbalances
        bid_n = full.loc[full["percentage"]<0,"notional"].sum()
        ask_n = full.loc[full["percentage"]>0,"notional"].sum()
        imb_n = (bid_n-ask_n)/tot_not if tot_not else np.nan
        bid_d = full.loc[full["percentage"]<0,"depth"].sum()
        ask_d = full.loc[full["percentage"]>0,"depth"].sum()
        imb_d = (bid_d-ask_d)/tot_dep if tot_dep else np.nan

        # 4) Momente
        n, d = full["notional"], full["depth"]
        moments = {
            "not_mean":  n.mean(),
            "not_var":   n.var(ddof=0),
            "not_skew":  skew(n, bias=False)   if len(n)>1 else np.nan,
            "not_kurt":  kurtosis(n, bias=False) if len(n)>1 else np.nan,
            "dep_mean":  d.mean(),
            "dep_var":   d.var(ddof=0),
            "dep_skew":  skew(d, bias=False)   if len(d)>1 else np.nan,
            "dep_kurt":  kurtosis(d, bias=False) if len(d)>1 else np.nan,
        }

        # 5) Intraday‐Segmente
        seg1 = full.between_time("00:00","07:59")[["notional","depth"]].sum(min_count=1)
        seg2 = full.between_time("08:00","15:59")[["notional","depth"]].sum(min_count=1)
        seg3 = full.between_time("16:00","23:59")[["notional","depth"]].sum(min_count=1)

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
            "has_00_08":          not pd.isna(seg1["notional"]),
            "has_08_16":          not pd.isna(seg2["notional"]),
            "has_16_24":          not pd.isna(seg3["notional"]),
        })

    df = pd.DataFrame(rows).set_index("date")
    df.index.name = "date"
    return df

def add_rolling_micro(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("→ Adding rolling & microstructure features (7/14/21/30)")
    # 7/14/21d Imbalances
    for w in (7,14,21):
        for base in ("notional_imbalance","depth_imbalance"):
            col = f"{base}_roll_{w}d"
            roll = df[base].rolling(window=w, min_periods=w).mean()
            df[col]          = roll.fillna(0)
            df[f"has_{col}"] = roll.notna()
        logging.info("   • computed imbalance rolls %dd", w)

    # VPIN (50-Tick)
    df["vpin"] = df.notional_imbalance.abs().rolling(window=50, min_periods=1).mean()
    logging.info("   • computed VPIN (50)")

    # 30-Tage Microstructure
    w = 30
    df["mid_price"] = (df.total_notional/df.total_depth).replace([np.inf,-np.inf], np.nan)
    df["ret"]       = df.mid_price.pct_change().abs().fillna(0)

    # Kyle-Lambda
    kl = []
    for i in range(len(df)):
        if i < w:
            kl.append(0)
        else:
            sub = df.iloc[i-w+1:i+1]
            X   = sub.total_notional.diff().values.reshape(-1,1)
            y   = sub.mid_price.diff().abs().values
            m   = (~np.isnan(X.flatten())) & (~np.isnan(y))
            kl.append(LinearRegression().fit(X[m], y[m]).coef_[0] if m.sum()>=2 else 0)
    df[f"kyle_lambda_roll_{w}d"]     = kl
    df[f"has_kyle_lambda_roll_{w}d"] = [i>=w-1 for i in range(len(df))]
    logging.info("   • computed Kyle-Lambda %dd", w)

    # Amihud
    ai      = df.ret / df.total_notional.replace(0, np.nan)
    roll_ai = ai.rolling(window=w, min_periods=w).mean().fillna(0)
    df[f"amihud_roll_{w}d"]    = roll_ai
    df[f"has_amihud_roll_{w}d"] = roll_ai.notna()
    logging.info("   • computed Amihud %dd", w)

    # Liquidity-Slope
    ls = []
    for i in range(len(df)):
        if i < w:
            ls.append(0)
        else:
            sub = df.iloc[i-w+1:i+1]
            X   = sub.rel_depth_1pct.values.reshape(-1,1)
            y   = sub.spread_pct.values
            m   = (~np.isnan(X.flatten())) & (~np.isnan(y))
            ls.append(LinearRegression().fit(X[m], y[m]).coef_[0] if m.sum()>=2 else 0)
    df[f"liq_slope_roll_{w}d"]     = ls
    df[f"has_liq_slope_roll_{w}d"] = [i>=w-1 for i in range(len(df))]
    logging.info("   • computed Liquidity-Slope %dd", w)

    return df.drop(columns=["mid_price","ret"], errors="ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",   required=True)
    parser.add_argument("--start-date",  required=True)
    parser.add_argument("--end-date",    required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output_file)
    symbol  = os.path.basename(args.input_dir)
    sd      = pd.to_datetime(args.start_date).tz_localize("UTC")
    ed      = pd.to_datetime(args.end_date).tz_localize("UTC")

    # Resume/Append Logic
    pattern = os.path.join(out_dir, f"{symbol}-features-*.parquet")
    files   = glob.glob(pattern)
    if files:
        best   = max(files, key=lambda f: pd.read_parquet(f).index.max())
        df_old = pd.read_parquet(best)
        new_sd = (df_old.index.max() + pd.Timedelta(days=1)).normalize()
        logging.info("→ Append mode, resuming at %s", new_sd.date())
    else:
        df_old = None
        inc    = pd.to_datetime(INCEPTION[symbol]).tz_localize("UTC")
        new_sd = max(sd, inc).normalize()
        logging.info("→ Fresh mode, starting at %s", new_sd.date())

    if new_sd > ed:
        logging.info("ℹ️ Nothing to append (new start > end).")
        return

    df_new = extract_for_days(args.input_dir, new_sd, ed)
    df_all = pd.concat([df_old, df_new]).sort_index() if df_old is not None else df_new
    df_upd = add_rolling_micro(df_all)

    os.makedirs(out_dir, exist_ok=True)
    df_upd.to_parquet(args.output_file, compression="snappy")
    logging.info("✅ Wrote %d days → %s", len(df_upd), args.output_file)

if __name__ == "__main__":
    main()
