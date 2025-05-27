#!/usr/bin/env python3
import argparse
import os
import glob
import gzip
import re
import shutil
import datetime
import subprocess
import logging

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from utils import init_logger, save_parquet

logger = init_logger("funding_pipeline")

# --- Config ---
LOCAL_BASE       = "raw/funding"                  # Basis-Verzeichnis für Downloads
OUTPUT_DIR       = "features/funding"             # Ausgabe-Parquets
SYMBOL_START     = {                              # Inception-Monate
    "BTCUSDT": "2020-01",
    "ETHUSDT": "2020-01",
    "BNBUSDT": "2020-02",
    "XRPUSDT": "2020-01",
    "SOLUSDT": "2020-09",
    "ENAUSDT": "2024-04",
}
ROLL_HOURS       = 8
SMA_DAYS         = 7

BASE_FUNDING_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
BASE_PREMIUM_URL = "https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines"

def download_monthly_data(symbol: str, sd: pd.Timestamp, ed: pd.Timestamp):
    """
    Lade alle monatlichen FundingRate- und PremiumIndexKlines-ZIPs von Binance
    zwischen sd (inclusive) und ed (inclusive) herunter und entpacke sie nach LOCAL_BASE.
    """
    start_m = sd.to_period("M")
    end_m   = ed.to_period("M")

    # FundingRate
    out_f = os.path.join(LOCAL_BASE, "fundingRate", symbol)
    os.makedirs(out_f, exist_ok=True)
    m = start_m
    while m <= end_m:
        period = str(m)  # z.B. '2020-01'
        zipname = f"{symbol}-fundingRate-{period}.zip"
        url     = f"{BASE_FUNDING_URL}/{symbol}/{zipname}"
        target  = os.path.join(out_f, zipname)
        logger.info(f"→ DOWNLOAD Funding {zipname}")
        res = subprocess.run(["curl","-sSf",url,"-o",target])
        if res.returncode == 0:
            subprocess.run(["unzip","-q","-o",target,"-d",out_f], check=True)
            os.remove(target)
        else:
            logger.warning(f"{zipname} nicht gefunden")
        m += 1

    # PremiumIndexKlines (1h)
    out_p = os.path.join(LOCAL_BASE, "premiumIndexKlines", symbol, "1h")
    os.makedirs(out_p, exist_ok=True)
    m = start_m
    while m <= end_m:
        period = str(m)
        zipname = f"{symbol}-1h-{period}.zip"
        url     = f"{BASE_PREMIUM_URL}/{symbol}/1h/{zipname}"
        target  = os.path.join(out_p, zipname)
        logger.info(f"→ DOWNLOAD Premium {zipname}")
        res = subprocess.run(["curl","-sSf",url,"-o",target])
        if res.returncode == 0:
            subprocess.run(["unzip","-q","-o",target,"-d",out_p], check=True)
            os.remove(target)
        else:
            logger.warning(f"{zipname} nicht gefunden")
        m += 1

def list_local_csvs(symbol: str, kind: str):
    """
    Liste der lokal entpackten CSVs für FundingRate bzw. PremiumIndexKlines.
    """
    if kind == "fundingRate":
        return sorted(glob.glob(f"{LOCAL_BASE}/fundingRate/{symbol}/{symbol}-fundingRate-*.csv"))
    else:  # premiumIndexKlines
        return sorted(glob.glob(f"{LOCAL_BASE}/premiumIndexKlines/{symbol}/1h/{symbol}-1h-*.csv"))

def load_and_concat_funding(symbol: str, sd: pd.Timestamp):
    files = list_local_csvs(symbol, "fundingRate")
    dfs = []
    for fn in files:
        # nur Perioden >= sd laden
        per = re.search(rf"{symbol}-fundingRate-(\d{{4}}-\d{{2}})", fn).group(1)
        if pd.Period(per, "M") < sd.to_period("M"):
            continue
        logger.info(f"Lade Funding-CSV {fn}")
        opener = gzip.open if fn.endswith(".gz") else open
        with opener(fn, "rt") as f:
            df = pd.read_csv(f)
        df.columns = [c.lower() for c in df.columns]
        df["fundingtime"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True, errors="coerce")
        df["fundingrate"] = df["last_funding_rate"]
        dfs.append(df[["fundingtime","fundingrate"]])
    all_df = pd.concat(dfs, ignore_index=True).sort_values("fundingtime").drop_duplicates("fundingtime")
    all_df.index = all_df["fundingtime"]
    return all_df[["fundingrate"]]

def load_and_concat_premium(symbol: str, idx: pd.DatetimeIndex):
    files = list_local_csvs(symbol, "premiumIndexKlines")
    frames = []
    for fn in files:
        per = re.search(rf"{symbol}-1h-(\d{{4}}-\d{{2}})", fn).group(1)
        if pd.Period(per, "M") < idx.min().to_period("M"):
            continue
        logger.info(f"Lade Premium-CSV {fn}")
        df = pd.read_csv(fn)
        cols = [c.lower() for c in df.columns]
        cols = [c.replace("opentime","open_time").replace("closetime","close_time") for c in cols]
        df.columns = cols
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
        series = df.set_index("open_time")["close"]
        frames.append(series)
    all_prem = pd.concat(frames).sort_index().drop_duplicates()
    return all_prem.reindex(idx).ffill()

def compute_features(fund_df: pd.DataFrame) -> pd.DataFrame:
    hourly = fund_df["fundingrate"].resample("1h").mean().ffill()
    out = pd.DataFrame({"fundingRate": hourly})
    out["fundingRate_8h"] = out["fundingRate"].rolling(ROLL_HOURS).sum()
    window = SMA_DAYS * 24
    out["sma7d"]  = out["fundingRate_8h"].rolling(window).mean()
    out["zscore"] = (out["fundingRate_8h"] - out["sma7d"]) / out["fundingRate_8h"].rolling(window).std()
    out["flip"] = np.sign(out["fundingRate_8h"]).diff().abs().fillna(0).astype(int)
    out["has_sma"]    = out["sma7d"].notna().astype(int)
    out["has_zscore"] = out["zscore"].notna().astype(int)
    out["sma7d"]      = out["sma7d"].fillna(0)
    out["zscore"]     = out["zscore"].fillna(0)
    out["flip_cumsum"]      = out["flip"].cumsum()
    out["hours_since_flip"] = out.groupby("flip_cumsum").cumcount()
    out.drop(columns="flip_cumsum", inplace=True)
    out["basis"] = np.nan
    return out

def process_symbol(symbol: str, start_date: str = None, end_date: str = None):
    logger.info(f"=== Verarbeitung {symbol} ===")
    # Termine festlegen
    if start_date and end_date:
        sd = pd.to_datetime(start_date + "-01").tz_localize("UTC")
        ed = pd.to_datetime(end_date + "-01").tz_localize("UTC")
    else:
        inception = SYMBOL_START[symbol]
        sd = pd.to_datetime(inception + "-01").tz_localize("UTC")
        ed = pd.Timestamp.utcnow().normalize().tz_localize("UTC")
    # Download der Rohdaten
    download_monthly_data(symbol, sd, ed)
    # Einlesen & Features
    df_f = load_and_concat_funding(symbol, sd)
    feats = compute_features(df_f)
    prem  = load_and_concat_premium(symbol, feats.index)
    feats["basis"] = prem
    # Resume/Append falls Parquet schon existiert
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pattern = f"{OUTPUT_DIR}/{symbol}-funding-features-*.parquet"
    files = glob.glob(pattern)
    if files:
        latest = max(files, key=lambda f: pd.read_parquet(f).index.max())
        old = pd.read_parquet(latest)
        merged = pd.concat([old, feats]).sort_index().drop_duplicates(keep="first")
    else:
        merged = feats
    # Dynamischer Dateiname anhand realer Daten
    real_sd = merged.index.min().date()
    real_ed = merged.index.max().date()
    out_fp  = f"{OUTPUT_DIR}/{symbol}-funding-features-{real_sd}_to_{real_ed}.parquet"
    save_parquet(merged, out_fp)
    logger.info(f"Gespeichert {len(merged)} Zeilen in {out_fp}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",     required=True, help="z.B. BTCUSDT")
    p.add_argument("--start-date", default=None, help="YYYY-MM für historisch (optional)")
    p.add_argument("--end-date",   default=None, help="YYYY-MM für historisch (optional)")
    args = p.parse_args()
    process_symbol(args.symbol, args.start_date, args.end_date)

if __name__ == "__main__":
    main()
