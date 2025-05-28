#!/usr/bin/env python3
import os
import glob
import argparse
import subprocess
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Logger & Parquet-Saver ───────────────────────────────────────────────────
def init_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
        logger.addHandler(h)
    return logger

def save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy")

logger = init_logger("funding_pipeline")

# ── Konfiguration ─────────────────────────────────────────────────────────────
LOCAL_BASE = "data/futures/um/monthly"
OUTPUT_DIR = "features/funding"
SYMBOL_START = {
    "BTCUSDT": "2020-01",
    "ETHUSDT": "2020-01",
    "BNBUSDT": "2020-02",
    "XRPUSDT": "2020-01",
    "SOLUSDT": "2020-09",
    "ENAUSDT": "2024-04",
}
ROLL_HOURS = 8
SMA_DAYS   = 7

# ── Hilfsfunktion: Flexible CSV-Leser für FundingRate & PremiumIndex ────────
def read_csv_flexible(path: str, kind: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=0)
    lower = [c.lower() for c in df.columns]

    if kind == "fundingRate":
        # neue Binance-Variante
        if {"calc_time", "last_funding_rate"}.issubset(lower):
            df = df.rename(columns={"calc_time": "timestamp", "last_funding_rate": "fundingRate"})
        # ältere Varianten (timestamp + fundingRate)
        else:
            colmap = {c.lower(): c for c in df.columns}
            # timestamp
            for alt in ("timestamp","fundingtime","funding_time"):
                if alt in colmap:
                    df = df.rename(columns={colmap[alt]: "timestamp"})
                    break
            # fundingRate
            for alt in ("funding_rate","fundingrate"):
                if alt in colmap:
                    df = df.rename(columns={colmap[alt]: "fundingRate"})
                    break
        required = {"timestamp","fundingRate"}

    else:  # premiumIndexKlines
        # Standard-Header
        if {"open_time","close"}.issubset(lower):
            df = df.rename(columns={"open_time": "timestamp", "close": "close"})
        # CamelCase-Fallback
        else:
            colmap = {c.lower(): c for c in df.columns}
            if "opentime" in colmap and "close" in colmap:
                df = df.rename(columns={colmap["opentime"]: "timestamp", colmap["close"]: "close"})
        required = {"timestamp","close"}

    if not required.issubset(df.rename(str.lower, axis=1).columns):
        raise ValueError(f"{path}: Fehlende Spalten nach Umbenennung {required}")

    # Timestamp → datetime UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    return df.set_index("timestamp").sort_index()

# ── Datei-Listing & Download ─────────────────────────────────────────────────
def list_monthly_files(symbol: str, kind: str) -> list[str]:
    base = f"{LOCAL_BASE}/{kind}/{symbol}"
    if kind == "premiumIndexKlines":
        base += "/1h"
        pattern = f"{symbol}-1h-*.csv"
    else:
        pattern = f"{symbol}-fundingRate-*.csv"
    return sorted(glob.glob(f"{base}/{pattern}"))

def download_and_unzip(symbol: str, kind: str, start_month: str, end_month: str):
    base_url = "https://data.binance.vision/data/futures/um/monthly"
    curr = pd.Period(start_month, "M")
    last = pd.Period(end_month, "M")

    while curr <= last:
        per = curr.strftime("%Y-%m")
        if kind == "fundingRate":
            zip_name = f"{symbol}-fundingRate-{per}.zip"
            url      = f"{base_url}/fundingRate/{symbol}/{zip_name}"
            dst_dir  = f"{LOCAL_BASE}/fundingRate/{symbol}"
            dst      = f"{dst_dir}/{symbol}-fundingRate-{per}.csv"
        else:
            zip_name = f"{symbol}-1h-{per}.zip"
            url      = f"{base_url}/premiumIndexKlines/{symbol}/1h/{zip_name}"
            dst_dir  = f"{LOCAL_BASE}/premiumIndexKlines/{symbol}/1h"
            dst      = f"{dst_dir}/{symbol}-1h-{per}.csv"

        os.makedirs(dst_dir, exist_ok=True)
        logger.info(f"→ DOWNLOAD {zip_name}")
        res = subprocess.run(["curl","-sSf",url,"-o","tmp.zip"], capture_output=True)
        if res.returncode == 0:
            subprocess.run(["unzip","-p","tmp.zip"], stdout=open(dst,"wb"), check=True)
            os.remove("tmp.zip")
        else:
            logger.warning(f"{zip_name} nicht gefunden, übersprungen")
        curr += 1

# ── Laden & Zusammenfügen ────────────────────────────────────────────────────
def load_and_concat_funding(symbol: str) -> pd.DataFrame:
    files = list_monthly_files(symbol, "fundingRate")
    dfs = []
    for fn in files:
        logger.info(f"Lade Funding-CSV {fn}")
        try:
            df = read_csv_flexible(fn, "fundingRate")
            dfs.append(df[["fundingRate"]])
        except Exception as e:
            logger.error(f"{fn}: {e}")
    if not dfs:
        raise ValueError("Keine validen Funding-CSV-Dateien gefunden.")
    return pd.concat(dfs).sort_index().drop_duplicates()

def load_and_concat_premium(symbol: str, idx: pd.DatetimeIndex) -> pd.Series:
    files = list_monthly_files(symbol, "premiumIndexKlines")
    series = []
    for fn in files:
        logger.info(f"Lade Premium-Index-CSV {fn}")
        try:
            df = read_csv_flexible(fn, "premiumIndexKlines")
            series.append(df["close"])
        except Exception as e:
            logger.error(f"{fn}: {e}")
    if not series:
        raise ValueError("Keine validen Premium-Index-CSV-Dateien gefunden.")
    all_p = pd.concat(series).sort_index().drop_duplicates()
    return all_p.reindex(idx, method="ffill")

# ── Feature-Berechnung ──────────────────────────────────────────────────────
def compute_features(fund: pd.DataFrame) -> pd.DataFrame:
    hourly = fund["fundingRate"].resample("1h").mean().ffill()
    out = pd.DataFrame({"fundingRate": hourly})
    out["fundingRate_8h"] = out["fundingRate"].rolling(ROLL_HOURS).sum()
    w = SMA_DAYS * 24
    out["sma7d"]  = out["fundingRate_8h"].rolling(w).mean()
    out["zscore"] = (out["fundingRate_8h"] - out["sma7d"]) / out["fundingRate_8h"].rolling(w).std()
    out["flip"]   = np.sign(out["fundingRate_8h"]).diff().abs().fillna(0).astype(int)
    out["has_sma"]    = out["sma7d"].notna().astype(int)
    out["has_zscore"] = out["zscore"].notna().astype(int)
    out["sma7d"]  = out["sma7d"].fillna(0)
    out["zscore"] = out["zscore"].fillna(0)
    out["hours_since_flip"] = out["flip"].cumsum().groupby(out["flip"].cumsum()).cumcount()
    out["basis"] = np.nan
    return out

# ── Haupt-Logik ─────────────────────────────────────────────────────────────
def process_symbol(symbol: str, start_date: str=None, end_date: str=None):
    logger.info(f"=== Verarbeitung {symbol} ===")
    # Zeitraum ermitteln
    if start_date and end_date:
        sd, ed = start_date[:7], end_date[:7]
    else:
        sd = SYMBOL_START[symbol]
        ed = (datetime.utcnow().date() - timedelta(days=1)).strftime("%Y-%m")

    # Download
    download_and_unzip(symbol, "fundingRate", sd, ed)
    download_and_unzip(symbol, "premiumIndexKlines", sd, ed)

    # Laden & Features
    df_f = load_and_concat_funding(symbol)
    feats = compute_features(df_f)
    prem  = load_and_concat_premium(symbol, feats.index)
    feats["basis"] = prem

    # Atomarer Write: tmp → delete old → rename
    tmp = os.path.join(OUTPUT_DIR, f"{symbol}-funding-features-temp.parquet")
    save_parquet(feats, tmp)
    for old in glob.glob(os.path.join(OUTPUT_DIR, f"{symbol}-funding-features-*.parquet")):
        if old != tmp:
            os.remove(old)
    real_sd = feats.index.min().date()
    real_ed = feats.index.max().date()
    final = os.path.join(OUTPUT_DIR, f"{symbol}-funding-features-{real_sd}_to_{real_ed}.parquet")
    os.rename(tmp, final)
    logger.info(f"✅ {symbol}: geschrieben {len(feats)} Zeilen nach {final}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",     required=True)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date",   default=None)
    args = p.parse_args()
    process_symbol(args.symbol, args.start_date, args.end_date)

if __name__ == "__main__":
    main()
