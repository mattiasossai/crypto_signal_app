#!/usr/bin/env python3
import argparse
import os
import glob
import gzip
import re
import subprocess
import logging
import datetime

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ─── Utility-Funktionen ───────────────────────────────────────────────────────

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

# ─── Konfiguration ─────────────────────────────────────────────────────────────

logger = init_logger("funding_pipeline")

LOCAL_BASE       = "raw/funding"
OUTPUT_DIR       = "features/funding"
SYMBOL_START     = {
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

# ─── Download-Logik ohne pd.Period, mit Inception-Capping ──────────────────────

def download_monthly_data(symbol: str, sd: pd.Timestamp, ed: pd.Timestamp):
    y, m = sd.year, sd.month
    ye, me = ed.year, ed.month

    out_f = os.path.join(LOCAL_BASE, "fundingRate", symbol)
    os.makedirs(out_f, exist_ok=True)
    while (y < ye) or (y == ye and m <= me):
        period = f"{y:04d}-{m:02d}"
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
        if m > 12:
            m = 1
            y += 1

    out_p = os.path.join(LOCAL_BASE, "premiumIndexKlines", symbol, "1h")
    os.makedirs(out_p, exist_ok=True)
    y, m = sd.year, sd.month
    while (y < ye) or (y == ye and m <= me):
        period = f"{y:04d}-{m:02d}"
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
        if m > 12:
            m = 1
            y += 1

# ─── CSV-Einlese-Funktionen ────────────────────────────────────────────────────

def list_local_csvs(symbol: str, kind: str) -> list[str]:
    if kind == "fundingRate":
        return sorted(glob.glob(f"{LOCAL_BASE}/fundingRate/{symbol}/{symbol}-fundingRate-*.csv"))
    else:
        return sorted(glob.glob(f"{LOCAL_BASE}/premiumIndexKlines/{symbol}/1h/{symbol}-1h-*.csv"))

def load_and_concat_funding(symbol: str, sd: pd.Timestamp) -> pd.DataFrame:
    files = list_local_csvs(symbol, "fundingRate")
    dfs = []
    for fn in files:
        per = re.search(rf"{symbol}-fundingRate-(\d{{4}}-\d{{2}})", fn).group(1)
        year, month = map(int, per.split("-"))
        if (year, month) < (sd.year, sd.month):
            continue
        logger.info(f"Lade Funding-CSV {fn}")
        opener = gzip.open if fn.endswith(".gz") else open
        with opener(fn, "rt") as f:
            df = pd.read_csv(f)
        df.columns = [c.lower() for c in df.columns]
        df["fundingtime"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True, errors="coerce")
        df["fundingrate"] = df["last_funding_rate"]
        dfs.append(df[["fundingtime","fundingrate"]])
    if not dfs:
        raise ValueError("Keine Funding-Dateien gefunden.")
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.sort_values("fundingtime").drop_duplicates("fundingtime")
    all_df.index = all_df["fundingtime"]
    return all_df[["fundingrate"]]

def load_and_concat_premium(symbol: str, idx: pd.DatetimeIndex) -> pd.Series:
    expected = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","count",
        "taker_buy_volume","taker_buy_quote_volume","ignore"
    ]
    files = list_local_csvs(symbol, "premiumIndexKlines")
    frames = []
    for fn in files:
        logger.info(f"Lade Premium-Index-CSV {fn}")
        df = pd.read_csv(fn)
        cols = [c.lower() for c in df.columns]
        cols = [c.replace("opentime","open_time").replace("closetime","close_time") for c in cols]
        df.columns = cols
        if not set(expected).issubset(df.columns):
            df = pd.read_csv(fn, header=None, names=expected)
        if "open_time" not in df.columns:
            raise ValueError(f"{fn}: Spalte 'open_time' fehlt")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
        frames.append(df.set_index("open_time")["close"])
    if not frames:
        raise ValueError("Keine Premium-Index-Dateien gefunden.")
    all_prem = pd.concat(frames).sort_index().drop_duplicates()
    return all_prem.reindex(idx).ffill()

# ─── Feature-Berechnung ───────────────────────────────────────────────────────

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
    out["sma7d"]  = out["sma7d"].fillna(0)
    out["zscore"] = out["zscore"].fillna(0)
    out["flip_cumsum"]      = out["flip"].cumsum()
    out["hours_since_flip"] = out.groupby("flip_cumsum").cumcount()
    out.drop(columns="flip_cumsum", inplace=True)
    out["basis"] = np.nan
    return out

# ─── Hauptfunktion ────────────────────────────────────────────────────────────

def process_symbol(symbol: str, start_date: str = None, end_date: str = None):
    logger.info(f"=== Verarbeitung {symbol} ===")

    if start_date and end_date:
        sd = pd.to_datetime(start_date + "-01").tz_localize("UTC")
        ed = pd.to_datetime(end_date   + "-01").tz_localize("UTC")
    else:
        sd = None
        ed = pd.Timestamp.utcnow().normalize().tz_localize("UTC")

    inception = pd.to_datetime(SYMBOL_START[symbol] + "-01").tz_localize("UTC")
    sd = inception if sd is None else max(sd, inception)

    download_monthly_data(symbol, sd, ed)

    df_f  = load_and_concat_funding(symbol, sd)
    feats = compute_features(df_f)
    prem  = load_and_concat_premium(symbol, feats.index)
    feats["basis"] = prem

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/{symbol}-funding-features.parquet"
 
     # 1) Wenn schon ein Parquet existiert: alte & neue Zeilen mergen
     if os.path.exists(out_path):
         existing = pd.read_parquet(out_path)
         merged   = pd.concat([existing, feats]).sort_index()
         # doppelte Zeitpunkte rauswerfen
         merged   = merged[~merged.index.duplicated(keep="first")]
 
         # ✋ wenn nach dem Merge keine neue Zeile dazugekommen ist → abbrechen
         if len(merged) == len(existing):
             logger.info(f"ℹ️ {symbol}: Keine neuen Funding-Daten – nichts zu tun.")
             return
 
         # ansonsten: überschreiben
         save_parquet(merged, out_path)
         logger.info(f"♻️ {symbol}: Funding-Parquet aktualisiert, +{len(merged)-len(existing)} neue Zeilen")
     else:
         # noch kein File vorhanden → initialer Write
         save_parquet(feats, out_path)
         logger.info(f"✅ {symbol}: Initiales Funding-Parquet erstellt, {len(feats)} Zeilen")
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",     required=True, help="z.B. BTCUSDT")
    p.add_argument("--start-date", default=None, help="YYYY-MM, optional")
    p.add_argument("--end-date",   default=None, help="YYYY-MM, optional")
    args = p.parse_args()
    process_symbol(args.symbol, args.start_date, args.end_date)

if __name__ == "__main__":
    main()
