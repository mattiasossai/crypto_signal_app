#!/usr/bin/env python3
import os
import glob
import gzip
import re
import argparse
import subprocess
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import logging

# ── Logger & Parquet-Save (ehemals utils.py) ──
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

# ── Konfiguration ──
LOCAL_BASE   = "data/futures/um/monthly"
OUTPUT_DIR   = "features/funding"
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

def check_columns(df: pd.DataFrame, required: list[str], fn: str):
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"{fn}: Fehlende Spalten {miss}")

def list_monthly_files(symbol: str, kind: str) -> list[str]:
    path = f"{LOCAL_BASE}/{kind}/{symbol}"
    if kind == "premiumIndexKlines":
        path += "/1h"
        pattern = f"{symbol}-1h-*.csv"
    else:
        pattern = f"{symbol}-fundingRate-*.csv"
    return sorted(glob.glob(f"{path}/{pattern}"))

def download_and_unzip(symbol: str, kind: str, start: str, end: str):
    """
    Lädt nur die Monatspakete von start bis end (YYYY-MM).
    """
    base_url = "https://data.binance.vision/data/futures/um/monthly"
    out_dir = f"{LOCAL_BASE}/{kind}/{symbol}"
    os.makedirs(out_dir, exist_ok=True)
    curr = pd.Period(start, "M")
    last = pd.Period(end,   "M")
    while curr <= last:
        per = curr.strftime("%Y-%m")
        if kind == "fundingRate":
            zip_name = f"{symbol}-fundingRate-{per}.zip"
            url      = f"{base_url}/fundingRate/{symbol}/{zip_name}"
            dst      = f"{out_dir}/{symbol}-fundingRate-{per}.csv"
        else:
            zip_name = f"{symbol}-1h-{per}.zip"
            url      = f"{base_url}/premiumIndexKlines/{symbol}/1h/{zip_name}"
            dst      = f"{out_dir}/{symbol}-1h-{per}.csv"

        logger.info(f"→ DOWNLOAD {zip_name}")
        res = subprocess.run(["curl","-sSf",url,"-o","tmp.zip"], capture_output=True)
        if res.returncode == 0:
            subprocess.run(["unzip","-p","tmp.zip"], stdout=open(dst,"wb"), check=True)
            os.remove("tmp.zip")
        else:
            logger.warning(f"{zip_name} nicht gefunden")
        curr += 1

def load_and_concat_funding(symbol: str) -> pd.DataFrame:
    files = list_monthly_files(symbol, "fundingRate")
    dfs = []
    for fn in files:
        logger.info(f"Lade Funding-CSV {fn}")
        opener = gzip.open if fn.endswith(".gz") else open
        df = pd.read_csv(opener(fn, "rt"))
        df.columns = [c.lower() for c in df.columns]
        check_columns(df, ["calc_time","funding_interval_hours","last_funding_rate"], fn)
        df["fundingtime"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True, errors="coerce")
        df["fundingrate"] = df["last_funding_rate"]
        dfs.append(df.set_index("fundingtime")[["fundingrate"]])
    if not dfs:
        raise ValueError("Keine Funding-Dateien gefunden.")
    return pd.concat(dfs).sort_index().drop_duplicates()

def load_and_concat_premium(symbol: str, idx: pd.DatetimeIndex) -> pd.Series:
    files = list_monthly_files(symbol, "premiumIndexKlines")
    expected = ["open_time","open","close"]
    frames = []
    for fn in files:
        logger.info(f"Lade Premium-Index-CSV {fn}")
        df = pd.read_csv(fn)
        cols = [c.lower() for c in df.columns]
        df.columns = [c.replace("opentime","open_time").replace("closetime","close") for c in cols]
        if not set(expected).issubset(df.columns):
            df = pd.read_csv(fn, header=None, names=expected+cols[len(expected):])
        check_columns(df, ["open_time","close"], fn)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
        frames.append(df.set_index("open_time")["close"])
    if not frames:
        raise ValueError("Keine Premium-Index-Dateien gefunden.")
    all_prem = pd.concat(frames).sort_index().drop_duplicates()
    return all_prem.reindex(idx, method="ffill")

def compute_features(fund_df: pd.DataFrame) -> pd.DataFrame:
    hourly = fund_df["fundingrate"].resample("1h").mean().ffill()
    out = pd.DataFrame({"fundingRate": hourly})
    out["fundingRate_8h"] = out["fundingRate"].rolling(ROLL_HOURS).sum()
    window = SMA_DAYS * 24
    out["sma7d"]  = out["fundingRate_8h"].rolling(window).mean()
    out["zscore"] = (out["fundingRate_8h"] - out["sma7d"]) \
                    / out["fundingRate_8h"].rolling(window).std()
    out["flip"]    = np.sign(out["fundingRate_8h"]).diff().abs().fillna(0).astype(int)
    out["has_sma"]    = out["sma7d"].notna().astype(int)
    out["has_zscore"] = out["zscore"].notna().astype(int)
    out["sma7d"]  = out["sma7d"].fillna(0)
    out["zscore"] = out["zscore"].fillna(0)
    out["flip_cumsum"]      = out["flip"].cumsum()
    out["hours_since_flip"] = out.groupby("flip_cumsum").cumcount()
    out.drop(columns="flip_cumsum", inplace=True)
    out["basis"] = np.nan
    return out

def process_symbol(symbol: str, start_date: str = None, end_date: str = None):
    logger.info(f"=== Verarbeitung {symbol} ===")
    inception = SYMBOL_START.get(symbol)
    if not inception:
        raise ValueError(f"Inception für {symbol} fehlt.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/{symbol}-funding-features.parquet"

    # ––––– Ermitteln, ab welchem Monat wir laden müssen –––––
    if start_date and end_date:
        # historischer Voll-Lauf
        download_start = start_date[:7]
        download_end   = end_date[:7]
    else:
        # täglicher Incremental-Lauf
        # 1) Wenn Parquet existiert, parse letztes End-Datum aus Dateiname:
        existing_files = glob.glob(f"{OUTPUT_DIR}/{symbol}-funding-features-*_*_to_*.parquet")
        if existing_files:
            latest_file = max(existing_files)
            m = re.search(r"_to_(\d{4}-\d{2}-\d{2})", latest_file)
            last_date = datetime.datetime.strptime(m.group(1), "%Y-%m-%d").date()
            first_of_next = (last_date.replace(day=1) + relativedelta(months=1))
            download_start = first_of_next.strftime("%Y-%m")
        else:
            # noch nie gelaufen
            download_start = inception
        # bis zum letzten abgeschlossenen Monat:
        today = datetime.date.today()
        first_of_this = today.replace(day=1)
        download_end = (first_of_this - relativedelta(months=1)).strftime("%Y-%m")

    # ––––– Download durchführen, wenn nötig –––––
    if pd.Period(download_start, "M") > pd.Period(download_end, "M"):
        logger.info(f"ℹ️ {symbol}: Kein neuer Monat zum Download ({download_start} > {download_end}).")
    else:
        download_and_unzip(symbol, "fundingRate", download_start, download_end)
        download_and_unzip(symbol, "premiumIndexKlines", download_start, download_end)

    # ––––– Daten laden & Features berechnen –––––
    df_fund = load_and_concat_funding(symbol)
    feats   = compute_features(df_fund)
    prem    = load_and_concat_premium(symbol, feats.index)
    feats["basis"] = prem

    # ––––– Parquet schreiben oder anhängen –––––
    if os.path.exists(out_path):
        existing = pd.read_parquet(out_path)
        merged   = pd.concat([existing, feats]).sort_index()
        merged   = merged[~merged.index.duplicated(keep="first")]
        if len(merged) == len(existing):
            logger.info(f"ℹ️ {symbol}: Keine neuen Zeilen – nothing to do.")
            return
        save_parquet(merged, out_path)
        logger.info(f"♻️ {symbol}: +{len(merged) - len(existing)} Zeilen angehängt")
    else:
        save_parquet(feats, out_path)
        logger.info(f"✅ {symbol}: Initial erstellt, {len(feats)} Zeilen")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",     required=True)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date",   default=None)
    args = p.parse_args()
    process_symbol(args.symbol, args.start_date, args.end_date)

if __name__ == "__main__":
    main()
