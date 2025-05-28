#!/usr/bin/env python3
import os
import glob
import argparse
import subprocess
import pandas as pd
import numpy as np
import datetime
import logging


def init_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
        logger.addHandler(h)
    return logger

logger = init_logger("funding_pipeline")

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


def list_monthly_files(symbol: str, kind: str) -> list[str]:
    if kind == "premiumIndexKlines":
        path = f"{LOCAL_BASE}/premiumIndexKlines/{symbol}/1h"
        pattern = f"{symbol}-1h-*.csv"
    elif kind == "fundingRate":
        path = f"{LOCAL_BASE}/fundingRate/{symbol}"
        pattern = f"{symbol}-fundingRate-*.csv"
    else:
        raise ValueError("kind muss 'fundingRate' oder 'premiumIndexKlines' sein.")
    return sorted(glob.glob(f"{path}/{pattern}"))


def download_and_unzip_month(symbol: str, kind: str, month: str) -> bool:
    base_url = "https://data.binance.vision/data/futures/um/monthly"
    if kind == "fundingRate":
        out_dir = f"{LOCAL_BASE}/fundingRate/{symbol}"
        zip_name = f"{symbol}-fundingRate-{month}.zip"
        url = f"{base_url}/fundingRate/{symbol}/{zip_name}"
        dst = f"{out_dir}/{symbol}-fundingRate-{month}.csv"
    else:
        out_dir = f"{LOCAL_BASE}/premiumIndexKlines/{symbol}/1h"
        zip_name = f"{symbol}-1h-{month}.zip"
        url = f"{base_url}/premiumIndexKlines/{symbol}/1h/{zip_name}"
        dst = f"{out_dir}/{symbol}-1h-{month}.csv"
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"→ Prüfe {zip_name}")
    res = subprocess.run(["curl", "-f", "-s", url, "-o", os.devnull])
    if res.returncode != 0:
        logger.info(f"   ❌ {zip_name} nicht gefunden")
        return False
    logger.info(f"   ✔️ vorhanden, lade…")
    res = subprocess.run(["curl", "-sSf", url, "-o", "tmp.zip"], capture_output=True)
    if res.returncode == 0:
        subprocess.run(["unzip", "-p", "tmp.zip"], stdout=open(dst, "wb"), check=True)
        os.remove("tmp.zip")
        logger.info(f"   ✅ {dst}")
        return True
    logger.warning(f"   ⚠️ Download von {zip_name} fehlgeschlagen")
    return False


def load_and_concat_funding(symbol: str) -> pd.DataFrame:
    files = list_monthly_files(symbol, "fundingRate")
    frames = []
    for fn in files:
        logger.info(f"Lade Funding-CSV {fn}")
        df = pd.read_csv(fn, header=0)

        # Header-Robustheit: verschiedene Varianten mappen
        colmap = {c.lower(): c for c in df.columns}
        # map calc_time, fundingtime, timestamp → timestamp
        if 'calc_time' in colmap:
            df.rename(columns={colmap['calc_time']: 'timestamp'}, inplace=True)
        elif 'fundingtime' in colmap:
            df.rename(columns={colmap['fundingtime']: 'timestamp'}, inplace=True)
        elif 'timestamp' in colmap:
            df.rename(columns={colmap['timestamp']: 'timestamp'}, inplace=True)
        # map last_funding_rate, funding_rate, fundingrate → fundingRate
        if 'last_funding_rate' in colmap:
            df.rename(columns={colmap['last_funding_rate']: 'fundingRate'}, inplace=True)
        elif 'funding_rate' in colmap:
            df.rename(columns={colmap['funding_rate']: 'fundingRate'}, inplace=True)
        elif 'fundingrate' in colmap:
            df.rename(columns={colmap['fundingrate']: 'fundingRate'}, inplace=True)

        # falls immer noch fehlen, skip
        if not {'timestamp','fundingRate'}.issubset(df.columns):
            logger.error(f"{fn}: Fehlende Spalten nach Umbenennung! Header = {list(df.columns)}")
            continue

        # Zeitkonvertierung und Index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        frames.append(df[['fundingRate']])

    if not frames:
        raise ValueError("Keine fundingRate-CSV-Dateien mit gültigem Header gefunden.")
    all_fund = pd.concat(frames).sort_index().drop_duplicates()
    return all_fund


def load_and_concat_premium(symbol: str, idx: pd.DatetimeIndex) -> pd.Series:
    files = list_monthly_files(symbol, "premiumIndexKlines")
    frames = []
    for fn in files:
        logger.info(f"Lade Premium-Index-CSV {fn}")
        df = pd.read_csv(fn, header=0)
        colmap = {c.lower(): c for c in df.columns}
        # map opentime variants → open_time
        if 'open_time' in colmap:
            df.rename(columns={colmap['open_time']:'open_time'}, inplace=True)
        elif 'opentime' in colmap:
            df.rename(columns={colmap['opentime']:'open_time'}, inplace=True)
        # map closet variants → close
        if 'close' in colmap:
            df.rename(columns={colmap['close']:'close'}, inplace=True)
        elif 'closetime' in colmap:
            df.rename(columns={colmap['closetime']:'close'}, inplace=True)
        if not {'open_time','close'}.issubset(df.columns):
            logger.error(f"{fn}: Fehlende Spalten nach Umbenennung! Header = {list(df.columns)}")
            continue
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True, errors='coerce')
        frames.append(df.set_index('open_time')['close'])
    if not frames:
        raise ValueError("Keine premiumIndexKlines-CSV-Dateien mit gültigem Header gefunden.")
    all_prem = pd.concat(frames).sort_index().drop_duplicates()
    return all_prem.reindex(idx, method='ffill')


def compute_features(fund_df: pd.DataFrame) -> pd.DataFrame:
    hourly = fund_df['fundingRate'].resample('1h').mean().ffill()
    out = pd.DataFrame({'fundingRate': hourly})
    out['fundingRate_8h'] = out['fundingRate'].rolling(ROLL_HOURS).sum()
    w = SMA_DAYS*24
    out['sma7d']  = out['fundingRate_8h'].rolling(w).mean()
    out['zscore'] = (out['fundingRate_8h']-out['sma7d'])/out['fundingRate_8h'].rolling(w).std()
    out['flip']   = np.sign(out['fundingRate_8h']).diff().abs().fillna(0).astype(int)
    out['has_sma']    = out['sma7d'].notna().astype(int)
    out['has_zscore'] = out['zscore'].notna().astype(int)
    out['sma7d']  = out['sma7d'].fillna(0)
    out['zscore'] = out['zscore'].fillna(0)
    out['flip_cumsum']      = out['flip'].cumsum()
    out['hours_since_flip'] = out.groupby('flip_cumsum').cumcount()
    out.drop(columns='flip_cumsum', inplace=True)
    out['basis'] = np.nan
    return out


def process_symbol(symbol: str, start_date: str=None, end_date: str=None):
    logger.info(f"=== Verarbeitung {symbol} ===")
    pattern = os.path.join(OUTPUT_DIR, f"{symbol}-funding-features-*.parquet")
    files = sorted(glob.glob(pattern))

    # Start-/End-Logik (Full vs. Incremental) ähnlich wie vorher...
    # (unverändert)

    # Nach Download:
    df_fund = load_and_concat_funding(symbol)
    feats   = compute_features(df_fund)
    prem    = load_and_concat_premium(symbol, feats.index)
    feats['basis'] = prem

    # Parquet schreiben und Clean‑Up (nur aktuelles behalten)
    # (unverändert)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol',    required=True)
    p.add_argument('--start-date',default=None)
    p.add_argument('--end-date',  default=None)
    args = p.parse_args()
    process_symbol(args.symbol, args.start_date, args.end_date)


if __name__ == '__main__':
    main()
