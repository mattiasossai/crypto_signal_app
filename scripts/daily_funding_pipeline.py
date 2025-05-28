#!/usr/bin/env python3
import os
import glob
import gzip
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

def check_columns(df: pd.DataFrame, required: list[str], fn: str):
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"{fn}: Fehlende Spalten {miss}")

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
    elif kind == "premiumIndexKlines":
        out_dir = f"{LOCAL_BASE}/premiumIndexKlines/{symbol}/1h"
        zip_name = f"{symbol}-1h-{month}.zip"
        url = f"{base_url}/premiumIndexKlines/{symbol}/1h/{zip_name}"
        dst = f"{out_dir}/{symbol}-1h-{month}.csv"
    else:
        raise ValueError("kind muss 'fundingRate' oder 'premiumIndexKlines' sein.")
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
    else:
        logger.warning(f"   ⚠️ Download von {zip_name} fehlgeschlagen")
        return False

def load_and_concat_funding(symbol: str) -> pd.DataFrame:
    files = list_monthly_files(symbol, "fundingRate")
    frames = []
    for fn in files:
        logger.info(f"Lade Funding-CSV {fn}")
        df = pd.read_csv(fn)

        # ----------- HEADER ROBUSTHEIT WIE BEI BOOKDEPTH -----------
        cols = [c.lower() for c in df.columns]
        if len(df.columns) == 2 and str(df.columns[0]).isdigit():
            df.columns = ["timestamp", "fundingRate"]
        else:
            # Mögliche Namensvarianten abdecken
            colmap = {c.lower(): c for c in df.columns}
            # timestamp/ fundingTime Varianten
            if "timestamp" not in colmap:
                for alt in ["fundingtime", "funding_time", "time"]:
                    if alt in colmap:
                        df.rename(columns={colmap[alt]: "timestamp"}, inplace=True)
            if "fundingrate" not in colmap:
                for alt in ["funding_rate", "fundingrate"]:
                    if alt in colmap:
                        df.rename(columns={colmap[alt]: "fundingRate"}, inplace=True)
            # Falls Header immer noch fehlt, breche ab mit Log
            if not {"timestamp", "fundingRate"}.issubset(df.columns):
                logger.error(f"{fn}: Fehlende Spalten nach Umbenennung! Header = {list(df.columns)}")
                continue

        # ----------- ZEITKONVERTIERUNG -----------
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        frames.append(df[["fundingRate"]])
    if not frames:
        raise ValueError("Keine Funding-CSV-Dateien gefunden oder alle haben falsche Header.")
    all_funding = pd.concat(frames).sort_index().drop_duplicates()
    return all_funding

def load_and_concat_premium(symbol: str, idx: pd.DatetimeIndex) -> pd.Series:
    files = list_monthly_files(symbol, "premiumIndexKlines")
    expected = ["open_time", "open", "close"]
    frames = []
    for fn in files:
        logger.info(f"Lade Premium-Index-CSV {fn}")
        df = pd.read_csv(fn)
        cols = [c.lower() for c in df.columns]
        df.columns = [c.replace("opentime", "open_time").replace("closetime", "close") for c in cols]
        if not set(expected).issubset(df.columns):
            df = pd.read_csv(fn, header=None, names=expected+df.columns[len(expected):])
        check_columns(df, ["open_time", "close"], fn)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
        frames.append(df.set_index("open_time")["close"])
    if not frames:
        raise ValueError("Keine Premium-Index-Dateien gefunden.")
    all_prem = pd.concat(frames).sort_index().drop_duplicates()
    return all_prem.reindex(idx, method="ffill")

def compute_features(fund_df: pd.DataFrame) -> pd.DataFrame:
    hourly = fund_df["fundingRate"].resample("1h").mean().ffill()
    out = pd.DataFrame({"fundingRate": hourly})
    out["fundingRate_8h"] = out["fundingRate"].rolling(ROLL_HOURS).sum()
    window = SMA_DAYS * 24
    out["sma7d"]   = out["fundingRate_8h"].rolling(window).mean()
    out["zscore"]  = (out["fundingRate_8h"] - out["sma7d"]) / out["fundingRate_8h"].rolling(window).std()
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
    inception = SYMBOL_START[symbol]
    parquet_dir = OUTPUT_DIR
    pattern = os.path.join(parquet_dir, f"{symbol}-funding-features-*.parquet")
    files = sorted(glob.glob(pattern))

    # --- Full-History-Modus: Lade gezielt gewünschten Bereich ---
    if start_date and end_date:
        start_month = pd.Period(start_date, "M")
        end_month = pd.Period(end_date, "M")
        months = []
        curr = start_month
        while curr <= end_month:
            months.append(curr.strftime("%Y-%m"))
            curr += 1

        got_any = False
        for month in months:
            got_funding = download_and_unzip_month(symbol, "fundingRate", month)
            got_premium = download_and_unzip_month(symbol, "premiumIndexKlines", month)
            if got_funding and got_premium:
                got_any = True
        if not got_any:
            logger.info(f"❌ Für {symbol} keine neuen Daten im Zeitraum {start_date} bis {end_date} – nichts zu tun.")
            return

        # Features für alle geladenen Monate berechnen
        df_fund = load_and_concat_funding(symbol)
        feats   = compute_features(df_fund)
        prem    = load_and_concat_premium(symbol, feats.index)
        feats["basis"] = prem

        real_sd = feats.index.min().strftime("%Y-%m-%d")
        real_ed = feats.index.max().strftime("%Y-%m-%d")
        out_file = os.path.join(parquet_dir, f"{symbol}-funding-features-{real_sd}_to_{real_ed}.parquet")
        feats.to_parquet(out_file, engine="pyarrow", compression="snappy")
        logger.info(f"✅ Neues Parquet (Full-History) gespeichert: {out_file}")

        # Clean-Up: Nur neues Parquet behalten
        for old in glob.glob(pattern):
            if old != out_file:
                os.remove(old)
                logger.info(f"🗑️ Altes Parquet entfernt: {old}")
        return

    # --- Inkrementeller Modus: Prüfe, ob neuer Monat vorhanden ist ---
    if files:
        latest_file = max(files, key=os.path.getmtime)
        existing = pd.read_parquet(latest_file)
        last_idx = existing.index.max()
        last_month = last_idx.to_period("M")
        next_month = (last_month + 1).strftime("%Y-%m")
    else:
        existing = None
        next_month = inception

    # Prüfe neuen Monat, nur wenn ZIP auf Binance vorhanden!
    got_funding = download_and_unzip_month(symbol, "fundingRate", next_month)
    got_premium = download_and_unzip_month(symbol, "premiumIndexKlines", next_month)

    if not (got_funding and got_premium):
        logger.info(f"❌ Für {symbol} keine neuen Daten für {next_month} gefunden – nichts zu tun.")
        return

    # Features aus allen bisherigen + neuem Monat berechnen
    df_fund = load_and_concat_funding(symbol)
    feats   = compute_features(df_fund)
    prem    = load_and_concat_premium(symbol, feats.index)
    feats["basis"] = prem

    real_sd = feats.index.min().strftime("%Y-%m-%d")
    real_ed = feats.index.max().strftime("%Y-%m-%d")
    out_file = os.path.join(parquet_dir, f"{symbol}-funding-features-{real_sd}_to_{real_ed}.parquet")
    feats.to_parquet(out_file, engine="pyarrow", compression="snappy")
    logger.info(f"✅ Neues Parquet gespeichert: {out_file}")

    # Clean-Up: Nur das aktuelle Parquet behalten
    for old in glob.glob(pattern):
        if old != out_file:
            os.remove(old)
            logger.info(f"🗑️ Altes Parquet entfernt: {old}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",     required=True)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date",   default=None)
    args = p.parse_args()
    process_symbol(args.symbol, args.start_date, args.end_date)

if __name__ == "__main__":
    main()
