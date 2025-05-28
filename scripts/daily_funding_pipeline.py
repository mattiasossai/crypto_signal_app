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
    """
    Lädt eine Monats-ZIP herunter und extrahiert die .csv.
    """
    base_url = "https://data.binance.vision/data/futures/um/monthly"
    if kind == "fundingRate":
        out_dir, zip_name = f"{LOCAL_BASE}/fundingRate/{symbol}", f"{symbol}-fundingRate-{month}.zip"
        url = f"{base_url}/fundingRate/{symbol}/{zip_name}"
        dst = f"{out_dir}/{symbol}-fundingRate-{month}.csv"
    else:
        out_dir, zip_name = f"{LOCAL_BASE}/premiumIndexKlines/{symbol}/1h", f"{symbol}-1h-{month}.zip"
        url = f"{base_url}/premiumIndexKlines/{symbol}/1h/{zip_name}"
        dst = f"{out_dir}/{symbol}-1h-{month}.csv"

    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"→ Prüfe {zip_name}")
    # Existenz prüfen
    if subprocess.run(["curl","-f","-s",url,"-o",os.devnull]).returncode != 0:
        logger.info(f"   ❌ {zip_name} nicht gefunden")
        return False

    logger.info(f"   ✔️ vorhanden, lade…")
    # herunterladen & entpacken in dst
    if subprocess.run(["curl","-sSf",url,"-o","tmp.zip"]).returncode == 0:
        subprocess.run(["unzip","-p","tmp.zip"], stdout=open(dst,"wb"), check=True)
        os.remove("tmp.zip")
        logger.info(f"   ✅ {dst}")
        return True
    else:
        logger.warning(f"   ⚠️ Download von {zip_name} fehlgeschlagen")
        return False

def load_and_concat_funding(symbol: str) -> pd.DataFrame:
    """
    Liest alle fundingRate-CSV, erkennt Header-Varianten robust
    und liefert ein DataFrame mit datetime‐Index und Spalte 'fundingRate'.
    """
    files = list_monthly_files(symbol, "fundingRate")
    frames = []
    for fn in files:
        logger.info(f"Lade Funding-CSV {fn}")
        df = pd.read_csv(fn)

        # 1) Zwei-Spalten-CSV (kein Header) erkennen
        if len(df.columns)==2 and str(df.columns[0]).isdigit():
            df.columns = ["timestamp","fundingRate"]
        else:
            # 2) Varianten für Timestamp/FundingRate mappen
            colmap = {c.lower():c for c in df.columns}
            # timestamp
            for alt in ["timestamp","fundingtime","funding_time","time"]:
                if alt in colmap:
                    df.rename(columns={colmap[alt]:"timestamp"}, inplace=True)
                    break
            # fundingRate
            for alt in ["fundingrate","funding_rate","last_funding_rate"]:
                if alt in colmap:
                    df.rename(columns={colmap[alt]:"fundingRate"}, inplace=True)
                    break
            # prüfen
            if not {"timestamp","fundingRate"}.issubset(df.columns):
                logger.error(f"{fn}: Fehlende Spalten nach Umbenennung! Header = {list(df.columns)}")
                continue

        # Timestamp → datetime index
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"],unit="ms",utc=True,errors="coerce")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"],utc=True,errors="coerce")
        df.set_index("timestamp",inplace=True)
        df.sort_index(inplace=True)

        frames.append(df[["fundingRate"]])

    if not frames:
        raise ValueError("Keine Funding-CSV-Dateien gefunden oder alle fehlerhaft.")
    all_f = pd.concat(frames).sort_index().drop_duplicates()
    return all_f

def load_and_concat_premium(symbol: str, idx: pd.DatetimeIndex) -> pd.Series:
    """
    Liest alle premiumIndexKlines-CSV, mappt Header robust
    und resampled auf die funding‐Indexzeiten.
    """
    files = list_monthly_files(symbol, "premiumIndexKlines")
    expected = ["open_time","close"]
    frames = []
    for fn in files:
        logger.info(f"Lade Premium-Index-CSV {fn}")
        df = pd.read_csv(fn)
        # Header-Varianten vereinheitlichen
        cols = [c.lower().replace("opentime","open_time").replace("closetime","close") for c in df.columns]
        df.columns = cols
        # falls Header komplett fehlt, erzwinge positionsbasiert
        if not set(expected).issubset(df.columns):
            df = pd.read_csv(fn, header=None, names=expected+df.columns[len(expected):])
        check_columns(df, expected, fn)
        df["open_time"] = pd.to_datetime(df["open_time"],unit="ms",utc=True,errors="coerce")
        frames.append(df.set_index("open_time")["close"])

    if not frames:
        raise ValueError("Keine Premium-Index-Dateien gefunden.")
    all_p = pd.concat(frames).sort_index().drop_duplicates()
    return all_p.reindex(idx,method="ffill")

def compute_features(fund_df: pd.DataFrame) -> pd.DataFrame:
    hourly = fund_df["fundingRate"].resample("1h").mean().ffill()
    out = pd.DataFrame({"fundingRate":hourly})
    out["fundingRate_8h"] = out["fundingRate"].rolling(ROLL_HOURS).sum()
    w = SMA_DAYS*24
    out["sma7d"]  = out["fundingRate_8h"].rolling(w).mean()
    out["zscore"] = (out["fundingRate_8h"]-out["sma7d"]) / out["fundingRate_8h"].rolling(w).std()
    out["flip"]   = np.sign(out["fundingRate_8h"]).diff().abs().fillna(0).astype(int)
    out["has_sma"]    = out["sma7d"].notna().astype(int)
    out["has_zscore"] = out["zscore"].notna().astype(int)
    out["sma7d"]  = out["sma7d"].fillna(0)
    out["zscore"] = out["zscore"].fillna(0)
    out["flip_cumsum"]      = out["flip"].cumsum()
    out["hours_since_flip"] = out.groupby("flip_cumsum").cumcount()
    out.drop(columns="flip_cumsum",inplace=True)
    out["basis"] = np.nan
    return out

def process_symbol(symbol: str, start_date: str=None, end_date: str=None):
    logger.info(f"=== Verarbeitung {symbol} ===")
    inception   = SYMBOL_START[symbol]
    parquet_dir = OUTPUT_DIR
    os.makedirs(parquet_dir, exist_ok=True)
    pattern = os.path.join(parquet_dir, f"{symbol}-funding-features-*.parquet")
    existing_files = sorted(glob.glob(pattern))

    # Historical-Modus?
    if start_date and end_date:
        # lade jeden Monat im Zeitraum
        start, end = pd.Period(start_date,"M"), pd.Period(end_date,"M")
        months = [p.strftime("%Y-%m") for p in pd.period_range(start,end,freq="M")]
        got = False
        for m in months:
            if download_and_unzip_month(symbol,"fundingRate",m) and download_and_unzip_month(symbol,"premiumIndexKlines",m):
                got = True
        if not got:
            logger.info("❌ Keine Daten im gewählten Zeitraum.")
            return
        df_f = load_and_concat_funding(symbol)
        feats = compute_features(df_f)
        feats["basis"] = load_and_concat_premium(symbol,feats.index)
        real_sd = feats.index.min().strftime("%Y-%m-%d")
        real_ed = feats.index.max().strftime("%Y-%m-%d")
        out_file = os.path.join(parquet_dir,f"{symbol}-funding-features-{real_sd}_to_{real_ed}.parquet")

        # atomar schreiben
        tmp = out_file + ".tmp"
        feats.to_parquet(tmp,engine="pyarrow",compression="snappy")
        os.replace(tmp,out_file)

        # alte löschen
        for old in existing_files:
            if old != out_file: os.remove(old)
        logger.info(f"✅ Full-History Parquet: {out_file}")
        return

    # inkrementell
    if existing_files:
        latest = max(existing_files,key=os.path.getmtime)
        df_old = pd.read_parquet(latest)
        last_month = df_old.index.max().to_period("M") + 1
        next_month = last_month.strftime("%Y-%m")
    else:
        next_month = inception

    if not (download_and_unzip_month(symbol,"fundingRate",next_month)
            and download_and_unzip_month(symbol,"premiumIndexKlines",next_month)):
        logger.info(f"❌ Keine neuen Daten für {next_month}.")
        return

    df_f = load_and_concat_funding(symbol)
    feats = compute_features(df_f)
    feats["basis"] = load_and_concat_premium(symbol,feats.index)
    real_sd = feats.index.min().strftime("%Y-%m-%d")
    real_ed = feats.index.max().strftime("%Y-%m-%d")
    out_file = os.path.join(parquet_dir,f"{symbol}-funding-features-{real_sd}_to_{real_ed}.parquet")

    # atomar schreiben + aufräumen
    tmp = out_file + ".tmp"
    feats.to_parquet(tmp,engine="pyarrow",compression="snappy")
    os.replace(tmp,out_file)
    for old in existing_files:
        if old != out_file: os.remove(old)
    logger.info(f"✅ Neues Parquet: {out_file}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",    required=True)
    p.add_argument("--start-date",default=None)
    p.add_argument("--end-date",  default=None)
    args = p.parse_args()
    process_symbol(args.symbol, args.start_date, args.end_date)

if __name__ == "__main__":
    main()
