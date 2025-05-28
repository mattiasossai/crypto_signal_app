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

# ── Integrierte Logger- und Speicher-Funktionen (war vorher in utils.py) ──
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
    if kind == "premium":
        path += "/1h"
        pattern = f"{symbol}-1h-*.csv"
    else:
        pattern = f"{symbol}-fundingRate-*.csv"
    return sorted(glob.glob(f"{path}/{pattern}"))

def remote_exists(symbol: str, kind: str, period: str) -> bool:
    """
    Prüft per HEAD-Request, ob die ZIP auf dem Server existiert.
    """
    base = "https://data.binance.vision/data/futures/um/monthly"
    if kind == "funding":
        zip_name = f"{symbol}-fundingRate-{period}.zip"
        url = f"{base}/fundingRate/{symbol}/{zip_name}"
    else:
        zip_name = f"{symbol}-1h-{period}.zip"
        url = f"{base}/premiumIndexKlines/{symbol}/1h/{zip_name}"
    res = subprocess.run(["curl","-sSfI", url], capture_output=True)
    return res.returncode == 0

def download_and_unzip(symbol: str, kind: str, period: str):
    """
    Lädt genau eine ZIP herunter und entpackt sie direkt nach CSV.
    """
    base = "https://data.binance.vision/data/futures/um/monthly"
    out_dir = f"{LOCAL_BASE}/{kind}/{symbol}"
    os.makedirs(out_dir, exist_ok=True)

    if kind == "funding":
        zip_name = f"{symbol}-fundingRate-{period}.zip"
        url = f"{base}/fundingRate/{symbol}/{zip_name}"
        dst = f"{out_dir}/{symbol}-fundingRate-{period}.csv"
    else:
        zip_name = f"{symbol}-1h-{period}.zip"
        url = f"{base}/premiumIndexKlines/{symbol}/1h/{zip_name}"
        dst = f"{out_dir}/{symbol}-1h-{period}.csv"

    logger.info(f"→ Prüfe {zip_name}")
    if remote_exists(symbol, kind, period):
        logger.info(f"   ✔️ vorhanden, lade herunter…")
        subprocess.run(["curl","-sSf",url,"-o","tmp.zip"], check=True)
        subprocess.run(["unzip","-p","tmp.zip"], stdout=open(dst,"wb"), check=True)
        os.remove("tmp.zip")
        logger.info(f"   ✅ gespeichert nach {dst}")
        return True
    else:
        logger.info(f"   ⚠️ {zip_name} nicht gefunden, skip.")
        return False

def load_and_concat_funding(symbol: str) -> pd.DataFrame:
    files = list_monthly_files(symbol, "funding")
    dfs = []
    for fn in files:
        logger.info(f"Lade Funding-CSV {fn}")
        opener = gzip.open if fn.endswith(".gz") else open
        df = pd.read_csv(opener(fn, "rt"))
        df.columns = [c.lower() for c in df.columns]
        check_columns(df, ["calc_time","last_funding_rate"], fn)
        df["fundingtime"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True, errors="coerce")
        df["fundingrate"] = df["last_funding_rate"]
        dfs.append(df.set_index("fundingtime")[["fundingrate"]])
    if not dfs:
        raise ValueError("Keine Funding-Dateien gefunden.")
    return pd.concat(dfs).sort_index().drop_duplicates()

def load_and_concat_premium(symbol: str, idx: pd.DatetimeIndex) -> pd.Series:
    files = list_monthly_files(symbol, "premium")
    expected = ["open_time","close"]
    frames = []
    for fn in files:
        logger.info(f"Lade Premium-Index-CSV {fn}")
        df = pd.read_csv(fn)
        cols = [c.lower() for c in df.columns]
        df.columns = [c.replace("opentime","open_time").replace("closetime","close") for c in cols]
        if not set(expected).issubset(df.columns):
            # kein Header
            df = pd.read_csv(fn, header=None, names=expected + list(df.columns[len(expected):]))
        check_columns(df, expected, fn)
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

def process_symbol(symbol: str, start_date: str=None, end_date: str=None):
    logger.info(f"=== Verarbeitung {symbol} ===")
    inception = SYMBOL_START[symbol]

    # Zielpfad
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/{symbol}-funding-features.parquet"

    # === 1) Historisch oder Inkrementell? ===
    if start_date and end_date:
        # kompletter historischer Lauf
        months = pd.period_range(start_date, end_date, freq="M").strftime("%Y-%m")
        to_fetch_funding = list(months)
        to_fetch_premium = list(months)
    else:
        # inkrementeller Lauf: nur *nächsten* Monat prüfen
        if os.path.exists(out_path):
            existing = pd.read_parquet(out_path)
            last_ts = existing.index.max()
            next_period = (last_ts.to_period("M") + 1)
        else:
            next_period = pd.Period(inception, "M")
        per = next_period.strftime("%Y-%m")

        # nur einen Monat, wenn remote existiert
        to_fetch_funding = [per] if remote_exists(symbol, "funding", per) else []
        to_fetch_premium = [per] if remote_exists(symbol, "premium", per) else []

        if not to_fetch_funding and not to_fetch_premium:
            logger.info(f"ℹ️ {symbol}: Kein neuer Monat ({per}) zum Download.")
            return

    # === 2) Download ===
    for per in to_fetch_funding:
        download_and_unzip(symbol, "funding", per)
    for per in to_fetch_premium:
        download_and_unzip(symbol, "premium", per)

    # === 3) Laden & Features ===
    df_fund = load_and_concat_funding(symbol)
    feats   = compute_features(df_fund)
    prem    = load_and_concat_premium(symbol, feats.index)
    feats["basis"] = prem

    # === 4) Schreiben/Appenden ===
    if os.path.exists(out_path):
        existing = pd.read_parquet(out_path)
        merged   = pd.concat([existing, feats]).sort_index()
        merged   = merged[~merged.index.duplicated(keep="first")]
        if len(merged) == len(existing):
            logger.info(f"ℹ️ {symbol}: Keine neuen Zeilen – nothing to do.")
            return
        save_parquet(merged, out_path)
        logger.info(f"♻️ {symbol}: +{len(merged)-len(existing)} Zeilen angehängt")
    else:
        save_parquet(feats, out_path)
        logger.info(f"✅ {symbol}: Initial erstellt, {len(feats)} Zeilen")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",     required=True)
    p.add_argument("--start-date", default=None, help="YYYY-MM")
    p.add_argument("--end-date",   default=None, help="YYYY-MM")
    args = p.parse_args()
    process_symbol(args.symbol, args.start_date, args.end_date)

if __name__ == "__main__":
    main()
