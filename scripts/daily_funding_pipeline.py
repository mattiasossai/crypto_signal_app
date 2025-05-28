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

# ── Logger & Parquet-Save ──
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

def remote_exists(url: str) -> bool:
    """Prüft per HEAD-Request, ob die ZIP auf Binance existiert."""
    res = subprocess.run(
        ["curl", "-s", "-I", "-f", url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return res.returncode == 0

def list_monthly_files(symbol: str, kind: str) -> list[str]:
    path = f"{LOCAL_BASE}/{kind}/{symbol}"
    if kind == "premiumIndexKlines":
        path += "/1h"
        pat = f"{symbol}-1h-*.csv"
    else:
        pat = f"{symbol}-fundingRate-*.csv"
    return sorted(glob.glob(f"{path}/{pat}"))

def download_and_unzip(symbol: str, kind: str, start: str, end: str):
    """
    Lädt nur vorhandene ZIPs von YYYY-MM=start bis end herunter
    und entpackt sie direkt ins CSV-Format.
    """
    base = "https://data.binance.vision/data/futures/um/monthly"
    out  = f"{LOCAL_BASE}/{kind}/{symbol}"
    os.makedirs(out, exist_ok=True)

    curr = pd.Period(start, "M")
    last = pd.Period(end,   "M")
    while curr <= last:
        per     = curr.strftime("%Y-%m")
        if kind=="fundingRate":
            zipn = f"{symbol}-fundingRate-{per}.zip"
            url  = f"{base}/fundingRate/{symbol}/{zipn}"
            dst  = f"{out}/{symbol}-fundingRate-{per}.csv"
        else:
            zipn = f"{symbol}-1h-{per}.zip"
            url  = f"{base}/premiumIndexKlines/{symbol}/1h/{zipn}"
            dst  = f"{out}/{symbol}-1h-{per}.csv"

        logger.info(f"→ Prüfe {zipn}")
        if remote_exists(url):
            logger.info("  ✔️ vorhanden, lade…")
            subprocess.run(["curl","-sSf",url,"-o","tmp.zip"], check=True)
            subprocess.run(
                ["unzip","-p","tmp.zip"],
                stdout=open(dst,"wb"), check=True
            )
            os.remove("tmp.zip")
            logger.info(f"  ✅ {dst}")
        else:
            logger.info("  ℹ️ nicht gefunden, überspringe.")
        curr += 1

def check_columns(df: pd.DataFrame, req: list[str], fn: str):
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"{fn}: Fehlende Spalten {miss}")

def load_and_concat_funding(symbol: str) -> pd.DataFrame:
    files = list_monthly_files(symbol, "fundingRate")
    dfs = []
    for fn in files:
        logger.info(f"Lade Funding {fn}")
        opener = gzip.open if fn.endswith(".gz") else open
        df = pd.read_csv(opener(fn,"rt"))
        df.columns = [c.lower() for c in df.columns]
        check_columns(df, ["calc_time","funding_interval_hours","last_funding_rate"], fn)
        df["fundingtime"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
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
        logger.info(f"Lade Premium {fn}")
        df = pd.read_csv(fn)
        cols = [c.lower() for c in df.columns]
        df.columns = [c.replace("opentime","open_time").replace("closetime","close") for c in cols]
        if not set(expected).issubset(df.columns):
            df = pd.read_csv(fn, header=None, names=expected+df.columns[len(expected):])
        check_columns(df, ["open_time","close"], fn)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        frames.append(df.set_index("open_time")["close"])
    if not frames:
        raise ValueError("Keine Premium-Index-Dateien gefunden.")
    prem = pd.concat(frames).sort_index().drop_duplicates()
    return prem.reindex(idx, method="ffill")

def compute_features(fund_df: pd.DataFrame) -> pd.DataFrame:
    hourly = fund_df["fundingrate"].resample("1h").mean().ffill()
    out = pd.DataFrame({"fundingRate": hourly})
    out["fundingRate_8h"] = out["fundingRate"].rolling(ROLL_HOURS).sum()
    w = SMA_DAYS*24
    out["sma7d"]  = out["fundingRate_8h"].rolling(w).mean()
    out["zscore"] = (out["fundingRate_8h"]-out["sma7d"])/out["fundingRate_8h"].rolling(w).std()
    out["flip"]   = np.sign(out["fundingRate_8h"]).diff().abs().fillna(0).astype(int)
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/{symbol}-funding-features.parquet"

    # 1) Historisch nur wenn explizit über CLI:
    if start_date and end_date:
        download_start = start_date[:7]
        download_end   = end_date[:7]
    else:
        # 2) Inkrementell: nur letzten abgeschlossenen Monat
        last_month = (pd.Timestamp.utcnow().to_period("M") - 1).strftime("%Y-%m")
        if os.path.exists(out_path):
            existing      = pd.read_parquet(out_path)
            last_ts       = existing.index.max()
            next_month    = (last_ts.to_period("M")+1).strftime("%Y-%m")
            download_start = next_month
        else:
            # hier nur letzten Monat, nicht vom Inception
            download_start = last_month
        download_end = last_month

    # 3) Download falls neuer Monat
    if pd.Period(download_start,"M") <= pd.Period(download_end,"M"):
        download_and_unzip(symbol, "fundingRate", download_start, download_end)
        download_and_unzip(symbol, "premiumIndexKlines", symbol, download_start, download_end)
    else:
        logger.info(f"ℹ️ {symbol}: kein neuer Monat ({download_start}>{download_end})")

    # 4) Daten laden & Feature-Berechnung
    df_fund = load_and_concat_funding(symbol)
    feats   = compute_features(df_fund)
    prem    = load_and_concat_premium(symbol, feats.index)
    feats["basis"] = prem

    # 5) Parquet schreiben / anhängen
    if os.path.exists(out_path):
        existing = pd.read_parquet(out_path)
        merged   = pd.concat([existing, feats]).sort_index()
        merged   = merged[~merged.index.duplicated(keep="first")]
        if len(merged)==len(existing):
            logger.info("ℹ️ keine neuen Zeilen – done")
            return
        save_parquet(merged, out_path)
        logger.info(f"♻️ angehängt +{len(merged)-len(existing)} Zeilen")
    else:
        save_parquet(feats, out_path)
        logger.info(f"✅ initial erstellt, {len(feats)} Zeilen")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",     required=True)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date",   default=None)
    args = p.parse_args()
    process_symbol(args.symbol, args.start_date, args.end_date)
