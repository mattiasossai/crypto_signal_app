#!/usr/bin/env python3
import os
import glob
import argparse
import subprocess
import logging
import pandas as pd
import re
import numpy as np
from datetime import datetime, timedelta

def init_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
        logger.addHandler(h)
    return logger

logger = init_logger("funding_pipeline")

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
SMA_DAYS = 7

def list_monthly_files(symbol: str, kind: str) -> list[str]:
    path = os.path.join(LOCAL_BASE, kind, symbol)
    if kind == "premiumIndexKlines":
        path = os.path.join(path, "1h")
        pattern = f"{symbol}-1h-*.csv"
    else:
        pattern = f"{symbol}-fundingRate-*.csv"
    return sorted(glob.glob(os.path.join(path, pattern)))

def download_and_unzip_month(symbol: str, kind: str, month: str) -> bool:
    """
    Lädt die ZIP-Datei für `symbol`/`kind` und Monat im Format YYYY-MM,
    entpackt per `unzip -p` direkt in die CSV im LOCAL_BASE-Verzeichnis.
    """
    base = "https://data.binance.vision/data/futures/um/monthly"
    if kind == "fundingRate":
        zip_name = f"{symbol}-fundingRate-{month}.zip"
        url = f"{base}/fundingRate/{symbol}/{zip_name}"
        dst_dir = os.path.join(LOCAL_BASE, "fundingRate", symbol)
        dst = os.path.join(dst_dir, f"{symbol}-fundingRate-{month}.csv")
    else:
        zip_name = f"{symbol}-1h-{month}.zip"
        url = f"{base}/premiumIndexKlines/{symbol}/1h/{zip_name}"
        dst_dir = os.path.join(LOCAL_BASE, "premiumIndexKlines", symbol, "1h")
        dst = os.path.join(dst_dir, f"{symbol}-1h-{month}.csv")
    os.makedirs(dst_dir, exist_ok=True)

    logger.info(f"→ Prüfe {zip_name}")
    # erst Verfügbarkeit checken
    if subprocess.run(["curl","-f","-s",url,"-o",os.devnull]).returncode != 0:
        logger.info(f"   ❌ {zip_name} nicht gefunden")
        return False

    # dann herunterladen + entpacken
    logger.info(f"   ✔️ vorhanden, lade…")
    if subprocess.run(["curl","-sSf",url,"-o","tmp.zip"]).returncode == 0:
        subprocess.run(["unzip","-p","tmp.zip"], stdout=open(dst,"wb"), check=True)
        os.remove("tmp.zip")
        logger.info(f"   ✅ {dst}")
        return True
    else:
        logger.warning(f"   ⚠️ Download von {zip_name} fehlgeschlagen")
        return False

# ── Flexible CSV-Leser für FundingRate & PremiumIndexKlines ──
def read_csv_flexible(path: str, kind: str) -> pd.DataFrame:
    """
    Liest FundingRate- oder PremiumIndexKlines-CSV ein,
    erkennt headerlose Dateien und normalisiert:
      • Index: UTC-Datetime aus 'timestamp'
      • kind='fundingRate'       → Spalte 'fundingRate'
      • kind='premiumIndexKlines'→ Spalte 'close'
    """

    # 1) Prüfen auf headerlos
    sample = pd.read_csv(path, nrows=1, header=None).iloc[0].tolist()
    headerless = all(str(x).replace('.', '', 1).lstrip('-').isdigit() for x in sample)

    # 2) Einlesen
    df = pd.read_csv(path, header=None if headerless else 0)

    # ── headerlose Zuordnung vorziehen ──────────────────────────────
    if kind == "fundingRate" and headerless:
        df.columns = ["timestamp","funding_interval_hours","last_funding_rate"][:df.shape[1]]
    if kind == "premiumIndexKlines" and headerless:
        cols = [
            "open_time","open","high","low","close","volume",
            "close_time","quote_volume","count",
            "taker_buy_volume","taker_buy_quote_volume","ignore"
        ]
        df.columns = cols[:df.shape[1]]

    # Jetzt erst die normierte Spalten-Map bauen
    norm_map = {}
    for col in df.columns:
        key = re.sub(r"[_\s]", "", str(col).lower())
        norm_map[key] = col

    # ── FundingRate ─────────────────────────────────────────────────
    if kind == "fundingRate":
        # Timestamp-Feld finden
        for k in ("calctime","timestamp","fundingtime"):
            if k in norm_map:
                ts = norm_map[k]; break
        else:
            raise ValueError(f"{path}: Kein Timestamp-Feld in {list(norm_map)}")

        # Rate-Feld finden
        for k in ("lastfundingrate","fundingrate"):
            if k in norm_map:
                fr = norm_map[k]; break
        else:
            raise ValueError(f"{path}: Kein FundingRate-Feld in {list(norm_map)}")

        df = df.rename(columns={ts:"timestamp", fr:"fundingRate"})
        df = df[["timestamp","fundingRate"]]

    # ── PremiumIndexKlines ──────────────────────────────────────────
    else:
        # Timestamp-Feld finden
        for k in ("opentime","open_time"):
            if k in norm_map:
                ts = norm_map[k]; break
        else:
            raise ValueError(f"{path}: Kein open_time/opentime in {list(norm_map)}")

        # Close-Feld finden
        if "close" in norm_map:
            cl = norm_map["close"]
        else:
            raise ValueError(f"{path}: Kein close-Feld in {list(norm_map)}")

        df = df.rename(columns={ts:"timestamp", cl:"close"})
        df = df[["timestamp","close"]]

    # ── Timestamp → datetime UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    df = df.set_index("timestamp").sort_index()

    return df

def load_and_concat_funding(symbol: str) -> pd.DataFrame:
    files = list_monthly_files(symbol, "fundingRate")
    frames = []
    for fn in files:
        logger.info(f"Lade Funding-CSV {fn}")
        try:
            df = read_csv_flexible(fn, "fundingRate")
        except Exception as e:
            logger.error(f"{fn}: {e}")
            continue
        frames.append(df[["fundingRate"]])
    if not frames:
        raise ValueError("Keine validen Funding-CSV-Dateien gefunden.")
    df_all = pd.concat(frames).sort_index().drop_duplicates()
    return df_all

def load_and_concat_premium(symbol: str, idx: pd.DatetimeIndex) -> pd.Series:
    files = list_monthly_files(symbol, "premiumIndexKlines")
    frames = []
    for fn in files:
        logger.info(f"Lade Premium-Index-CSV {fn}")
        try:
            df = read_csv_flexible(fn, "premiumIndexKlines")
        except Exception as e:
            logger.error(f"{fn}: {e}")
            continue
        frames.append(df["close"])
    if not frames:
        raise ValueError("Keine Premium-Index-Dateien gefunden.")
    all_prem = pd.concat(frames).sort_index().drop_duplicates()
    # auf Funding-Stunden resamplen + ffill
    return all_prem.reindex(idx, method="ffill")

def compute_features(fund_df: pd.DataFrame) -> pd.DataFrame:
    hourly = fund_df["fundingRate"].resample("1h").mean().ffill()
    out = pd.DataFrame({"fundingRate": hourly})
    out["fundingRate_8h"] = out["fundingRate"].rolling(ROLL_HOURS).sum()
    win = SMA_DAYS * 24
    out["sma7d"] = out["fundingRate_8h"].rolling(win).mean()
    out["zscore"] = (out["fundingRate_8h"] - out["sma7d"]) \
                    / out["fundingRate_8h"].rolling(win).std()
    out["flip"] = np.sign(out["fundingRate_8h"]).diff().abs().fillna(0).astype(int)
    out["has_sma"] = out["sma7d"].notna().astype(int)
    out["has_zscore"] = out["zscore"].notna().astype(int)
    out["sma7d"]  = out["sma7d"].fillna(0)
    out["zscore"] = out["zscore"].fillna(0)
    out["flip_cumsum"] = out["flip"].cumsum()
    out["hours_since_flip"] = out.groupby("flip_cumsum").cumcount()
    out.drop(columns="flip_cumsum", inplace=True)
    out["basis"] = np.nan
    return out

def process_symbol(symbol: str, start_date: str=None, end_date: str=None):
    logger.info(f"=== Verarbeitung {symbol} ===")
    inception = SYMBOL_START[symbol]

    # bestimmen, welche Monate wir brauchen
    if start_date and end_date:
        start = pd.Period(start_date, "M")
        end = pd.Period(end_date, "M")
    else:
        start = pd.Period(inception, "M")
        yesterday = datetime.utcnow().date() - timedelta(days=1)
        end = pd.Period(yesterday, "M")

    # Download aller Monate
    for per in pd.period_range(start, end, freq="M"):
        m = per.strftime("%Y-%m")
        download_and_unzip_month(symbol, "fundingRate", m)
        download_and_unzip_month(symbol, "premiumIndexKlines", m)

    # Laden & Features
    df_fund = load_and_concat_funding(symbol)
    feats = compute_features(df_fund)
    feats["basis"] = load_and_concat_premium(symbol, feats.index)

    # Speichern – nur ein Parquet im OUTPUT_DIR
    sd = feats.index.min().strftime("%Y-%m-%d")
    ed = feats.index.max().strftime("%Y-%m-%d")
    out = os.path.join(OUTPUT_DIR, f"{symbol}-funding-features-{sd}_to_{ed}.parquet")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    feats.to_parquet(out, engine="pyarrow", compression="snappy")
    logger.info(f"✅ {symbol}: geschrieben {len(feats)} Zeilen nach {out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    args = p.parse_args()
    process_symbol(args.symbol, args.start_date, args.end_date)

if __name__ == "__main__":
    main()
