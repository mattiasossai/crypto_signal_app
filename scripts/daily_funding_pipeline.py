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
    if kind == "premiumIndexKlines":
        path += "/1h"
        pattern = f"{symbol}-1h-*.csv"
    else:
        pattern = f"{symbol}-fundingRate-*.csv"
    return sorted(glob.glob(f"{path}/{pattern}"))

def download_and_unzip(symbol: str, kind: str, start: str, end: str):
    """
    Lädt nur die Monate von start bis end (YYYY-MM) herunter,
    überspringt bestehende CSV-Dateien.
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
            dst      = f"{out_dir}/{symbol}-fundingRate-{per}.csv"
            url      = f"{base_url}/fundingRate/{symbol}/{zip_name}"
        else:
            zip_name = f"{symbol}-1h-{per}.zip"
            dst      = f"{out_dir}/{symbol}-1h-{per}.csv"
            url      = f"{base_url}/premiumIndexKlines/{symbol}/1h/{zip_name}"

        if os.path.exists(dst):
            logger.info(f"   ✔️ {os.path.basename(dst)} existiert, überspringe…")
        else:
            logger.info(f"→ Prüfe {zip_name}")
            res = subprocess.run(["curl","-sSf",url,"-o","tmp.zip"], capture_output=True)
            if res.returncode == 0:
                subprocess.run(["unzip","-p","tmp.zip"], stdout=open(dst,"wb"), check=True)
                os.remove("tmp.zip")
                logger.info(f"   ✅ gespeichert nach {dst}")
            else:
                logger.warning(f"   ❌ {zip_name} nicht gefunden")
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
    if not files:
        logger.warning("Keine Premium-Index-Dateien gefunden – basis wird NaN")
        return pd.Series(data=np.nan, index=idx, name="basis")

    expected = ["open_time","open","close"]
    frames = []
    for fn in files:
        logger.info(f"Lade Premium-Index-CSV {fn}")
        df = pd.read_csv(fn)
        cols = [c.lower() for c in df.columns]
        df.columns = [c.replace("opentime","open_time").replace("closetime","close") for c in cols]
        if not set(expected).issubset(df.columns):
            # falls ohne Header
            df = pd.read_csv(fn, header=None, names=expected + cols[len(expected):])
        check_columns(df, ["open_time","close"], fn)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
        frames.append(df.set_index("open_time")["close"])
    all_prem = pd.concat(frames).sort_index().drop_duplicates()
    return all_prem.reindex(idx, method="ffill")

def compute_features(fund_df: pd.DataFrame) -> pd.DataFrame:
    hourly = fund_df["fundingrate"].resample("1h").mean().ffill()
    out = pd.DataFrame({"fundingRate": hourly})

    # 8h Funding-Rate
    out["fundingRate_8h"] = out["fundingRate"].rolling(ROLL_HOURS).sum()

    # 7d SMA & Z-Score
    window = SMA_DAYS * 24
    out["sma7d"]  = out["fundingRate_8h"].rolling(window).mean()
    out["zscore"] = (
        (out["fundingRate_8h"] - out["sma7d"])
         / out["fundingRate_8h"].rolling(window).std()
    )

    # Flip-Flag + Stunden seit letztem Flip
    out["flip"] = np.sign(out["fundingRate_8h"]).diff().abs().fillna(0).astype(int)
    out["flip_cumsum"]      = out["flip"].cumsum()
    out["hours_since_flip"] = out.groupby("flip_cumsum").cumcount()
    out.drop(columns="flip_cumsum", inplace=True)

    # Flags und Imputation
    out["has_sma"]    = out["sma7d"].notna().astype(int)
    out["has_zscore"] = out["zscore"].notna().astype(int)
    out["sma7d"]      = out["sma7d"].fillna(0)
    out["zscore"]     = out["zscore"].fillna(0)

    # Platzhalter Basis
    out["basis"] = np.nan
    return out

def process_symbol(symbol: str, start_date: str = None, end_date: str = None):
    logger.info(f"=== Verarbeitung {symbol} ===")
    inception_str = SYMBOL_START.get(symbol)
    if not inception_str:
        raise ValueError(f"Inception für {symbol} fehlt.")
    inception_period = pd.Period(inception_str, "M")

    # Prepare output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Ermitteln, welche Monate geladen werden müssen
    if start_date and end_date:
        # voller historischer Lauf
        download_start = pd.Period(start_date, "M")
        download_end   = pd.Period(end_date,   "M")
        if download_start < inception_period:
            download_start = inception_period
    else:
        # inkrementeller Lauf
        out_pattern = os.path.join(OUTPUT_DIR, f"{symbol}-funding-features-*_to_*.parquet")
        existing_files = glob.glob(out_pattern)
        if existing_files:
            # lade letzten Timestamp
            latest = max(existing_files, key=lambda f: f)
            # parse Enddatum aus Dateiname
            end_str = os.path.basename(latest).split("_to_")[-1].replace(".parquet","")
            last_period = pd.Period(end_str, "M")
            download_start = last_period + 1
        else:
            download_start = inception_period
        # bis zum letzten abgeschlossenen Monat
        download_end = pd.Timestamp.now(tz="UTC").to_period("M") - 1

    if download_start > download_end:
        logger.info(f"ℹ️ {symbol}: Kein neuer Monat zum Download ({download_start} > {download_end}).")
    else:
        # 2) Nur wirklich fehlende CSVs ziehen
        download_and_unzip(symbol, "fundingRate",          download_start.strftime("%Y-%m"), download_end.strftime("%Y-%m"))
        download_and_unzip(symbol, "premiumIndexKlines",   download_start.strftime("%Y-%m"), download_end.strftime("%Y-%m"))

    # 3) Daten laden & Features berechnen
    df_fund = load_and_concat_funding(symbol)
    feats   = compute_features(df_fund)
    prem    = load_and_concat_premium(symbol, feats.index)
    feats["basis"] = prem

    # 4) Parquet dynamisch benennen wie beim BookDepth
    #    nach Datumsbeginn und -ende in feats.index
    real_start = feats.index.min().date().isoformat()
    real_end   = feats.index.max().date().isoformat()
    tmp_file   = os.path.join(OUTPUT_DIR, f"{symbol}-funding-features-temp.parquet")
    final_file = os.path.join(
        OUTPUT_DIR,
        f"{symbol}-funding-features-{real_start}_to_{real_end}.parquet"
    )

    feats.to_parquet(tmp_file, engine="pyarrow", compression="snappy")

    # alte Versionen löschen
    for old in glob.glob(os.path.join(OUTPUT_DIR, f"{symbol}-funding-features-*_to_*.parquet")):
        if old != final_file:
            os.remove(old)

    os.rename(tmp_file, final_file)
    logger.info(f"✅ {symbol}: geschrieben {len(feats)} Zeilen nach {final_file}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",     required=True)
    p.add_argument("--start-date", default=None, help="YYYY-MM, optional für historischen Lauf")
    p.add_argument("--end-date",   default=None, help="YYYY-MM, optional für historischen Lauf")
    args = p.parse_args()
    process_symbol(args.symbol, args.start_date, args.end_date)

if __name__ == "__main__":
    main()
