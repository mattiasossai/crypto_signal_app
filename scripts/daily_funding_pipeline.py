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
    Liest eine FundingRate- oder PremiumIndexKlines-CSV vollautomatisch ein,
    erkennt headerless-Dateien und alle gängigen Header-Varianten
    und liefert einen DataFrame mit:
      - Index: UTC-Datetime aus 'timestamp'
      - für kind='fundingRate': Spalte 'fundingRate'
      - für kind='premiumIndexKlines': Spalte 'close'
    """

    # 1) Probezeile einlesen (immer ohne Header), um headerless zu detektieren
    sample = pd.read_csv(path, nrows=1, header=None).iloc[0].tolist()
    headerless = all(str(x).replace('.', '', 1).lstrip('-').isdigit() for x in sample)

    # 2) ganze Datei einlesen: ohne Header, wenn headerless, sonst mit Header=0
    df = pd.read_csv(path, header=None if headerless else 0)

    # 3) Roh‐Spalten bei headerless sofort benennen
    if headerless:
        if kind == "fundingRate":
            # fundingRate-Dateien haben (meist) 3 Spalten
            names = ["calc_time", "funding_interval_hours", "last_funding_rate"]
        else:
            # premiumIndexKlines hat 12 Spalten
            names = [
                "open_time","open","high","low","close","volume",
                "close_time","quote_volume","count",
                "taker_buy_volume","taker_buy_quote_volume","ignore"
            ]
        # falls Binance mal Spalten weglässt oder anfügt, kürzen wir oder übernehmen alle
        df.columns = names[:df.shape[1]]

    # 4) Jetzt eine Norm‐Map aufbauen (bereinigt um Unterstriche, Groß-/Kleinschrift, Spaces)
    norm_map = {}
    for col in df.columns:
        key = re.sub(r"[\s_]", "", str(col).lower())
        norm_map[key] = col

    # 5) Spalten-Umbenennung für FundingRate
    if kind == "fundingRate":
        # Timestamp‐Spalte finden
        for cand in ("calctime","timestamp","fundingtime","funding_time"):
            if cand in norm_map:
                ts = norm_map[cand]
                break
        else:
            raise ValueError(f"{path}: Kein Timestamp-Feld gefunden (habe: {list(norm_map)})")

        # FundingRate‐Spalte finden
        for cand in ("lastfundingrate","fundingrate","funding_rate"):
            if cand in norm_map:
                fr = norm_map[cand]
                break
        else:
            raise ValueError(f"{path}: Kein FundingRate-Feld gefunden (habe: {list(norm_map)})")

        df = df.rename(columns={ts: "timestamp", fr: "fundingRate"})
        df = df[["timestamp", "fundingRate"]]

    # 6) Spalten-Umbenennung für PremiumIndexKlines
    else:
        # Timestamp‐Spalte finden
        for cand in ("opentime","open_time"):
            if cand in norm_map:
                ts = norm_map[cand]
                break
        else:
            raise ValueError(f"{path}: Kein open_time/opentime gefunden (habe: {list(norm_map)})")

        # Close‐Spalte finden
        if "close" in norm_map:
            cl = norm_map["close"]
        else:
            raise ValueError(f"{path}: Kein close-Feld gefunden (habe: {list(norm_map)})")

        df = df.rename(columns={ts: "timestamp", cl: "close"})
        df = df[["timestamp", "close"]]

    # 7) Timestamp → datetime UTC, als Index setzen
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
        errors="coerce"
    )
    df = df.set_index("timestamp").sort_index()

    # 8) Final-Check
    required = {"timestamp", "fundingRate"} if kind=="fundingRate" else {"timestamp","close"}
    missing = required - set(df.reset_index().columns)
    if missing:
        raise ValueError(f"{path}: Fehlende Spalten nach Umbenennung: {missing}")

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
