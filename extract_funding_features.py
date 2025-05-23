import os
import glob
import gzip
import re
import pandas as pd
import numpy as np
from utils import init_logger, save_parquet

logger = init_logger("funding_features")

# --- Verzeichnisse & Inception ---
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

# --- Feature-Parameter ---
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

def load_and_concat_funding(files: list[str]) -> pd.DataFrame:
    dfs = []
    for fn in files:
        logger.info(f"Lade Funding-CSV {fn}")
        opener = gzip.open if fn.endswith(".gz") else open
        with opener(fn, "rt") as f:
            df = pd.read_csv(f)
        df.columns = [c.lower() for c in df.columns]
        check_columns(df, ["calc_time","funding_interval_hours","last_funding_rate"], fn)
        df["fundingtime"] = pd.to_datetime(df["calc_time"], unit="ms")
        df["fundingrate"] = df["last_funding_rate"]
        dfs.append(df[["fundingtime","fundingrate"]])
    if not dfs:
        raise ValueError("Keine Funding-Dateien gefunden.")
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.sort_values("fundingtime").drop_duplicates(["fundingtime"])
    all_df.index = all_df["fundingtime"]
    return all_df[["fundingrate"]]

def load_and_concat_premium(symbol: str, idx: pd.DatetimeIndex) -> pd.Series:
    expected = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","count",
        "taker_buy_volume","taker_buy_quote_volume","ignore"
    ]
    files = list_monthly_files(symbol, "premiumIndexKlines")
    frames = []
    for fn in files:
        logger.info(f"Lade Premium-Index-CSV {fn}")
        # Versuch 1: mit Header einlesen
        df = pd.read_csv(fn)
        cols = [c.lower() for c in df.columns]
        # CamelCase → snake_case
        cols = [c.replace("opentime","open_time")
                  .replace("closetime","close_time") for c in cols]
        df.columns = cols
        if not set(expected).issubset(df.columns):
            # fehlender Header: manuell spaltennamen setzen
            df = pd.read_csv(fn, header=None, names=expected)
        check_columns(df, ["open_time","close"], fn)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        series = df.set_index("open_time")["close"]
        frames.append(series)
    if not frames:
        raise ValueError("Keine Premium-Index-Dateien gefunden.")
    all_prem = pd.concat(frames).sort_index().drop_duplicates()
    # auf funding-Stunden resamplen und ffill
    all_prem = all_prem.reindex(idx).ffill()
    return all_prem

def compute_features(fund_df: pd.DataFrame) -> pd.DataFrame:
    # 1H-Resample + ffill
    hourly = fund_df["fundingrate"].resample("1h").mean().ffill()
    out = pd.DataFrame({"fundingRate": hourly})

    # 1) 8h Funding-Rate rollierend
    out["fundingRate_8h"] = out["fundingRate"].rolling(ROLL_HOURS).sum()

    # 2) 7d SMA + Z-Score
    window = SMA_DAYS * 24
    out["sma7d"]   = out["fundingRate_8h"].rolling(window).mean()
    out["zscore"]  = (
        (out["fundingRate_8h"] - out["sma7d"])
        / out["fundingRate_8h"].rolling(window).std()
    )

    # 3) Flip-Flag (Vorzeichenwechsel)
    out["flip"] = np.sign(out["fundingRate_8h"]).diff().abs().fillna(0).astype(int)

    # 4) has_sma & has_zscore dynamisch prüfen
    out["has_sma"]    = out["sma7d"].notna().astype(int)
    out["has_zscore"] = out["zscore"].notna().astype(int)

    # 5) Imputation für ML: sma7d & zscore mit 0 füllen (keine inplace-Warnung)
    out["sma7d"]  = out["sma7d"].fillna(0)
    out["zscore"] = out["zscore"].fillna(0)

    # 6) hours_since_flip: Stunden seit letztem flip==1
    out["flip_cumsum"]      = out["flip"].cumsum()
    out["hours_since_flip"] = out.groupby("flip_cumsum").cumcount()
    out.drop(columns="flip_cumsum", inplace=True)

    # 7) placeholder für basis
    out["basis"] = np.nan

    return out

def process_symbol(symbol: str):
    logger.info(f"=== Verarbeitung {symbol} ===")
    start = SYMBOL_START.get(symbol)
    if not start:
        raise ValueError(f"Inception für {symbol} fehlt.")

    files_fund = [
        fn for fn in list_monthly_files(symbol, "fundingRate")
        if pd.Period(re.search(
              rf"{symbol}-fundingRate-(\d{{4}}-\d{{2}})", fn
           ).group(1),"M")
           >= pd.Period(start,"M")
    ]
    if not files_fund:
        logger.info("Keine Funding-Dateien gefunden – überspringe.")
        return

    df_fund = load_and_concat_funding(files_fund)
    feats   = compute_features(df_fund)

    prem = load_and_concat_premium(symbol, feats.index)
    feats["basis"] = prem

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/{symbol}-funding-features.parquet"
    if os.path.exists(out_path):
        existing = pd.read_parquet(out_path)
        merged   = pd.concat([existing, feats]).sort_index()
        merged   = merged[~merged.index.duplicated(keep="first")]
        save_parquet(merged, out_path)
        logger.info(f"Angehängt: {len(feats)} Zeilen")
    else:
        save_parquet(feats, out_path)
        logger.info(f"Initial: {len(feats)} Zeilen")

if __name__ == "__main__":
    symbol = os.getenv("SYMBOL")
    if not symbol:
        raise SystemExit("Bitte Umgebungsvariable SYMBOL setzen.")
    process_symbol(symbol)
