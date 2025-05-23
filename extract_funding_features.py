import os
import glob
import gzip
import re
import pandas as pd
import numpy as np
from utils import init_logger, save_parquet

logger = init_logger("funding_features")

# --- CONFIG ---
S3_PATH    = "s3://data.binance.vision/data/futures/um/monthly/fundingRate"
LOCAL_DIR  = "data/futures/um/monthly/fundingRate"
OUTPUT_DIR = "features/funding"
SYMBOLS = {
    "BTCUSDT": "2020-01",
    "ETHUSDT": "2020-01",
    "BNBUSDT": "2020-02",
    "XRPUSDT": "2020-01",
    "SOLUSDT": "2020-09",
    "ENAUSDT": "2024-04",
}
ROLL_HOURS = 8
SMA_DAYS   = 7

def sync_s3():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    os.system(f"aws s3 sync {S3_PATH} {LOCAL_DIR}")

def list_files(symbol: str):
    pattern = os.path.join(LOCAL_DIR, f"{symbol}-fundingRate-*.csv*")
    return sorted(glob.glob(pattern))

def parse_period(fn: str, symbol: str) -> pd.Period:
    m = re.search(rf"{symbol}-fundingRate-(\d{{4}}-\d{{2}})", fn)
    return pd.Period(m.group(1), freq="M") if m else None

def load_and_concat(files: list[str]) -> pd.DataFrame:
    dfs = []
    for fn in files:
        logger.info(f"Lade {fn}")
        open_fn = gzip.open if fn.endswith(".gz") else open
        with open_fn(fn, "rt") as f:
            df = pd.read_csv(f)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("fundingTime").drop_duplicates(["fundingTime"])
    df.set_index("fundingTime", inplace=True)
    return df

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    hourly = df["fundingRate"].resample("1H").mean().ffill()
    feats = pd.DataFrame({"fundingRate": hourly})
    feats["fundingRate_8h"] = feats["fundingRate"].rolling(window=ROLL_HOURS).sum()

    window = SMA_DAYS * 24
    feats["sma7d"] = feats["fundingRate_8h"].rolling(window=window).mean()
    feats["zscore"] = (
        (feats["fundingRate_8h"] - feats["sma7d"])
        / feats["fundingRate_8h"].rolling(window=window).std()
    )
    feats["flip"] = np.sign(feats["fundingRate_8h"]).diff().fillna(0).abs().astype(int)
    feats["basis"] = np.nan  # hier später Spot-vs-Futures-Basis ergänzen

    return feats

def process_symbol(symbol: str):
    logger.info(f"=== Verarbeitung {symbol} ===")
    outpath = os.path.join(OUTPUT_DIR, f"{symbol}-funding-features.parquet")
    files = [f for f in list_files(symbol)
             if parse_period(f, symbol) >= pd.Period(SYMBOLS[symbol], freq="M")]

    if os.path.exists(outpath):
        existing = pd.read_parquet(outpath)
        last = existing.index.max().to_period("M")
        new_files = [f for f in files if parse_period(f, symbol) > last]
        if not new_files:
            logger.info("Keine neuen Daten – überspringe.")
            return
        raw_new = load_and_concat(new_files)
        feats_new = compute_features(raw_new)
        merged = pd.concat([existing, feats_new])
        merged = merged[~merged.index.duplicated(keep="first")].sort_index()
        save_parquet(merged, outpath)
        logger.info(f"Angehängt: {len(feats_new)} Zeilen")
    else:
        raw   = load_and_concat(files)
        feats = compute_features(raw)
        save_parquet(feats, outpath)
        logger.info(f"Initiales Parquet geschrieben ({len(feats)} Zeilen)")

def main():
    sync_s3()
    for sym in SYMBOLS:
        process_symbol(sym)

if __name__ == "__main__":
    main()
