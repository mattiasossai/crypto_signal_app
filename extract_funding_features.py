import os, glob, gzip, re
import pandas as pd, numpy as np
from utils import init_logger, save_parquet

logger = init_logger("funding_features")

LOCAL_BASE  = "data/futures/um/monthly/fundingRate"
OUTPUT_DIR  = "features/funding"
SYMBOLS     = {
    "BTCUSDT": "2020-01",
    "ETHUSDT": "2020-01",
    "BNBUSDT": "2020-02",
    "XRPUSDT": "2020-01",
    "SOLUSDT": "2020-09",
    "ENAUSDT": "2024-04",
}
ROLL_HOURS = 8
SMA_DAYS   = 7

def list_files(symbol: str):
    # Holt alle CSV-Dateien für das Symbol
    return sorted(glob.glob(f"{LOCAL_BASE}/{symbol}/{symbol}-fundingRate-*.csv"))

def load_and_concat(files: list[str]) -> pd.DataFrame:
    dfs = []
    for fn in files:
        logger.info(f"Lade {fn}")
        opener = gzip.open if fn.endswith(".gz") else open
        with opener(fn, "rt") as f:
            df = pd.read_csv(f)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("fundingTime").drop_duplicates(["fundingTime"])
    return df.set_index("fundingTime")

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # Auf Stunden-Basis resamplen und ffill
    hourly = df["fundingRate"].resample("1H").mean().ffill()
    out = pd.DataFrame({"fundingRate": hourly})
    # 1) 8-Stunden Funding-Rate (rollierend)
    out["fundingRate_8h"] = out["fundingRate"].rolling(ROLL_HOURS).sum()
    # 2) 7-Tage SMA + Z-Score
    window = SMA_DAYS * 24
    out["sma7d"]   = out["fundingRate_8h"].rolling(window).mean()
    out["zscore"] = (
        (out["fundingRate_8h"] - out["sma7d"])
        / out["fundingRate_8h"].rolling(window).std()
    )
    # 3) Flip-Flag (Vorzeichenwechsel)
    out["flip"] = np.sign(out["fundingRate_8h"]).diff().abs().fillna(0).astype(int)
    # 4) Platzhalter für Spot-vs-Futures Basis
    out["basis"] = np.nan
    return out

def process_symbol(symbol: str):
    logger.info(f"=== Verarbeitung {symbol} ===")
    out_path = f"{OUTPUT_DIR}/{symbol}-funding-features.parquet"
    # Nur Dateien ab dem Startmonat
    files = [
        fn for fn in list_files(symbol)
        if pd.Period(re.search(rf"{symbol}-fundingRate-(\d{{4}}-\d{{2}})", fn).group(1), "M")
           >= pd.Period(SYMBOLS[symbol], "M")
    ]

    if os.path.exists(out_path):
        # inkrementell: nur neue Monate anhängen
        existing = pd.read_parquet(out_path)
        last_mon = existing.index.max().to_period("M")
        new_files = [
            fn for fn in files
            if pd.Period(re.search(rf"{symbol}-fundingRate-(\d{{4}}-\d{{2}})", fn).group(1), "M")
               > last_mon
        ]
        if not new_files:
            logger.info("Keine neuen Dateien – überspringe.")
            return
        df_new  = compute_features(load_and_concat(new_files))
        merged  = pd.concat([existing, df_new]).sort_index()
        merged  = merged[~merged.index.duplicated(keep="first")]
        save_parquet(merged, out_path)
        logger.info(f"An bestehendes Parquet angehängt: {len(df_new)} Zeilen")
    else:
        # initial: alle historischen Daten verarbeiten
        df_all = compute_features(load_and_concat(files))
        save_parquet(df_all, out_path)
        logger.info(f"Initiales Parquet geschrieben: {len(df_all)} Zeilen")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sym in SYMBOLS:
        process_symbol(sym)
