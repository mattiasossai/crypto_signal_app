#!/usr/bin/env python3
import argparse
import os
import glob
import shutil
import datetime
import subprocess
import logging

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression

# ——————————————— Logger konfigurieren ———————————————
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Inception-Daten pro Symbol
INCEPTION = {
    "BTCUSDT": "2023-01-01",
    "ETHUSDT": "2023-01-01",
    "BNBUSDT": "2023-01-01",
    "SOLUSDT": "2023-01-01",
    "XRPUSDT": "2023-01-06",
    "ENAUSDT": "2024-04-02",
}

def download_day(symbol: str, day, raw_dir: str) -> bool:
    """
    Lade die ZIP-Datei für einen Tag herunter und entpacke sie.
    Return True bei Erfolg, False bei 404 oder Fehler.
    """
    ds = day.strftime("%Y-%m-%d")
    zip_name = f"{symbol}-bookDepth-{ds}.zip"
    zip_path = os.path.join(raw_dir, zip_name)
    url = f"https://data.binance.vision/data/futures/um/daily/bookDepth/{symbol}/{zip_name}"
    logger.info(f"→ FETCH {symbol} {ds}: {url}")
    res = subprocess.run(["curl", "-sSf", url, "-o", zip_path], capture_output=True)
    if res.returncode == 0:
        subprocess.run(["unzip", "-q", "-o", zip_path, "-d", raw_dir], check=True)
        os.remove(zip_path)
        logger.info(f"{symbol} {ds}: ZIP erfolgreich geladen und entpackt")
        return True
    else:
        logger.warning(f"{symbol} {ds}: ZIP nicht gefunden (Status {res.returncode})")
        return False

def download_and_unzip(symbol: str, days, raw_dir: str):
    """
    Lade alle Tage herunter und entpacke sie.
    """
    os.makedirs(raw_dir, exist_ok=True)
    results = []
    for day in days:
        ok = download_day(symbol, day, raw_dir)
        results.append(ok)
    return results

def parse_csv_to_df(csv_fp: str, day: pd.Timestamp):
    """
    Robustes Einlesen eines BookDepth-CSV:
     - Headerful (timestamp,percentage,depth,notional)
     - fallback headerless
    Liefert DataFrame (Index=timestamp) und has_data-Flag.
    """
    EXPECTED = ["timestamp", "percentage", "depth", "notional"]
    path = pd.Path(csv_fp) if hasattr(pd, 'Path') else None

    if not os.path.exists(csv_fp):
        logger.warning(f"{csv_fp}: Datei fehlt, übersprungen")
        empty = pd.DataFrame([], columns=EXPECTED[1:], index=pd.DatetimeIndex([], tz="UTC"))
        return empty, False

    # 1) Versuch Headerful
    try:
        df = pd.read_csv(csv_fp, header=0)
    except Exception as e:
        logger.error(f"{csv_fp}: Fehler beim Einlesen headerful: {e}")
        empty = pd.DataFrame([], columns=EXPECTED[1:], index=pd.DatetimeIndex([], tz="UTC"))
        return empty, False

    lower = [c.lower() for c in df.columns]
    if set(EXPECTED).issubset(lower):
        logger.info(f"{os.path.basename(csv_fp)}: Headerful erkannt → {df.columns.tolist()}")
        # remap auf konsistente Namen
        rename_map = {
            df.columns[lower.index("timestamp")]: "timestamp",
            df.columns[lower.index("percentage")]: "percentage",
            df.columns[lower.index("depth")]:     "depth",
            df.columns[lower.index("notional")]: "notional",
        }
        df = df.rename(columns=rename_map)[EXPECTED]
    else:
        logger.warning(f"{os.path.basename(csv_fp)}: Header unvollständig, fallback headerless")
        df = pd.read_csv(csv_fp, header=None)
        if df.shape[1] < len(EXPECTED):
            logger.error(f"{os.path.basename(csv_fp)}: headerless erwartet ≥{len(EXPECTED)} Spalten, hat {df.shape[1]}")
            empty = pd.DataFrame([], columns=EXPECTED[1:], index=pd.DatetimeIndex([], tz="UTC"))
            return empty, False
        df = df.iloc[:, :len(EXPECTED)]
        df.columns = EXPECTED

    # Timestamp konvertieren
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms" if pd.api.types.is_numeric_dtype(df["timestamp"]) else None,
        utc=True,
        errors="coerce"
    )
    n_bad = df["timestamp"].isna().sum()
    if n_bad:
        logger.warning(f"{os.path.basename(csv_fp)}: {n_bad} ungültige timestamps (NaT)")

    df = df.set_index("timestamp").sort_index()

    # Tages-Slice
    next_day = day + pd.Timedelta(days=1)
    sl = df.loc[(df.index >= day) & (df.index < next_day)]
    logger.info(f"{os.path.basename(csv_fp)}: gelesen {len(df)} Zeilen, gesliced {len(sl)}")
    return sl, not sl.empty

def extract_raw_for_days(symbol: str, raw_dir: str, start: pd.Timestamp, end: pd.Timestamp):
    """
    Für alle Tage im Zeitraum Daten extrahieren und Features berechnen.
    """
    days = pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")
    rows = []
    for day in days:
        csv_fp = os.path.join(raw_dir, f"{symbol}-bookDepth-{day.strftime('%Y-%m-%d')}.csv")
        sl, has_data = parse_csv_to_df(csv_fp, day)

        # — Tages-Features wie bisher —
        tot_not = sl["notional"].sum()
        tot_dep = sl["depth"].sum()
        m1  = sl["percentage"].abs() <= 1.0
        m10 = sl["percentage"].abs() <= 10.0
        n1, d1   = sl.loc[m1, "notional"].sum(), sl.loc[m1, "depth"].sum()
        n10, d10 = sl.loc[m10, "notional"].sum(), sl.loc[m10, "depth"].sum()
        rel_n1 = n1/tot_not if tot_not else np.nan
        rel_d1 = d1/tot_dep if tot_dep else np.nan
        p_min = sl.loc[sl["percentage"]>0, "percentage"].min() or 0
        n_max = sl.loc[sl["percentage"]<0, "percentage"].max() or 0
        spread_pct = p_min + abs(n_max)
        bid_n = sl.loc[sl["percentage"]<0, "notional"].sum()
        ask_n = sl.loc[sl["percentage"]>0, "notional"].sum()
        imb_n = (bid_n-ask_n)/tot_not if tot_not else np.nan
        bid_d = sl.loc[sl["percentage"]<0, "depth"].sum()
        ask_d = sl.loc[sl["percentage"]>0, "depth"].sum()
        imb_d = (bid_d-ask_d)/tot_dep if tot_dep else np.nan

        n, d = sl["notional"], sl["depth"]
        moments = {
            "not_mean":     n.mean(),
            "not_var":      n.var(ddof=0),
            "not_skew":     skew(n, bias=False)  if len(n)>1 else np.nan,
            "not_kurt":     kurtosis(n, bias=False) if len(n)>1 else np.nan,
            "dep_mean":     d.mean(),
            "dep_var":      d.var(ddof=0),
            "dep_skew":     skew(d, bias=False)  if len(d)>1 else np.nan,
            "dep_kurt":     kurtosis(d, bias=False) if len(d)>1 else np.nan,
        }

        seg1 = sl.between_time("00:00","07:59")[["notional","depth"]].sum(min_count=1)
        seg2 = sl.between_time("08:00","15:59")[["notional","depth"]].sum(min_count=1)
        seg3 = sl.between_time("16:00","23:59")[["notional","depth"]].sum(min_count=1)

        rows.append({
            "date":               day,
            "file_exists":        has_data,
            "has_notional":       tot_not>0,
            "has_depth":          tot_dep>0,
            "total_notional":     tot_not,
            "total_depth":        tot_dep,
            "notional_1pct":      n1,
            "depth_1pct":         d1,
            "rel_notional_1pct":  rel_n1,
            "rel_depth_1pct":     rel_d1,
            "notional_10pct":     n10,
            "depth_10pct":        d10,
            "spread_pct":         spread_pct,
            "notional_imbalance": imb_n,
            "depth_imbalance":    imb_d,
            **moments,
            "notional_00_08":     seg1["notional"],
            "depth_00_08":        seg1["depth"],
            "notional_08_16":     seg2["notional"],
            "depth_08_16":        seg2["depth"],
            "notional_16_24":     seg3["notional"],
            "depth_16_24":        seg3["depth"],
            "has_00_08":          not pd.isna(seg1["notional"]),
            "has_08_16":          not pd.isna(seg2["notional"]),
            "has_16_24":          not pd.isna(seg3["notional"]),
            "has_data":           has_data,
        })

    df = pd.DataFrame(rows).set_index("date")
    df.index.name = "date"
    return df

def add_rolling_micro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnung rollierender Fenster und Microstructure Features.
    Ohne fillna(0) – stattdessen has_* Flags.
    """
    for w in (7, 14, 21):
        for base in ("notional_imbalance", "depth_imbalance"):
            col = f"{base}_roll_{w}d"
            roll = df[base].rolling(window=w, min_periods=w).mean()
            df[col]          = roll
            df[f"has_{col}"] = roll.notna().astype(int)

    # VPIN
    df["vpin"] = df.notional_imbalance.abs().rolling(window=50, min_periods=1).mean()

    # Amihud (min_periods = window)
    ai      = df.ret / df.total_notional.replace(0, np.nan)
    roll_ai = ai.rolling(window=30, min_periods=30).mean()
    df["amihud_roll_30d"]     = roll_ai
    df["has_amihud_roll_30d"] = roll_ai.notna().astype(int)
    
    kl = []
    for i in range(len(df)):
        if i < w:
            kl.append(0)
        else:
            sub = df.iloc[i-w+1:i+1]
            X = sub.total_notional.diff().values.reshape(-1,1)
            y = sub.mid_price.diff().abs().values
            m = (~np.isnan(X.flatten())) & (~np.isnan(y))
            kl.append(LinearRegression().fit(X[m],y[m]).coef_[0] if m.sum()>=2 else 0)
    df[f"kyle_lambda_roll_{w}d"]     = kl
    df[f"has_kyle_lambda_roll_{w}d"] = [i>=w-1 for i in range(len(df))]

    ai      = df.ret / df.total_notional.replace(0,np.nan)
    roll_ai = ai.rolling(window=w, min_periods=w).mean().fillna(0)
    df[f"amihud_roll_{w}d"]    = roll_ai
    df[f"has_amihud_roll_{w}d"] = roll_ai.notna()

    ls = []
    for i in range(len(df)):
        if i < w:
            ls.append(0)
        else:
            sub = df.iloc[i-w+1:i+1]
            X = sub.rel_depth_1pct.values.reshape(-1,1)
            y = sub.spread_pct.values
            m = (~np.isnan(X.flatten())) & (~np.isnan(y))
            ls.append(LinearRegression().fit(X[m],y[m]).coef_[0] if m.sum()>=2 else 0)
    df[f"liq_slope_roll_{w}d"]     = ls
    df[f"has_liq_slope_roll_{w}d"] = [i>=w-1 for i in range(len(df))]

    return df.drop(columns=["mid_price","ret"], errors="ignore")

def process_symbol(symbol: str, start_date: str, end_date: str):
    # a) parse dates
    if start_date and end_date:
        sd = pd.to_datetime(start_date).tz_localize("UTC")
        ed = pd.to_datetime(end_date).tz_localize("UTC")
        # ─── Neu: Inception capping auch für historical ───
        inception = pd.to_datetime(INCEPTION[symbol]).tz_localize("UTC")
        if sd < inception:
            print(f"{symbol}: historical start {sd.date()} vor Inception {inception.date()}, cappe um.")
            sd = inception
    else:
        # daily/resume logic wie gehabt
        y = datetime.datetime.utcnow().date() - datetime.timedelta(days=1)
        sd = None
        ed = pd.to_datetime(y).tz_localize("UTC")

    out_dir = f"features/bookDepth/{symbol}"
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, f"{symbol}-features-*.parquet")
    files = glob.glob(pattern)

    if sd is None:
        if files:
            latest = max(files, key=lambda f: pd.read_parquet(f).index.max())
            df_old = pd.read_parquet(latest)
            if "date" in df_old.columns:
                df_old["date"] = pd.to_datetime(df_old["date"], utc=True)
                df_old = df_old.set_index("date").sort_index()
            else:
                df_old.index = pd.to_datetime(df_old.index, utc=True)
                df_old = df_old.sort_index()
            sd = (df_old.index.max() + pd.Timedelta(days=1)).normalize()
            out_file = latest  # Nur temporär; später wird der Dateiname angepasst
        else:
            df_old = pd.DataFrame()
            sd = pd.to_datetime(INCEPTION[symbol]).tz_localize("UTC")
            out_file = None  # Noch kein Dateiname, wird nachher generiert
    else:
        df_old = pd.DataFrame()
        out_file = None

    if sd.tzinfo is None: sd = sd.tz_localize("UTC")
    if sd > ed:
        print(f"ℹ️ {symbol}: nichts zu tun ({sd.date()} > {ed.date()})")
        return

    days = pd.date_range(sd.normalize(), ed.normalize(), freq="D", tz="UTC")
    rawdir = f"raw/bookDepth/{symbol}"
    print(f"→ {symbol}: Downloading {len(days)} days…")
    download_and_unzip(symbol, days, rawdir)

    df_new = extract_raw_for_days(symbol, rawdir, sd, ed)
    shutil.rmtree(rawdir)

    # ✋ Wenn df_new gar keine echten Tage liefert, brechen wir ab
    if not df_new['file_exists'].any():
        print(f"ℹ️ {symbol}: Keine neuen BookDepth-Daten ab {sd.date()}, nichts zu tun.")
        return

    df_all = pd.concat([df_old, df_new]).sort_index()
    df_upd = add_rolling_micro(df_all)

    # 1. Schreibe temporär
    tmp_out_file = os.path.join(out_dir, f"{symbol}-features-temp.parquet")
    df_upd.to_parquet(tmp_out_file, compression="snappy")

    # 2. Lösche alle alten Parquets (außer temp) und sammle sie
    removed = []
    for old in glob.glob(os.path.join(out_dir, f"{symbol}-features-*_to_*.parquet")):
        if old != tmp_out_file:
            os.remove(old)
            removed.append(old)

    # 3. Benenne temporäres File in finalen Namen um
    real_sd = df_upd.index.min().date()
    real_ed = df_upd.index.max().date()
    final_out_file = os.path.join(
        out_dir,
        f"{symbol}-features-{real_sd}_to_{real_ed}.parquet"
    )
    os.rename(tmp_out_file, final_out_file)

    # 4. Git-Staging: füge neues File hinzu und entferne die alten
    import subprocess
    subprocess.run(["git", "add", final_out_file], check=True)
    for old in removed:
        subprocess.run(["git", "rm", "--quiet", old], check=True)

    print(f"✅ {symbol}: written {len(df_upd)} days to {final_out_file}")
    
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",      required=True)
    p.add_argument("--start-date",  default=None)
    p.add_argument("--end-date",    default=None)
    args = p.parse_args()
    process_symbol(args.symbol, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
