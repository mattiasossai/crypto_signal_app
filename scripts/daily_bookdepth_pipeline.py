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

    if not os.path.exists(csv_fp):
        logger.warning(f"{os.path.basename(csv_fp)}: Datei fehlt, übersprungen")
        empty = pd.DataFrame([], columns=EXPECTED[1:], index=pd.DatetimeIndex([], tz="UTC"))
        return empty, False

    # 1) Versuch Headerful
    try:
        df = pd.read_csv(csv_fp, header=0)
    except Exception as e:
        logger.error(f"{os.path.basename(csv_fp)}: Fehler beim Einlesen headerful: {e}")
        empty = pd.DataFrame([], columns=EXPECTED[1:], index=pd.DatetimeIndex([], tz="UTC"))
        return empty, False

    lower = [c.lower() for c in df.columns]
    if set(EXPECTED).issubset(lower):
        logger.info(f"{os.path.basename(csv_fp)}: Headerful erkannt → {df.columns.tolist()}")
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

        # ─── Neuer Block beginnt hier (ersetzt den alten Abschnitt bis rows.append) ───

        # 1) Totale Notional- und Depth-Summen
        tot_not = sl["notional"].sum()
        tot_dep = sl["depth"].sum()

        # 2) Relativ‐Anteile für ±1 % (wie bisher)
        m1  = sl["percentage"].abs() <= 1.0
        n1, d1 = sl.loc[m1, "notional"].sum(), sl.loc[m1, "depth"].sum()
        rel_n1 = n1 / tot_not if tot_not else np.nan
        rel_d1 = d1 / tot_dep if tot_dep else np.nan

        # ─── Neu: Relativ‐Anteile für ±2 %, ±3 %, ±4 %, ±5 % ───
        n2  = sl.loc[sl["percentage"].abs() <= 2.0, "notional"].sum()
        d2  = sl.loc[sl["percentage"].abs() <= 2.0, "depth"].sum()
        rel_n2 = n2 / tot_not if tot_not else np.nan
        rel_d2 = d2 / tot_dep if tot_dep else np.nan

        n3  = sl.loc[sl["percentage"].abs() <= 3.0, "notional"].sum()
        d3  = sl.loc[sl["percentage"].abs() <= 3.0, "depth"].sum()
        rel_n3 = n3 / tot_not if tot_not else np.nan
        rel_d3 = d3 / tot_dep if tot_dep else np.nan

        n4  = sl.loc[sl["percentage"].abs() <= 4.0, "notional"].sum()
        d4  = sl.loc[sl["percentage"].abs() <= 4.0, "depth"].sum()
        rel_n4 = n4 / tot_not if tot_not else np.nan
        rel_d4 = d4 / tot_dep if tot_dep else np.nan

        n5  = sl.loc[sl["percentage"].abs() <= 5.0, "notional"].sum()
        d5  = sl.loc[sl["percentage"].abs() <= 5.0, "depth"].sum()
        rel_n5 = n5 / tot_not if tot_not else np.nan
        rel_d5 = d5 / tot_dep if tot_dep else np.nan

        # 3) Engste Ask/Bid für spread_pct (wie bisher)
        pos = sl.loc[sl["percentage"] > 0, "percentage"].sort_values()
        neg = sl.loc[sl["percentage"] < 0, "percentage"].sort_values()
        if len(pos) >= 1 and len(neg) >= 1:
            p_ask = pos.iloc[0]
            p_bid = neg.iloc[-1]
            spread_pct = p_ask + abs(p_bid)
        else:
            spread_pct = np.nan

        # 4) Absolute USD‐Spread aus Mid‐Approximation (spread_abs) ─ neu
        if not np.isnan(spread_pct):
            mid_price = tot_not / tot_dep if tot_dep else np.nan
            ask_px    = mid_price * (1 + p_ask/100)
            bid_px    = mid_price * (1 + p_bid/100)
            spread_abs = ask_px - bid_px
        else:
            spread_abs = np.nan

        # 5) VWAP‐Spread (spread_vwap) über alle Ebenen ±1…±5 % ─ neu
        if not np.isnan(spread_pct):
            mid_price = tot_not / tot_dep if tot_dep else np.nan
            sl2 = sl.copy()
            sl2["price"] = mid_price * (1 + sl2["percentage"]/100)

            bids = sl2[sl2["percentage"] < 0]
            asks = sl2[sl2["percentage"] > 0]
            if (not bids.empty) and (not asks.empty):
                vwap_bid = (bids["price"]  * bids["notional"]).sum() / bids["notional"].sum()
                vwap_ask = (asks["price"] * asks["notional"]).sum() / asks["notional"].sum()
                spread_vwap = vwap_ask - vwap_bid
            else:
                spread_vwap = np.nan
        else:
            spread_vwap = np.nan

        # 6) Bid/Ask‐Imbalance (wie bisher)
        bid_n = sl.loc[sl["percentage"] < 0, "notional"].sum()
        ask_n = sl.loc[sl["percentage"] > 0, "notional"].sum()
        imb_n = (bid_n - ask_n) / tot_not if tot_not else np.nan

        bid_d = sl.loc[sl["percentage"] < 0, "depth"].sum()
        ask_d = sl.loc[sl["percentage"] > 0, "depth"].sum()
        imb_d = (bid_d - ask_d) / tot_dep if tot_dep else np.nan

        # 7) Momente (unverändert)
        n, d = sl["notional"], sl["depth"]
        moments = {
            "not_mean":   n.mean(),
            "not_var":    n.var(ddof=0),
            "not_skew":   skew(n, bias=False)  if len(n) > 1 else np.nan,
            "not_kurt":   kurtosis(n, bias=False) if len(n) > 1 else np.nan,
            "dep_mean":   d.mean(),
            "dep_var":    d.var(ddof=0),
            "dep_skew":   skew(d, bias=False)  if len(d) > 1 else np.nan,
            "dep_kurt":   kurtosis(d, bias=False) if len(d) > 1 else np.nan,
        }

        # 8) Time‐Slice‐Segmente (unverändert)
        seg1 = sl.between_time("00:00","07:59")[["notional","depth"]].sum(min_count=1)
        seg2 = sl.between_time("08:00","15:59")[["notional","depth"]].sum(min_count=1)
        seg3 = sl.between_time("16:00","23:59")[["notional","depth"]].sum(min_count=1)

        # 9) rows.append mit allen neuen und bisherigen Feldern
        rows.append({
            "date":               day,
            "file_exists":        has_data,
            "has_notional":       tot_not > 0,
            "has_depth":          tot_dep > 0,
            "total_notional":     tot_not,
            "total_depth":        tot_dep,
            # Duplication‐Flag (wird weiter unten endgültig gesetzt)
            "dup_flag":           None,

            # → rel_notional_Xpct / rel_depth_Xpct (X=1…5)
            "notional_1pct":      n1,
            "depth_1pct":         d1,
            "rel_notional_1pct":  rel_n1,
            "rel_depth_1pct":     rel_d1,

            "notional_2pct":      n2,   # neu
            "depth_2pct":         d2,   # neu
            "rel_notional_2pct":  rel_n2,# neu
            "rel_depth_2pct":     rel_d2,# neu

            "notional_3pct":      n3,   # neu
            "depth_3pct":         d3,   # neu
            "rel_notional_3pct":  rel_n3,# neu
            "rel_depth_3pct":     rel_d3,# neu

            "notional_4pct":      n4,   # neu
            "depth_4pct":         d4,   # neu
            "rel_notional_4pct":  rel_n4,# neu
            "rel_depth_4pct":     rel_d4,# neu

            "notional_5pct":      n5,   # neu
            "depth_5pct":         d5,   # neu
            "rel_notional_5pct":  rel_n5,# neu
            "rel_depth_5pct":     rel_d5,# neu

            # → Spread‐Features
            "spread_pct":         spread_pct,   # bleibt (z.B. 2.0)
            "spread_abs":         spread_abs,   # neu: abs. USD‐Spread
            "spread_vwap":        spread_vwap,  # neu: VWAP‐Spread

            # → Imbalance (wie bisher)
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

        # ─── Ende des neuen Blocks ───

    df = pd.DataFrame(rows).set_index("date")
    df.index.name = "date"

    # ─── Duplication-Flag endgültig setzen (unverändert) ───
    prev_not = None
    prev_dep = None
    df["dup_flag"] = 0
    prev_not = None
    prev_dep = None
    for idx in df.index:
        cur_not = df.at[idx, "total_notional"]
        cur_dep = df.at[idx, "total_depth"]

        # Wenn wir einen Vortag haben, prüfen wir auf Duplikat
        if (prev_not is not None) and (prev_dep is not None):
            if (cur_not == prev_not) and (cur_dep == prev_dep):
                df.at[idx, "dup_flag"] = 1

        # Ganz wichtig: prev_not und prev_dep hier AUßERHALB des if setzen,
        # damit sie ab dem 2. Schleifendurchlauf nicht mehr None sind.
        prev_not = cur_not
        prev_dep = cur_dep

    return df


def add_rolling_micro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnung rollierender Fenster und Microstructure Features.
    Ohne fillna(0) – stattdessen has_* Flags und ML‐freundliches Lückenhandling.
    """
    # ── Rollende Imbalance-Fenster (wie vorher) ──
    def mean_if_real_change(x):
        changes = x.diff().abs() > 0
        if changes.sum() >= 1:
            return x.mean()
        return np.nan

    for w in (7, 14, 21):
        for base in ("notional_imbalance", "depth_imbalance"):
            col = f"{base}_roll_{w}d"
            df[col] = (
                df[base]
                  .rolling(window=w, min_periods=w)
                  .apply(mean_if_real_change, raw=False)
            )
            df[f"has_{col}"] = df[col].notna().astype(int)

    # VPIN (mindestens 1 Wert)
    df["vpin"] = df.notional_imbalance.abs().rolling(window=50, min_periods=1).mean()

    # MidPrice + Return (ret) für Kyle λ und Amihud
    df["mid_price"] = (df.total_notional / df.total_depth).replace([np.inf, -np.inf], np.nan)
    df["ret"]       = df.mid_price.pct_change(fill_method=None).abs()

    # Kyle Lambda
    w = 30
    kl = []
    for i in range(len(df)):
        if i < w:
            kl.append(np.nan)
        else:
            sub = df.iloc[i-w+1:i+1]
            X = sub.total_notional.diff().values.reshape(-1,1)
            y = sub.mid_price.diff().abs().values
            m = (~np.isnan(X.flatten())) & (~np.isnan(y))
            coef = LinearRegression().fit(X[m], y[m]).coef_[0] if m.sum() >= 2 else np.nan
            kl.append(coef)
    df[f"kyle_lambda_roll_{w}d"]     = kl
    df[f"has_kyle_lambda_roll_{w}d"] = pd.Series(kl).notna().astype(int).values

    # Amihud (ohne Fillna)
    ai      = df.ret / df.total_notional.replace(0, np.nan)
    roll_ai = ai.rolling(window=w, min_periods=w).mean()
    df[f"amihud_roll_{w}d"]     = roll_ai
    df[f"has_amihud_roll_{w}d"] = roll_ai.notna().astype(int)

    # ── Liquidity Slopes über 30 Tage für rel_depth_1pct … rel_depth_5pct ──
    window = 30
    # Lege leere Listen an für alle 5 Level
    slopes = {f"rel{level}": [] for level in (1,2,3,4,5)}

    # Iteriere durch jeden Datumseintrag
    for i in range(len(df)):
        if i < window:
            # Weniger als 30 Tage vorhanden → immer NaN
            for level in (1,2,3,4,5):
                slopes[f"rel{level}"].append(np.nan)
        else:
            sub = df.iloc[i-window+1 : i+1]
            # Für jedes Level X führen wir Regression: spread_vwap  ~  rel_depth_Xpct
            for level in (1,2,3,4,5):
                col_in = f"rel_depth_{level}pct"
                y_vals = sub["spread_vwap"].values
                X_vals = sub[col_in].values.reshape(-1, 1)
                # Filtere nur gültige Paare
                mask = (~np.isnan(X_vals.flatten())) & (~np.isnan(y_vals))
                if mask.sum() >= 2:
                    coef = LinearRegression().fit(X_vals[mask], y_vals[mask]).coef_[0]
                else:
                    coef = np.nan
                slopes[f"rel{level}"].append(coef)

    # Weise die berechneten Slope‐Listen dem DataFrame zu
    for level in (1,2,3,4,5):
        col_out = f"liq_slope_roll_{window}d_rel{level}pct"
        flag_out = f"has_liq_slope_roll_{window}d_rel{level}pct"
        df[col_out] = slopes[f"rel{level}"]
        df[flag_out] = df[col_out].notna().astype(int)

    # Entferne nur MidPrice (Ret bleibt erhalten, falls gewünscht)
    return df.drop(columns=["mid_price"], errors="ignore")
    
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
