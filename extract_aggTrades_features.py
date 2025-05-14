#!/usr/bin/env python3
import os
import glob
import argparse
import logging
import pandas as pd

# ─── Logging Setup ───────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ─── 1) Helfer: CSV einlesen mit oder ohne Header ────────────────────────────────
def read_agg_csv(path: str) -> pd.DataFrame:
    """
    Liest eine aggTrades-CSV-Datei ein.
    Erkennt automatisch, ob Header in der ersten Zeile sind oder nicht.
    Erwartete Spalten (in dieser Reihenfolge):
      0 agg_trade_id
      1 price
      2 quantity
      3 first_trade_id
      4 last_trade_id
      5 transact_time
      6 is_buyer_maker
    Wir brauchen nur columns 2,5,6.
    """
    names = [
        "agg_trade_id", "price", "quantity",
        "first_trade_id", "last_trade_id",
        "transact_time", "is_buyer_maker"
    ]
    try:
        # Versuch: mit Header
        df = pd.read_csv(path, usecols=["quantity","transact_time","is_buyer_maker"])
        logging.debug(f"→ {os.path.basename(path)}: mit Header eingelesen")
    except ValueError:
        # Fallback: ohne Header, selbst benennen
        df = pd.read_csv(
            path, header=None, names=names,
            usecols=["quantity","transact_time","is_buyer_maker"],
        )
        logging.debug(f"→ {os.path.basename(path)}: ohne Header eingelesen")
    return df

# ─── 2) Feature-Extraktion ───────────────────────────────────────────────────────
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aus den Trades in df (index timestamp) die täglichen Kern-Features berechnen.
    """
    # Menge & Flag korrekt typisieren
    df["quantity"] = df["quantity"].astype(float)
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)

    # 1D-Resample-Grundgerüst
    daily = pd.DataFrame(index=df.index.normalize().unique()).sort_index()
    daily.index.name = "date"

    # Volumen-Summen
    grouped = df.groupby(df.index.normalize())
    daily["total_volume"] = grouped["quantity"].sum()
    daily["buy_volume"]   = grouped.apply(lambda g: g.loc[~g["is_buyer_maker"], "quantity"].sum())
    daily["sell_volume"]  = grouped.apply(lambda g: g.loc[g["is_buyer_maker"],  "quantity"].sum())

    # Imbalance (geclippt)
    imbalance = (daily["buy_volume"] - daily["sell_volume"]) / daily["total_volume"]
    daily["imbalance"] = imbalance.clip(-0.9, 0.9).fillna(0.0)

    # Max Volumen pro 1h / 4h Bucket
    vol1h = df["quantity"].resample("1H").sum()
    daily["max_vol_1h"] = vol1h.resample("1D").max().fillna(0.0)
    vol4h = df["quantity"].resample("4H").sum()
    daily["max_vol_4h"] = vol4h.resample("1D").max().fillna(0.0)

    # Median Trades per Minute
    trades_per_min = df["quantity"].resample("1T").count()
    daily["avg_trades_per_min"] = trades_per_min.resample("1D").median().fillna(0.0)

    return daily.reset_index()

# ─── 3) Main ────────────────────────────────────────────────────────────────────
def main(input_dir: str, output_file: str, start_date: str, end_date: str):
    # --- Input CSVs sammeln
    pattern = os.path.join(input_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        logging.error(f"Kein CSV in {input_dir}")
        return

    # --- Alle CSVs einlesen & zusammenhängen
    df_list = []
    for f in files:
        try:
            tmp = read_agg_csv(f)
        except Exception as e:
            logging.error(f"Error reading {f}: {e}")
            continue
        # Zeitstempel als DatetimeIndex
        tmp["timestamp"] = pd.to_datetime(tmp["transact_time"], unit="ms", utc=True)
        tmp = tmp.set_index("timestamp")
        df_list.append(tmp)

    if not df_list:
        logging.error("No valid data after reading; exit.")
        return

    df = pd.concat(df_list).sort_index()
    logging.info(f"Combined DataFrame: {df.shape}")

    # --- Nach globalem Zeitfenster filtern (inkl. Overlap: einen Tag früher)
    sd = pd.to_datetime(start_date).tz_localize("UTC") - pd.Timedelta(days=1)
    ed = pd.to_datetime(end_date).tz_localize("UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    df = df.loc[(df.index >= sd) & (df.index <= ed)]
    logging.info(f"After filtering between {sd.date()} and {ed.date()}: {df.shape}")

    # --- Feature-Extraktion auf Tagesbasis
    daily = extract_features(df)

    # --- Drop Overlap-Tag (alle Daten < start_date weglöschen)
    daily = daily[daily["date"] >= pd.to_datetime(start_date)]
    if daily.empty:
        logging.warning("No trades in the true date range; output will be empty.")

    # --- In Parquet speichern
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    daily.to_parquet(output_file, index=False, compression="snappy")
    logging.info(f"Wrote features to {output_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Extract daily aggTrades features from Binance aggTrades CSVs"
    )
    p.add_argument("--input-dir",   required=True, help="Folder with aggTrades CSVs")
    p.add_argument("--output-file", required=True, help="Target Parquet file")
    p.add_argument("--start-date",  required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date",    required=True, help="YYYY-MM-DD")
    args = p.parse_args()

    main(args.input_dir, args.output_file, args.start_date, args.end_date)
