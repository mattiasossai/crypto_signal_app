#!/usr/bin/env python3
import os, glob, argparse, logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Symbol-spezifische Inception-Daten
INCEPTION = {
    "BTCUSDT": "2023-01-01",
    "ETHUSDT": "2023-01-01",
    "BNBUSDT": "2023-01-01",
    "SOLUSDT": "2023-01-01",
    "XRPUSDT": "2023-01-06",
    "ENAUSDT": "2024-04-02",
}

def fp_contains_header(fp: str) -> bool:
    with open(fp, "r") as f:
        return not f.readline().split(",")[0].isdigit()

def process_day(fp: str, sym: str) -> dict:
    """Liest eine Tages-CSV ein und gibt die aggregierten Features als Dict zurück."""
    date_str = os.path.basename(fp).split(f"{sym}-bookDepth-")[1].replace(".csv", "")
    df = pd.read_csv(
        fp,
        header=0 if fp_contains_header(fp) else None,
        names=["timestamp", "percentage", "depth", "notional"]
    )
    # Zeitstempel → Datetime
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)

    # 1) Totals
    total_notional = df["notional"].sum()
    total_depth    = df["depth"].sum()

    # 2) 1%-Bin
    mask1 = df["percentage"].abs() <= 1.0
    notional_1pct = df.loc[mask1, "notional"].sum()
    depth_1pct    = df.loc[mask1, "depth"].sum()

    # 3) 10%-Bin
    mask10 = df["percentage"].abs() <= 10.0
    notional_10pct = df.loc[mask10, "notional"].sum()
    depth_10pct    = df.loc[mask10, "depth"].sum()

    # 4) Spread (robust gegen leere Gruppen)
    pos = df.loc[df["percentage"] > 0, "percentage"]
    neg = df.loc[df["percentage"] < 0, "percentage"]
    pos_min = pos.min() if not pos.empty else 0
    neg_max = neg.max() if not neg.empty else 0
    spread_pct = pos_min + abs(neg_max)

    # 5) Imbalances
    bid_notional = df.loc[df["percentage"] < 0, "notional"].sum()
    ask_notional = df.loc[df["percentage"] > 0, "notional"].sum()
    notional_imbalance = (bid_notional - ask_notional) / total_notional if total_notional else 0

    bid_depth = df.loc[df["percentage"] < 0, "depth"].sum()
    ask_depth = df.loc[df["percentage"] > 0, "depth"].sum()
    depth_imbalance = (bid_depth - ask_depth) / total_depth if total_depth else 0

    return {
        "date": date_str,
        "total_notional": total_notional,
        "notional_1pct": notional_1pct,
        "depth_1pct": depth_1pct,
        "notional_10pct": notional_10pct,
        "depth_10pct": depth_10pct,
        "spread_pct": spread_pct,
        "notional_imbalance": notional_imbalance,
        "depth_imbalance": depth_imbalance
    }

def main(input_dir, output_file, start_date, end_date):
    sym = os.path.basename(input_dir)
    inc_date = pd.to_datetime(INCEPTION.get(sym, start_date))
    all_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))

    rows = []
    for fp in all_files:
        day = pd.to_datetime(os.path.basename(fp).split(f"{sym}-bookDepth-")[1][:10])
        # Filter auf Inception & Nutzer-Zeitraum
        if day < inc_date or day < pd.to_datetime(start_date) or day > pd.to_datetime(end_date):
            continue
        logging.info("Processing %s", fp)
        rows.append(process_day(fp, sym))

    if not rows:
        logging.error("⚠️ Keine Daten gefunden im Zeitraum; beende.")
        return

    # DataFrame zusammenbauen und speichern
    df_out = pd.DataFrame(rows).set_index("date").sort_index()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_out.to_parquet(output_file, compression="snappy")
    logging.info("✅ Gespeichert %s (%d Tage)", output_file, len(df_out))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--start-date",  required=True)
    p.add_argument("--end-date",    required=True)
    args = p.parse_args()
    main(args.input_dir, args.output_file, args.start_date, args.end_date)
