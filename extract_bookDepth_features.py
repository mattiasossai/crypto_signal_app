#!/usr/bin/env python3
import os, glob, argparse
import pandas as pd

# Symbol-spezifische Inception
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
        first = f.readline().split(",")[0]
    return not first.isdigit()

def extract_features(input_dir: str, start_date: str, end_date: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not files:
        raise SystemExit("⚠️ Keine .csv-Dateien im Input-Verzeichnis gefunden")

    sym = os.path.basename(input_dir)
    inc_date = pd.to_datetime(INCEPTION.get(sym, start_date))

    # Daten einlesen
    dfs = []
    for fp in files:
        day_str = os.path.basename(fp).replace(f"{sym}-bookDepth-", "").replace(".csv", "")
        day = pd.to_datetime(day_str)
        if day < inc_date:
            continue

        df = pd.read_csv(
            fp,
            header=0 if fp_contains_header(fp) else None,
            names=["timestamp","percentage","depth","notional"],
            dtype={"timestamp": int, "percentage": float, "depth": float, "notional": float},
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        dfs.append(df)

    # zusammenführen & trimmen
    full = pd.concat(dfs).sort_index()
    sd = pd.to_datetime(start_date).tz_localize("UTC")
    ed = pd.to_datetime(end_date).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    full = full[sd:ed]

    # Feature-Berechnung
    total_notional = full["notional"].sum()
    total_depth    = full["depth"].sum()

    mask1  = full["percentage"].abs() <= 1.0
    mask10 = full["percentage"].abs() <= 10.0

    feats = {
        "total_notional":      total_notional,
        "notional_1pct":       full.loc[mask1, "notional"].sum(),
        "depth_1pct":          full.loc[mask1, "depth"].sum(),
        "notional_10pct":      full.loc[mask10, "notional"].sum(),
        "depth_10pct":         full.loc[mask10, "depth"].sum(),
        "spread_pct":          full.loc[full["percentage"]>0, "percentage"].min() \
                                + abs(full.loc[full["percentage"]<0, "percentage"].max()),
        "notional_imbalance":  (full.loc[full["percentage"]<0, "notional"].sum()
                                - full.loc[full["percentage"]>0, "notional"].sum())
                               / (total_notional or 1),
        "depth_imbalance":     (full.loc[full["percentage"]<0, "depth"].sum()
                               - full.loc[full["percentage"]>0, "depth"].sum())
                               / (total_depth or 1),
    }

    # Index auf Window-Start in ms seit Epoch
    idx = int(sd.value // 10**6)
    df_out = pd.DataFrame([feats], index=[idx])
    df_out.index.name = "timestamp"
    return df_out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--start-date",  required=True)
    p.add_argument("--end-date",    required=True)
    args = p.parse_args()

    out = extract_features(args.input_dir, args.start_date, args.end_date)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    out.to_parquet(args.output_file, compression="snappy")
    print(f"✅ Wrote features to {args.output_file}")

if __name__ == "__main__":
    main()
