#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd

# Symbol-specific inception dates
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
    # 1) Alle CSVs einlesen
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    sym = os.path.basename(input_dir)
    inc_date = pd.to_datetime(INCEPTION.get(sym, start_date))

    dfs = []
    for fp in files:
        day_str = os.path.basename(fp).replace(f"{sym}-bookDepth-", "").replace(".csv", "")
        day = pd.to_datetime(day_str, format="%Y-%m-%d")
        if day < inc_date:
            continue
        df = pd.read_csv(
            fp,
            header=0 if fp_contains_header(fp) else None,
            names=["timestamp","percentage","depth","notional"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        dfs.append(df)

    if not dfs:
        raise SystemExit("⚠️ Keine Daten gefunden")

    full = pd.concat(dfs).sort_index()

    # 2) Window zuschneiden
    sd = pd.to_datetime(start_date).tz_localize("UTC")
    ed = (pd.to_datetime(end_date).tz_localize("UTC")
          + pd.Timedelta(days=1)
          - pd.Timedelta(milliseconds=1))
    full = full[sd:ed]

    # 3) Tägliche Aggregation und Lücken mit 0 auffüllen
    all_days = pd.date_range(sd.normalize(), ed.normalize(), freq="D", tz="UTC")
    daily = full.resample("D").agg({
        "notional": "sum",
        "depth":    "sum"
    }).reindex(all_days, fill_value=0)

    # 4) 1%-Bin und 10%-Bin pro Tag
    mask1  = full["percentage"].abs() <= 1.0
    mask10 = full["percentage"].abs() <= 10.0
    daily_1pct  = full[mask1].resample("D").agg({"notional":"sum","depth":"sum"}).reindex(all_days, fill_value=0)
    daily_10pct = full[mask10].resample("D").agg({"notional":"sum","depth":"sum"}).reindex(all_days, fill_value=0)

    # 5) Spread pro Tag
    daily_spread = full.resample("D").apply(
        lambda df: (df.loc[df["percentage"]>0,"percentage"].min(fill_value=0)
                    + abs(df.loc[df["percentage"]<0,"percentage"].max(fill_value=0)))
    ).reindex(all_days, fill_value=0)

    # 6) Feature-Berechnung als Tagesmittel (stabil gegenüber Lücken)
    expected_days = len(all_days)
    feats = {
        # Summen geteilt durch erwartete Tage → tägliches Mittel
        "avg_notional_per_day":      daily["notional"].sum()  / expected_days,
        "avg_depth_per_day":         daily["depth"].sum()     / expected_days,
        "avg_notional_1pct_day":     daily_1pct["notional"].sum() / expected_days,
        "avg_depth_1pct_day":        daily_1pct["depth"].sum()    / expected_days,
        "avg_notional_10pct_day":    daily_10pct["notional"].sum()/ expected_days,
        "avg_depth_10pct_day":       daily_10pct["depth"].sum()   / expected_days,
        "avg_spread_pct_per_day":    daily_spread.sum()          / expected_days,
        # Imbalance über das gesamte Fenster (kann weiterhin absolute Ratio sein)
        "notional_imbalance": (
            full.loc[full["percentage"]<0, "notional"].sum()
          - full.loc[full["percentage"]>0, "notional"].sum()
        ) / full["notional"].sum() if full["notional"].sum()>0 else 0,
        "depth_imbalance": (
            full.loc[full["percentage"]<0, "depth"].sum()
          - full.loc[full["percentage"]>0, "depth"].sum()
        ) / full["depth"].sum()     if full["depth"].sum()>0    else 0,
    }

    df_out = pd.DataFrame([feats], index=[int(sd.value // 10**6)])
    df_out.index.name = "timestamp"
    return df_out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--start-date",  required=True)
    parser.add_argument("--end-date",    required=True)
    args = parser.parse_args()

    out = extract_features(args.input_dir, args.start_date, args.end_date)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    out.to_parquet(args.output_file, compression="snappy")
    print(f"✅ Wrote features to {args.output_file}")

if __name__ == "__main__":
    main()
