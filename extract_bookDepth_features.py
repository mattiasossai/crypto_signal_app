#!/usr/bin/env python3
import os, glob, argparse
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
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    dfs = []
    sym = os.path.basename(input_dir)
    inc_date = pd.to_datetime(INCEPTION.get(sym, start_date))

    for fp in files:
        day_str = os.path.basename(fp).replace(".csv", "")
        day = pd.to_datetime(day_str)
        if day < inc_date:
            continue

        df = pd.read_csv(
            fp,
            header=0 if fp_contains_header(fp) else None,
            names=["timestamp","percentage","depth","notional"]
        )
        # timestamp in ms → datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        dfs.append(df)

    if not dfs:
        raise SystemExit("⚠️ Keine Daten gefunden")

    full = pd.concat(dfs).sort_index()

    # schneide auf window zu
    sd = pd.to_datetime(start_date).tz_localize("UTC")
    ed = (pd.to_datetime(end_date).tz_localize("UTC")
          + pd.Timedelta(days=1)
          - pd.Timedelta(milliseconds=1))
    full = full[sd:ed]

    # 1) globale Summen
    total_notional = full["notional"].sum()
    total_depth    = full["depth"].sum()

    # 2) 1%-Bin: percentage ≤ +1% und ≥ -1%
    mask1 = full["percentage"].abs() <= 1.0
    notional_1pct = full.loc[mask1, "notional"].sum()
    depth_1pct    = full.loc[mask1, "depth"].sum()

    # 3) 10%-Bin
    mask10 = full["percentage"].abs() <= 10.0
    notional_10pct = full.loc[mask10, "notional"].sum()
    depth_10pct    = full.loc[mask10, "depth"].sum()

    # 4) Spread in %-Punkten: ask_min_pos + |bid_max_neg|
    pos_min = full.loc[full["percentage"]>0, "percentage"].min()
    neg_max = full.loc[full["percentage"]<0, "percentage"].max()
    spread_pct = pos_min + abs(neg_max)

    # 5) Depth- & Notional-Imbalance
    bid_notional = full.loc[full["percentage"]<0, "notional"].sum()
    ask_notional = full.loc[full["percentage"]>0, "notional"].sum()
    notional_imbalance = (bid_notional - ask_notional) / total_notional

    bid_depth = full.loc[full["percentage"]<0, "depth"].sum()
    ask_depth = full.loc[full["percentage"]>0, "depth"].sum()
    depth_imbalance = (bid_depth - ask_depth) / total_depth

    # Baue den Output-DataFrame
    feats = {
        "total_notional":      total_notional,
        "notional_1pct":       notional_1pct,
        "depth_1pct":          depth_1pct,
        "notional_10pct":      notional_10pct,
        "depth_10pct":         depth_10pct,
        "spread_pct":          spread_pct,
        "notional_imbalance":  notional_imbalance,
        "depth_imbalance":     depth_imbalance
    }

    df_out = pd.DataFrame([feats], index=[int(sd.value // 10**6)])
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
