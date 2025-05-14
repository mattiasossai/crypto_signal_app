#!/usr/bin/env python3
import argparse
import os, glob
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

def extract_features(input_dir, start_date, end_date):
    # load all CSVs
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    dfs = []
    for fp in files:
        day = os.path.basename(fp).replace(".csv","")
        # skip vor Inception
        sym = os.path.basename(input_dir)
        if pd.to_datetime(day) < pd.to_datetime(INCEPTION.get(sym, start_date)):
            continue
        # read (alte CSVs ohne Header → header=None)
        df = pd.read_csv(fp,
                         header=0 if fp_contains_header(fp:=fp) else None,
                         names=["timestamp","percentage","depth","notional"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        dfs.append(df)

    if not dfs:
        raise SystemExit("⚠️ Keine Daten gefunden")

    full = pd.concat(dfs)
    # filter window
    sd = pd.to_datetime(start_date).tz_localize("UTC")
    ed = pd.to_datetime(end_date).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    full = full[sd:ed]

    # aggregations
    total_notional = full["notional"].sum()
    n1 = full.loc[full["percentage"]<=0.01, :]
    n10 = full.loc[full["percentage"]<=0.10, :]

    feats = {
        "total_notional": total_notional,
        "notional_1pct":   n1["notional"].sum(),
        "depth_1pct":      n1["depth"].sum(),
        "notional_10pct":  n10["notional"].sum(),
        "depth_10pct":     n10["depth"].sum(),
    }
    return pd.DataFrame([feats], index=[int(sd.value/10**6)])  # timestamp in ms

def fp_contains_header(fp):
    # prüfe erste Zeile, ob sie nicht-numerisch beginnt
    with open(fp,"r") as f:
        first = f.readline().split(",")[0]
    return not first.isdigit()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--start-date",  required=True)
    p.add_argument("--end-date",    required=True)
    args = p.parse_args()

    out = extract_features(args.input_dir, args.start_date, args.end_date)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    out.to_parquet(args.output_file)
    print(f"✅ Wrote features to {args.output_file}")

if __name__=="__main__":
    main()
