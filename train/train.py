#!/usr/bin/env python3
# train/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, layers
import joblib

from train.features import (
    SYMBOLS,
    INTERVALS,
    load_candles,
    load_metrics,
    add_indicators,
    add_candlestick_patterns,
    join_metrics,
    generate_labels
)
from train.snapshots import load_snapshots

def main():
    # ─── 1) Metrics laden ────────────────────────────────────────────────────────
    print("⏳ Loading metrics …")
    df_oi = load_metrics("open_interest")
    df_fr = load_metrics("funding_rate")
    df_metrics = pd.merge(df_oi, df_fr, on=["symbol","date"], how="outer")

    all_dfs = []
    # ─── 2) Pro Symbol & Intervall ───────────────────────────────────────────────
    for symbol in SYMBOLS:
        # 2.a) Tages-Snapshots nur für dieses Symbol
        print(f"⏳ Loading snapshots for {symbol}")
        df_snap = load_snapshots(
            base_dir="snapshots/futures",
            symbols=[symbol],
            top_n=10
        )
        # normalize date
        df_snap["date"] = pd.to_datetime(df_snap["date"]).dt.normalize()

        for interval in INTERVALS:
            print(f"⏳ Processing {symbol} @ {interval}")
            # Candle-Daten + Indikatoren
            df = load_candles(symbol, interval)
            df = add_indicators(df)
            df = add_candlestick_patterns(df)

            # Merge Metrics
            df = join_metrics(df, df_metrics)

            # Labels erzeugen
            df = generate_labels(df)

            # Merge Snapshots auf Tagesdatum
            tmp = df.reset_index().rename(columns={"index":"timestamp"})
            tmp["date"] = tmp["timestamp"].dt.normalize()
            df = tmp.merge(df_snap, how="left", on=["symbol","date"]).set_index("timestamp")

            # Fehlende Snapshot-Features (vor Start) mit 0 füllen
            snap_cols = [c for c in df.columns if c.startswith("snap_")]
            df[snap_cols] = df[snap_cols].fillna(0.0)

            # Symbol & Interval als Spalten
            df["symbol"]   = symbol
            df["interval"] = interval

            all_dfs.append(df)

    # ─── 3) Gesamten DataFrame zusammenführen ────────────────────────────────────
    df_full = pd.concat(all_dfs)
    print("✅ Feature matrix ready – rows:", len(df_full))

    # ─── 4) Train/Test-Split ─────────────────────────────────────────────────────
    X = df_full.drop(columns=["symbol","interval","future_close","return","label"])
    y = df_full["label"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, shuffle=False, test_size=0.2)

    # ─── 5) Skalierung ────────────────────────────────────────────────────────────
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # ─── 6) Modell definieren ────────────────────────────────────────────────────
    model = Sequential([
        layers.Input(shape=(Xtr_s.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ─── 7) Training ──────────────────────────────────────────────────────────────
    print("⏳ Training model …")
    model.fit(
        Xtr_s, ytr,
        validation_data=(Xte_s, yte),
        epochs=25,
        batch_size=256
    )

    # ─── 8) Export Scaler & Modell ───────────────────────────────────────────────
    print("✅ Saving scaler and model …")
    joblib.dump(scaler, "scaler.pkl")
    model.save("model.h5")

    print("🎉 Done.")

if __name__ == "__main__":
    main()
