#!/usr/bin/env python3
# train/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, layers
import joblib

from features import (
    SYMBOLS, INTERVALS,
    load_candles, load_metrics,
    add_indicators, add_candlestick_patterns,
    join_metrics, generate_labels
)

def main():
    # 1) Load Metrics
    print("⏳ Loading metrics …")
    df_oi = load_metrics("open_interest")
    df_fr = load_metrics("funding_rate")
    df_metrics = pd.merge(df_oi, df_fr, on=["symbol","date"], how="outer")

    all_dfs = []
    # 2) Per Symbol & Interval Prozess
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            print(f"⏳ Processing {symbol} @ {interval}")
            df = load_candles(symbol, interval)
            df = add_indicators(df)
            df = add_candlestick_patterns(df)
            df = join_metrics(df, df_metrics)
            df = generate_labels(df)
            df["symbol"]   = symbol
            df["interval"] = interval
            all_dfs.append(df)

    df_full = pd.concat(all_dfs)
    print("✅ Feature matrix ready – rows:", len(df_full))

    # 3) Train/Test-Split
    X = df_full.drop(columns=["symbol","interval","future_close","return","label"])
    y = df_full["label"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, shuffle=False, test_size=0.2)

    # 4) Skalierung
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # 5) Modell definieren
    model = Sequential([
        layers.Input(shape=(Xtr_s.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64,  activation="relu"),
        layers.Dense(3,   activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 6) Training
    print("⏳ Training model …")
    model.fit(Xtr_s, ytr, validation_data=(Xte_s, yte), epochs=25, batch_size=256)

    # 7) Export
    print("✅ Saving scaler and model …")
    joblib.dump(scaler, "scaler.pkl")
    model.save("model.h5")

    print("🎉 Done.")

if __name__ == "__main__":
    main()
