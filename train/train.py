#!/usr/bin/env python3
# train/train.py

import os, zipfile, json, glob
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Sequential

# ─── KONFIGURATION ─────────────────────────────────────────────────────────────
SYMBOLS   = ["BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","SOLUSDT","ENAUSDT"]
INTERVALS = ["1m","5m","15m","1h","4h"]
PARTS     = ["part1","part2"]
HIST_DIR  = "historical"
METR_DIR  = "metrics"
TARGET_FUTURE_INTERVAL = "1h"   # Label auf 1h-Bewegung

# ─── 1) HISTORISCHE CANDLESTICKS LADEN ─────────────────────────────────────────
def load_historical():
    df_list = []
    for sym in SYMBOLS:
        for iv in INTERVALS:
            for part in PARTS:
                base = os.path.join(HIST_DIR, sym, iv, part)
                if not os.path.isdir(base): continue
                for z in os.listdir(base):
                    if not z.endswith(".zip"): continue
                    zp = os.path.join(base, z)
                    with zipfile.ZipFile(zp) as zf:
                        fname = zf.namelist()[0]
                        # Binance-Vision-Format: erste Zeile Header, dann OHLCV:
                        df = pd.read_csv(
                          zf.open(fname),
                          header=None, skiprows=1,
                          names=[
                            "Open_time","Open","High","Low","Close","Volume",
                            "Close_time","Quote_av","Num_trades","TBAV","TQAV","Ignore"
                          ]
                        )
                    df["timestamp"] = pd.to_datetime(df["Open_time"], unit="ms")
                    df.set_index("timestamp", inplace=True)
                    df = df[["Open","High","Low","Close","Volume"]].astype(float)
                    df["symbol"]   = sym
                    df["interval"] = iv
                    df_list.append(df)
    if not df_list:
        raise RuntimeError("Keine historischen Daten gefunden in 'historical/'")
    return pd.concat(df_list)

# ─── 2) TÄGLICHE METRIKEN LADEN ─────────────────────────────────────────────────
def load_metrics():
    # erwartet JSON-Antworten vom Worker, aggregiere pro Tag+Symbol
    recs = []
    for part in PARTS:
        for metric in ["open_interest","funding_rate","liquidity"]:
            base = os.path.join(METR_DIR, part, metric)
            if not os.path.isdir(base): continue
            for fn in os.listdir(base):
                if not fn.endswith(".json"): continue
                # Dateiname: e.g. BTCUSDT_2020-01-01.json oder BTCUSDT_2020-01-01_08.json
                parts = fn.rstrip(".json").split("_")
                sym   = parts[0]
                date  = parts[1]
                hour  = parts[2] if len(parts)>2 else None
                data = json.load(open(os.path.join(base,fn)))
                # data ist meist Liste von Werten oder ein Objekt; wir summarizen zu einem Tages-Wert
                # Hier als Beispiel: nehmen wir den Mittelwert aller 'rate' Felder, falls vorhanden
                if isinstance(data, list) and all(isinstance(x, dict) for x in data):
                    vals = []
                    for x in data:
                        # versuche diverse Keys
                        for k in ("openInterest","fundingRate","liquidity"):
                            if k in x:
                                vals.append(float(x[k]))
                                break
                    val = float(np.mean(vals)) if vals else np.nan
                elif isinstance(data, dict) and metric in data:
                    val = float(data[metric])
                else:
                    # fallback: Länge der Liste
                    val = float(len(data)) if isinstance(data, list) else np.nan
                recs.append({
                    "symbol": sym,
                    "date": pd.to_datetime(date),
                    "metric": metric,
                    "value": val
                })
    if not recs:
        raise RuntimeError("Keine Metriken gefunden in 'metrics/'")
    return pd.DataFrame(recs)

# ─── 3) FEATURE ENGINEERING ─────────────────────────────────────────────────────
def engineer_features(df_hist, df_metrics):
    # 3.1 Candlestick-Indikatoren pro Zeile
    df = df_hist.copy()
    o = df["Open"]; c = df["Close"]; h = df["High"]; l = df["Low"]; v = df["Volume"]

    # RSI14
    delta = c.diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    rsi_u = up.ewm(span=14, adjust=False).mean()
    rsi_d = down.ewm(span=14, adjust=False).mean()
    df["RSI_14"] = 100 - 100/(1 + rsi_u/rsi_d)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD_line"] = ema12 - ema26
    df["MACD_signal"] = df["MACD_line"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD_line"] - df["MACD_signal"]

    # EMA50/200 + Crossover
    df["EMA_50"]  = c.ewm(span=50,  adjust=False).mean()
    df["EMA_200"] = c.ewm(span=200, adjust=False).mean()
    df["Bullish_Cross"] = (
      (df["EMA_50"].shift(1) < df["EMA_200"].shift(1)) &
      (df["EMA_50"] >= df["EMA_200"])
    ).astype(int)
    df["Bearish_Cross"] = (
      (df["EMA_50"].shift(1) > df["EMA_200"].shift(1)) &
      (df["EMA_50"] <= df["EMA_200"])
    ).astype(int)

    # SMA50
    df["SMA_50"] = c.rolling(50).mean()

    # Bollinger Bands (20,2)
    df["BB_mid"] = c.rolling(20).mean()
    df["BB_std"] = c.rolling(20).std(ddof=0)
    df["BB_up"]  = df["BB_mid"] + 2*df["BB_std"]
    df["BB_lo"]  = df["BB_mid"] - 2*df["BB_std"]
    df["PercentB"]          = (c - df["BB_lo"])/(df["BB_up"] - df["BB_lo"])
    df["Dist_to_BB_up"]     = df["BB_up"] - c
    df["Dist_to_BB_down"]   = c - df["BB_lo"]

    # Stochastic %K/%D
    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df["Sto_K"] = (c - low14)/(high14 - low14)*100
    df["Sto_D"] = df["Sto_K"].rolling(3).mean()

    # Candlestick-Patterns
    prev_o = o.shift(1); prev_c = c.shift(1)
    # Bullish Engulf
    bull1 = (prev_c < prev_o) & (c > o)
    bull2 = (c >= prev_o) & (o <= prev_c)
    df["Bull_Engulf"]  = (bull1 & bull2).astype(int)
    # Bearish Engulf
    bear1 = (prev_c > prev_o) & (c < o)
    bear2 = (o >= prev_c) & (c <= prev_o)
    df["Bear_Engulf"] = (bear1 & bear2).astype(int)
    # Doji
    df["Doji"]        = (abs(c - o) <= 0.001*c).astype(int)

    # 3.2 Multi-Timeframe: Baseline 1m, die anderen ff-fill
    # Wir erstellen einen MultiIndex DataFrame pro Symbol, dann reindex
    dfs = []
    for sym in SYMBOLS:
        df_sym = df[df["symbol"] == sym]
        # Einzelne IVs
        df_iv = {}
        for iv in INTERVALS:
            df_iv[iv] = df_sym[df_sym["interval"] == iv].copy()
            # suffix für Spalten
            suf = "_"+iv
            df_iv[iv].columns = [col+suf for col in df_iv[iv].columns]
            # Index = timestamp
        # Basis ist 1m
        base = df_iv["1m"]
        merged = base
        for iv in ["5m","15m","1h","4h"]:
            merged = merged.join(
              df_iv[iv].reindex(base.index, method="ffill"),
              how="left",
              rsuffix="_"+iv
            )
        dfs.append(merged)
    Xfull = pd.concat(dfs)

    # 3.3 Merge der täglichen Metriken (auf Tagesdatum gerundet)
    df_met = load_metrics()
    # Pivot: rows = date+symbol, cols = metric
    pivot = df_met.pivot_table(
      index=["symbol","date"], columns="metric", values="value"
    ).reset_index()
    pivot["date"] = pd.to_datetime(pivot["date"])
    # verknüpfe: erzeugen in Xfull Spalte 'date' aus index
    Xfull = Xfull.reset_index().rename(columns={"index":"timestamp"})
    Xfull["date"] = Xfull["timestamp"].dt.normalize()
    # Wir haben symbol_1m Spalte => so holen wir symbol:
    Xfull = Xfull.merge(pivot, how="left",
      left_on=["symbol_1m","date"], right_on=["symbol","date"])
    Xfull.drop(columns=["symbol","date"], inplace=True)

    # 3.4 Label erzeugen: nächste 1h-Schlusskurs-Bewegung
    # wir referenzieren auf die Close_1h Spalte im 1h-DataFrame
    Xfull = Xfull.sort_values("timestamp")
    Xfull["future_1h_close"] = Xfull["Close_1h"].shift(- int(60/1))  # 60 min / 1min
    # % change
    Xfull["pct_change_1h"] = (
      (Xfull["future_1h_close"] - Xfull["Close_1m"]) /
      Xfull["Close_1m"]
    )
    # Label: 2 = Long (>1%), 0 = Short (<–1%), 1 = Neutral
    def label_row(x):
        if x > 0.01: return 2
        if x < -0.01: return 0
        return 1
    Xfull["label"] = Xfull["pct_change_1h"].apply(label_row)

    # Drop rows with NaNs (z.B. am Anfang, Ende)
    Xfull.dropna(inplace=True)
    return Xfull

# ─── 4) TRAINING ─────────────────────────────────────────────────────────────────
def train_and_export(df_features):
    # Features/Labels trennen
    X = df_features.drop(columns=["timestamp","symbol_1m","interval_1m",
                                  "Open_time_1m","future_1h_close","pct_change_1h","label"])
    y = df_features["label"].astype(int)

    # Split (kein Shuffling für Zeitserien)
    Xtr, Xte, ytr, yte = train_test_split(
      X, y, test_size=0.2, shuffle=False
    )

    # Skalieren
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # Modell
    model = Sequential([
      layers.Input(shape=(Xtr_s.shape[1],)),
      layers.Dense(128, activation="relu"),
      layers.Dropout(0.3),
      layers.Dense(64, activation="relu"),
      layers.Dense(3, activation="softmax")
    ])
    model.compile(
      optimizer="adam",
      loss="sparse_categorical_crossentropy",
      metrics=["accuracy"]
    )

    # Training
    model.fit(
      Xtr_s, ytr,
      validation_data=(Xte_s, yte),
      epochs=25, batch_size=256
    )

    # Export Scaler
    import joblib
    joblib.dump(scaler, "scaler.pkl")

    # Export TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

    print("✅ Fertig: scaler.pkl + model.tflite erstellt")

# ─── MAIN ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("⏳ Lade historische Daten …")
    df_hist = load_historical()
    print("⏳ Engineering der Features …")
    df_feat = engineer_features(df_hist, None)
    print("⏳ Starte Training …")
    train_and_export(df_feat)
