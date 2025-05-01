#!/usr/bin/env python3
import os
import sys
import io
import zipfile
import pickle
import logging
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas_ta as ta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# === Konfiguration ===
PROXY_URL = os.getenv("PROXY_URL", "").rstrip("/")
if not PROXY_URL:
    logging.error("Missing PROXY_URL environment variable")
    sys.exit(1)

SYMBOLS = [
    'BTCUSDT','ETHUSDT','BNBUSDT','XRPUSDT','ADAUSDT',
    'DOGEUSDT','SOLUSDT','AVAXUSDT','DOTUSDT','MATICUSDT',
    'TRXUSDT','LINKUSDT','FTTUSDT','ETCUSDT','UNIUSDT'
]
INTERVALS = ['1m','5m','15m','1h','4h']
# Lade 30 Tage Historie (bis gestern, um 404 für heutiges Zip zu vermeiden)
yesterday = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1)
DATES = pd.date_range(end=yesterday, periods=30).strftime('%Y-%m-%d')

# === Daten laden (tägliche ZIP von data.binance.vision) ===
def load_historical(symbol: str, interval: str) -> pd.DataFrame:
    dfs = []
    for date in DATES:
        url = f"{PROXY_URL}?symbol={symbol}&interval={interval}&date={date}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                fname = z.namelist()[0]
                # skiprows=1 um den Header "open,close,..." zu überspringen
                df = pd.read_csv(z.open(fname), header=None, skiprows=1)
            df.columns = ['t','o','h','l','c','v','ct','qv','nt','tb','tq','x']
            df['c'] = df['c'].astype(float)
            df['v'] = df['v'].astype(float)
            df.index = pd.to_datetime(df['t'], unit='ms')
            dfs.append(df[['c','v']])
        except Exception as e:
            logging.warning(f"{symbol} {interval} {date} skipped: {e}")
    if not dfs:
        return pd.DataFrame()
    # zusammenhängen, Duplikate entfernen, maximal 5000 Zeilen behalten
    all_hist = pd.concat(dfs).drop_duplicates()
    return all_hist.tail(5000)

# === Feature-Berechnung ===
def build_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty or len(df) < 50:
        return pd.DataFrame()
    bb = ta.bbands(df['c'], length=20, std=2)
    return pd.DataFrame({
        f"{prefix}_c":     df['c'],
        f"{prefix}_v":     df['v'],
        f"{prefix}_rsi":   ta.rsi(df['c'], length=14),
        f"{prefix}_macd":  ta.macd(df['c'], fast=12, slow=26, signal=9)['MACD_12_26_9'],
        f"{prefix}_ema200": ta.ema(df['c'], length=200),
        f"{prefix}_sma50":  ta.sma(df['c'], length=50),
        f"{prefix}_bb_up":  bb['BBU_20_2.0'],
        f"{prefix}_bb_mid": bb['BBM_20_2.0'],
        f"{prefix}_bb_low": bb['BBL_20_2.0'],
    }).dropna()

# === Pipeline ===
frames = []
for sym in SYMBOLS:
    sym_feats = []
    for iv in INTERVALS:
        hist = load_historical(sym, iv)
        feat = build_features(hist, iv)
        if feat.empty:
            logging.warning(f"{sym} {iv}: no valid features")
        else:
            sym_feats.append(feat)
    if not sym_feats:
        logging.warning(f"{sym}: no data at all → skip symbol")
        continue

    df_sym = pd.concat(sym_feats, axis=1, join='inner')
    df_sym['future'] = df_sym['1h_c'].shift(-1)
    df_sym.dropna(inplace=True)
    df_sym['label'] = ((df_sym['future'] - df_sym['1h_c']) / df_sym['1h_c']) \
                       .apply(lambda x: 2 if x > 0.01 else (0 if x < -0.01 else 1))
    frames.append(df_sym)

if not frames:
    logging.error("Keine Trainingsdaten gefunden → abort")
    sys.exit(1)

data = pd.concat(frames).dropna()
# Features / Labels trennen
feat_cols = [c for c in data.columns if any(s in c for s in ['_c','_v','_rsi','_macd','_ema200','_sma50','_bb_'])]
X = data[feat_cols].values
y = tf.keras.utils.to_categorical(data['label'], num_classes=3)

# Skalieren
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
Xtr, Xvl, ytr, yvl = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Modell definieren & trainieren ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(Xtr.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax'),
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(Xtr, ytr, validation_data=(Xvl, yvl), epochs=30, batch_size=256)

# === Speichern ===
model.save('model.keras')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

logging.info("✅ Training done: model.keras, scaler.pkl, model.tflite")
