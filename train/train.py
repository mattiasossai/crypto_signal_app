import os
import sys
import io
import zipfile
import pickle
import requests

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas_ta as ta

# === Konfiguration ===
# Liste der Top-Futures
SYMBOLS = [
    'BTCUSDT','ETHUSDT','BNBUSDT','XRPUSDT','ADAUSDT',
    'DOGEUSDT','SOLUSDT','AVAXUSDT','DOTUSDT','MATICUSDT',
    'TRXUSDT','LINKUSDT','FTTUSDT','ETCUSDT','UNIUSDT',
    'AXSUSDT','SUIUSDT','ENAUSDT','TRUMPUSDT'
]
# Zeitebenen, die wir nutzen wollen
INTERVALS = ['1m','5m','15m','1h','4h']
# Wir laden jeweils 30 Tage Historie
DATES = pd.date_range(
    end=pd.Timestamp.utcnow().floor('D'),
    periods=30
).strftime('%Y-%m-%d')

# === Funktion zum Laden historischer ZIP-Daten ===
def load_historical(symbol: str, interval: str, date: str) -> pd.DataFrame | None:
    """
    Lädt das ZIP-Archiv für das gegebene Symbol, Interval und Datum
    von Binance S3 und gibt einen DataFrame mit 'c' (Close) und 'v' (Volume) zurück.
    Bei Fehlern (404, schlechtes ZIP) None zurückgeben.
    """
    url = (
        f"https://data.binance.vision/data/futures/um/daily/klines/"
        f"{symbol}/{interval}/{symbol}-{interval}-{date}.zip"
    )
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        print(f"⚠️ {symbol} {interval} {date} skipped: HTTP {resp.status_code}")
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                # Wir skippen die Kopfzeile und benennen Spalten manuell
                df = pd.read_csv(
                    f,
                    header=None,
                    skiprows=1,
                    names=['t','o','h','l','c','v','ct','qv','nt','tb','tq','x']
                )
    except zipfile.BadZipFile:
        print(f"⚠️ {symbol} {interval} {date} skipped: Bad ZIP")
        return None

    # Konvertiere und setze Index
    df['c'] = df['c'].astype(float)
    df['v'] = df['v'].astype(float)
    df.index = pd.to_datetime(df['t'], unit='ms')
    return df[['c','v']]

# === Feature-Engineering ===
def build_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    bb = ta.bbands(df['c'], length=20, std=2)
    return pd.DataFrame({
        f"{prefix}_c":    df['c'],
        f"{prefix}_v":    df['v'],
        f"{prefix}_rsi":  ta.rsi(df['c'], length=14),
        f"{prefix}_macd": ta.macd(df['c'], fast=12, slow=26, signal=9)['MACD_12_26_9'],
        f"{prefix}_ema200": ta.ema(df['c'], length=200),
        f"{prefix}_sma50":  ta.sma(df['c'], length=50),
        f"{prefix}_bb_up":   bb['BBU_20_2.0'],
        f"{prefix}_bb_mid":  bb['BBM_20_2.0'],
        f"{prefix}_bb_low":  bb['BBL_20_2.0'],
    }).dropna()

# === 1) Daten sammeln und Feature-Matrizen bauen ===
all_frames = []
for sym in SYMBOLS:
    per_symbol = []
    for iv in INTERVALS:
        # Lade alle 30 Tage, baue Features und füge an
        feats_list = []
        for dt in DATES:
            hist = load_historical(sym, iv, dt)
            if hist is None:
                continue
            feats = build_features(hist, iv)
            feats_list.append(feats)
        if not feats_list:
            continue
        merged_iv = pd.concat(feats_list).drop_duplicates().iloc[-5000:]
        per_symbol.append(merged_iv)
    if not per_symbol:
        print(f"⚠️ {sym}: keine Daten über alle Intervalle")
        continue
    # Wir joinen alle Intervalle aneinander (inner join)
    df_sym = pd.concat(per_symbol, axis=1, join='inner').dropna()
    # Zielvariable: Close 1h der nächsten Kerze
    df_sym['future'] = df_sym['1h_c'].shift(-1)
    df_sym.dropna(inplace=True)
    df_sym['label'] = (
        (df_sym['future'] - df_sym['1h_c']) / df_sym['1h_c']
    ).apply(lambda x: 2 if x > 0.01 else (0 if x < -0.01 else 1))
    all_frames.append(df_sym)

# === 2) Validierung ===
if not all_frames:
    print("⚠️ Kein Trainings-Frame erstellt. Abbruch.")
    sys.exit(1)
data = pd.concat(all_frames).dropna()
if data.empty:
    print("⚠️ Kombinierte Daten sind leer. Abbruch.")
    sys.exit(1)

# === 3) Features & Labels aufspalten ===
feature_cols = [c for c in data.columns if c not in ['future','label']]
X = data[feature_cols].values
y = tf.keras.utils.to_categorical(data['label'], num_classes=3)

# Standardisieren
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
Xtr, Xvl, ytr, yvl = train_test_split(
    X_scaled, y,
    test_size=0.2, random_state=42
)

# === 4) Modell definieren & trainieren ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(Xtr.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(Xtr, ytr, validation_data=(Xvl, yvl), epochs=30, batch_size=256)

# === 5) Speichern: .keras, Scaler & TFLite ===
# 5.1 native Keras-Format
model.save('model.keras')
# 5.2 Scaler speichern
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
# 5.3 TFLite-Konvertierung
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Training abgeschlossen: model.keras, scaler.pkl, model.tflite erzeugt.")
