# train/train.py

import os
import sys
import io
import zipfile
import pickle

import pandas as pd
import numpy as np
import requests
import tensorflow as tf
import pandas_ta as ta

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Config / Constants ---
PROXY = os.getenv('PROXY_URL')
if not PROXY:
    print("❌ ERROR: PROXY_URL environment variable not set")
    sys.exit(1)

# Top-20 USD-M Futures auf Binance
SYMBOLS = [
    'BTCUSDT','ETHUSDT','BNBUSDT','XRPUSDT','ADAUSDT',
    'DOGEUSDT','SOLUSDT','AVAXUSDT','DOTUSDT','MATICUSDT',
    'TRXUSDT','LINKUSDT','FTTUSDT','ETCUSDT','UNIUSDT',
    'AXSUSDT','SUIUSDT','ENAUSDT','TRUMPUSDT'
]
INTERVALS = ['1m','5m','15m','1h','4h']

# Wir laden je Tag 30 Tage Historie
DATES = pd.date_range(
    end=pd.Timestamp.utcnow().floor('D'),
    periods=30
).strftime('%Y-%m-%d').tolist()

# --- Helper Functions ---

def load_historical(symbol: str, interval: str, date: str) -> pd.DataFrame:
    """
    Lädt die ZIP-Datei über den Proxy, entpackt sie und gibt
    ein DataFrame mit Spalten ['c','v'] zurück.
    """
    url = f"{PROXY}?symbol={symbol}&interval={interval}&date={date}"
    r = requests.get(url)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    csv_name = z.namelist()[0]
    df = pd.read_csv(z.open(csv_name), header=None)
    df.columns = ['t','o','h','l','c','v','ct','qv','nt','tb','tq','x']
    df['c'] = df['c'].astype(float)
    df['v'] = df['v'].astype(float)
    df.index = pd.to_datetime(df['t'], unit='ms')
    return df[['c','v']]

def build_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Berechnet technische Indikatoren für eine Kerzenserie und
    gibt ein DataFrame zurück, dessen Spalten mit prefix_ beginnen.
    """
    bb = ta.bbands(df['c'], length=20, std=2)
    return pd.DataFrame({
        f"{prefix}_c":      df['c'],
        f"{prefix}_v":      df['v'],
        f"{prefix}_rsi":    ta.rsi(df['c'], length=14),
        f"{prefix}_macd":   ta.macd(df['c'], fast=12, slow=26, signal=9)['MACD_12_26_9'],
        f"{prefix}_ema200": ta.ema(df['c'], length=200),
        f"{prefix}_sma50":  ta.sma(df['c'], length=50),
        f"{prefix}_bb_up":  bb['BBU_20_2.0'],
        f"{prefix}_bb_mid": bb['BBM_20_2.0'],
        f"{prefix}_bb_low": bb['BBL_20_2.0'],
    }).dropna()

# --- Daten sammeln & Features bauen ---
all_frames = []
for sym in SYMBOLS:
    symbol_frames = []
    for iv in INTERVALS:
        for d in DATES:
            try:
                df_hist = load_historical(sym, iv, d)
                df_feat = build_features(df_hist, iv)
                symbol_frames.append(df_feat)
            except Exception as e:
                # Bei 404 o.Ä. überspringen
                print(f"⚠️ {sym} {iv} {d} skipped: {e}")
    if not symbol_frames:
        continue
    merged = pd.concat(symbol_frames, axis=1, join='inner')
    # Future-Preis (nächste Kerze 1h) für Label
    merged['future'] = merged['1h_c'].shift(-1)
    merged.dropna(inplace=True)
    # Label: 2 = Long, 1 = Neutral, 0 = Short
    merged['label'] = (
        (merged['future'] - merged['1h_c']) / merged['1h_c']
    ).apply(lambda x: 2 if x > 0.01 else (0 if x < -0.01 else 1))
    all_frames.append(merged)

# Exit, falls keine Daten
if not all_frames:
    print("❌ Keine historischen Daten gefunden. Abbruch.")
    sys.exit(1)

# --- Dataset zusammenführen & aufbereiten ---
data = pd.concat(all_frames).dropna()
if data.empty:
    print("❌ Kombinierte Daten leer. Abbruch.")
    sys.exit(1)

# Merkmale und Ziel
feature_cols = [c for c in data.columns if c not in ['future','label']]
X = data[feature_cols].values
y = tf.keras.utils.to_categorical(data['label'].astype(int), num_classes=3)

# Skalierung & Split
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
Xtr, Xvl, ytr, yvl = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Modell definieren & trainieren ---
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

# --- Modelle & Artefakte speichern ---
# 1) Keras-Modell
model.save('model.keras')
# 2) Scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
# 3) TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Modell-Training & Export abgeschlossen: model.keras, scaler.pkl, model.tflite")
