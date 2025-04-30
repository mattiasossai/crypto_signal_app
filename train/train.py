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
SYMBOLS = [
    'BTCUSDT','ETHUSDT','BNBUSDT','XRPUSDT','ADAUSDT',
    'DOGEUSDT','SOLUSDT','AVAXUSDT','DOTUSDT','MATICUSDT',
    'TRXUSDT','LINKUSDT','FTTUSDT','ETCUSDT','UNIUSDT',
    'AXSUSDT','SUIUSDT','ENAUSDT','TRUMPUSDT'
]
INTERVALS = ['1m','5m','15m','1h','4h']
DATES = pd.date_range(
    end=pd.Timestamp.utcnow().floor('D'),
    periods=30
).strftime('%Y-%m-%d')

def load_historical(symbol: str, interval: str, date: str) -> pd.DataFrame | None:
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
                df = pd.read_csv(
                    f,
                    header=None,
                    skiprows=1,
                    names=['t','o','h','l','c','v','ct','qv','nt','tb','tq','x']
                )
    except Exception as e:
        print(f"⚠️ {symbol} {interval} {date} skipped: {e}")
        return None

    df['c'] = df['c'].astype(float)
    df['v'] = df['v'].astype(float)
    df.index = pd.to_datetime(df['t'], unit='ms')
    return df[['c','v']]

def build_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame | None:
    try:
        bb   = ta.bbands(df['c'], length=20, std=2)
        rsi  = ta.rsi(df['c'], length=14)
        macd = ta.macd(df['c'], fast=12, slow=26, signal=9)
        ema200 = ta.ema(df['c'], length=200)
        sma50  = ta.sma(df['c'], length=50)
    except Exception:
        return None

    # Prüfen, ob Indikatoren erzeugt wurden
    if macd is None or bb is None or 'MACD_12_26_9' not in macd:
        return None

    feat = pd.DataFrame({
        f"{prefix}_c":      df['c'],
        f"{prefix}_v":      df['v'],
        f"{prefix}_rsi":    rsi,
        f"{prefix}_macd":   macd['MACD_12_26_9'],
        f"{prefix}_ema200": ema200,
        f"{prefix}_sma50":  sma50,
        f"{prefix}_bb_up":  bb['BBU_20_2.0'],
        f"{prefix}_bb_mid": bb['BBM_20_2.0'],
        f"{prefix}_bb_low": bb['BBL_20_2.0'],
    }).dropna()

    return feat if len(feat) > 50 else None

# === 1) Daten sammeln + Features bauen ===
all_frames = []

for sym in SYMBOLS:
    feats_per_interval: dict[str, pd.DataFrame] = {}

    for iv in INTERVALS:
        daily_feats = []
        for dt in DATES:
            hist = load_historical(sym, iv, dt)
            if hist is None:
                continue
            feats = build_features(hist, iv)
            if feats is None:
                continue
            daily_feats.append(feats)

        if not daily_feats:
            print(f"⚠️ {sym} {iv}: keine validen Tages-Features")
            continue

        merged_iv = pd.concat(daily_feats).drop_duplicates().iloc[-5000:]
        feats_per_interval[iv] = merged_iv

    # Wenn nicht alle Intervalle da sind ODER speziell 1h fehlt, überspringen
    if set(feats_per_interval.keys()) < set(INTERVALS):
        missing = set(INTERVALS) - set(feats_per_interval.keys())
        print(f"⚠️ {sym}: fehlt Intervalle {missing} → übersprungen")
        continue

    # === 2) Symbol-spezifisches DataFrame ===
    df_sym = pd.concat(feats_per_interval.values(), axis=1, join='inner').dropna()
    df_sym['future'] = df_sym['1h_c'].shift(-1)
    df_sym.dropna(inplace=True)
    df_sym['label'] = (
        (df_sym['future'] - df_sym['1h_c']) / df_sym['1h_c']
    ).apply(lambda x: 2 if x > 0.01 else (0 if x < -0.01 else 1))

    all_frames.append(df_sym)

# === 3) Validierung ===
if not all_frames:
    print("⚠️ Kein Trainings-Frame erstellt. Abbruch.")
    sys.exit(1)

data = pd.concat(all_frames).dropna()
if data.empty:
    print("⚠️ Kombinierte Daten sind leer. Abbruch.")
    sys.exit(1)

# === 4) Features / Labels ===
feature_cols = [c for c in data.columns if c not in ['future','label']]
X = data[feature_cols].values
y = tf.keras.utils.to_categorical(data['label'], num_classes=3)

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
Xtr, Xvl, ytr, yvl = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === 5) Modell bauen & trainieren ===
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

# === 6) Speichern ===
model.save('model.keras')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Fertig: model.keras, scaler.pkl & model.tflite erstellt.")
