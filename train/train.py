import os, sys, pickle
import pandas as pd, numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas_ta as ta
import requests

# Proxy-URL aus GitHub Secret
PROXY_URL = os.getenv('PROXY_URL')

# Top-15 USD-M Futures (volumen-basiert)
SYMBOLS   = ['BTCUSDT','ETHUSDT','BNBUSDT','XRPUSDT','ADAUSDT',
             'SOLUSDT','DOGEUSDT','AVAXUSDT','DOTUSDT','MATICUSDT',
             'LINKUSDT','TRXUSDT','ETCUSDT','UNIUSDT','SUIUSDT']
INTERVALS = ['1m','5m','15m','1h','4h']
LIMIT     = 500

def fetch_klines(symbol, interval, mode='rest', date=None):
    """
    Modus 'zip' oder 'rest'. Für zip: über PROXY_URL?symbol&interval&mode=zip&date=YYYY-MM-DD
    Für rest: PROXY_URL?symbol&interval&mode=rest&limit=500
    """
    params = {
      'symbol': symbol,
      'interval': interval,
      'mode': mode
    }
    if mode == 'zip':
        params['date'] = date or pd.Timestamp.utcnow().strftime('%Y-%m-%d')
    else:
        params['limit'] = str(LIMIT)

    resp = requests.get(PROXY_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()  # bei rest: Liste von Klines

# 1) Daten sammeln & Feature-Berechnung
frames = []
for sym in SYMBOLS:
    ints = []
    for iv in INTERVALS:
        try:
            data = fetch_klines(sym, iv, mode='rest')
            df = pd.DataFrame(data, columns=[
              't','o','h','l','c','v','ignore1','ignore2','ignore3','ignore4','ignore5','ignore6'
            ])
            df['c'] = df['c'].astype(float)
            df['v'] = df['v'].astype(float)
            df.set_index(pd.to_datetime(df['t'], unit='ms'), inplace=True)
        except Exception as e:
            print(f"⚠️ {sym} {iv} REST skipped: {e}")
            continue

        # Technische Indikatoren (Beispiel)
        bb = ta.bbands(df['c'], length=20, std=2)
        feat = pd.DataFrame({
          f"{iv}_c": df['c'],
          f"{iv}_v": df['v'],
          f"{iv}_rsi": ta.rsi(df['c'], length=14),
          f"{iv}_macd": ta.macd(df['c'], fast=12, slow=26, signal=9)['MACD_12_26_9'],
          f"{iv}_ema200": ta.ema(df['c'], length=200),
          f"{iv}_sma50": ta.sma(df['c'], length=50),
          f"{iv}_bb_up": bb['BBU_20_2.0'],
          f"{iv}_bb_mid": bb['BBM_20_2.0'],
          f"{iv}_bb_low": bb['BBL_20_2.0']
        }).dropna()

        if feat.empty:
            print(f"⚠️ {sym} {iv}: keine validen Features")
            continue
        ints.append(feat)

    if not ints:
        print(f"⚠️ {sym}: keine Daten in keinem Interval")
        continue

    merged = pd.concat(ints, axis=1, join='inner')
    merged['future'] = merged[f"1h_c"].shift(-1)
    merged.dropna(inplace=True)
    merged['label'] = ((merged['future'] - merged['1h_c']) / merged['1h_c'])\
                      .apply(lambda x: 2 if x > 0.01 else (0 if x < -0.01 else 1))
    frames.append(merged)

# 2) Validierung
if not frames:
    print("⚠️ Keine Trainingsdaten – Abbruch.")
    sys.exit(1)
data = pd.concat(frames).dropna()
if data.empty:
    print("⚠️ Nach Merge keine Daten – Abbruch.")
    sys.exit(1)

# 3) Features / Labels
feature_cols = [c for c in data.columns if c not in ['future','label']]
X = data[feature_cols].values
y = tf.keras.utils.to_categorical(data['label'], num_classes=3)

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
Xtr, Xvl, ytr, yvl = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4) Modell definieren & trainieren
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(Xtr.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(Xtr, ytr, validation_data=(Xvl, yvl), epochs=30, batch_size=256)

# 5) Speichern
model.save('model.keras')
with open('scaler.pkl','wb') as f:
    pickle.dump(scaler, f)

# 6) In TFLite konvertieren
conv = tf.lite.TFLiteConverter.from_keras_model(model)
tfl = conv.convert()
with open('model.tflite','wb') as f:
    f.write(tfl)

print("✅ train.py fertig: model.keras, scaler.pkl, model.tflite")
