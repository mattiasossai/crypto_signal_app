import os, sys, pickle, zipfile, io
from datetime import datetime, timedelta

import pandas as pd, numpy as np, tensorflow as tf
from binance.client import Client, BinanceAPIException
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas_ta as ta
import requests

# ─── Konfiguration ──────────────────────────────────────────────────────────────

PROXY_URL          = os.getenv('PROXY_URL')         # Dein Worker-URL
BINANCE_API_KEY    = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

SYMBOLS   = ['BTCUSDT','ETHUSDT','BNBUSDT','XRPUSDT','ADAUSDT',
             'DOGEUSDT','SOLUSDT','AVAXUSDT','DOTUSDT','MATICUSDT',
             'TRXUSDT','LINKUSDT','FTTUSDT','ETCUSDT','UNIUSDT']
INTERVALS = ['1m','5m','15m','1h','4h']

# für REST-Fallback: 5 Jahre in ms
PAST_MS   = int((datetime.utcnow() - timedelta(days=5*365)).timestamp() * 1000)
LIMIT_API = 1500

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# ─── Hilfsfunktionen ────────────────────────────────────────────────────────────

def load_zip(symbol, interval, date):
    """Versuch ZIP via Worker → DataFrame."""
    url = f"{PROXY_URL}?symbol={symbol}&interval={interval}&date={date}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        raise IOError(f"Worker-ZIP fehlgeschlagen ({r.status_code})")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = z.namelist()[0]
        df = pd.read_csv(z.open(name), header=None)
    df.columns = ['t','o','h','l','c','v','ct','qv','nt','tb','tq','x']
    df['c'], df['v'] = df['c'].astype(float), df['v'].astype(float)
    df.index = pd.to_datetime(df['t'], unit='ms')
    return df[['c','v']]

def load_api(symbol, interval):
    """Hole Kerzen der letzten 5 Jahre per Futures-API."""
    print(f"🔄 API-Fallback: {symbol} {interval}")
    parts = []
    start = PAST_MS
    while True:
        kl = client.futures_klines(symbol=symbol, interval=interval, startTime=start, limit=LIMIT_API)
        if not kl: break
        df = pd.DataFrame(kl, columns=['t','o','h','l','c','v','ct','qv','nt','tb','tq','x'])
        df['c'], df['v'] = df['c'].astype(float), df['v'].astype(float)
        df.index = pd.to_datetime(df['t'], unit='ms')
        parts.append(df[['c','v']])
        last = df['t'].iat[-1] + 1
        if last >= int(datetime.utcnow().timestamp()*1000): break
        start = last
    if not parts:
        raise IOError("API lieferte null Daten")
    return pd.concat(parts).drop_duplicates()

def build_features(df, pre):
    """Technische Indikatoren für einen DF."""
    bb = ta.bbands(df['c'], length=20, std=2)
    return pd.DataFrame({
        f"{pre}_c": df['c'],
        f"{pre}_v": df['v'],
        f"{pre}_rsi": ta.rsi(df['c'],14),
        f"{pre}_macd": ta.macd(df['c'],12,26,9)['MACD_12_26_9'],
        f"{pre}_ema200": ta.ema(df['c'],200),
        f"{pre}_sma50": ta.sma(df['c'],50),
        f"{pre}_bb_up": bb['BBU_20_2.0'],
        f"{pre}_bb_mid": bb['BBM_20_2.0'],
        f"{pre}_bb_low": bb['BBL_20_2.0']
    }).dropna()

# ─── 1) Daten sammeln ───────────────────────────────────────────────────────────

all_frames = []
today = datetime.utcnow().strftime("%Y-%m-%d")

for sym in SYMBOLS:
    sym_frames = []
    for iv in INTERVALS:
        # 30 Tage via Worker, sonst API
        try:
            days = []
            for d in range(30):
                date = (datetime.utcnow() - timedelta(days=d)).strftime("%Y-%m-%d")
                days.append(load_zip(sym, iv, date))
            hist = pd.concat(days).drop_duplicates().sort_index()
            print(f"✅ {sym} {iv} via Worker geladen")
        except Exception as e:
            print(f"⚠️ {sym} {iv} Worker-Fehler: {e}")
            hist = load_api(sym, iv)

        feat = build_features(hist, iv)
        if feat.empty:
            print(f"⚠️ {sym} {iv}: keine Features → übersprungen")
            continue
        sym_frames.append(feat)

    if not sym_frames:
        print(f"⚠️ Dummy-Fallback SYMBOL {sym}: keine Intervalle geladen")
        continue

    merged = pd.concat(sym_frames, axis=1, join='inner').dropna()
    merged['future'] = merged['1h_c'].shift(-1)
    merged.dropna(inplace=True)
    merged['label'] = ((merged['future']-merged['1h_c'])/merged['1h_c'])\
        .apply(lambda x:2 if x>0.01 else (0 if x< -0.01 else 1))
    all_frames.append(merged)

# ─── 2) Fallback auf Dummy? ───────────────────────────────────────────────────────

if not all_frames:
    print("❌ Keine Daten für _alle_ Symbole → Dummy-Modell")
    model = tf.keras.Sequential([tf.keras.layers.Input(1), tf.keras.layers.Dense(3,activation='softmax')])
    model.compile('adam','categorical_crossentropy')
    model.save('model.keras')
    pickle.dump(StandardScaler(), open('scaler.pkl','wb'))
    tfl = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    open('model.tflite','wb').write(tfl)
    sys.exit(0)

data = pd.concat(all_frames).dropna()
if data.empty:
    print("❌ Daten leer nach Merge → Dummy-Modell")
    model = tf.keras.Sequential([tf.keras.layers.Input(1), tf.keras.layers.Dense(3,activation='softmax')])
    model.compile('adam','categorical_crossentropy')
    model.save('model.keras')
    pickle.dump(StandardScaler(), open('scaler.pkl','wb'))
    tfl = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    open('model.tflite','wb').write(tfl)
    sys.exit(0)

# ─── 3) Train/Test ───────────────────────────────────────────────────────────────

X = data[[c for c in data.columns if c not in ['future','label']]].values
y = tf.keras.utils.to_categorical(data['label'],3)
scaler = StandardScaler().fit(X)
X_s = scaler.transform(X)
Xtr, Xvl, ytr, yvl = train_test_split(X_s,y,test_size=0.2,random_state=42)

# ─── 4) Modell ──────────────────────────────────────────────────────────────────

model = tf.keras.Sequential([
    tf.keras.layers.Input(Xtr.shape[1]),
    tf.keras.layers.Dense(128,'relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64,'relu'),
    tf.keras.layers.Dense(3,'softmax'),
])
model.compile('adam','categorical_crossentropy',['accuracy'])
model.fit(Xtr,ytr,validation_data=(Xvl,yvl),epochs=30,batch_size=256)

# ─── 5) Speichern ───────────────────────────────────────────────────────────────

model.save('model.keras')
pickle.dump(scaler, open('scaler.pkl','wb'))
tfl_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
open('model.tflite','wb').write(tfl_model)

print("🎉 model.keras, scaler.pkl und model.tflite erfolgreich erzeugt")
