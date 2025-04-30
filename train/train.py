import os, sys, pickle, zipfile, io
import pandas as pd, numpy as np, tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas_ta as ta
import requests

# Worker‐Proxy URL (aus Schritt 1)
PROXY_URL = "https://<your-worker-subdomain>.workers.dev"

# Top-20 USD-M Futures
SYMBOLS   = ['BTCUSDT','ETHUSDT','BNBUSDT','XRPUSDT','ADAUSDT',
             'DOGEUSDT','SOLUSDT','AVAXUSDT','DOTUSDT','MATICUSDT',
             'TRXUSDT','LINKUSDT','FTTUSDT','ETCUSDT','UNIUSDT',
             'AXSUSDT','SUIUSDT','ENAUSDT','TRUMPUSDT']
INTERVALS = ['1m','5m','15m','1h','4h']
# Lade 30 Tage Historie
DATES = pd.date_range(end=pd.Timestamp.utcnow().floor('D'), periods=30).strftime('%Y-%m-%d')

def load_historical(symbol, interval):
    dfs=[]
    for d in DATES:
        url = f"{PROXY_URL}?symbol={symbol}&interval={interval}&limit=500"
        # S3‐Archiv:
        url = f"https://data.binance.vision/data/futures/um/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{d}.zip"
        r = requests.get(url, stream=True)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            name=z.namelist()[0]
            df = pd.read_csv(z.open(name), header=None)
        df.columns=['t','o','h','l','c','v','ct','qv','nt','tb','tq','x']
        df['c']=df['c'].astype(float)
        df['v']=df['v'].astype(float)
        df.index=pd.to_datetime(df['t'], unit='ms')
        dfs.append(df[['c','v']])
    return pd.concat(dfs).drop_duplicates().last(5000)  # max 5000 rows

def build_features(df, prefix):
    bb=ta.bbands(df['c'], length=20, std=2)
    return pd.DataFrame({
      f"{prefix}_c": df['c'],
      f"{prefix}_v": df['v'],
      f"{prefix}_rsi": ta.rsi(df['c'], length=14),
      f"{prefix}_macd": ta.macd(df['c'], fast=12, slow=26, signal=9)['MACD_12_26_9'],
      f"{prefix}_ema200": ta.ema(df['c'], length=200),
      f"{prefix}_sma50": ta.sma(df['c'], length=50),
      f"{prefix}_bb_up": bb['BBU_20_2.0'],
      f"{prefix}_bb_mid": bb['BBM_20_2.0'],
      f"{prefix}_bb_low": bb['BBL_20_2.0']
    }).dropna()

# 1) Data sammeln
frames=[]
for sym in SYMBOLS:
  ints=[]
  for iv in INTERVALS:
    hist=load_historical(sym,iv)
    feat=build_features(hist, iv)
    ints.append(feat)
  merged=pd.concat(ints, axis=1, join='inner')
  merged['future']=merged['1h_c'].shift(-1)
  merged=merged.dropna()
  merged['label']=((merged['future']-merged['1h_c'])/merged['1h_c'])\
                   .apply(lambda x:2 if x>0.01 else (0 if x< -0.01 else 1))
  frames.append(merged)

# 2) Validate
if not frames:
  print("⚠️ Keine historischen Daten!")
  sys.exit(1)
data=pd.concat(frames).dropna()
if data.empty:
  print("⚠️ Daten leer nach Merge!")
  sys.exit(1)

# 3) Features/Labels
feat_cols=[c for c in data.columns if c not in ['future','label']]
X=data[feat_cols].values
y=tf.keras.utils.to_categorical(data['label'], num_classes=3)
scaler=StandardScaler().fit(X)
X_scaled=scaler.transform(X)
X_tr,X_val,y_tr,y_val=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

# 4) Modell bauen
model=tf.keras.Sequential([
  tf.keras.layers.Input(shape=(X_tr.shape[1],)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_tr,y_tr,validation_data=(X_val,y_val),epochs=30,batch_size=256)

# 5) Speicher
model.save('model.keras')
with open('scaler.pkl','wb') as f: pickle.dump(scaler,f)
# TFLite
conv=tf.lite.TFLiteConverter.from_keras_model(model)
tfl=conv.convert()
open('model.tflite','wb').write(tfl)
print("✅ train.py done: model.keras, scaler.pkl, model.tflite")
