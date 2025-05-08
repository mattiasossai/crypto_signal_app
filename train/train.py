#!/usr/bin/env python3
# train/train.py

import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, layers
from features import load_candles, load_metrics, add_indicators, join_metrics, generate_labels

# 1) Dataset aufbauen
rows = []
for symbol in ["BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","SOLUSDT","ENAUSDT"]:
    for interval in ["1m","5m","15m","1h","4h"]:
        df = load_candles(symbol, interval)
        df = add_indicators(df)
        oi = load_metrics('open_interest')
        fr = load_metrics('funding_rate')
        df = join_metrics(df, oi, fr)
        df = generate_labels(df)
        rows.append(df)

df_all = pd.concat(rows)

# 2) Features / Labels splitten
X = df_all.drop(columns=['future_close','return','label'])
y = df_all['label']

# 3) Train/Test Split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4) Skalierung
scaler = StandardScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr)
Xte_s = scaler.transform(Xte)

# 5) Modell definieren
model = Sequential([layers.Input(Xtr_s.shape[1]),
                    layers.Dense(128, activation='relu'),
                    layers.Dropout(0.3),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(3, activation='softmax')])
model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])

# 6) Trainieren
model.fit(Xtr_s, ytr, validation_data=(Xte_s, yte), epochs=20, batch_size=256)

# 7) Export
import joblib, tensorflow as tf
joblib.dump(scaler, 'scaler.pkl')
ct = tf.lite.TFLiteConverter.from_keras_model(model)
open('model.tflite','wb').write(ct.convert())
