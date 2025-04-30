import os, pickle, asyncio, requests
import pandas as pd, numpy as np, tensorflow as tf
import pandas_ta as ta
from fastapi import FastAPI

# Lade TFLite & Scaler
inter = tf.lite.Interpreter(model_path="../model.tflite")
inter.allocate_tensors()
inp=inter.get_input_details()[0]
oup=inter.get_output_details()[0]
with open("../scaler.pkl","rb") as f: scaler=pickle.load(f)

# Konfig
SYMS = ['BTCUSDT','ETHUSDT',…,'TRUMPUSDT']
IFS  = ['1m','5m','15m','1h','4h']
PROXY_URL = "https://<your-worker-subdomain>.workers.dev"

# Speicher
windows={sym:{iv:pd.DataFrame(columns=['c']) for iv in IFS} for sym in SYMS}
last_signals=[]

def fetch_latest(sym,iv):
    url = f"{PROXY_URL}?symbol={sym}&interval={iv}"
    df = pd.DataFrame(requests.get(url).json(),
      columns=['ot','o','h','l','c','v','ct','qv','nt','tb','tq','x'])
    df['c']=df['c'].astype(float)
    return df[['c','v']].iloc[-200:]  # WINDOW

def compute_and_infer(sym):
    feats=[]
    for iv in IFS:
        df=windows[sym][iv]
        feats+=[
          df['c'].iloc[-1],
          ta.rsi(df['c'],14).iloc[-1],
          ta.macd(df['c'],12,26,9)['MACD_12_26_9'].iloc[-1],
          ta.ema(df['c'],200).iloc[-1],
          ta.sma(df['c'],50).iloc[-1]
        ]
    X=np.array(feats,dtype=np.float32).reshape(1,-1)
    X=scaler.transform(X)
    inter.set_tensor(inp['index'],X); inter.invoke()
    p=inter.get_tensor(oup['index'])[0]
    sig="NEU"; 
    if p[2]>0.6: sig="LONG"
    if p[0]>0.6: sig="SHORT"
    entry=windows[sym]['1m']['c'].iloc[-1]
    tp=entry*(1.01 if sig=="LONG" else 0.99)
    sl=entry*(0.99 if sig=="LONG" else 1.01)
    rec={"symbol":sym,"signal":sig,"prob":p.tolist(),"entry":entry,"tp":tp,"sl":sl}
    last_signals.append(rec)
    return rec

app=FastAPI()

@app.on_event("startup")
async def startup():
    asyncio.create_task(poller())

async def poller():
    while True:
        for sym in SYMS:
            for iv in IFS:
                df=fetch_latest(sym,iv)
                windows[sym][iv]=df
            rec=compute_and_infer(sym)
            print(rec)
        await asyncio.sleep(60)  # jede Minute

@app.get("/signals")
def get_signals():
    return last_signals[-50:]

# Run mit: uvicorn service.real_time_service:app --host 0.0.0.0 --port 8000
