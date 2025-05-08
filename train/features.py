#!/usr/bin/env python3
# train/features.py

import os
import glob
import json
import pandas as pd
import numpy as np
import pandas_ta as ta

# ---------- 1) Load Functions ----------

def load_candles(symbol: str, interval: str) -> pd.DataFrame:
    path = f"historical/{symbol}/{interval}/*.csv"
    files = sorted(glob.glob(path))
    df_list = []
    for f in files:
        tmp = pd.read_csv(f, parse_dates=['timestamp'], index_col='timestamp')
        df_list.append(tmp[['open','high','low','close','volume']].rename(
            columns=str.capitalize
        ))
    df = pd.concat(df_list)
    df['symbol'] = symbol
    df['interval'] = interval
    return df


def load_metrics(metric: str) -> pd.DataFrame:
    recs = []
    for part in ['part1','part2']:
        base = f"metrics/{part}/{metric}/*.json"
        for fn in glob.glob(base):
            sym, date = os.path.basename(fn).replace('.json','').split('_',1)
            data = json.load(open(fn))
            # Flatten list of dicts or single dict
            if isinstance(data, list):
                vals = [float(x.get(metric,0)) for x in data if isinstance(x,dict)]
                val = np.mean(vals) if vals else np.nan
            else:
                val = float(data.get(metric, np.nan))
            recs.append({
                'symbol': sym,
                'date': pd.to_datetime(date),
                metric: val
            })
    df = pd.DataFrame(recs)
    return df.pivot(index=['symbol','date'], columns=[], values=[metric]).reset_index()

# ---------- 2) Technical Indicators ----------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # EMA/SMA
    df['EMA50'] = ta.ema(df['Close'], length=50)
    df['EMA200'] = ta.ema(df['Close'], length=200)
    df['SMA50'] = ta.sma(df['Close'], length=50)

    # RSI
    df['RSI14'] = ta.rsi(df['Close'], length=14)

    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)

    # Bollinger Bands
    bb = ta.bbands(df['Close'], length=20, std=2)
    df = pd.concat([df, bb], axis=1)

    # ATR
    df['ATR14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # Stochastic %K/%D
    sto = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
    df['Sto_K'] = sto['STOCHk_14_3_3']
    df['Sto_D'] = sto['STOCHd_14_3_3']

    # Candlestick Patterns (Engulfing, Doji)
    df['Bull_Engulf'] = ta.cdl_engulfing(df['Open'], df['High'], df['Low'], df['Close'])
    df['Bear_Engulf'] = df['Bull_Engulf'] * -1
    df['Doji'] = ta.cdl_doji(df['Open'], df['High'], df['Low'], df['Close'])

    return df

# ---------- 3) Metrics Joiner ----------

def join_metrics(df: pd.DataFrame, oi: pd.DataFrame, fr: pd.DataFrame) -> pd.DataFrame:
    # Merge on symbol + date
    df = df.copy()
    df['date'] = df.index.normalize()
    df = df.reset_index().merge(
        oi, how='left', left_on=['symbol','date'], right_on=['symbol','date']
    ).merge(
        fr, how='left', left_on=['symbol','date'], right_on=['symbol','date']
    )
    return df.set_index('timestamp')

# ---------- 4) Labels ----------

def generate_labels(df: pd.DataFrame, horizon: int=60) -> pd.DataFrame:
    df = df.copy().sort_index()
    df['future_close'] = df['Close'].shift(-horizon)
    df['return'] = df['future_close'] / df['Close'] - 1
    df['label'] = np.where(df['return']>0.01, 2,
                     np.where(df['return']<-0.01, 0, 1))
    # TP/SL
    df['TP'] = df['Close'] + 2 * df['ATR14']
    df['SL'] = df['Close'] - 1 * df['ATR14']
    return df.dropna()
