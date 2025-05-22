#!/usr/bin/env python3
import os
import glob
import sys
import warnings
import logging
import argparse

import pandas as pd
import numpy as np
import pandas_ta as ta
import talib

# ─── Warnungen unterdrücken ─────────────────────────────────────────────────────
warnings.filterwarnings(
    "ignore",
    message="Converting to PeriodArray/Index representation will drop timezone information"
)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Settings ─────────────────────────────────────────────────────────────────
RAW_ROOT      = 'raw/klines'
FEATURES_ROOT = 'features/klines'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# ─── Indicators ────────────────────────────────────────────────────────────────
def add_indicators(df):
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    for n in (20, 50, 100, 200):
        df[f'EMA{n}'] = ta.ema(c, length=n)
        df[f'SMA{n}'] = ta.sma(c, length=n)
    df['VWAP']   = ta.vwap(h, l, c, v)
    df['OBV']    = ta.obv(c, v)
    df['MFI14']  = ta.mfi(h, l, c, v, length=14).astype(float)
    adx = ta.adx(h, l, c, length=14)
    if isinstance(adx, pd.DataFrame):
        for col in adx.columns:
            df[col] = adx[col]
    df['CCI14']  = ta.cci(h, l, c, length=14)
    df['RSI14']  = ta.rsi(c, length=14)
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if isinstance(macd, pd.DataFrame):
        for col in macd.columns:
            df[col] = macd[col]
    bb = ta.bbands(c, length=20, std=2)
    if isinstance(bb, pd.DataFrame):
        for col in bb.columns:
            df[col] = bb[col]
    df['ATR14'] = ta.atr(h, l, c, length=14)
    st = ta.supertrend(h, l, c, length=10, multiplier=3.0)
    if isinstance(st, pd.DataFrame):
        for col in st.columns:
            df[col] = st[col]
        if "SUPERT_10_3.0" in st:
            df['Supertrend'] = st["SUPERT_10_3.0"]
    else:
        df['Supertrend'] = np.nan
    df['WillR14'] = ta.willr(h, l, c, length=14)
    df['CMF20']   = ta.cmf(h, l, c, v, length=20)
    psar = ta.psar(h, l, c, step=0.02, max_step=0.2)
    if isinstance(psar, pd.DataFrame):
        for col in psar.columns:
            df[col] = psar[col]
        if "PSARl_0.02_0.2" in psar:
            df['PSAR'] = psar["PSARl_0.02_0.2"]
    else:
        df['PSAR'] = np.nan
    df['UO'] = ta.uo(h, l, c, s7=7, s14=14, s28=28)
    tsi = ta.tsi(c, r=25, s=13)
    if isinstance(tsi, pd.DataFrame):
        for col in tsi.columns:
            df[col] = tsi[col]
        if "TSI_25_13" in tsi:
            df['TSI'] = tsi["TSI_25_13"]
    else:
        df['TSI'] = tsi
    df['EMA50_200_Ratio'] = df['EMA50'] / df['EMA200']
    df['VWAP_Close_Ratio'] = df['VWAP'] / df['Close']
    df['ST_Close_Ratio']   = df['Supertrend'] / df['Close']
    return df

# ─── Fibonacci Retracement ────────────────────────────────────────────────────
def add_fibonacci_levels(df, lookbacks=(20,50,100)):
    for lb in lookbacks:
        highs = df['High'].rolling(lb, min_periods=1).max()
        lows  = df['Low'].rolling(lb, min_periods=1).min()
        df[f'Fibo_236_{lb}'] = highs - (highs - lows) * 0.236
        df[f'Fibo_382_{lb}'] = highs - (highs - lows) * 0.382
        df[f'Fibo_500_{lb}'] = highs - (highs - lows) * 0.5
        df[f'Fibo_618_{lb}'] = highs - (highs - lows) * 0.618
        df[f'Fibo_786_{lb}'] = highs - (highs - lows) * 0.786
    return df

# ─── Candlestick Patterns via TA-Lib ───────────────────────────────────────────
def add_candlestick_patterns(df):
    o = df["Open"].values; h = df["High"].values
    l = df["Low"].values;  c = df["Close"].values
    df["Bull_Engulf"]  = talib.CDLENGULFING(o,h,l,c)
    df["Bear_Engulf"]  = -talib.CDLENGULFING(o,h,l,c)
    df["Doji"]         = talib.CDLDOJI(o,h,l,c)
    df["Hammer"]       = talib.CDLHAMMER(o,h,l,c)
    df["ShootingStar"] = talib.CDLSHOOTINGSTAR(o,h,l,c)
    return df

# ─── CSV Loader mit Header-Check ───────────────────────────────────────────────
def concat_csvs(symbol, interval):
    path = os.path.join(RAW_ROOT, symbol)
    files = sorted(glob.glob(os.path.join(path, f"{symbol}-{interval}-*.csv")))
    dfs = []
    for fn in files:
        try:
            df = pd.read_csv(fn, header=0)
            if list(df.columns[:6]) == ["open_time","open","high","low","close","volume"]:
                df = df.iloc[: , :6]
                df.columns = ["timestamp","open","high","low","close","volume"]
            else:
                df = pd.read_csv(fn, header=None).iloc[: , :6]
                df.columns = ["timestamp","open","high","low","close","volume"]
            dfs.append(df)
        except Exception as e:
            logging.warning(f"Failed load {fn}: {e}")
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
    df = df.sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')
    df.rename(columns=str.capitalize, inplace=True)
    return df

# ─── Hauptprozess ─────────────────────────────────────────────────────────────
def process_symbol_interval(symbol, interval):
    df = concat_csvs(symbol, interval)
    if df.empty:
        logging.error(f"No data for {symbol}-{interval}, abort.")
        sys.exit(1)

    df = add_indicators(df)
    df = add_fibonacci_levels(df, lookbacks=(20,50,100))
    df = add_candlestick_patterns(df)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            c if isinstance(c, str) else "_".join(c).strip()
            for c in df.columns
        ]

    df['symbol']         = symbol
    df['interval']       = interval
    df['data_available'] = True

    out_dir = os.path.join(FEATURES_ROOT, symbol, interval)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"features-{symbol}-{interval}.parquet")
    df.to_parquet(out_file, compression='snappy')
    logging.info(f"Saved {out_file} ({len(df)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol',   required=True)
    parser.add_argument('--interval', required=True)
    args = parser.parse_args()
    process_symbol_interval(args.symbol, args.interval)
