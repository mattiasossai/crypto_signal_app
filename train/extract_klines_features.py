#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import argparse

RAW_ROOT      = 'raw/klines'
FEATURES_ROOT = 'features/klines'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def add_indicators(df):
    c = df['Close']; h = df['High']; l = df['Low']; v = df['Volume']
    for n in (20, 50, 100, 200):
        df[f'EMA{n}'] = ta.ema(c, length=n)
        df[f'SMA{n}'] = ta.sma(c, length=n)
    df['VWAP'] = ta.vwap(h, l, c, v)
    df['OBV'] = ta.obv(c, v)
    df['MFI14'] = ta.mfi(h, l, c, v, length=14).astype(float)

    adx = ta.adx(h, l, c, length=14)
    if isinstance(adx, pd.DataFrame):
        for col in adx.columns:
            df[col] = adx[col]

    df['CCI14'] = ta.cci(h, l, c, length=14)
    df['RSI14'] = ta.rsi(c, length=14)

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
    df['CMF20'] = ta.cmf(h, l, c, v, length=20)

    psar = ta.psar(h, l, c, step=0.02, max_step=0.2)
    if isinstance(psar, pd.DataFrame):
        for col in psar.columns:
            df[col] = psar[col]
        if "PSARl_0.02_0.2" in psar:
            df['PSAR'] = psar["PSARl_0.02_0.2"]
    else:
        df['PSAR'] = np.nan

    uo = ta.uo(h, l, c, s7=7, s14=14, s28=28)
    df['UO'] = uo

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
    df['ST_Close_Ratio'] = df['Supertrend'] / df['Close']
    return df

def add_fibonacci_levels(df, lookbacks=(20, 50, 100)):
    for lb in lookbacks:
        highs = df['High'].rolling(window=lb, min_periods=1).max()
        lows = df['Low'].rolling(window=lb, min_periods=1).min()
        df[f'Fibo_236_{lb}'] = highs - (highs - lows) * 0.236
        df[f'Fibo_382_{lb}'] = highs - (highs - lows) * 0.382
        df[f'Fibo_500_{lb}'] = highs - (highs - lows) * 0.5
        df[f'Fibo_618_{lb}'] = highs - (highs - lows) * 0.618
        df[f'Fibo_786_{lb}'] = highs - (highs - lows) * 0.786
    return df

def add_candlestick_patterns(df):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    df["Bull_Engulf"] = ta.cdl_engulfing(o, h, l, c)
    df["Bear_Engulf"] = -df["Bull_Engulf"]
    df["Doji"] = ta.cdl_doji(o, h, l, c)
    df["Hammer"] = ta.cdl_hammer(o, h, l, c)
    df["ShootingStar"] = ta.cdl_shootingstar(o, h, l, c)
    return df

def concat_csvs(symbol, interval):
    dir_path = os.path.join(RAW_ROOT, symbol)
    pattern = f"{symbol}-{interval}-*.csv"
    csv_files = sorted(glob.glob(os.path.join(dir_path, pattern)))
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, header=0)
            expected_cols = ["open_time", "open", "high", "low", "close", "volume"]
            if list(df.columns[:6]) == expected_cols:
                df = df.iloc[:, :6]
                df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
            else:
                df = pd.read_csv(f, header=None)
                df = df.iloc[:, :6]
                df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
            dfs.append(df)
        except Exception as e:
            logging.warning(f"Fehler beim Laden {f}: {e}")
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
    df = df.sort_values('timestamp').drop_duplicates('timestamp')
    df = df.set_index('timestamp')
    df.rename(columns=str.capitalize, inplace=True)
    return df

def process_symbol_interval(symbol, interval):
    feature_dir = os.path.join(FEATURES_ROOT, symbol, interval)
    os.makedirs(feature_dir, exist_ok=True)
    out_file = os.path.join(feature_dir, f'features-{symbol}-{interval}.parquet')

    df = concat_csvs(symbol, interval)
    if df.empty:
        logging.error(f"Keine CSVs für {symbol}-{interval} in {os.path.join(RAW_ROOT, symbol)}. Featurefile wird nicht erzeugt.")
        return

    df = add_indicators(df)
    df = add_fibonacci_levels(df, lookbacks=(20, 50, 100))
    df = add_candlestick_patterns(df)
    df['symbol'] = symbol
    df['interval'] = interval
    df['data_available'] = True

    df.to_parquet(out_file, compression='snappy')
    logging.info(f"Feature-Parquet gespeichert: {out_file} ({len(df)} Zeilen)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, required=True)
    parser.add_argument('--interval', type=str, required=True)
    args = parser.parse_args()

    process_symbol_interval(args.symbol, args.interval)
