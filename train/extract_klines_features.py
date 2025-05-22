#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import argparse

def process_symbol_interval(symbol, interval):
    # ... DEIN KOMPLETTER CODE wie oben ...
    print(f"Processing {symbol} {interval}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, required=True)
    parser.add_argument('--interval', type=str, required=True)
    args = parser.parse_args()

    process_symbol_interval(args.symbol, args.interval)

# ─── Einstellungen ───────────────────────
RAW_ROOT      = 'raw/klines'
FEATURES_ROOT = 'features/klines'
SYMBOLS       = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ENAUSDT"]
INTERVALS     = ["1m", "5m", "15m", "1h", "4h"]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# ─── Technische Indikatoren (ohne rollierende Fenster) ───
def add_indicators(df):
    c = df['Close']; h = df['High']; l = df['Low']; v = df['Volume']

    # Moving Averages
    for n in (20, 50, 100, 200):
        df[f'EMA{n}'] = ta.ema(c, length=n)
        df[f'SMA{n}'] = ta.sma(c, length=n)

    # VWAP, OBV, MFI
    df['VWAP'] = ta.vwap(h, l, c, v)
    df['OBV'] = ta.obv(c, v)
    df['MFI14'] = ta.mfi(h, l, c, v, length=14)

    # Trendstärke/Volatilität/Momentum
    adx = ta.adx(h, l, c, length=14)
    df = pd.concat([df, adx], axis=1)
    df['CCI14'] = ta.cci(h, l, c, length=14)
    df['RSI14'] = ta.rsi(c, length=14)
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    bb = ta.bbands(c, length=20, std=2)
    df = pd.concat([df, bb], axis=1)
    df['ATR14'] = ta.atr(h, l, c, length=14)

    # Weitere Indikatoren
    st = ta.supertrend(h, l, c, length=10, multiplier=3.0)
    df['Supertrend'] = st["SUPERT_10_3.0"] if "SUPERT_10_3.0" in st else np.nan
    df['WillR14'] = ta.willr(h, l, c, length=14)
    df['CMF20'] = ta.cmf(h, l, c, v, length=20)
    psar = ta.psar(h, l, c, step=0.02, max_step=0.2)
    df['PSAR'] = psar["PSARl_0.02_0.2"] if "PSARl_0.02_0.2" in psar else np.nan
    df['UO'] = ta.uo(h, l, c, s7=7, s14=14, s28=28)
    df['TSI'] = ta.tsi(c, r=25, s=13)

    # Ratio-Features
    df['EMA50_200_Ratio'] = df['EMA50'] / df['EMA200']
    df['VWAP_Close_Ratio'] = df['VWAP'] / df['Close']
    df['ST_Close_Ratio'] = df['Supertrend'] / df['Close']

    return df

# ─── Candlestick Patterns ───
def add_candlestick_patterns(df):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    df["Bull_Engulf"] = ta.cdl_engulfing(o, h, l, c)
    df["Bear_Engulf"] = -df["Bull_Engulf"]
    df["Doji"] = ta.cdl_doji(o, h, l, c)
    df["Hammer"] = ta.cdl_hammer(o, h, l, c)
    df["ShootingStar"] = ta.cdl_shootingstar(o, h, l, c)
    return df

# ─── Hilfsfunktion: CSVs zusammenführen ───
def concat_csvs(csv_files):
    dfs = []
    for f in csv_files:
        try:
            # Auto-Header-Erkennung
            df = pd.read_csv(f)
            # Falls keine Header oder doppelte Header: prüfen
            if not set(['timestamp', 'open', 'high', 'low', 'close', 'volume']).issubset(df.columns):
                df = pd.read_csv(f, header=None)
                df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
            dfs.append(df)
        except Exception as e:
            logging.warning(f"Fehler beim Laden {f}: {e}")
    if not dfs:
        return pd.DataFrame()  # Leeres DataFrame
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
    df = df.sort_values('timestamp').drop_duplicates('timestamp')
    df = df.set_index('timestamp')
    df.rename(columns=str.capitalize, inplace=True)  # Open, High, Low, Close, Volume
    return df

# ─── Hauptprozess ───
def process_symbol_interval(symbol, interval):
    raw_dir = os.path.join(RAW_ROOT, symbol, interval)
    feature_dir = os.path.join(FEATURES_ROOT, symbol, interval)
    os.makedirs(feature_dir, exist_ok=True)
    out_file = os.path.join(feature_dir, f'features-{symbol}-{interval}.parquet')

    csv_files = sorted(glob.glob(os.path.join(raw_dir, '*.csv')))
    if not csv_files:
        logging.error(f"Keine CSVs für {symbol}-{interval} in {raw_dir}. Featurefile wird nicht erzeugt.")
        return

    df = concat_csvs(csv_files)
    if df.empty:
        logging.error(f"Leeres DataFrame für {symbol}-{interval}. Featurefile wird nicht erzeugt.")
        return

    df = add_indicators(df)
    df = add_candlestick_patterns(df)
    df['symbol'] = symbol
    df['interval'] = interval
    df['data_available'] = True

    df.to_parquet(out_file, compression='snappy')
    logging.info(f"Feature-Parquet gespeichert: {out_file} ({len(df)} Zeilen)")

if __name__ == '__main__':
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            process_symbol_interval(symbol, interval)
