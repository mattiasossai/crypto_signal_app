#!/usr/bin/env python3
# train/features.py

import os, glob, json
import pandas as pd
import numpy as np
import pandas_ta as ta

# ─── 1) Konfiguration ────────────────────────────────────────────────────────────
SYMBOLS   = ["BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","SOLUSDT","ENAUSDT"]
INTERVALS = ["1m","5m","15m","1h","4h"]
PARTS     = ["part1","part2"]

# ─── 2) Candle-Daten laden ───────────────────────────────────────────────────────
def load_candles(symbol: str, interval: str, hist_dir="historical") -> pd.DataFrame:
    """
    Lädt alle CSVs unter historical/<symbol>/<interval>/*.csv
    und gibt einen DataFrame mit index=timestamp und Spalten Open, High, Low, Close, Volume.
    """
    path = os.path.join(hist_dir, symbol, interval, "*.csv")
    files = sorted(glob.glob(path))
    if not files:
        raise FileNotFoundError(f"No candles for {symbol}/{interval}")
    df_list = []
    for fn in files:
        df = pd.read_csv(fn, parse_dates=["timestamp"], index_col="timestamp")
        df = df[["open","high","low","close","volume"]].rename(
            columns=str.capitalize
        )
        df_list.append(df)
    out = pd.concat(df_list)
    out["symbol"]   = symbol
    out["interval"] = interval
    return out

# ─── 3) Metrics laden ────────────────────────────────────────────────────────────
def load_metrics(metric: str, metr_dir="metrics") -> pd.DataFrame:
    """
    Lädt JSONs für 'open_interest' oder 'funding_rate' aus metrics/part*/<metric>/*.json
    und pivotiert zu symbol | date | <metric>.
    """
    recs = []
    for part in PARTS:
        base = os.path.join(metr_dir, part, metric, "*.json")
        for fn in glob.glob(base):
            sym, date_str = os.path.basename(fn).replace(".json","").split("_",1)
            date = pd.to_datetime(date_str)
            data = json.load(open(fn))
            if isinstance(data, list):
                vals = [float(d.get(metric,0)) for d in data if isinstance(d,dict)]
                val = np.mean(vals) if vals else np.nan
            else:
                val = float(data.get(metric, np.nan))
            recs.append({"symbol":sym, "date":date, "metric":metric, "value":val})
    df = pd.DataFrame(recs)
    return df.pivot(index=["symbol","date"], columns="metric", values="value").reset_index()

# ─── 4) Technische Indikatoren ──────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet:
      • EMA20/50/100/200, SMA20/50/100/200
      • VWAP, OBV, MFI14
      • ADX14, CCI14
      • RSI14, MACD12,26,9, BB20, ATR14
      • Supertrend10,3; WillR14; CMF20; Parabolic SAR; Ultimate Osc; TSI
      • Ratio-Features (z.B. EMA50/EMA200, VWAP/Close, ST/Close)
    """
    df = df.copy()
    c = df["Close"]; h = df["High"]; l = df["Low"]; v = df["Volume"]

    # Moving Averages
    for n in (20,50,100,200):
        df[f"EMA{n}"] = ta.ema(c, length=n)
        df[f"SMA{n}"] = ta.sma(c, length=n)

    # VWAP & Volumen-Indikatoren
    df["VWAP"]   = ta.vwap(h, l, c, v)
    df["OBV"]    = ta.obv(c, v)
    df["MFI14"]  = ta.mfi(h, l, c, v, length=14)

    # Trend-Stärke
    adx = ta.adx(h, l, c, length=14)
    df = pd.concat([df, adx], axis=1)       # ADX_14, DMP_14, DMN_14

    # Volatility & Momentum
    df["CCI14"]  = ta.cci(h, l, c, length=14)
    df["RSI14"]  = ta.rsi(c, length=14)
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)      # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    bb   = ta.bbands(c, length=20, std=2)
    df = pd.concat([df, bb], axis=1)        # BBM_20_2.0, BBUP_20_2.0, BBLW_20_2.0, BBP_20_2.0
    df["ATR14"]  = ta.atr(h, l, c, length=14)

    # Erweiterte Indikatoren
    st = ta.supertrend(h, l, c, length=10, multiplier=3.0)
    df["Supertrend"]     = st["SUPERT_10_3.0"]
    df["WillR14"]        = ta.willr(h, l, c, length=14)
    df["CMF20"]          = ta.cmf(h, l, c, v, length=20)
    psar = ta.psar(h, l, c, step=0.02, max_step=0.2)
    df["PSAR"]           = psar["PSARl_0.02_0.2"]
    df["UO"]             = ta.uo(h, l, c, s7=7, s14=14, s28=28)
    df["TSI"]            = ta.tsi(c, r=25, s=13)

    # Ratio-Features
    df["EMA50_200_Ratio"] = df["EMA50"] / df["EMA200"]
    df["VWAP_Close_Ratio"] = df["VWAP"] / df["Close"]
    df["ST_Close_Ratio"]   = df["Supertrend"] / df["Close"]

    return df

# ─── 5) Candlestick Patterns ────────────────────────────────────────────────────
def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erkennung via pandas_ta:
      • Bullish/Bearish Engulfing, Doji, Hammer, Shooting Star
    """
    df = df.copy()
    o,c,h,l = df["Open"], df["Close"], df["High"], df["Low"]

    df["Bull_Engulf"]  = ta.cdl_engulfing(o, h, l, c)
    df["Bear_Engulf"]  = -df["Bull_Engulf"]
    df["Doji"]         = ta.cdl_doji(o, h, l, c)
    df["Hammer"]       = ta.cdl_hammer(o, h, l, c)
    df["ShootingStar"] = ta.cdl_shootingstar(o, h, l, c)

    return df

# ─── 6) Metrics joinen ──────────────────────────────────────────────────────────
def join_metrics(df: pd.DataFrame, dfm: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt open_interest & funding_rate per symbol+date hinzu.
    Erwartet index timestamp und Spalte 'symbol'.
    """
    tmp = df.copy().reset_index().rename(columns={"index":"timestamp"})
    tmp["date"] = tmp["timestamp"].dt.normalize()
    out = tmp.merge(dfm, how="left", on=["symbol","date"])
    return out.set_index("timestamp")

# ─── 7) Labels generieren ───────────────────────────────────────────────────────
def generate_labels(df: pd.DataFrame,
                    horizon: int = 60,
                    threshold: float = 0.01) -> pd.DataFrame:
    """
    Erstellt:
      • label: 2=Long (>threshold), 1=Neutral, 0=Short (<-threshold)
      • TP  = Close + 2*ATR14
      • SL  = Close - 1*ATR14
    horizon = Minuten-Versatz auf 1m-Basis
    """
    df = df.copy().sort_index()
    df["future_close"] = df["Close"].shift(-horizon)
    df["return"]       = df["future_close"] / df["Close"] - 1
    df["label"]        = np.where(df["return"] >  threshold, 2,
                          np.where(df["return"] < -threshold, 0, 1))
    df["TP"]           = df["Close"] + 2 * df["ATR14"]
    df["SL"]           = df["Close"] - 1 * df["ATR14"]
    return df.dropna(subset=["label"])
