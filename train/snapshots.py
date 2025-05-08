#!/usr/bin/env python3
# train/snapshots.py

import glob, json, os
import pandas as pd

def load_snapshots(base_dir="snapshots/futures", symbols=None, top_n=10):
    """
    Lädt alle JSON-Snapshots aus snapshots/futures/<symbol>/*.json
    und berechnet pro Datei:
     - snap_bidN, snap_askN    (Summe der Top-N-Orders)
     - snap_spread             (asks0.price - bids0.price)
     - snap_mid                (Mittel aus bids0 und asks0)
     - snap_imbalance          ((bidN-askN)/(bidN+askN))
    Gibt DataFrame mit Spalten [symbol, date, ...features...] zurück.
    """
    symbols = symbols or []
    rows = []
    for sym in (symbols if symbols else os.listdir(base_dir)):
        path = os.path.join(base_dir, sym, "*.json")
        for fn in glob.glob(path):
            date = pd.to_datetime(os.path.basename(fn).rstrip(".json"))
            data = json.load(open(fn))
            bids = [(float(p), float(q)) for p,q in data.get("bids",[])]
            asks = [(float(p), float(q)) for p,q in data.get("asks",[])]
            if not bids or not asks:
                continue
            bid_sum = sum(q for _,q in bids[:top_n])
            ask_sum = sum(q for _,q in asks[:top_n])
            spread   = asks[0][0] - bids[0][0]
            mid      = (asks[0][0] + bids[0][0]) / 2
            imb      = (bid_sum - ask_sum) / (bid_sum + ask_sum) if (bid_sum+ask_sum)>0 else 0.0

            rows.append({
                "symbol": sym,
                "date": date.normalize(),
                f"snap_bid{top_n}": bid_sum,
                f"snap_ask{top_n}": ask_sum,
                "snap_spread": spread,
                "snap_mid": mid,
                "snap_imbalance": imb
            })
    return pd.DataFrame(rows)
