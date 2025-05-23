import os
import logging
import pandas as pd

def init_logger(name: str) -> logging.Logger:
    """
    Initialisiert einen Stream-Logger für Konsolenausgabe.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger

def save_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Speichert das DataFrame als Parquet (snappy-komprimiert) unter `path`.
    Legt fehlende Ordner automatisch an.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # engine pyarrow vorausgesetzt
    df.to_parquet(path, engine="pyarrow", compression="snappy")
