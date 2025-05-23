import os
import logging
import pandas as pd

def init_logger(name: str) -> logging.Logger:
    """
    Erzeugt einen einfachen Stream-Logger (INFO-Level).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
        logger.addHandler(h)
    return logger

def save_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Speichert das DataFrame als snappy-komprimiertes Parquet.
    Legt dabei fehlende Ordner automatisch an.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy")
