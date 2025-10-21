# src/preprocessing/data_loader.py
from typing import Optional
import pandas as pd
import os


def load_csv(path: str, parse_dates: Optional[list] = None) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame and parse dates.
    Raises FileNotFoundError if path missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    parse_dates = parse_dates or ["LastUpdated"]
    df = pd.read_csv(path, parse_dates=parse_dates)
    return df


def load_lot(df: pd.DataFrame, lot_col: str = "SystemCodeNumber", lot_id=None) -> pd.DataFrame:
    """
    Return rows for a single lot id. If lot_id is None, returns the df unchanged.
    """
    if lot_id is None:
        return df.copy()
    if lot_col not in df.columns:
        raise KeyError(f"Lot column '{lot_col}' not found in DataFrame.")
    return df[df[lot_col] == lot_id].copy()
