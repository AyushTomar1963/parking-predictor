# src/preprocessing/time_series.py
from typing import Optional
import pandas as pd
import numpy as np


def ensure_datetime_index(df: pd.DataFrame, ts_col: str = "LastUpdated") -> pd.DataFrame:
    """
    Ensure DataFrame has a datetime index named 'LastUpdated'.
    If ts_col exists, convert and set as index; else assume index already datetime.
    """
    df = df.copy()
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.set_index(ts_col)
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("Index is not datetime and 'LastUpdated' column not found/parsable.")
    df = df.sort_index()
    return df


def process_lot_data(
    lot_df: pd.DataFrame,
    ts_col: str = "LastUpdated",
    occupancy_col: str = "Occupancy",
    resample_rule: str = "H",
    interp_method: str = "time",
    dropna: bool = True,
    default_capacity: int = 600,
) -> pd.DataFrame:
    """
    Process a single lot's raw DataFrame into a regular time-indexed DataFrame
    with occupancy interpolated and basic calendar features added.

    Steps:
      - ensure datetime index
      - resample (aggregate by mean) to regular frequency (default hourly)
      - interpolate missing values
      - add calendar features: hour_of_day, day_of_week, is_weekend
      - add capacity column (preserves from raw data or uses default)

    Args:
        lot_df: Raw parking lot DataFrame
        ts_col: Timestamp column name
        occupancy_col: Occupancy column name
        resample_rule: Resampling frequency (default: 'H' for hourly)
        interp_method: Interpolation method
        dropna: Whether to drop NaN rows
        default_capacity: Default capacity if not present in data

    Returns:
      df_processed: DataFrame indexed by timestamp with Occupancy, calendar features, and Capacity.
    """
    df = lot_df.copy()
    df = ensure_datetime_index(df, ts_col=ts_col)

    # If occupancy column missing, raise
    if occupancy_col not in df.columns:
        raise KeyError(f"Occupancy column '{occupancy_col}' not found in DataFrame.")

    # Check if Capacity column exists in raw data
    has_capacity = "Capacity" in df.columns
    if has_capacity:
        capacity_series = df["Capacity"].resample(resample_rule).first().ffill()

    # Resample + aggregate occupancy
    s = df[occupancy_col].resample(resample_rule).mean()

    # Interpolate missing values (time-based) and forward/backward fill small gaps
    s_interp = s.interpolate(method=interp_method).ffill().bfill()

    df_processed = s_interp.to_frame(name=occupancy_col)

    # Calendar features
    df_processed["hour_of_day"] = df_processed.index.hour
    df_processed["day_of_week"] = df_processed.index.dayofweek
    df_processed["is_weekend"] = (df_processed["day_of_week"] >= 5).astype(int)

    # Add Capacity column
    if has_capacity:
        df_processed["Capacity"] = capacity_series
    else:
        df_processed["Capacity"] = default_capacity
        print(f"   ℹ️  No Capacity column found in raw data, using default: {default_capacity}")

    if dropna:
        df_processed = df_processed.dropna()

    return df_processed


def create_lag_features(df: pd.DataFrame, lags: list = None, target_col: str = "Occupancy") -> pd.DataFrame:
    """
    Create lagged features for time series prediction.
    
    Args:
        df: DataFrame with target column
        lags: List of lag periods to create (default: [1, 2, 3, 24, 48])
        target_col: Name of target column to create lags from
    
    Returns:
        DataFrame with lag features added, NaN rows dropped
    """
    if lags is None:
        lags = [1, 2, 3, 24, 48]
    
    df = df.copy()
    
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    return df.dropna()