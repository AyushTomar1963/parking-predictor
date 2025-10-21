"""
Data utilities for loading, parsing datetime, resampling, etc.
"""
import pandas as pd
from datetime import datetime


def load_data(file_path):
    """Load CSV data and parse datetime columns."""
    df = pd.read_csv(file_path)
    return df


def parse_datetime(df, datetime_col):
    """Parse datetime column."""
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    return df


def resample_data(df, datetime_col, freq='H'):
    """Resample time series data."""
    df = df.set_index(datetime_col)
    df_resampled = df.resample(freq).mean()
    return df_resampled
