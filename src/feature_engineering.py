"""
Feature engineering: time features, lag features, weather, events, etc.
"""
import pandas as pd


def add_time_features(df, datetime_col):
    """Add time-based features like hour, day of week, month, etc."""
    df['hour'] = df[datetime_col].dt.hour
    df['day_of_week'] = df[datetime_col].dt.dayofweek
    df['day'] = df[datetime_col].dt.day
    df['month'] = df[datetime_col].dt.month
    df['year'] = df[datetime_col].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df


def add_lag_features(df, target_col, lags=[1, 2, 3, 6, 12, 24]):
    """Add lag features for time series prediction."""
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df


def add_rolling_features(df, target_col, windows=[3, 6, 12, 24]):
    """Add rolling window statistics."""
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
    return df
