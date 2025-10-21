# src/prediction/direct_predictor.py
"""
Direct multi-step forecasting:
- train_direct_models: trains one model per horizon (1..H)
- predict_direct: uses those models to predict for each horizon
- DirectForecaster: small wrapper over the dict-of-models approach
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


def train_direct_models(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    max_horizon: int,
    params: Dict[str, Any] = None,
    num_boost_round: int = 200
) -> Dict[int, Any]:
    """
    Train a separate model for each horizon h in 1..max_horizon.
    Returns dict {h: model}.
    """
    import lightgbm as lgb

    params = params or {
        'objective': 'regression',
        'metric': 'l2',
        'verbosity': -1,
        'seed': 42
    }

    models: Dict[int, Any] = {}

    for h in range(1, max_horizon + 1):
        df_h = df.copy()
        df_h['target_h'] = df_h[target].shift(-h)
        df_h = df_h.dropna(subset=features + ['target_h'])
        if df_h.empty:
            # no training data for this horizon
            continue

        X = df_h[features]
        y = df_h['target_h']

        train_data = lgb.Dataset(X, label=y)
        mdl = lgb.train(params, train_data, num_boost_round=num_boost_round)
        models[h] = mdl

    return models


def predict_direct(models: Dict[int, Any], df_block: pd.DataFrame, features: List[str]) -> Dict[int, pd.Series]:
    """
    Predict with direct models. Returns dict {h: pd.Series indexed by df_block.index}.
    """
    preds: Dict[int, pd.Series] = {}

    # validate features are present once
    missing = [c for c in features if c not in df_block.columns]
    if missing:
        raise ValueError(f"predict_direct: missing feature columns: {missing}")

    # drop rows with NaNs in required features
    X = df_block[features].dropna()
    if X.empty:
        # nothing to predict
        return {}

    for h, model in models.items():
        arr = model.predict(X)
        preds[h] = pd.Series(arr, index=X.index)

    return preds


class DirectForecaster:
    def __init__(self, models: Dict[int, Any], features: List[str], max_horizon: int):
        self.models = models
        self.features = features
        self.max_horizon = max_horizon

    def forecast(self, df: pd.DataFrame) -> Dict[int, pd.Series]:
        return predict_direct(self.models, df, self.features)

    def forecast_horizon(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        if horizon not in self.models:
            raise ValueError(f"Horizon {horizon} not available")
        X = df[self.features].dropna()
        arr = self.models[horizon].predict(X)
        return pd.Series(arr, index=X.index)
