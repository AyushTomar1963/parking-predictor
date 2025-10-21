"""
recursive_predictor.py

Implements the recursive strategy for multi-step time series prediction.
Extracted from Main.ipynb - recursive forecasting logic.

In recursive prediction, we predict one step ahead, then use that prediction
as input for the next step, continuing iteratively for the entire forecast horizon.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any


def predict_recursive(model, df_start: pd.DataFrame, features: List[str], max_horizon: int) -> pd.Series:
    """
    Generate multi-step forecasts using recursive strategy.
    
    This function iteratively predicts one step ahead, then uses that prediction
    as input (updating lag features) for the next step.
    
    Args:
        model: Trained model with a predict() method (e.g., LightGBM, XGBoost)
        df_start: DataFrame containing the most recent observations with all features
                 Must include lag features and calendar features
        features: List of feature names to use for prediction
        max_horizon: Number of hours/steps to forecast into the future
        
    Returns:
        pd.Series: Forecasted values indexed by future timestamps
        
    Example:
        >>> from src.models import LightGBMModel
        >>> model = LightGBMModel()
        >>> model.fit(X_train, y_train)
        >>> 
        >>> # Prepare seed data with last 48 hours
        >>> seed_df = df.iloc[-48:].copy()
        >>> 
        >>> # Predict next 24 hours recursively
        >>> predictions = predict_recursive(
        ...     model, seed_df, 
        ...     features=['lag_1', 'lag_2', 'lag_24', 'hour_of_day', 'day_of_week'],
        ...     max_horizon=24
        ... )
        
    Note:
        - Requires lag features in the input DataFrame
        - Updates lag features dynamically with predictions
        - Calendar features (hour, day_of_week) computed from timestamp
    
    Based on: Main.ipynb Cell #18 (predict_recursive function)
    """
    df_work = df_start.copy()
    preds = []
    
    # Iteratively predict each step
    for h in range(1, max_horizon + 1):
        # Get features from the latest row
        X = df_work[features].iloc[[-1]]
        
        # Predict next value
        yhat = float(model.predict(X)[0])
        preds.append(yhat)
        
        # Prepare next row by updating lag features and calendar features
        next_index = df_work.index[-1] + pd.Timedelta(hours=1)
        new_row = {}
        
        # Update lag features by shifting
        for col in df_work.columns:
            if col.startswith('lag_'):
                lag_k = int(col.split('_')[1])
                if lag_k == 1:
                    # lag_1 becomes the current prediction
                    new_row[col] = yhat
                else:
                    # lag_k becomes previous lag_{k-1}
                    prev_lag = f'lag_{lag_k - 1}'
                    new_row[col] = df_work.iloc[-1].get(prev_lag, np.nan)
            elif col == 'Occupancy':
                new_row[col] = yhat
        
        # Compute calendar features for next timestamp
        new_row['hour_of_day'] = next_index.hour
        new_row['day_of_week'] = next_index.dayofweek
        new_row['is_weekend'] = int(next_index.dayofweek >= 5)
        
        # Create DataFrame row and append
        new_df_row = pd.DataFrame(new_row, index=[next_index])
        
        # Ensure columns align
        for c in df_work.columns:
            if c not in new_df_row.columns:
                new_df_row[c] = np.nan
        new_df_row = new_df_row[df_work.columns]
        
        # Concatenate to working dataframe
        df_work = pd.concat([df_work, new_df_row])
    
    # Create proper index for predictions
    future_index = [df_start.index[-1] + pd.Timedelta(hours=i) for i in range(1, max_horizon + 1)]
    
    return pd.Series(preds, index=future_index)


def train_recursive_model(df: pd.DataFrame, features: List[str], target: str, 
                         params: Dict[str, Any] = None, num_boost_round: int = 200):
    """
    Train a single 1-step ahead model for recursive forecasting.
    
    In recursive strategy, we only need one model that predicts the next step.
    This model is then used repeatedly, feeding predictions back as inputs.
    
    Args:
        df: DataFrame with features and target
        features: List of feature column names
        target: Target column name (e.g., 'Occupancy')
        params: LightGBM parameters dict (optional)
        num_boost_round: Number of boosting rounds
        
    Returns:
        Trained LightGBM model (booster object)
        
    Example:
        >>> # Train recursive model
        >>> model = train_recursive_model(
        ...     train_df, 
        ...     features=['lag_1', 'lag_2', 'lag_24', 'hour_of_day'],
        ...     target='Occupancy',
        ...     num_boost_round=200
        ... )
        >>> 
        >>> # Use for recursive prediction
        >>> predictions = predict_recursive(model, seed_df, features, max_horizon=24)
    
    Based on: Main.ipynb Cell #18 (train_recursive_model function)
    """
    import lightgbm as lgb
    
    params = params or {
        'objective': 'regression',
        'metric': 'l2',
        'verbosity': -1,
        'seed': 42
    }
    
    df_r = df.dropna()
    X = df_r[features]
    y = df_r[target]
    
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=num_boost_round)
    
    return model


class RecursiveForecaster:
    """
    Wrapper class for recursive forecasting strategy.
    
    This class encapsulates the recursive prediction logic, making it easier
    to use in production pipelines.
    
    Example:
        >>> from src.models import LightGBMModel
        >>> from src.prediction.recursive_predictor import RecursiveForecaster
        >>> 
        >>> # Train model
        >>> model = LightGBMModel()
        >>> model.fit(X_train, y_train)
        >>> 
        >>> # Create forecaster
        >>> forecaster = RecursiveForecaster(
        ...     model=model,
        ...     features=['lag_1', 'lag_2', 'lag_24', 'hour_of_day', 'day_of_week']
        ... )
        >>> 
        >>> # Generate forecast
        >>> predictions = forecaster.forecast(seed_data, horizon=24)
    """
    
    def __init__(self, model, features: List[str]):
        """
        Initialize recursive forecaster.
        
        Args:
            model: Trained model with predict() method
            features: List of feature names used for prediction
        """
        self.model = model
        self.features = features
    
    def forecast(self, seed_data: pd.DataFrame, horizon: int) -> pd.Series:
        """
        Generate multi-step forecast using recursive strategy.
        
        Args:
            seed_data: DataFrame with recent observations (must contain all features)
            horizon: Number of steps to forecast
            
        Returns:
            pd.Series: Forecasted values
        """
        return predict_recursive(self.model, seed_data, self.features, horizon)
    
    def forecast_with_confidence(self, seed_data: pd.DataFrame, horizon: int, 
                                 n_simulations: int = 100) -> Dict[str, pd.Series]:
        """
        Generate forecast with confidence intervals using Monte Carlo simulation.
        
        Args:
            seed_data: DataFrame with recent observations
            horizon: Number of steps to forecast
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with 'mean', 'lower', 'upper' prediction bands
            
        Note:
            This is a simple implementation. For production, consider using
            quantile regression or more sophisticated uncertainty quantification.
        """
        predictions = []
        
        for _ in range(n_simulations):
            # Add small random noise to seed data to create variations
            noisy_seed = seed_data.copy()
            if 'Occupancy' in noisy_seed.columns:
                noise = np.random.normal(0, noisy_seed['Occupancy'].std() * 0.1, len(noisy_seed))
                noisy_seed['Occupancy'] = noisy_seed['Occupancy'] + noise
            
            # Generate prediction
            pred = predict_recursive(self.model, noisy_seed, self.features, horizon)
            predictions.append(pred.values)
        
        # Compute statistics
        predictions_array = np.array(predictions)
        
        return {
            'mean': pd.Series(predictions_array.mean(axis=0), index=pred.index),
            'lower': pd.Series(np.percentile(predictions_array, 5, axis=0), index=pred.index),
            'upper': pd.Series(np.percentile(predictions_array, 95, axis=0), index=pred.index)
        }

