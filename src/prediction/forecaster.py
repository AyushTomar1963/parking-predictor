"""
forecaster.py

High-level interface for multi-step time series forecasting.
Provides unified interface for both direct and recursive strategies.

This module orchestrates the prediction workflow, handling:
- Strategy selection (direct vs recursive)
- Model loading
- Feature preparation
- Prediction generation
- Result formatting

Extracted from Main.ipynb Cell #16-18
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Literal
from pathlib import Path

# Import prediction strategies
from .recursive_predictor import predict_recursive, train_recursive_model
from .direct_predictor import predict_direct, train_direct_models, DirectForecaster


class TimeSeriesForecaster:
    """
    Unified interface for multi-step time series forecasting.
    
    Supports both recursive and direct forecasting strategies:
    - Recursive: Single model, iterative 1-step predictions
    - Direct: Separate models for each horizon
    
    Attributes:
        strategy: Forecasting strategy ('recursive' or 'direct')
        features: List of feature names
        target: Target variable name
        max_horizon: Maximum forecast horizon
        model(s): Trained model(s)
    
    Example - Recursive Strategy:
        >>> from src.prediction.forecaster import TimeSeriesForecaster
        >>> 
        >>> # Initialize forecaster
        >>> forecaster = TimeSeriesForecaster(
        ...     strategy='recursive',
        ...     features=['lag_1', 'lag_2', 'lag_24', 'hour_of_day'],
        ...     target='Occupancy',
        ...     max_horizon=24
        ... )
        >>> 
        >>> # Train model
        >>> forecaster.fit(train_df)
        >>> 
        >>> # Generate forecast
        >>> predictions = forecaster.predict(test_df.iloc[-1:], steps=24)
    
    Example - Direct Strategy:
        >>> forecaster = TimeSeriesForecaster(
        ...     strategy='direct',
        ...     features=['lag_1', 'lag_2', 'lag_24', 'hour_of_day'],
        ...     target='Occupancy',
        ...     max_horizon=24
        ... )
        >>> 
        >>> forecaster.fit(train_df)
        >>> predictions = forecaster.predict(test_df)
    """
    
    def __init__(
        self,
        strategy: Literal['recursive', 'direct'] = 'recursive',
        features: List[str] = None,
        target: str = 'Occupancy',
        max_horizon: int = 24,
        model_params: Dict[str, Any] = None
    ):
        """
        Initialize time series forecaster.
        
        Args:
            strategy: 'recursive' or 'direct'
            features: List of feature names
            target: Target variable name
            max_horizon: Maximum forecast horizon
            model_params: LightGBM parameters (optional)
        """
        self.strategy = strategy
        self.features = features or []
        self.target = target
        self.max_horizon = max_horizon
        self.model_params = model_params or {
            'objective': 'regression',
            'metric': 'l2',
            'verbosity': -1,
            'seed': 42
        }
        
        # Model storage
        self.model = None  # For recursive
        self.models = None  # For direct
        self.is_fitted = False
    
    def fit(
        self,
        df: pd.DataFrame,
        num_boost_round: int = 200,
        **kwargs
    ) -> 'TimeSeriesForecaster':
        """
        Train forecasting model(s).
        
        Args:
            df: Training data with features and target
            num_boost_round: Number of boosting rounds
            **kwargs: Additional arguments for training
            
        Returns:
            self: Fitted forecaster
        """
        if self.strategy == 'recursive':
            self.model = train_recursive_model(
                df=df,
                features=self.features,
                target=self.target,
                params=self.model_params,
                num_boost_round=num_boost_round
            )
        
        elif self.strategy == 'direct':
            self.models = train_direct_models(
                df=df,
                features=self.features,
                target=self.target,
                max_horizon=self.max_horizon,
                params=self.model_params,
                num_boost_round=num_boost_round
            )
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.is_fitted = True
        return self
    
    def predict(
        self,
        df: pd.DataFrame,
        steps: int = None
    ) -> Union[pd.Series, Dict[int, pd.Series]]:
        """
        Generate multi-step forecasts.
        
        Args:
            df: Input data with features
                - Recursive: Use last row as seed (df.iloc[-1:])
                - Direct: Use entire dataframe
            steps: Number of steps to forecast (for recursive)
                   If None, uses max_horizon
        
        Returns:
            - Recursive: pd.Series with future predictions
            - Direct: Dictionary mapping horizon to predictions
        """
        if not self.is_fitted:
            raise ValueError("Forecaster not fitted. Call fit() first.")
        
        if self.strategy == 'recursive':
            steps = steps or self.max_horizon
            return predict_recursive(
                model=self.model,
                df_start=df,
                features=self.features,
                max_horizon=steps
            )
        
        elif self.strategy == 'direct':
            return predict_direct(
                models=self.models,
                df_block=df,
                features=self.features
            )
    
    def predict_single_horizon(
        self,
        df: pd.DataFrame,
        horizon: int
    ) -> Union[float, pd.Series]:
        """
        Predict for a specific horizon.
        
        Args:
            df: Input data
            horizon: Forecast horizon (1 to max_horizon)
            
        Returns:
            Prediction(s) for specified horizon
        """
        if not self.is_fitted:
            raise ValueError("Forecaster not fitted. Call fit() first.")
        
        if self.strategy == 'recursive':
            predictions = predict_recursive(
                model=self.model,
                df_start=df,
                features=self.features,
                max_horizon=horizon
            )
            return predictions.iloc[-1]  # Return last (horizon-th) prediction
        
        elif self.strategy == 'direct':
            if horizon not in self.models:
                raise ValueError(f"Horizon {horizon} not available")
            
            X = df[self.features].dropna()
            pred = self.models[horizon].predict(X)
            return pd.Series(pred, index=X.index) if len(pred) > 1 else pred[0]
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save trained model(s) to disk.
        
        Args:
            path: Directory path to save models
        """
        import joblib
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.strategy == 'recursive':
            joblib.dump(self.model, path / 'recursive_model.pkl')
        
        elif self.strategy == 'direct':
            for h, model in self.models.items():
                joblib.dump(model, path / f'direct_model_h{h}.pkl')
        
        # Save metadata
        metadata = {
            'strategy': self.strategy,
            'features': self.features,
            'target': self.target,
            'max_horizon': self.max_horizon,
            'model_params': self.model_params
        }
        joblib.dump(metadata, path / 'metadata.pkl')
    
    def load(self, path: Union[str, Path]) -> 'TimeSeriesForecaster':
        """
        Load trained model(s) from disk.
        
        Args:
            path: Directory path to load models from
            
        Returns:
            self: Loaded forecaster
        """
        import joblib
        from glob import glob
        
        path = Path(path)
        
        # Load metadata
        metadata = joblib.load(path / 'metadata.pkl')
        self.strategy = metadata['strategy']
        self.features = metadata['features']
        self.target = metadata['target']
        self.max_horizon = metadata['max_horizon']
        self.model_params = metadata['model_params']
        
        if self.strategy == 'recursive':
            self.model = joblib.load(path / 'recursive_model.pkl')
        
        elif self.strategy == 'direct':
            self.models = {}
            for model_file in glob(str(path / 'direct_model_h*.pkl')):
                h = int(Path(model_file).stem.split('h')[1])
                self.models[h] = joblib.load(model_file)
        
        self.is_fitted = True
        return self
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained model(s).
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Forecaster not fitted")
        
        if self.strategy == 'recursive':
            importance = self.model.feature_importance(importance_type='gain')
            return pd.DataFrame({
                'feature': self.features,
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        elif self.strategy == 'direct':
            # Average importance across all horizon models
            all_importance = []
            for h, model in self.models.items():
                importance = model.feature_importance(importance_type='gain')
                all_importance.append(importance)
            
            avg_importance = np.mean(all_importance, axis=0)
            return pd.DataFrame({
                'feature': self.features,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)


def multi_step_forecast(
    model: Any,
    data: pd.DataFrame,
    features: List[str],
    steps: int = 24,
    strategy: str = 'recursive'
) -> pd.Series:
    """
    Convenience function for multi-step forecasting.
    
    This is a simplified interface that wraps the strategy-specific functions.
    
    Args:
        model: Trained model (for recursive) or dict of models (for direct)
        data: Input data with features
        features: List of feature names
        steps: Number of steps to forecast
        strategy: 'recursive' or 'direct'
        
    Returns:
        pd.Series with predictions
        
    Example:
        >>> from src.prediction.forecaster import multi_step_forecast
        >>> 
        >>> # Recursive prediction
        >>> predictions = multi_step_forecast(
        ...     model=trained_model,
        ...     data=test_df.iloc[-1:],
        ...     features=['lag_1', 'lag_2', 'lag_24'],
        ...     steps=24,
        ...     strategy='recursive'
        ... )
        >>> 
        >>> # Direct prediction (returns dict)
        >>> predictions = multi_step_forecast(
        ...     model=trained_models_dict,
        ...     data=test_df,
        ...     features=['lag_1', 'lag_2', 'lag_24'],
        ...     steps=24,
        ...     strategy='direct'
        ... )
    """
    if strategy == 'recursive':
        return predict_recursive(
            model=model,
            df_start=data,
            features=features,
            max_horizon=steps
        )
    
    elif strategy == 'direct':
        predictions_dict = predict_direct(
            models=model,
            df_block=data,
            features=features
        )
        # Return only up to requested steps
        return {h: pred for h, pred in predictions_dict.items() if h <= steps}
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def create_forecaster(
    strategy: str,
    train_df: pd.DataFrame,
    features: List[str],
    target: str = 'Occupancy',
    max_horizon: int = 24,
    **kwargs
) -> TimeSeriesForecaster:
    """
    Factory function to create and train a forecaster.
    
    Args:
        strategy: 'recursive' or 'direct'
        train_df: Training data
        features: List of feature names
        target: Target variable name
        max_horizon: Maximum forecast horizon
        **kwargs: Additional arguments for training
        
    Returns:
        Trained TimeSeriesForecaster
        
    Example:
        >>> from src.prediction.forecaster import create_forecaster
        >>> 
        >>> # Create and train in one step
        >>> forecaster = create_forecaster(
        ...     strategy='recursive',
        ...     train_df=train_data,
        ...     features=['lag_1', 'lag_2', 'lag_24', 'hour_of_day'],
        ...     target='Occupancy',
        ...     max_horizon=24,
        ...     num_boost_round=200
        ... )
        >>> 
        >>> # Use for prediction
        >>> predictions = forecaster.predict(test_data.iloc[-1:])
    """
    forecaster = TimeSeriesForecaster(
        strategy=strategy,
        features=features,
        target=target,
        max_horizon=max_horizon
    )
    
    forecaster.fit(train_df, **kwargs)
    
    return forecaster


