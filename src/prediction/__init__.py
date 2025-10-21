"""
src/prediction module

Multi-step time series forecasting for parking occupancy prediction.
Implements both recursive and direct forecasting strategies.

Extracted from Main.ipynb notebook.
"""

# Import main classes and functions for easy access
from .forecaster import (
    TimeSeriesForecaster,
    multi_step_forecast,
    create_forecaster
)

from .recursive_predictor import (
    predict_recursive,
    train_recursive_model
)

from .direct_predictor import (
    predict_direct,
    train_direct_models,
    DirectForecaster
)

__all__ = [
    # Main forecaster class
    'TimeSeriesForecaster',
    
    # Convenience functions
    'multi_step_forecast',
    'create_forecaster',
    
    # Recursive strategy
    'predict_recursive',
    'train_recursive_model',
    
    # Direct strategy
    'predict_direct',
    'train_direct_models',
    'DirectForecaster',
]
