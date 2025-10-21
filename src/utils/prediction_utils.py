"""
Forecasting utilities, evaluation, batch predictions.
"""
import numpy as np
import pandas as pd


def forecast_next_hours(model, current_data, hours=24):
    """Generate hourly forecasts for the next N hours."""
    predictions = model.predict(steps=hours)
    return predictions


def batch_predict(model, data_batches):
    """Run predictions on multiple batches."""
    predictions = []
    for batch in data_batches:
        pred = model.predict(batch)
        predictions.append(pred)
    return np.concatenate(predictions)


def calculate_confidence_interval(predictions, std_dev, confidence=0.95):
    """Calculate prediction confidence intervals."""
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    lower_bound = predictions - z_score * std_dev
    upper_bound = predictions + z_score * std_dev
    
    return lower_bound, upper_bound


def prepare_forecast_df(predictions, start_time, freq='H'):
    """Prepare forecast results as a DataFrame."""
    time_index = pd.date_range(start=start_time, periods=len(predictions), freq=freq)
    df = pd.DataFrame({
        'timestamp': time_index,
        'predicted_occupancy': predictions
    })
    return df
