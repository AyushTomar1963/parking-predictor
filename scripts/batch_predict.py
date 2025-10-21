"""
Batch Prediction Script

Generate predictions for multiple hours/days ahead using trained models.
Useful for forecasting and capacity planning.

Usage:
    python scripts/batch_predict.py --hours 24 --model lightgbm
    python scripts/batch_predict.py --hours 168 --model all --output predictions.csv
"""

import argparse
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.time_series_processor import process_lot_data
from src.models import LightGBMModel, XGBoostModel, ARIMAXModel
from src.queueing import get_queueing_inputs, get_booking_confirmation


def create_lag_features(df, lags=[1, 2, 3, 24, 48]):
    """Create lag features for ML models."""
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['Occupancy'].shift(lag)
    return df.dropna()


def recursive_forecast_ml(model, last_data, features, steps):
    """
    Generate multi-step forecast using recursive strategy.
    
    For each step:
    1. Predict next value
    2. Update lag features with prediction
    3. Move to next time step
    """
    predictions = []
    current_data = last_data.copy()
    
    for step in range(steps):
        # Predict next value
        X_next = current_data[features].iloc[[-1]]
        pred = float(model.predict(X_next)[0])
        predictions.append(pred)
        
        # Update data for next iteration
        next_index = current_data.index[-1] + pd.Timedelta(hours=1)
        
        # Shift lag features
        new_row = {}
        for col in current_data.columns:
            if col.startswith('lag_'):
                lag_num = int(col.split('_')[1])
                if lag_num == 1:
                    new_row[col] = pred
                else:
                    # Get previous lag value
                    prev_lag = f'lag_{lag_num - 1}'
                    new_row[col] = current_data.iloc[-1].get(prev_lag, np.nan)
            elif col == 'Occupancy':
                new_row[col] = pred
        
        # Time-based features for next hour
        new_row['hour_of_day'] = next_index.hour
        new_row['day_of_week'] = next_index.dayofweek
        new_row['is_weekend'] = int(next_index.dayofweek >= 5)
        
        # Append new row
        new_df = pd.DataFrame(new_row, index=[next_index])
        current_data = pd.concat([current_data, new_df])
    
    # Create index for predictions
    start_index = last_data.index[-1] + pd.Timedelta(hours=1)
    pred_index = pd.date_range(start=start_index, periods=steps, freq='H')
    
    return pd.Series(predictions, index=pred_index)


def batch_predict_lightgbm(model_path, last_data, features, steps):
    """Generate batch predictions using LightGBM."""
    print(f"\n   Predicting with LightGBM...")
    
    model = LightGBMModel()
    model.load_model(model_path)
    
    predictions = recursive_forecast_ml(model, last_data, features, steps)
    
    return predictions


def batch_predict_xgboost(model_path, last_data, features, steps):
    """Generate batch predictions using XGBoost."""
    print(f"\n   Predicting with XGBoost...")
    
    model = XGBoostModel()
    model.load_model(model_path)
    
    predictions = recursive_forecast_ml(model, last_data, features, steps)
    
    return predictions


def batch_predict_arima(model_path, steps):
    """Generate batch predictions using ARIMA."""
    print(f"\n   Predicting with ARIMA...")
    
    model = ARIMAXModel()
    model.load_model(model_path)
    
    predictions = model.predict(steps=steps)
    
    return predictions


def add_booking_probabilities(predictions_df, capacity, raw_data):
    """Add booking probability columns to predictions."""
    print("\n   Calculating booking probabilities...")
    
    # Estimate queueing parameters
    hourly_rates, service_rate_mu = get_queueing_inputs(
        raw_data, capacity=capacity,
        timestamp_col='LastUpdated', occupancy_col='Occupancy'
    )
    
    # Calculate booking probability for each predicted hour
    booking_results = []
    
    for idx, row in predictions_df.iterrows():
        hour_of_day = idx.hour
        
        for model_col in [c for c in predictions_df.columns if c.endswith('_prediction')]:
            pred_occ = row[model_col]
            
            result = get_booking_confirmation(
                predicted_occupancy=pred_occ,
                capacity=capacity,
                hour_of_day=hour_of_day,
                hourly_arrival_rates=hourly_rates,
                service_rate_mu=service_rate_mu
            )
            
            # Add booking metrics
            model_name = model_col.replace('_prediction', '')
            predictions_df.loc[idx, f'{model_name}_available_slots'] = result['available_slots']
            predictions_df.loc[idx, f'{model_name}_prob_get_spot'] = result['prob_get_spot']
            predictions_df.loc[idx, f'{model_name}_expected_wait_min'] = result['expected_wait_minutes']
    
    return predictions_df


def main(args):
    """Main batch prediction pipeline."""
    
    print("=" * 60)
    print("PARKING PREDICTOR - BATCH PREDICTION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Forecast horizon: {args.hours} hours")
    print(f"Lot: {args.lot}")
    print("=" * 60)
    
    # ========== 1. LOAD DATA ==========
    print("\n[1/4] Loading data...")
    df = pd.read_csv(args.data_path)
    df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])
    
    # Keep raw data for queueing calculations
    raw_data = df.copy()
    
    # Filter by lot
    if args.lot:
        df = df[df['SystemCodeNumber'] == args.lot].copy()
    
    df = df.set_index('LastUpdated')
    print(f"   ✓ Loaded {len(df)} records")
    
    # ========== 2. PREPROCESS ==========
    print("\n[2/4] Preprocessing data...")
    processed_data = process_lot_data(df)
    
    # Create features for ML models
    df_ml = create_lag_features(processed_data, lags=[1, 2, 3, 24, 48])
    features = ['lag_1', 'lag_2', 'lag_3', 'lag_24', 'lag_48', 
                'hour_of_day', 'day_of_week', 'is_weekend']
    
    # Use last available data as starting point
    last_data = df_ml.iloc[-100:]  # Keep enough history for recursive forecasting
    
    print(f"   ✓ Using last {len(last_data)} records as seed data")
    print(f"   ✓ Last timestamp: {last_data.index[-1]}")
    
    # ========== 3. GENERATE PREDICTIONS ==========
    print(f"\n[3/4] Generating {args.hours}-hour forecast...")
    
    model_dir = Path(args.model_dir)
    all_predictions = {}
    
    # Predict with specified model(s)
    if args.model == 'lightgbm' or args.model == 'all':
        model_path = model_dir / 'lightgbm_model.txt'
        if model_path.exists():
            preds = batch_predict_lightgbm(model_path, last_data, features, args.hours)
            all_predictions['lightgbm'] = preds
        else:
            print(f"   ⚠️  LightGBM model not found at {model_path}")
    
    if args.model == 'xgboost' or args.model == 'all':
        model_path = model_dir / 'xgboost_model.json'
        if model_path.exists():
            preds = batch_predict_xgboost(model_path, last_data, features, args.hours)
            all_predictions['xgboost'] = preds
        else:
            print(f"   ⚠️  XGBoost model not found at {model_path}")
    
    if args.model == 'arima' or args.model == 'all':
        model_path = model_dir / 'arima_model.pkl'
        if model_path.exists():
            preds = batch_predict_arima(model_path, args.hours)
            # Create proper index for ARIMA predictions
            start_time = last_data.index[-1] + pd.Timedelta(hours=1)
            pred_index = pd.date_range(start=start_time, periods=args.hours, freq='H')
            preds = pd.Series(preds, index=pred_index)
            all_predictions['arima'] = preds
        else:
            print(f"   ⚠️  ARIMA model not found at {model_path}")
    
    if not all_predictions:
        print("\n   ❌ No models found for prediction!")
        return
    
    # ========== 4. CREATE PREDICTIONS DATAFRAME ==========
    print("\n[4/4] Formatting predictions...")
    
    # Combine all predictions into DataFrame
    predictions_df = pd.DataFrame()
    
    for model_name, preds in all_predictions.items():
        predictions_df[f'{model_name}_prediction'] = preds
    
    # Add time-based features
    predictions_df['hour_of_day'] = predictions_df.index.hour
    predictions_df['day_of_week'] = predictions_df.index.dayofweek
    predictions_df['day_name'] = predictions_df.index.day_name()
    predictions_df['is_weekend'] = (predictions_df['day_of_week'] >= 5).astype(int)
    
    # Add booking probabilities if requested
    if args.booking_prob:
        predictions_df = add_booking_probabilities(
            predictions_df, args.capacity, 
            raw_data[raw_data['SystemCodeNumber'] == args.lot].reset_index()
        )
    
    print(f"   ✓ Generated predictions for {len(predictions_df)} hours")
    
    # ========== 5. SAVE PREDICTIONS ==========
    if args.output:
        output_path = Path(args.output)
        predictions_df.to_csv(output_path)
        print(f"\n   ✓ Saved predictions to {output_path}")
    
    # ========== 6. DISPLAY SUMMARY ==========
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    
    print(f"\nForecast Period: {predictions_df.index[0]} to {predictions_df.index[-1]}")
    print(f"Total Hours: {len(predictions_df)}")
    
    for model_name in all_predictions.keys():
        col_name = f'{model_name}_prediction'
        print(f"\n{model_name.upper()} Predictions:")
        print(f"  Mean: {predictions_df[col_name].mean():.2f}")
        print(f"  Min:  {predictions_df[col_name].min():.2f}")
        print(f"  Max:  {predictions_df[col_name].max():.2f}")
        print(f"  Std:  {predictions_df[col_name].std():.2f}")
    
    # Show first few predictions
    print("\n" + "-" * 60)
    print("First 10 predictions:")
    print("-" * 60)
    display_cols = [c for c in predictions_df.columns if c.endswith('_prediction')]
    print(predictions_df[display_cols].head(10).to_string())
    
    print("\n" + "=" * 60)
    print("BATCH PREDICTION COMPLETED SUCCESSFULLY! ✓")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate batch predictions')
    
    parser.add_argument('--data-path', type=str, default='data/raw/dataset.csv',
                        help='Path to input CSV file')
    parser.add_argument('--lot', type=str, default='BHMBCCMKT01',
                        help='Parking lot system code number')
    parser.add_argument('--capacity', type=int, default=600,
                        help='Parking lot capacity')
    parser.add_argument('--model', type=str, default='lightgbm',
                        choices=['lightgbm', 'xgboost', 'arima', 'all'],
                        help='Model to use for prediction')
    parser.add_argument('--model-dir', type=str, default='data/models',
                        help='Directory containing trained models')
    parser.add_argument('--hours', type=int, default=24,
                        help='Number of hours to forecast')
    parser.add_argument('--output', type=str, default='data/processed/predictions.csv',
                        help='Output CSV file path')
    parser.add_argument('--booking-prob', action='store_true',
                        help='Calculate booking probabilities')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    main(args)
