"""
Train and Save Models Script

This script trains multiple models (LightGBM, ARIMA, XGBoost, LSTM) on parking data
and saves them for later use.

Usage:
    python scripts/train_models.py --lot BHMBCCMKT01 --capacity 600
    python scripts/train_models.py --models lightgbm,xgboost --split 0.8
"""

import argparse
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing.time_series_processor import create_lag_features, process_lot_data
from src.models import LightGBMModel, XGBoostModel, ARIMAXModel
from src.models.models import evaluate_model


def train_lightgbm(X_train, y_train, X_val, y_val, params=None):
    """Train LightGBM model."""
    print("\n" + "=" * 60)
    print("Training LightGBM Model")
    print("=" * 60)
    
    model = LightGBMModel(
        params=params or {
            'objective': 'regression',
            'metric': 'l2',
        },
        num_boost_round=200
    )
    
    model.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_metrics = evaluate_model(y_train, train_pred)
    val_metrics = evaluate_model(y_val, val_pred)
    
    print(f"\nTrain Metrics - MAE: {train_metrics['MAE']:.2f}, RMSE: {train_metrics['RMSE']:.2f}")
    print(f"Val Metrics   - MAE: {val_metrics['MAE']:.2f}, RMSE: {val_metrics['RMSE']:.2f}")
    
    return model, val_metrics


def train_xgboost(X_train, y_train, X_val, y_val, params=None):
    """Train XGBoost model."""
    print("\n" + "=" * 60)
    print("Training XGBoost Model")
    print("=" * 60)
    
    model = XGBoostModel(
        params=params or {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 7,
            'seed': 42
        },
        num_boost_round=200
    )
    
    model.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_metrics = evaluate_model(y_train, train_pred)
    val_metrics = evaluate_model(y_val, val_pred)
    
    print(f"\nTrain Metrics - MAE: {train_metrics['MAE']:.2f}, RMSE: {train_metrics['RMSE']:.2f}")
    print(f"Val Metrics   - MAE: {val_metrics['MAE']:.2f}, RMSE: {val_metrics['RMSE']:.2f}")
    
    return model, val_metrics


def train_arima(y_train, y_val, order=(1, 0, 0)):
    """Train ARIMA model."""
    print("\n" + "=" * 60)
    print("Training ARIMA Model")
    print("=" * 60)
    
    model = ARIMAXModel(order=order)
    model.fit(y_train)  # Simple positional argument, no keyword
    
    # Predict on validation set
    val_pred = model.predict(steps=len(y_val))
    
    val_metrics = evaluate_model(y_val.values, val_pred)
    
    print(f"\nVal Metrics - MAE: {val_metrics['MAE']:.2f}, RMSE: {val_metrics['RMSE']:.2f}")
    
    return model, val_metrics


def main(args):
    """Main training pipeline."""
    
    print("=" * 60)
    print("PARKING PREDICTOR - MODEL TRAINING PIPELINE")
    print("=" * 60)
    print(f"Lot ID: {args.lot}")
    print(f"Models to train: {args.models}")
    print(f"Train/Val split: {args.split}/{1-args.split}")
    print("=" * 60)
    
    # ========== 1. LOAD DATA ==========
    print("\n[1/5] Loading data...")
    df = pd.read_csv(args.data_path)
    df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])
    
    # Filter by lot if column exists (raw data) and lot specified
    if args.lot and 'SystemCodeNumber' in df.columns:
        df = df[df['SystemCodeNumber'] == args.lot].copy()
        print(f"   ✓ Filtered to lot: {args.lot}")
    
    df = df.set_index('LastUpdated')
    print(f"   ✓ Loaded {len(df)} records")
    
    # ========== 2. PREPROCESS ==========
    print("\n[2/5] Preprocessing data...")
    
    # Check if data already has calendar features (likely already processed)
    if 'hour_of_day' in df.columns and 'Occupancy' in df.columns:
        print("   ℹ  Data appears already processed, skipping preprocessing...")
        processed_data = df
    else:
        # Need to process raw data
        processed_data = process_lot_data(df)
    
    print(f"   ✓ Processed {len(processed_data)} records")
    
    # ========== 3. CREATE FEATURES ==========
    print("\n[3/5] Creating features...")
    
    # For ML models (LightGBM, XGBoost)
    df_ml = create_lag_features(processed_data, lags=[1, 2, 3, 24, 48])
    features = ['lag_1', 'lag_2', 'lag_3', 'lag_24', 'lag_48', 
                'hour_of_day', 'day_of_week', 'is_weekend']
    target = 'Occupancy'
    
    print(f"   ✓ Created {len(features)} features")
    print(f"   ✓ Features: {features}")
    
    # ========== 4. TRAIN/VAL SPLIT ==========
    print("\n[4/5] Splitting data...")
    split_point = int(len(df_ml) * args.split)
    
    # For ML models
    train_ml = df_ml.iloc[:split_point]
    val_ml = df_ml.iloc[split_point:]
    
    X_train = train_ml[features]
    y_train = train_ml[target]
    X_val = val_ml[features]
    y_val = val_ml[target]
    
    # For ARIMA
    y_train_arima = processed_data['Occupancy'].iloc[:split_point]
    y_val_arima = processed_data['Occupancy'].iloc[split_point:]
    
    print(f"   ✓ Train samples: {len(X_train)}")
    print(f"   ✓ Val samples: {len(X_val)}")
    
    # ========== 5. TRAIN MODELS ==========
    print("\n[5/5] Training models...")
    
    trained_models = {}
    all_metrics = {}
    
    model_list = [m.strip().lower() for m in args.models.split(',')]
    
    # Train LightGBM
    if 'lightgbm' in model_list or 'all' in model_list:
        model, metrics = train_lightgbm(X_train, y_train, X_val, y_val)
        trained_models['lightgbm'] = model
        all_metrics['lightgbm'] = metrics
        
        # Save model
        save_path = Path(args.output_dir) / 'lightgbm_model.txt'
        model.save_model(str(save_path))
        print(f"   ✓ Saved LightGBM model to {save_path}")
    
    # Train XGBoost
    if 'xgboost' in model_list or 'all' in model_list:
        model, metrics = train_xgboost(X_train, y_train, X_val, y_val)
        trained_models['xgboost'] = model
        all_metrics['xgboost'] = metrics
        
        # Save model
        save_path = Path(args.output_dir) / 'xgboost_model.json'
        model.save_model(str(save_path))
        print(f"   ✓ Saved XGBoost model to {save_path}")
    
    # Train ARIMA
    if 'arima' in model_list or 'all' in model_list:
        model, metrics = train_arima(y_train_arima, y_val_arima, order=(1, 0, 0))
        trained_models['arima'] = model
        all_metrics['arima'] = metrics
        
        # Save model
        save_path = Path(args.output_dir) / 'arima_model.pkl'
        model.save_model(str(save_path))
        print(f"   ✓ Saved ARIMA model to {save_path}")
    
    # ========== 6. COMPARE MODELS ==========
    if len(trained_models) > 1:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        
        for model_name, metrics in all_metrics.items():
            print(f"\n{model_name.upper()}:")
            print(f"  MAE:  {metrics['MAE']:.2f}")
            print(f"  RMSE: {metrics['RMSE']:.2f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    # ========== 7. SAVE METADATA ==========
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'lot_id': args.lot,
        'capacity': args.capacity,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'features': features,
        'models': list(trained_models.keys()),
        'metrics': {k: {mk: float(mv) for mk, mv in v.items()} 
                    for k, v in all_metrics.items()}
    }
    
    metadata_path = Path(args.output_dir) / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n   ✓ Saved metadata to {metadata_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY! ✓")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train parking prediction models')
    
    parser.add_argument('--data-path', type=str, default='data/raw/dataset.csv',
                        help='Path to input CSV file')
    parser.add_argument('--lot', type=str, default='BHMBCCMKT01',
                        help='Parking lot system code number')
    parser.add_argument('--capacity', type=int, default=600,
                        help='Parking lot capacity')
    parser.add_argument('--models', type=str, default='lightgbm,xgboost,arima',
                        help='Comma-separated list of models to train (lightgbm,xgboost,arima,all)')
    parser.add_argument('--split', type=float, default=0.8,
                        help='Train/validation split ratio')
    parser.add_argument('--output-dir', type=str, default='data/models',
                        help='Directory to save trained models')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
