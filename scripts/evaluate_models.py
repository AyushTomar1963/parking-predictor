"""
Evaluate Model Performance Script

This script loads trained models and evaluates them on test data with detailed metrics,
visualizations, and comparison reports.

Usage:
    python scripts/evaluate_models.py --model lightgbm
    python scripts/evaluate_models.py --model all --plot
"""

import argparse
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.time_series_processor import process_lot_data, create_lag_features
from src.models import LightGBMModel, XGBoostModel, ARIMAXModel
from src.models.models import evaluate_model, compare_models, get_best_model


def plot_predictions(y_true, predictions_dict, title="Model Predictions vs Actual", save_path=None):
    """Plot predictions from multiple models against actual values."""
    plt.figure(figsize=(14, 6))
    
    # Plot actual values
    plt.plot(y_true.index, y_true.values, label='Actual', linewidth=2, color='black', alpha=0.7)
    
    # Plot predictions from each model
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i, (model_name, preds) in enumerate(predictions_dict.items()):
        plt.plot(y_true.index, preds, label=model_name, 
                linewidth=1.5, alpha=0.7, color=colors[i % len(colors)])
    
    plt.xlabel('Time')
    plt.ylabel('Occupancy')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_residuals(y_true, y_pred, model_name, save_path=None):
    """Plot residual analysis."""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Residuals over time
    axes[0].plot(residuals.index if hasattr(residuals, 'index') else range(len(residuals)), 
                 residuals, alpha=0.6)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title(f'{model_name} - Residuals Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residual Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{model_name} - Residual Distribution')
    axes[1].grid(True, alpha=0.3)
    
    # Q-Q plot (residuals vs predicted)
    axes[2].scatter(y_pred, residuals, alpha=0.5)
    axes[2].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[2].set_xlabel('Predicted Values')
    axes[2].set_ylabel('Residuals')
    axes[2].set_title(f'{model_name} - Residuals vs Predicted')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved residual plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_lightgbm(model_path, X_test, y_test, plot_results=False, output_dir=None):
    """Evaluate LightGBM model."""
    print("\n" + "=" * 60)
    print("Evaluating LightGBM Model")
    print("=" * 60)
    
    # Load model
    model = LightGBMModel()
    model.load_model(model_path)
    print(f"   ✓ Loaded model from {model_path}")
    
    # Predict
    predictions = model.predict(X_test)
    
    # Evaluate
    metrics = evaluate_model(y_test, predictions)
    
    print(f"\nTest Metrics:")
    print(f"  MAE:  {metrics['MAE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    # Feature importance
    importance = model.get_feature_importance()
    print(f"\nTop 5 Important Features:")
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {feat}: {imp:.2f}")
    
    # Plot if requested
    if plot_results and output_dir:
        plot_residuals(y_test, predictions, 'LightGBM', 
                      save_path=Path(output_dir) / 'lightgbm_residuals.png')
    
    return predictions, metrics


def evaluate_xgboost(model_path, X_test, y_test, plot_results=False, output_dir=None):
    """Evaluate XGBoost model."""
    print("\n" + "=" * 60)
    print("Evaluating XGBoost Model")
    print("=" * 60)
    
    # Load model
    model = XGBoostModel()
    model.load_model(model_path)
    print(f"   ✓ Loaded model from {model_path}")
    
    # Predict
    predictions = model.predict(X_test)
    
    # Evaluate
    metrics = evaluate_model(y_test, predictions)
    
    print(f"\nTest Metrics:")
    print(f"  MAE:  {metrics['MAE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    # Feature importance
    importance = model.get_feature_importance()
    print(f"\nTop 5 Important Features:")
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {feat}: {imp:.2f}")
    
    # Plot if requested
    if plot_results and output_dir:
        plot_residuals(y_test, predictions, 'XGBoost', 
                      save_path=Path(output_dir) / 'xgboost_residuals.png')
    
    return predictions, metrics


def evaluate_arima(model_path, y_test, plot_results=False, output_dir=None):
    """Evaluate ARIMA model."""
    print("\n" + "=" * 60)
    print("Evaluating ARIMA Model")
    print("=" * 60)
    
    # Load model
    model = ARIMAXModel()
    model.load_model(model_path)
    print(f"   ✓ Loaded model from {model_path}")
    
    # Predict
    predictions = model.predict(steps=len(y_test))
    
    # Evaluate
    metrics = evaluate_model(y_test.values, predictions)
    
    print(f"\nTest Metrics:")
    print(f"  MAE:  {metrics['MAE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    # Plot if requested
    if plot_results and output_dir:
        plot_residuals(y_test.values, predictions, 'ARIMA', 
                      save_path=Path(output_dir) / 'arima_residuals.png')
    
    return predictions, metrics


def main(args):
    """Main evaluation pipeline."""
    
    print("=" * 60)
    print("PARKING PREDICTOR - MODEL EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Model directory: {args.model_dir}")
    print("=" * 60)
    
    # ========== 1. LOAD DATA ==========
    print("\n[1/4] Loading data...")
    df = pd.read_csv(args.data_path)
    df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])
    
    # Filter by lot if column exists (raw data) and lot specified
    if args.lot and 'SystemCodeNumber' in df.columns:
        df = df[df['SystemCodeNumber'] == args.lot].copy()
        print(f"   ✓ Filtered to lot: {args.lot}")
    elif args.lot:
        print(f"   ⚠️  No 'SystemCodeNumber' column found — assuming single-lot dataset.")
    
    df = df.set_index('LastUpdated')
    print(f"   ✓ Loaded {len(df)} records")
    
    # ========== 2. PREPROCESS ==========
    print("\n[2/4] Preprocessing data...")
    
    # Check if data already has calendar features (likely already processed)
    if 'hour_of_day' in df.columns and 'Occupancy' in df.columns:
        print("   ℹ  Data appears already processed, skipping preprocessing...")
        processed_data = df
    else:
        # Need to process raw data
        processed_data = process_lot_data(df)
    
    print(f"   ✓ Processed {len(processed_data)} records")
    
    # ========== 3. CREATE TEST SET ==========
    print("\n[3/4] Creating test set...")
    
    # For ML models - create lag features
    df_ml = create_lag_features(processed_data, lags=[1, 2, 3, 24, 48])
    print(f"   ✓ Created lag features. Shape before dropna: {df_ml.shape}")
    
    # Drop NaNs FIRST (before splitting) to ensure consistent sizes
    df_ml = df_ml.dropna()
    print(f"   ✓ Dropped NaNs. Shape after dropna: {df_ml.shape}")
    
    features = ['lag_1', 'lag_2', 'lag_3', 'lag_24', 'lag_48', 
                'hour_of_day', 'day_of_week', 'is_weekend']
    target = 'Occupancy'
    
    # Use last 20% as test set (on clean data)
    split_point = int(len(df_ml) * 0.8)
    test_ml = df_ml.iloc[split_point:]
    
    X_test = test_ml[features]
    y_test = test_ml[target]
    
    # For ARIMA - use same split point on processed_data
    y_test_arima = processed_data['Occupancy'].iloc[split_point:]
    
    print(f"   ✓ Test samples: {len(X_test)} (X_test and y_test match: {len(X_test) == len(y_test)})")
    
    # ========== 4. EVALUATE MODELS ==========
    print("\n[4/4] Evaluating models...")
    
    model_dir = Path(args.model_dir)
    all_predictions = {}
    all_metrics = {}
    
    # Evaluate specified model(s)
    if args.model == 'lightgbm' or args.model == 'all':
        model_path = model_dir / 'lightgbm_model.txt'
        if model_path.exists():
            preds, metrics = evaluate_lightgbm(
                model_path, X_test, y_test, args.plot, args.output_dir
            )
            all_predictions['LightGBM'] = preds
            all_metrics['LightGBM'] = metrics
        else:
            print(f"   ⚠️  LightGBM model not found at {model_path}")
    
    if args.model == 'xgboost' or args.model == 'all':
        model_path = model_dir / 'xgboost_model.json'
        if model_path.exists():
            preds, metrics = evaluate_xgboost(
                model_path, X_test, y_test, args.plot, args.output_dir
            )
            all_predictions['XGBoost'] = preds
            all_metrics['XGBoost'] = metrics
        else:
            print(f"   ⚠️  XGBoost model not found at {model_path}")
    
    if args.model == 'arima' or args.model == 'all':
        model_path = model_dir / 'arima_model.pkl'
        if model_path.exists():
            preds, metrics = evaluate_arima(
                model_path, y_test_arima, args.plot, args.output_dir
            )
            all_predictions['ARIMA'] = preds
            all_metrics['ARIMA'] = metrics
        else:
            print(f"   ⚠️  ARIMA model not found at {model_path}")
    
    # ========== 5. COMPARE MODELS ==========
    if len(all_metrics) > 1:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        
        comparison_table = []
        for model_name, metrics in all_metrics.items():
            comparison_table.append([
                model_name,
                f"{metrics['MAE']:.2f}",
                f"{metrics['RMSE']:.2f}",
                f"{metrics['MAPE']:.2f}%"
            ])
        
        # Print table
        print(f"\n{'Model':<15} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
        print("-" * 50)
        for row in comparison_table:
            print(f"{row[0]:<15} {row[1]:<10} {row[2]:<10} {row[3]:<10}")
        
        # Find best model
        best_model, best_score = get_best_model(all_metrics, metric='RMSE')
        print(f"\n🏆 Best Model: {best_model} (RMSE: {best_score:.2f})")
        
        # Plot comparison
        if args.plot and args.output_dir:
            plot_predictions(y_test, all_predictions, 
                           title="Model Comparison - Test Set",
                           save_path=Path(args.output_dir) / 'model_comparison.png')
    
    # ========== 6. SAVE EVALUATION REPORT ==========
    if args.output_dir:
        report = {
            'model': args.model,
            'test_samples': len(X_test),
            'metrics': {k: {mk: float(mv) for mk, mv in v.items()} 
                       for k, v in all_metrics.items()}
        }
        
        report_path = Path(args.output_dir) / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n   ✓ Saved evaluation report to {report_path}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY! ✓")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    
    parser.add_argument('--data-path', type=str, default='data/raw/dataset.csv',
                        help='Path to input CSV file')
    parser.add_argument('--lot', type=str, default='BHMBCCMKT01',
                        help='Parking lot system code number')
    parser.add_argument('--model', type=str, default='all',
                        choices=['lightgbm', 'xgboost', 'arima', 'all'],
                        help='Model to evaluate')
    parser.add_argument('--model-dir', type=str, default='data/models',
                        help='Directory containing trained models')
    parser.add_argument('--output-dir', type=str, default='data/models',
                        help='Directory to save evaluation results')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
