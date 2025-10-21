"""
Demo Pipeline Script

End-to-end demonstration: Load → Preprocess → Train → Predict → Calculate Booking Probability

This script demonstrates the complete parking prediction pipeline in action.

Usage:
    python scripts/demo_pipeline.py
    python scripts/demo_pipeline.py --lot BHMBCCMKT01 --capacity 600
"""

import pandas as pd
import sys
import os
import argparse
from pathlib import Path
import math

# Add parent directory to path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.time_series_processor import process_lot_data
from src.models import LightGBMModel, evaluate_model
from src.queueing import get_queueing_inputs, get_booking_confirmation


def run_demo_pipeline(csv_path='data/raw/dataset.csv', lot_id='BHMBCCMKT01', capacity=600):
    """
    Full demo pipeline showing the complete flow.
    
    Args:
        csv_path: Path to dataset CSV
        lot_id: Parking lot system code
        capacity: Lot capacity
    """
    
    print("=" * 60)
    print("PARKING PREDICTOR - DEMO PIPELINE")
    print("=" * 60)
    
    # ========== 1. LOAD DATA ==========
    print("\n[1/6] Loading data...")
    df = pd.read_csv(csv_path)
    df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])
    
    # Segregate by lot
    segregated_lots = {lot_id: group_df for lot_id, group_df in df.groupby('SystemCodeNumber')}
    one_lot_df = segregated_lots[lot_id]
    one_lot_df = one_lot_df.set_index('LastUpdated')
    
    print(f"   ✓ Loaded {len(df)} records")
    print(f"   ✓ Selected lot: {lot_id} ({len(one_lot_df)} records)")
    
    # ========== 2. PREPROCESS DATA ==========
    print("\n[2/6] Preprocessing data...")
    processed_lot_data = process_lot_data(one_lot_df)
    
    print(f"   ✓ Resampled to hourly frequency")
    print(f"   ✓ Created features: hour_of_day, day_of_week, is_weekend")
    print(f"   ✓ Processed records: {len(processed_lot_data)}")
    
    # ========== 3. CREATE LAG FEATURES ==========
    print("\n[3/6] Creating lag features...")
    df_model = processed_lot_data.copy()
    for lag in [1, 2, 3, 24, 48]:
        df_model[f'lag_{lag}'] = df_model['Occupancy'].shift(lag)
    df_model = df_model.dropna()
    
    features = ['lag_1', 'lag_2', 'lag_3', 'lag_24', 'lag_48', 'hour_of_day', 'day_of_week', 'is_weekend']
    target = 'Occupancy'
    
    print(f"   ✓ Created lag features: 1, 2, 3, 24, 48 hours")
    print(f"   ✓ Total features: {len(features)}")
    
    # ========== 4. TRAIN MODEL ==========
    print("\n[4/6] Training LightGBM model...")
    split = int(len(df_model) * 0.8)
    train, test = df_model.iloc[:split], df_model.iloc[split:]
    
    model = LightGBMModel(
        params={'objective': 'regression', 'metric': 'l2', 'verbosity': -1, 'seed': 42},
        num_boost_round=200
    )
    model.fit(train[features], train[target])
    
    print(f"   ✓ Training samples: {len(train)}")
    print(f"   ✓ Test samples: {len(test)}")
    
    # ========== 5. EVALUATE MODEL ==========
    print("\n[5/6] Evaluating model...")
    test_predictions = model.predict(test[features])
    metrics = evaluate_model(test[target], test_predictions)
    
    print(f"   ✓ Test MAE: {metrics['MAE']:.2f}")
    print(f"   ✓ Test RMSE: {metrics['RMSE']:.2f}")
    print(f"   ✓ Test MAPE: {metrics['MAPE']:.2f}%")
    
    # Make prediction for the last test point
    predicted_occupancy = test_predictions[-1]
    actual_occupancy = test[target].iloc[-1]
    print(f"   ✓ Last prediction: {predicted_occupancy:.2f} (actual: {actual_occupancy:.2f})")
    
    # ========== 6. CALCULATE BOOKING PROBABILITY ==========
    print("\n[6/6] Calculating booking probability...")
    
    # Estimate queueing parameters from historical data
    hourly_rates, service_rate_mu = get_queueing_inputs(
        one_lot_df.reset_index(), 
        capacity=capacity,
        timestamp_col='LastUpdated',
        occupancy_col='Occupancy'
    )
    
    # Get booking confirmation for predicted occupancy
    hour_of_day = test.index[-1].hour
    result = get_booking_confirmation(
        predicted_occupancy=predicted_occupancy,
        capacity=capacity,
        hour_of_day=hour_of_day,
        hourly_arrival_rates=hourly_rates,
        service_rate_mu=service_rate_mu
    )
    
    print(f"\n   📊 BOOKING PROBABILITY RESULTS:")
    print(f"   {'─' * 50}")
    print(f"   Time: Hour {hour_of_day}:00")
    print(f"   Predicted occupancy: {predicted_occupancy:.1f}/{capacity}")
    print(f"   Available slots: {result['available_slots']}")
    print(f"   Probability of getting spot: {result['prob_get_spot']*100:.1f}%")
    print(f"   Probability of waiting: {result['prob_wait']*100:.1f}%")
    if math.isfinite(result['expected_wait_minutes']):
        print(f"   Expected wait time: {result['expected_wait_minutes']:.1f} minutes")
    else:
        print(f"   Expected wait time: ∞ (lot full)")
    print(f"   {'─' * 50}")
    print(f"\n   💡 {result['recommendation']}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY! ✓")
    print("=" * 60)
    
    return {
        'model': model,
        'test_metrics': metrics,
        'prediction': predicted_occupancy,
        'booking_result': result
    }


def main(args):
    """Main entry point with argument parsing."""
    
    # Check if data file exists
    if not Path(args.data_path).exists():
        print(f"❌ Error: Data file not found at {args.data_path}")
        print(f"   Please provide correct path using --data-path argument")
        return
    
    # Run the demo pipeline
    results = run_demo_pipeline(
        csv_path=args.data_path,
        lot_id=args.lot,
        capacity=args.capacity
    )
    
    # Optionally save results
    if args.save_model:
        output_dir = Path('data/models')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / 'demo_lightgbm_model.txt'
        results['model'].save_model(str(model_path))
        print(f"\n   ✓ Saved trained model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo pipeline for parking prediction')
    
    parser.add_argument('--data-path', type=str, default='data/raw/dataset.csv',
                        help='Path to dataset CSV file')
    parser.add_argument('--lot', type=str, default='BHMBCCMKT01',
                        help='Parking lot system code number')
    parser.add_argument('--capacity', type=int, default=600,
                        help='Parking lot capacity')
    parser.add_argument('--save-model', action='store_true',
                        help='Save trained model after demo')
    
    args = parser.parse_args()
    
    main(args)
