"""
Reprocess Data with Capacity Column

This script reprocesses the raw parking data to include the Capacity column
in the processed output.

Usage:
    python scripts/reprocess_data.py
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.time_series_processor import process_lot_data, create_lag_features

def main():
    print("=" * 60)
    print("REPROCESSING DATA WITH CAPACITY COLUMN")
    print("=" * 60)
    
    # Paths
    raw_data_path = Path("data/raw/dataset.csv")
    processed_output = Path("data/processed/one_lot_data.csv")
    
    # Check if raw data exists
    if not raw_data_path.exists():
        print(f"❌ Raw data not found at: {raw_data_path}")
        print("   Please ensure data/raw/dataset.csv exists")
        return
    
    # Load raw data
    print(f"\n[1/4] Loading raw data from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)
    df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])
    print(f"   ✓ Loaded {len(df)} records")
    print(f"   ✓ Columns: {list(df.columns)}")
    
    # Check if SystemCodeNumber exists
    if 'SystemCodeNumber' in df.columns:
        lots = df['SystemCodeNumber'].unique()
        print(f"   ✓ Found {len(lots)} unique lots: {lots[:5]}...")
        
        # Filter to one lot (use first lot or BHMBCCMKT01 if exists)
        target_lot = 'BHMBCCMKT01' if 'BHMBCCMKT01' in lots else lots[0]
        print(f"\n[2/4] Filtering to lot: {target_lot}...")
        df_lot = df[df['SystemCodeNumber'] == target_lot].copy()
    else:
        print(f"\n[2/4] No SystemCodeNumber column, using all data...")
        df_lot = df.copy()
        target_lot = "single_lot"
    
    df_lot = df_lot.set_index('LastUpdated')
    print(f"   ✓ Filtered {len(df_lot)} records for lot {target_lot}")
    
    # Check capacity
    if 'Capacity' in df_lot.columns:
        capacity_values = df_lot['Capacity'].unique()
        print(f"   ✓ Capacity found in raw data: {capacity_values}")
        default_capacity = int(df_lot['Capacity'].mode()[0])
    else:
        print(f"   ⚠️  No Capacity column in raw data")
        default_capacity = 600
    
    # Process the data
    print(f"\n[3/4] Processing data with capacity={default_capacity}...")
    processed_data = process_lot_data(df_lot, default_capacity=default_capacity)
    print(f"   ✓ Processed {len(processed_data)} records")
    print(f"   ✓ Columns: {list(processed_data.columns)}")
    
    # Add lag features
    print(f"\n[4/4] Adding lag features...")
    df_with_lags = create_lag_features(processed_data, lags=[1, 2, 3, 24, 48])
    print(f"   ✓ Created lag features")
    print(f"   ✓ Final shape: {df_with_lags.shape}")
    print(f"   ✓ Final columns: {list(df_with_lags.columns)}")
    
    # Ensure output directory exists
    processed_output.parent.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    df_with_lags.to_csv(processed_output)
    print(f"\n✅ SUCCESS! Saved processed data to: {processed_output}")
    print(f"   📊 Records: {len(df_with_lags)}")
    print(f"   📋 Columns: {list(df_with_lags.columns)}")
    
    # Verify Capacity column
    if 'Capacity' in df_with_lags.columns:
        print(f"   ✓ Capacity column included: {df_with_lags['Capacity'].unique()}")
    else:
        print(f"   ❌ WARNING: Capacity column missing!")
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Retrain models: python scripts/train_models.py --model all --data-path data/processed/one_lot_data.csv")
    print("2. Restart API server: uvicorn app.main:app --reload --port 8000")
    print("=" * 60)

if __name__ == "__main__":
    main()
