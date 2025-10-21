"""
Data Pipeline (ETL) Script

Extract, Transform, Load pipeline for parking data.
Handles data loading, cleaning, preprocessing, and feature engineering.

Usage:
    python scripts/data_pipeline.py --input data/raw/dataset.csv --output data/processed/
    python scripts/data_pipeline.py --lot BHMBCCMKT01 --save-individual
"""

import argparse
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.time_series_processor import process_lot_data


def extract_data(file_path):
    """
    Extract: Load raw parking data from CSV.
    
    Returns:
        DataFrame with parking lot data
    """
    print("\n" + "=" * 60)
    print("EXTRACT: Loading Raw Data")
    print("=" * 60)
    
    df = pd.read_csv(file_path)
    
    print(f"   ✓ Loaded {len(df)} records")
    print(f"   ✓ Columns: {list(df.columns)}")
    print(f"   ✓ Date range: {df['LastUpdated'].min()} to {df['LastUpdated'].max()}")
    
    # Basic data info
    print(f"\n   Data Info:")
    print(f"   - Unique lots: {df['SystemCodeNumber'].nunique()}")
    print(f"   - Total capacity: {df['Capacity'].sum()}")
    print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


def transform_data(df, lot_id=None, validate=True):
    """
    Transform: Clean and process parking data.
    
    Steps:
        1. Convert timestamps
        2. Filter by lot (optional)
        3. Remove duplicates
        4. Handle missing values
        5. Validate data quality
        6. Resample to hourly
        7. Create features
    
    Returns:
        Dictionary of {lot_id: processed_dataframe}
    """
    print("\n" + "=" * 60)
    print("TRANSFORM: Cleaning and Processing Data")
    print("=" * 60)
    
    # Step 1: Convert timestamps
    print("\n[1/7] Converting timestamps...")
    df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])
    print(f"   ✓ Converted to datetime")
    
    # Step 2: Filter by lot (optional)
    if lot_id:
        print(f"\n[2/7] Filtering for lot: {lot_id}...")
        df = df[df['SystemCodeNumber'] == lot_id].copy()
        print(f"   ✓ Filtered to {len(df)} records")
    else:
        print(f"\n[2/7] Processing all lots...")
    
    # Step 3: Remove duplicates
    print("\n[3/7] Removing duplicates...")
    initial_count = len(df)
    df = df.drop_duplicates(subset=['SystemCodeNumber', 'LastUpdated'])
    removed = initial_count - len(df)
    print(f"   ✓ Removed {removed} duplicate records")
    
    # Step 4: Handle missing values
    print("\n[4/7] Handling missing values...")
    missing_before = df.isnull().sum().sum()
    
    # Fill missing occupancy with forward fill then backward fill
    df['Occupancy'] = df.groupby('SystemCodeNumber')['Occupancy'].fillna(method='ffill').fillna(method='bfill')
    
    # Fill missing capacity with mode per lot
    df['Capacity'] = df.groupby('SystemCodeNumber')['Capacity'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.median())
    )
    
    missing_after = df.isnull().sum().sum()
    print(f"   ✓ Handled {missing_before - missing_after} missing values")
    
    # Step 5: Validate data quality
    if validate:
        print("\n[5/7] Validating data quality...")
        
        # Check for negative occupancy
        negative_occ = (df['Occupancy'] < 0).sum()
        if negative_occ > 0:
            print(f"   ⚠️  Found {negative_occ} negative occupancy values - setting to 0")
            df.loc[df['Occupancy'] < 0, 'Occupancy'] = 0
        
        # Check for occupancy > capacity
        over_capacity = (df['Occupancy'] > df['Capacity']).sum()
        if over_capacity > 0:
            print(f"   ⚠️  Found {over_capacity} records where occupancy > capacity")
            df.loc[df['Occupancy'] > df['Capacity'], 'Occupancy'] = df.loc[df['Occupancy'] > df['Capacity'], 'Capacity']
        
        print(f"   ✓ Data validation complete")
    
    # Step 6 & 7: Process each lot
    print("\n[6/7] Resampling to hourly frequency...")
    print("[7/7] Creating time-based features...")
    
    processed_lots = {}
    
    # Get unique lots
    lots = df['SystemCodeNumber'].unique()
    
    for lot in lots:
        lot_df = df[df['SystemCodeNumber'] == lot].copy()
        lot_df = lot_df.set_index('LastUpdated')
        
        # Use the process_lot_data function
        processed = process_lot_data(lot_df)
        
        # Add lot metadata
        processed['SystemCodeNumber'] = lot
        processed['Capacity'] = lot_df['Capacity'].mode()[0] if not lot_df['Capacity'].mode().empty else lot_df['Capacity'].median()
        
        processed_lots[lot] = processed
        
        print(f"   ✓ Processed lot {lot}: {len(processed)} records")
    
    print(f"\n   ✓ Processed {len(processed_lots)} lots")
    
    return processed_lots


def load_data(processed_lots, output_dir, save_individual=True, save_combined=True):
    """
    Load: Save processed data to files.
    
    Args:
        processed_lots: Dictionary of {lot_id: processed_dataframe}
        output_dir: Directory to save processed files
        save_individual: Save individual lot files
        save_combined: Save combined file with all lots
    """
    print("\n" + "=" * 60)
    print("LOAD: Saving Processed Data")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Save individual lot files
    if save_individual:
        print("\n   Saving individual lot files...")
        for lot_id, lot_df in processed_lots.items():
            file_path = output_path / f"{lot_id}_processed.csv"
            lot_df.to_csv(file_path)
            saved_files.append(str(file_path))
            print(f"   ✓ Saved {lot_id} to {file_path}")
    
    # Save combined file
    if save_combined:
        print("\n   Saving combined file...")
        combined_df = pd.concat(processed_lots.values(), axis=0)
        combined_path = output_path / "all_lots_processed.csv"
        combined_df.to_csv(combined_path)
        saved_files.append(str(combined_path))
        print(f"   ✓ Saved combined data to {combined_path}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'num_lots': len(processed_lots),
        'lots': list(processed_lots.keys()),
        'total_records': sum(len(df) for df in processed_lots.values()),
        'files_saved': saved_files
    }
    
    metadata_path = output_path / "pipeline_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n   ✓ Saved metadata to {metadata_path}")
    
    return saved_files


def generate_summary_statistics(processed_lots, output_dir):
    """Generate and save summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    summary = []
    
    for lot_id, lot_df in processed_lots.items():
        stats = {
            'lot_id': lot_id,
            'records': len(lot_df),
            'capacity': int(lot_df['Capacity'].iloc[0]) if 'Capacity' in lot_df.columns else 'N/A',
            'avg_occupancy': float(lot_df['Occupancy'].mean()),
            'min_occupancy': float(lot_df['Occupancy'].min()),
            'max_occupancy': float(lot_df['Occupancy'].max()),
            'std_occupancy': float(lot_df['Occupancy'].std()),
            'utilization': float(lot_df['Occupancy'].mean() / lot_df['Capacity'].iloc[0] * 100) if 'Capacity' in lot_df.columns else 'N/A'
        }
        summary.append(stats)
    
    # Print summary
    print(f"\n{'Lot ID':<15} {'Records':<10} {'Capacity':<10} {'Avg Occ':<10} {'Utilization':<12}")
    print("-" * 70)
    
    for s in summary:
        util = f"{s['utilization']:.1f}%" if isinstance(s['utilization'], float) else s['utilization']
        print(f"{s['lot_id']:<15} {s['records']:<10} {s['capacity']:<10} {s['avg_occupancy']:<10.1f} {util:<12}")
    
    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_path = Path(output_dir) / "summary_statistics.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n   ✓ Saved summary statistics to {summary_path}")
    
    return summary


def main(args):
    """Main ETL pipeline."""
    
    print("=" * 60)
    print("PARKING PREDICTOR - ETL PIPELINE")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Lot filter: {args.lot if args.lot else 'All lots'}")
    print("=" * 60)
    
    # EXTRACT
    raw_df = extract_data(args.input)
    
    # TRANSFORM
    processed_lots = transform_data(raw_df, lot_id=args.lot, validate=not args.no_validate)
    
    # LOAD
    saved_files = load_data(
        processed_lots, 
        args.output,
        save_individual=args.save_individual,
        save_combined=not args.no_combined
    )
    
    # SUMMARY
    if args.summary:
        generate_summary_statistics(processed_lots, args.output)
    
    print("\n" + "=" * 60)
    print("ETL PIPELINE COMPLETED SUCCESSFULLY! ✓")
    print("=" * 60)
    print(f"\n   Processed {len(processed_lots)} lot(s)")
    print(f"   Saved {len(saved_files)} file(s)")
    print(f"   Output directory: {args.output}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ETL pipeline for parking data')
    
    parser.add_argument('--input', type=str, default='data/raw/dataset.csv',
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, default='data/processed/',
                        help='Output directory for processed files')
    parser.add_argument('--lot', type=str, default=None,
                        help='Filter for specific lot (optional)')
    parser.add_argument('--save-individual', action='store_true',
                        help='Save individual lot files')
    parser.add_argument('--no-combined', action='store_true',
                        help='Do not save combined file')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip data validation')
    parser.add_argument('--summary', action='store_true',
                        help='Generate summary statistics')
    
    args = parser.parse_args()
    
    main(args)
