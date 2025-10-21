# 📓 Notebook to Module Extraction Plan

This document tracks the extraction of code from `Main.ipynb` into modular Python files.

---

## 📊 Extraction Status

| Notebook Cell | Function/Code | Target Module | Status |
|---------------|---------------|---------------|--------|
| Cell 1 | Import libraries | N/A (used in scripts) | ✅ |
| Cell 4 | `segregated_lots` | `data_loader.py` | ⏳ |
| Cell 8 | `process_lot_data()` | `time_series_processor.py` | ⏳ |
| Cell 11 | LightGBM training | `lightgbm_model.py` | ✅ |
| Cell 12 | Multi-step forecasting | `forecaster.py` | ⏳ |
| Cell 15 | Queueing theory (Erlang-C) | `erlang_c.py` | ✅ |
| Cell 15 | `get_queueing_inputs()` | `queue_estimator.py` | ✅ |
| Cell 15 | `get_booking_confirmation()` | `booking_probability.py` | ✅ |

---

## ✅ Already Extracted

### 1. Queueing Theory (COMPLETE)
**From**: Cell #15 (`parking_queue_tools.py` section)  
**To**: `src/queueing/`

- ✅ `erlang_c.py` - `calculate_erlang_c()` function
- ✅ `queue_estimator.py` - `get_queueing_inputs()` function  
- ✅ `booking_probability.py` - `get_booking_confirmation()` function

**Code Location**: All queueing logic from lines 600-900 of Main.ipynb

### 2. LightGBM Model (COMPLETE)
**From**: Cells #11-14  
**To**: `src/models.py` - `LightGBMModel` class

Includes:
- Model initialization
- Training with validation
- Prediction
- Feature importance
- Model saving/loading

---

## ⏳ Pending Extraction

### 1. Data Processing Function
**From**: Cell #8 (lines 100-120)  
**To**: `src/preprocessing/time_series_processor.py`

**Function to extract**:
```python
def process_lot_data(lot_df):
    """
    Applies time series processing to a single parking lot's DataFrame.
    1. Sets 'LastUpdated' as the index.
    2. Resamples data to hourly frequency ('H'), taking the mean.
    3. Interpolates missing values.
    4. Engineers time-based features (hour, day of week, weekend).
    """
    df = lot_df.copy()
    df = df.sort_index()
    df_resampled = df['Occupancy'].resample('H').mean()
    df_resampled = df_resampled.interpolate(method='time')
    df_processed = df_resampled.to_frame()
    df_processed['hour_of_day'] = df_processed.index.hour
    df_processed['day_of_week'] = df_processed.index.dayofweek
    df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
    df_processed = df_processed.dropna()
    return df_processed
```

**Action Required**: Copy this function to `src/preprocessing/time_series_processor.py`

---

### 2. Data Loading Functions
**From**: Cells #2-5  
**To**: `src/preprocessing/data_loader.py`

**Functions to extract**:
```python
def load_parking_data(csv_path):
    """Load parking data from CSV."""
    df = pd.read_csv(csv_path)
    df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])
    return df

def segregate_by_lot(df):
    """Segregate data by parking lot."""
    return {lot_id: group_df for lot_id, group_df in df.groupby('SystemCodeNumber')}

def get_lot_data(df, lot_id):
    """Get data for specific parking lot."""
    segregated = segregate_by_lot(df)
    lot_df = segregated[lot_id]
    lot_df = lot_df.set_index('LastUpdated')
    return lot_df
```

**Action Required**: Create `src/preprocessing/data_loader.py` with these functions

---

### 3. Feature Engineering
**From**: Cell #13 (lag features creation)  
**To**: `src/preprocessing/feature_engineer.py`

**Functions to extract**:
```python
def create_lag_features(df, lags=[1, 2, 3, 24, 48]):
    """Create lag features for ML models."""
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['Occupancy'].shift(lag)
    return df.dropna()

def create_rolling_features(df, windows=[3, 24]):
    """Create rolling statistics features."""
    df = df.copy()
    for window in windows:
        df[f'rolling_mean_{window}'] = df['Occupancy'].rolling(window).mean()
        df[f'rolling_std_{window}'] = df['Occupancy'].rolling(window).std()
    return df
```

**Action Required**: Create `src/preprocessing/feature_engineer.py` with these functions

---

### 4. Multi-Step Forecasting
**From**: Cell #12 (Direct and Recursive strategies)  
**To**: `src/prediction/forecaster.py`

**Functions to extract**:
```python
def train_direct_models(df, features, target, max_horizon, params, num_boost_round):
    """Train separate model for each forecast horizon."""
    models = {}
    for h in range(1, max_horizon+1):
        df_h = df.copy()
        df_h['target_h'] = df_h[target].shift(-h)
        df_h = df_h.dropna()
        # Train model for horizon h
        ...
    return models

def predict_recursive(model, df_start, features, max_horizon):
    """Predict iteratively, feeding predictions back as features."""
    predictions = []
    df_work = df_start.copy()
    for h in range(1, max_horizon+1):
        # Predict next step
        # Update features with prediction
        ...
    return predictions
```

**Action Required**: Create `src/prediction/forecaster.py` with these functions

---

## 🎯 Extraction Priority

### High Priority (Do First)
1. ⏳ **Data Processing** (`time_series_processor.py`) - Used by all scripts
2. ⏳ **Data Loading** (`data_loader.py`) - Used by all scripts  
3. ⏳ **Feature Engineering** (`feature_engineer.py`) - Used by training scripts

### Medium Priority (Do Next)
4. ⏳ **Multi-Step Forecasting** (`forecaster.py`) - For advanced predictions
5. ⏳ **Visualization** (`visualization.py`) - For plotting
6. ⏳ **Metrics** (`metrics.py`) - Already in models.py, needs extraction

### Low Priority (Optional)
7. ⏳ **Data Cleaner** (`data_cleaner.py`) - Validation functions
8. ⏳ **Config Management** (`config.py`) - Load YAML configs
9. ⏳ **Logger** (`logger.py`) - Structured logging

---

## 🔧 How to Extract

For each function in the notebook:

1. **Copy the function** from the notebook cell
2. **Add docstring** with:
   - Purpose
   - Args
   - Returns
   - Example usage
3. **Add type hints** (optional but recommended)
4. **Create module file** if it doesn't exist
5. **Import in `__init__.py`** to make it accessible
6. **Update scripts** to use the new module
7. **Test** that everything still works

---

## 📝 Example Extraction

### Before (in notebook):
```python
# Cell #8
def process_lot_data(lot_df):
    df = lot_df.copy()
    df = df.sort_index()
    df_resampled = df['Occupancy'].resample('H').mean()
    df_resampled = df_resampled.interpolate(method='time')
    df_processed = df_resampled.to_frame()
    df_processed['hour_of_day'] = df_processed.index.hour
    df_processed['day_of_week'] = df_processed.index.dayofweek
    df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
    df_processed = df_processed.dropna()
    return df_processed
```

### After (in module):
```python
# src/preprocessing/time_series_processor.py

import pandas as pd
from typing import Union

def process_lot_data(lot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply time series processing to parking lot data.
    
    Processing steps:
    1. Sort by index (timestamp)
    2. Resample to hourly frequency using mean aggregation
    3. Interpolate missing values using time-weighted method
    4. Create time-based features (hour, day of week, weekend)
    5. Remove any remaining NaN values
    
    Args:
        lot_df: DataFrame with DatetimeIndex and 'Occupancy' column
        
    Returns:
        Processed DataFrame with:
            - 'Occupancy': Hourly resampled occupancy
            - 'hour_of_day': Hour (0-23)
            - 'day_of_week': Day of week (0=Monday, 6=Sunday)
            - 'is_weekend': Binary weekend indicator
    
    Example:
        >>> df = pd.read_csv('data.csv')
        >>> df = df.set_index('LastUpdated')
        >>> processed = process_lot_data(df)
        >>> print(processed.head())
    
    Based on: Main.ipynb Cell #8
    """
    # Create copy to avoid modifying original
    df = lot_df.copy()
    
    # Sort by timestamp
    df = df.sort_index()
    
    # Resample to hourly frequency
    df_resampled = df['Occupancy'].resample('H').mean()
    
    # Interpolate missing values
    df_resampled = df_resampled.interpolate(method='time')
    
    # Convert to DataFrame
    df_processed = df_resampled.to_frame()
    
    # Create time-based features
    df_processed['hour_of_day'] = df_processed.index.hour
    df_processed['day_of_week'] = df_processed.index.dayofweek
    df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
    
    # Remove NaN values
    df_processed = df_processed.dropna()
    
    return df_processed
```

---

## ✅ Checklist for Completing Extraction

### Preprocessing Module
- [ ] Create `src/preprocessing/data_loader.py`
- [ ] Create `src/preprocessing/data_cleaner.py`  
- [ ] Create `src/preprocessing/time_series_processor.py` ⚠️ **PRIORITY**
- [ ] Create `src/preprocessing/feature_engineer.py` ⚠️ **PRIORITY**
- [ ] Update `src/preprocessing/__init__.py`
- [ ] Test all functions work independently

### Prediction Module
- [ ] Create `src/prediction/forecaster.py`
- [ ] Create `src/prediction/direct_predictor.py`
- [ ] Create `src/prediction/recursive_predictor.py`
- [ ] Update `src/prediction/__init__.py`
- [ ] Test forecasting strategies

### Utils Module
- [ ] Create `src/utils/metrics.py` (extract from models.py)
- [ ] Create `src/utils/visualization.py`
- [ ] Create `src/utils/config.py`
- [ ] Create `src/utils/logger.py`
- [ ] Update `src/utils/__init__.py`

---

## 🚀 Quick Start for Extraction

### Step 1: Extract Data Processing (HIGHEST PRIORITY)
```bash
# Create the file
touch src/preprocessing/time_series_processor.py

# Copy the function from notebook Cell #8
# Add docstrings and type hints
# Test with: python -c "from src.preprocessing.time_series_processor import process_lot_data"
```

### Step 2: Update Scripts
```python
# In scripts that need it, change:
# from: inline function definition
# to: from src.preprocessing.time_series_processor import process_lot_data
```

### Step 3: Test
```bash
# Run demo to ensure it still works
python scripts/demo_pipeline.py
```

---

## 📚 References

- **Notebook**: `notebooks/Main.ipynb`
- **Target Structure**: `docs/DIRECTORY_STRUCTURE.md`
- **Migration Guide**: `docs/MIGRATION_GUIDE.md`

---

**Status**: ⏳ Extraction in progress  
**Priority**: Data processing functions (HIGH)  
**Next Action**: Extract `process_lot_data()` to `time_series_processor.py`

---

**Note**: The queueing theory extraction is complete! ✅  
The remaining extractions are for preprocessing and prediction modules.
