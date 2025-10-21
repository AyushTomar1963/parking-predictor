# Migration Guide

## Step-by-Step File Reorganization

### ✅ Already Created:
- New directory structure
- Configuration files
- Package __init__.py files

### 📦 Files to Move:

#### 1. Move existing `src/` files to proper subdirectories:

```bash
# From terminal in parking-predictor directory:

# Models
mv src/models.py src/models/lightgbm_model.py
# (Note: Will need to split ARIMA and LightGBM into separate files)

# Preprocessing
mv src/data_utils.py src/preprocessing/data_loader.py
mv src/feature_engineering.py src/preprocessing/feature_engineer.py

# Prediction
mv src/prediction_utils.py src/prediction/forecaster.py

# API
mv src/api_helpers.py src/api/response_formatter.py

# Booking (split into queueing package)
# src/booking_utils.py → will be split into:
#   - src/queueing/booking_probability.py
#   - src/queueing/erlang_c.py (from notebook Cell 21)
```

### 🔧 Manual Migration Steps:

#### Step 1: Create Queueing Package Files

Extract from notebook Cell 21:

1. **Create `src/queueing/erlang_c.py`:**
   - Copy Erlang-C functions from notebook
   - `_safe_pow_div()`, `calculate_erlang_c()`

2. **Create `src/queueing/queue_estimator.py`:**
   - Copy `get_queueing_inputs()` from notebook

3. **Create `src/queueing/booking_probability.py`:**
   - Copy `get_booking_confirmation()` from notebook
   - Copy booking logic from `src/booking_utils.py`

#### Step 2: Split Models File

**From `src/models.py`, create:**

1. **`src/models/arima_model.py`:**
   ```python
   class ARIMAXModel:
       # Copy ARIMAX class
   ```

2. **`src/models/lightgbm_model.py`:**
   ```python
   class LSTMModel:
       # Copy LSTM class (or remove if not used)
   
   def evaluate_model(y_true, y_pred):
       # Copy evaluation function
   ```

3. **`src/models/model_trainer.py`:**
   - Extract training logic from notebook Cell 20
   - `train_direct_models()`, `train_recursive_model()`

#### Step 3: Create Prediction Package

From notebook Cell 20, create:

1. **`src/prediction/forecaster.py`:**
   - Copy `forecast_next_hours()`, `batch_predict()` from `prediction_utils.py`
   - Add `make_lag_features()` from notebook

2. **`src/prediction/direct_predictor.py`:**
   - Copy `train_direct_models()`, `predict_direct()`

3. **`src/prediction/recursive_predictor.py`:**
   - Copy `train_recursive_model()`, `predict_recursive()`

#### Step 4: Create Preprocessing Package

1. **`src/preprocessing/time_series_processor.py`:**
   - Copy `process_lot_data()` from notebook Cell 8
   - Copy resampling and interpolation logic

2. **`src/preprocessing/data_cleaner.py`:**
   - Add data validation logic
   - Handle missing values, outliers

#### Step 5: Create Utils Package

1. **`src/utils/metrics.py`:**
   ```python
   def calculate_mae(y_true, y_pred):
       pass
   
   def calculate_rmse(y_true, y_pred):
       pass
   ```

2. **`src/utils/visualization.py`:**
   - Extract plotting code from notebook
   - Create reusable plot functions

3. **`src/utils/config.py`:**
   ```python
   import yaml
   
   def load_config(config_path):
       with open(config_path) as f:
           return yaml.safe_load(f)
   ```

4. **`src/utils/logger.py`:**
   ```python
   import logging
   
   def setup_logger(name, level="INFO"):
       # Logger configuration
       pass
   ```

#### Step 6: Create App Routes

1. **`app/routes/prediction_routes.py`:**
   ```python
   from fastapi import APIRouter
   
   router = APIRouter(prefix="/api/predict", tags=["predictions"])
   
   @router.get("/")
   async def predict_occupancy(hours: int = 24):
       # Prediction endpoint
       pass
   ```

2. **`app/routes/booking_routes.py`:**
   ```python
   from fastapi import APIRouter
   
   router = APIRouter(prefix="/api/booking", tags=["booking"])
   
   @router.post("/confirm")
   async def confirm_booking(...):
       # Booking endpoint
       pass
   ```

3. **`app/routes/analytics_routes.py`:**
   ```python
   from fastapi import APIRouter
   
   router = APIRouter(prefix="/api/analytics", tags=["analytics"])
   
   @router.get("/stats")
   async def get_statistics():
       # Analytics endpoint
       pass
   ```

#### Step 7: Update Main App

Update `app/main.py` to include new routes:

```python
from fastapi import FastAPI
from app.routes import prediction_routes, booking_routes, analytics_routes

app = FastAPI(title="Parking Predictor API")

# Include routers
app.include_router(prediction_routes.router)
app.include_router(booking_routes.router)
app.include_router(analytics_routes.router)
```

### 🔄 Update Import Statements

After moving files, update all imports:

**Old imports:**
```python
from src.models import ARIMAXModel
from src.prediction_utils import forecast_next_hours
from src.booking_utils import ParkingBookingSystem
```

**New imports:**
```python
from src.models.arima_model import ARIMAXModel
from src.prediction.forecaster import forecast_next_hours
from src.queueing.booking_probability import calculate_booking_probability
```

### 🧪 Testing After Migration

```bash
# 1. Run Python syntax check
python -m py_compile src/**/*.py

# 2. Try importing packages
python -c "from src.models.lightgbm_model import LSTMModel; print('✓ Models import OK')"
python -c "from src.queueing.erlang_c import calculate_erlang_c; print('✓ Queueing import OK')"

# 3. Run tests (if created)
pytest tests/

# 4. Start app
python app/main.py
```

### 📝 Checklist

- [ ] Create all new directories
- [ ] Move files to new locations
- [ ] Split `models.py` into separate files
- [ ] Extract queueing code from notebook
- [ ] Extract prediction code from notebook
- [ ] Create preprocessing modules
- [ ] Create utility modules
- [ ] Create API routes
- [ ] Update all import statements
- [ ] Update `app/main.py`
- [ ] Test imports
- [ ] Run application
- [ ] Update documentation
- [ ] Commit changes to git

### ⚠️ Important Notes:

1. **Keep Original Files**: Don't delete originals until migration is complete and tested
2. **Git Branch**: Create a new branch for migration: `git checkout -b feature/restructure`
3. **Test Incrementally**: Test after each major move
4. **Notebooks**: Keep notebooks in `notebooks/` folder untouched
5. **Backup**: Create a backup before starting: `cp -r parking-predictor parking-predictor-backup`

### 🎯 Priority Order:

1. ✅ **High Priority** (Core functionality):
   - Queueing package (needed for booking)
   - Models package (needed for predictions)
   - Prediction package (core feature)

2. **Medium Priority** (Improves organization):
   - Preprocessing package
   - API package
   - Utils package

3. **Low Priority** (Nice to have):
   - Scripts folder
   - Additional routes
   - Advanced features

### 🚀 Quick Start After Migration:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app/main.py

# Access the API
curl http://localhost:8000/api/predict?hours=24

# View API docs
open http://localhost:8000/docs
```

---

**Estimated Time**: 2-3 hours for complete migration
**Difficulty**: Medium
**Risk**: Low (if following incremental testing approach)
