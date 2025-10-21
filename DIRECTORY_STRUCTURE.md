# Parking Predictor - Directory Structure

## 📁 Complete Project Structure

```
parking-predictor/
│
├── 📂 app/                          # Flask/FastAPI Web Application
│   ├── __init__.py
│   ├── main.py                      # Main application entry point
│   ├── routes/                      # API route handlers
│   │   ├── __init__.py
│   │   ├── prediction_routes.py    # Prediction endpoints
│   │   ├── booking_routes.py       # Booking endpoints
│   │   └── analytics_routes.py     # Analytics & stats endpoints
│   ├── static/                      # Static assets (CSS, JS, images)
│   │   ├── css/
│   │   │   └── style.css
│   │   ├── js/
│   │   │   └── app.js
│   │   └── index.html              # Main frontend page
│   └── templates/                   # HTML templates (if using Jinja2)
│       └── dashboard.html
│
├── 📂 src/                          # Source code (business logic)
│   ├── __init__.py
│   │
│   ├── 📂 models/                   # ML Model implementations
│   │   ├── __init__.py
│   │   ├── arima_model.py          # ARIMA model class
│   │   ├── lightgbm_model.py       # LightGBM model class
│   │   ├── model_trainer.py        # Model training utilities
│   │   └── model_loader.py         # Load saved models
│   │
│   ├── 📂 preprocessing/            # Data preprocessing
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Load raw data
│   │   ├── data_cleaner.py         # Clean & validate data
│   │   ├── time_series_processor.py # Resample, interpolate
│   │   └── feature_engineer.py     # Create features (moved from root)
│   │
│   ├── 📂 prediction/               # Prediction logic
│   │   ├── __init__.py
│   │   ├── forecaster.py           # Multi-step forecasting
│   │   ├── direct_predictor.py     # Direct strategy
│   │   └── recursive_predictor.py  # Recursive strategy
│   │
│   ├── 📂 queueing/                 # Queueing theory implementation
│   │   ├── __init__.py
│   │   ├── erlang_c.py             # Erlang-C formula
│   │   ├── queue_estimator.py      # Estimate λ and μ from data
│   │   └── booking_probability.py  # Calculate booking success probability
│   │
│   ├── 📂 utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── metrics.py              # MAE, RMSE, etc.
│   │   ├── visualization.py        # Plotting utilities
│   │   ├── config.py               # Configuration management
│   │   └── logger.py               # Logging setup
│   │
│   └── 📂 api/                      # API helper functions
│       ├── __init__.py
│       ├── request_validator.py    # Validate API requests
│       ├── response_formatter.py   # Format API responses
│       └── error_handlers.py       # Error handling
│
├── 📂 data/                         # Data storage
│   ├── raw/                         # Original unprocessed data
│   │   └── dataset.csv             # Birmingham parking data
│   ├── processed/                   # Cleaned & processed data
│   │   ├── BHMBCCMKT01_processed.csv
│   │   └── all_lots_processed.csv
│   ├── models/                      # Saved trained models
│   │   ├── lightgbm_model.pkl
│   │   ├── arima_model.pkl
│   │   └── model_metadata.json
│   └── cache/                       # Temporary cache files
│       └── predictions_cache.json
│
├── 📂 notebooks/                    # Jupyter notebooks (NOT TOUCHED)
│   ├── arima.ipynb                 # Main analysis notebook
│   ├── exploratory_analysis.ipynb  # EDA
│   └── model_experiments.ipynb     # Model testing
│
├── 📂 tests/                        # Unit & integration tests
│   ├── __init__.py
│   ├── test_models.py              # Test ML models
│   ├── test_preprocessing.py       # Test data processing
│   ├── test_queueing.py            # Test queueing theory
│   ├── test_api.py                 # Test API endpoints
│   └── fixtures/                   # Test data fixtures
│       └── sample_data.csv
│
├── 📂 scripts/                      # Utility scripts
│   ├── train_models.py             # Train and save models
│   ├── evaluate_models.py          # Evaluate model performance
│   ├── batch_predict.py            # Batch prediction script
│   └── data_pipeline.py            # ETL pipeline
│
├── 📂 config/                       # Configuration files
│   ├── app_config.yaml             # App configuration
│   ├── model_config.yaml           # Model hyperparameters
│   └── logging_config.yaml         # Logging configuration
│
├── 📂 docs/                         # Documentation
│   ├── API.md                      # API documentation
│   ├── MODELS.md                   # Model documentation
│   ├── DEPLOYMENT.md               # Deployment guide
│   └── USER_GUIDE.md               # User guide
│
├── 📄 requirements.txt              # Python dependencies
├── 📄 requirements-dev.txt          # Development dependencies
├── 📄 setup.py                      # Package setup
├── 📄 .env.example                  # Environment variables template
├── 📄 .gitignore                    # Git ignore rules
├── 📄 README.md                     # Project overview
├── 📄 summary.md                    # Analysis summary
├── 📄 DIRECTORY_STRUCTURE.md        # This file
└── 📄 docker-compose.yml            # Docker setup (optional)

```

---

## 📋 File Migration Plan

### Files to Move:

1. **From `src/` root → `src/models/`:**
   - `models.py` → `src/models/lightgbm_model.py` & `arima_model.py` (split)

2. **From `src/` root → `src/preprocessing/`:**
   - `data_utils.py` → `src/preprocessing/data_loader.py`
   - `feature_engineering.py` → `src/preprocessing/feature_engineer.py`

3. **From `src/` root → `src/prediction/`:**
   - `prediction_utils.py` → `src/prediction/forecaster.py`

4. **From `src/` root → `src/api/`:**
   - `api_helpers.py` → `src/api/response_formatter.py`

5. **From `src/` root → `src/queueing/`:**
   - `booking_utils.py` → Split into:
     - `src/queueing/erlang_c.py` (Erlang-C from notebook)
     - `src/queueing/booking_probability.py` (booking logic)

6. **From `app/` → Restructure:**
   - Keep `main.py` but refactor to use new routes
   - Add `routes/` folder with split endpoints

---

## 🎯 Key Improvements

### 1. **Modular Organization**
   - Separate concerns: models, preprocessing, prediction, queueing
   - Each module has a single responsibility

### 2. **Scalable App Structure**
   - Routes separated by functionality
   - Easy to add new endpoints
   - Clear API structure

### 3. **Better Data Management**
   - Raw vs processed data separation
   - Model artifacts stored with metadata
   - Cache for performance optimization

### 4. **Testing Infrastructure**
   - Dedicated test folder with fixtures
   - Easy to run unit tests
   - Integration tests for API

### 5. **Configuration Management**
   - Centralized config files
   - Environment-based settings
   - Easy deployment configuration

### 6. **Documentation**
   - Comprehensive docs folder
   - API documentation
   - Deployment guides

---

## 🚀 Usage Guide

### Development Workflow:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (using notebook logic)
python scripts/train_models.py

# 3. Run tests
pytest tests/

# 4. Start development server
python app/main.py

# 5. Access application
# Browser: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Project Commands:

```bash
# Train models
python scripts/train_models.py --lot BHMBCCMKT01

# Evaluate models
python scripts/evaluate_models.py --model lightgbm

# Run batch predictions
python scripts/batch_predict.py --hours 24

# Run API server
uvicorn app.main:app --reload --port 8000
```

---

## 📦 Package Structure

### Main Packages:

1. **`src.models`** - Machine learning models
   - ARIMA, LightGBM implementations
   - Model training and evaluation

2. **`src.preprocessing`** - Data pipeline
   - Load, clean, transform data
   - Feature engineering

3. **`src.prediction`** - Forecasting
   - Direct and recursive strategies
   - Multi-step forecasting

4. **`src.queueing`** - Queueing theory
   - Erlang-C calculations
   - Booking probability estimation

5. **`src.utils`** - Utilities
   - Metrics, visualization, config
   - Common helper functions

6. **`app`** - Web application
   - API routes and endpoints
   - Frontend static files

---

## 🔧 Configuration Files

### `config/app_config.yaml`
```yaml
app:
  name: "Parking Predictor"
  version: "1.0.0"
  host: "0.0.0.0"
  port: 8000
  debug: true

database:
  path: "data/processed/"

cache:
  enabled: true
  ttl: 300  # seconds
```

### `config/model_config.yaml`
```yaml
lightgbm:
  num_boost_round: 200
  learning_rate: 0.05
  max_depth: 7
  num_leaves: 31

arima:
  order: [1, 0, 0]
  seasonal_order: [1, 1, 1, 24]

features:
  lags: [1, 2, 3, 24, 48]
  time_features: ["hour_of_day", "day_of_week", "is_weekend"]
```

---

## 🎨 Frontend Structure

### Static Files Organization:

```
app/static/
├── css/
│   ├── style.css           # Main styles
│   ├── dashboard.css       # Dashboard-specific
│   └── mobile.css          # Mobile responsive
├── js/
│   ├── app.js              # Main application logic
│   ├── charts.js           # Chart visualizations
│   ├── api.js              # API client
│   └── utils.js            # Helper functions
├── images/
│   ├── logo.png
│   └── icons/
└── index.html              # Single-page app
```

---

## 🧪 Testing Structure

```
tests/
├── unit/                   # Unit tests
│   ├── test_models.py
│   ├── test_preprocessing.py
│   └── test_queueing.py
├── integration/            # Integration tests
│   ├── test_api_endpoints.py
│   └── test_prediction_pipeline.py
├── fixtures/               # Test data
│   ├── sample_data.csv
│   └── mock_predictions.json
└── conftest.py            # Pytest configuration
```

---

## 📊 Data Flow

```
Raw Data (CSV)
    ↓
[Data Loader] → src/preprocessing/data_loader.py
    ↓
[Data Cleaner] → src/preprocessing/data_cleaner.py
    ↓
[Time Series Processor] → src/preprocessing/time_series_processor.py
    ↓
[Feature Engineer] → src/preprocessing/feature_engineer.py
    ↓
Processed Data → data/processed/
    ↓
[Model Trainer] → src/models/model_trainer.py
    ↓
Trained Models → data/models/
    ↓
[Forecaster] → src/prediction/forecaster.py
    ↓
Predictions
    ↓
[Queueing Calculator] → src/queueing/booking_probability.py
    ↓
Booking Probabilities
    ↓
[API Response] → app/routes/prediction_routes.py
    ↓
User (Web/Mobile App)
```

---

## 🌟 Best Practices Implemented

1. ✅ **Separation of Concerns** - Each module has a single responsibility
2. ✅ **DRY Principle** - No code duplication
3. ✅ **Configuration Management** - Externalized configs
4. ✅ **Error Handling** - Comprehensive error handling
5. ✅ **Logging** - Structured logging throughout
6. ✅ **Testing** - Unit and integration tests
7. ✅ **Documentation** - Inline and external docs
8. ✅ **Version Control** - Proper .gitignore
9. ✅ **Scalability** - Easy to add features
10. ✅ **Maintainability** - Clean, readable code

---

## 🚢 Deployment Options

### Option 1: Local Development
```bash
uvicorn app.main:app --reload
```

### Option 2: Production (Gunicorn)
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Option 3: Docker
```bash
docker-compose up --build
```

### Option 4: Cloud Deployment
- Heroku
- AWS Lambda
- Google Cloud Run
- Azure App Service

---

## 📚 Additional Resources

- **API Documentation**: `/docs` (auto-generated by FastAPI)
- **Model Notebooks**: `notebooks/arima.ipynb`
- **Analysis Summary**: `summary.md`
- **GitHub Repository**: [Link to repo]

---

## 🔄 Migration Steps (Detailed)

See `MIGRATION_GUIDE.md` for step-by-step instructions on:
1. Creating new directory structure
2. Moving files to new locations
3. Updating import statements
4. Testing after migration
5. Updating documentation

---

**Last Updated**: October 20, 2025
**Version**: 1.0.0
**Maintainer**: Parking Predictor Team
