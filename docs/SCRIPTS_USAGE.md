# Scripts Usage Guide

Complete guide for all utility scripts in the `scripts/` directory.

---

## 📂 Available Scripts

| Script | Purpose | Use Case |
|--------|---------|----------|
| `demo_pipeline.py` | End-to-end demo | Quick demonstration of complete pipeline |
| `data_pipeline.py` | ETL pipeline | Process raw data into clean datasets |
| `train_models.py` | Model training | Train and save models |
| `evaluate_models.py` | Model evaluation | Evaluate trained models with metrics |
| `batch_predict.py` | Batch predictions | Generate forecasts for multiple hours |

---

## 1. 🎯 demo_pipeline.py - Quick Demo

**Purpose**: Show the complete pipeline in action (Load → Preprocess → Train → Predict → Booking Probability)

### Basic Usage:
```bash
python scripts/demo_pipeline.py
```

### Advanced Usage:
```bash
# Specify custom lot and capacity
python scripts/demo_pipeline.py --lot BHMBCCMKT01 --capacity 600

# Save the trained model
python scripts/demo_pipeline.py --save-model

# Custom data path
python scripts/demo_pipeline.py --data-path data/raw/dataset.csv --lot BHMBCCMKT02
```

### Output:
- Console output with all pipeline steps
- Model training metrics (MAE, RMSE, MAPE)
- Booking probability prediction
- User-friendly recommendation

---

## 2. 🔄 data_pipeline.py - ETL Pipeline

**Purpose**: Extract, Transform, Load parking data into clean, processed datasets

### Basic Usage:
```bash
python scripts/data_pipeline.py
```

### Advanced Usage:
```bash
# Process all lots and save individual files
python scripts/data_pipeline.py --save-individual --summary

# Process specific lot only
python scripts/data_pipeline.py --lot BHMBCCMKT01 --save-individual

# Custom input/output paths
python scripts/data_pipeline.py \
  --input data/raw/dataset.csv \
  --output data/processed/ \
  --summary

# Skip validation (faster processing)
python scripts/data_pipeline.py --no-validate
```

### Output Files:
- `data/processed/{LOT_ID}_processed.csv` - Individual lot data
- `data/processed/all_lots_processed.csv` - Combined data
- `data/processed/pipeline_metadata.json` - Processing metadata
- `data/processed/summary_statistics.csv` - Summary stats (with --summary)

### What It Does:
1. **Extract**: Load raw CSV data
2. **Transform**:
   - Convert timestamps
   - Remove duplicates
   - Handle missing values
   - Validate data quality
   - Resample to hourly frequency
   - Create time-based features
3. **Load**: Save processed files

---

## 3. 🏋️ train_models.py - Model Training

**Purpose**: Train multiple ML models and save them for later use

### Basic Usage:
```bash
# Train default models (LightGBM, XGBoost, ARIMA)
python scripts/train_models.py
```

### Advanced Usage:
```bash
# Train specific models only
python scripts/train_models.py --models lightgbm,xgboost

# Train all available models
python scripts/train_models.py --models all

# Custom lot and capacity
python scripts/train_models.py --lot BHMBCCMKT01 --capacity 600

# Custom train/val split
python scripts/train_models.py --split 0.7

# Custom output directory
python scripts/train_models.py --output-dir models/production/
```

### Output Files:
- `data/models/lightgbm_model.txt` - Trained LightGBM model
- `data/models/xgboost_model.json` - Trained XGBoost model
- `data/models/arima_model.pkl` - Trained ARIMA model
- `data/models/model_metadata.json` - Training metadata

### Model Options:
- `lightgbm` - Fast gradient boosting (recommended for production)
- `xgboost` - Alternative gradient boosting
- `arima` - Statistical time series model
- `all` - Train all available models

---

## 4. 📊 evaluate_models.py - Model Evaluation

**Purpose**: Evaluate trained models with detailed metrics and visualizations

### Basic Usage:
```bash
# Evaluate all models
python scripts/evaluate_models.py --model all
```

### Advanced Usage:
```bash
# Evaluate specific model
python scripts/evaluate_models.py --model lightgbm

# Generate plots
python scripts/evaluate_models.py --model all --plot

# Custom model directory
python scripts/evaluate_models.py \
  --model-dir data/models/ \
  --output-dir results/ \
  --plot
```

### Output:
- Console metrics (MAE, RMSE, MAPE)
- Feature importance (for tree-based models)
- Model comparison table
- Best model recommendation

### With --plot flag:
- `{model}_residuals.png` - Residual analysis plots
- `model_comparison.png` - Predictions vs actual for all models
- `evaluation_report.json` - Detailed metrics JSON

---

## 5. 🔮 batch_predict.py - Batch Predictions

**Purpose**: Generate predictions for multiple hours ahead

### Basic Usage:
```bash
# Predict next 24 hours
python scripts/batch_predict.py --hours 24
```

### Advanced Usage:
```bash
# Predict next week (168 hours) with all models
python scripts/batch_predict.py --hours 168 --model all

# Include booking probabilities
python scripts/batch_predict.py --hours 24 --booking-prob

# Custom output file
python scripts/batch_predict.py \
  --hours 48 \
  --model lightgbm \
  --output predictions_48h.csv

# Specific lot and capacity
python scripts/batch_predict.py \
  --lot BHMBCCMKT01 \
  --capacity 600 \
  --hours 72 \
  --booking-prob
```

### Output File:
CSV file with columns:
- `timestamp` - Prediction timestamp
- `{model}_prediction` - Predicted occupancy for each model
- `hour_of_day`, `day_of_week`, `day_name`, `is_weekend` - Time features
- `{model}_available_slots` - Available spots (with --booking-prob)
- `{model}_prob_get_spot` - Booking probability (with --booking-prob)
- `{model}_expected_wait_min` - Expected wait time (with --booking-prob)

---

## 📈 Complete Workflow Examples

### Example 1: First Time Setup
```bash
# Step 1: Process raw data
python scripts/data_pipeline.py --save-individual --summary

# Step 2: Train models
python scripts/train_models.py --models all

# Step 3: Evaluate models
python scripts/evaluate_models.py --model all --plot

# Step 4: Generate predictions
python scripts/batch_predict.py --hours 24 --booking-prob
```

### Example 2: Quick Demo
```bash
# Run complete demo in one command
python scripts/demo_pipeline.py --save-model
```

### Example 3: Production Forecasting
```bash
# Train production model
python scripts/train_models.py \
  --models lightgbm \
  --output-dir models/production/

# Generate weekly forecast with booking probabilities
python scripts/batch_predict.py \
  --model lightgbm \
  --model-dir models/production/ \
  --hours 168 \
  --booking-prob \
  --output forecasts/week_ahead.csv
```

### Example 4: Model Comparison
```bash
# Train all models
python scripts/train_models.py --models all

# Evaluate and compare
python scripts/evaluate_models.py --model all --plot

# Batch predict with all models
python scripts/batch_predict.py --hours 48 --model all
```

---

## 🎯 Common Use Cases

### Use Case 1: Daily Operations
**Goal**: Update predictions for the next 24 hours

```bash
# Morning routine - generate today's predictions
python scripts/batch_predict.py \
  --hours 24 \
  --model lightgbm \
  --booking-prob \
  --output forecasts/today.csv
```

### Use Case 2: Weekly Planning
**Goal**: Forecast for the upcoming week

```bash
# Generate week-ahead forecast
python scripts/batch_predict.py \
  --hours 168 \
  --model all \
  --booking-prob \
  --output forecasts/this_week.csv
```

### Use Case 3: Model Retraining
**Goal**: Retrain models with latest data

```bash
# 1. Process new data
python scripts/data_pipeline.py --input data/raw/latest_dataset.csv

# 2. Retrain models
python scripts/train_models.py --models all

# 3. Evaluate performance
python scripts/evaluate_models.py --model all --plot
```

### Use Case 4: New Lot Analysis
**Goal**: Set up predictions for a new parking lot

```bash
# Process specific lot
python scripts/data_pipeline.py --lot BHMBCCMKT05 --save-individual

# Train models for that lot
python scripts/train_models.py --lot BHMBCCMKT05 --capacity 800

# Generate predictions
python scripts/batch_predict.py --lot BHMBCCMKT05 --capacity 800 --hours 24
```

---

## 🛠️ Troubleshooting

### Problem: "File not found" error
```bash
# Check if data file exists
ls data/raw/dataset.csv

# Create directories if needed
mkdir -p data/raw data/processed data/models
```

### Problem: "No models found" in evaluation
```bash
# Train models first
python scripts/train_models.py --models all
```

### Problem: Import errors
```bash
# Ensure you're in the project root directory
cd /path/to/parking-predictor

# Install dependencies
pip install -r requirements.txt
```

### Problem: Memory issues with large datasets
```bash
# Process one lot at a time
python scripts/data_pipeline.py --lot BHMBCCMKT01

# Use smaller training split
python scripts/train_models.py --split 0.5
```

---

## 📝 Tips & Best Practices

1. **Start with demo_pipeline.py** to understand the complete flow
2. **Process data first** with data_pipeline.py before training
3. **Train multiple models** to compare performance
4. **Use --plot flag** in evaluation to visualize results
5. **Include --booking-prob** for practical booking recommendations
6. **Save models regularly** for production use
7. **Version your models** by saving to dated directories

---

## 🎓 Next Steps

After running these scripts, you can:
1. Integrate predictions into a web API (see `app/` directory)
2. Set up automated daily forecasting (cron jobs)
3. Create dashboards with the prediction data
4. Implement real-time updates
5. Deploy to production

---

**Last Updated**: October 20, 2025  
**Version**: 1.0.0
