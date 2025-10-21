# Scripts Directory

Utility scripts for the Parking Predictor system.

---

## 📜 Quick Reference

| Script | Command | Description |
|--------|---------|-------------|
| **demo_pipeline.py** | `python scripts/demo_pipeline.py` | Complete end-to-end demo |
| **data_pipeline.py** | `python scripts/data_pipeline.py --summary` | ETL pipeline for data processing |
| **train_models.py** | `python scripts/train_models.py --models all` | Train ML models |
| **evaluate_models.py** | `python scripts/evaluate_models.py --model all --plot` | Evaluate model performance |
| **batch_predict.py** | `python scripts/batch_predict.py --hours 24 --booking-prob` | Generate batch predictions |

---

## 🚀 Quick Start

### 1. Run the Demo (Fastest)
```bash
python scripts/demo_pipeline.py
```

### 2. Complete Setup (Recommended)
```bash
# Process data
python scripts/data_pipeline.py --save-individual --summary

# Train models
python scripts/train_models.py --models lightgbm,xgboost

# Evaluate
python scripts/evaluate_models.py --model all --plot

# Predict
python scripts/batch_predict.py --hours 24 --booking-prob
```

---

## 📖 Detailed Documentation

See [SCRIPTS_USAGE.md](../docs/SCRIPTS_USAGE.md) for:
- Detailed usage examples
- All command-line arguments
- Complete workflows
- Troubleshooting guide

---

## 🎯 Common Commands

### Daily Forecasting
```bash
python scripts/batch_predict.py --hours 24 --booking-prob
```

### Weekly Planning
```bash
python scripts/batch_predict.py --hours 168 --model all
```

### Model Retraining
```bash
python scripts/train_models.py --models all
python scripts/evaluate_models.py --model all --plot
```

### Process New Data
```bash
python scripts/data_pipeline.py --input data/raw/new_data.csv --summary
```

---

## 📂 Output Locations

- **Processed Data**: `data/processed/`
- **Trained Models**: `data/models/`
- **Predictions**: `data/processed/predictions.csv`
- **Plots**: `data/models/*.png`

---

## 💡 Tips

1. Always run `data_pipeline.py` first to process raw data
2. Use `demo_pipeline.py` to quickly test the complete flow
3. Add `--plot` to evaluation for visual analysis
4. Include `--booking-prob` for booking recommendations
5. Train multiple models and compare performance

---

**Need Help?** Check [SCRIPTS_USAGE.md](../docs/SCRIPTS_USAGE.md) for detailed documentation.
