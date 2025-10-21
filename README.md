# 🚗 Parking Predictor - Smart Parking Occupancy Prediction System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A comprehensive parking occupancy prediction and booking system combining **machine learning forecasting** with **queueing theory** to provide accurate availability predictions and booking probabilities.

---

## 🌟 Features

### 🎯 Core Capabilities
- **Time Series Forecasting**: ARIMA and LightGBM models for hourly occupancy prediction
- **Multi-Step Prediction**: Direct and recursive strategies for 1-48 hour forecasts
- **Booking Probability**: Erlang-C queueing theory for real-time spot availability
- **REST API**: FastAPI-powered endpoints for easy integration
- **Web Dashboard**: Interactive UI for visualizations and bookings
- **Real-time Analytics**: Live statistics and trend analysis

### 📊 Advanced Features
- Multi-horizon forecasting (6h, 12h, 24h, 48h)
- Rolling-origin evaluation for robust testing
- Hourly arrival pattern detection
- Dynamic pricing recommendations
- Confidence intervals for predictions
- Historical trend visualization

---

## 📁 Project Structure

```
parking-predictor/
│
├── 📂 app/                          # Web Application
│   ├── main.py                      # FastAPI app entry
│   ├── routes/                      # API endpoints
│   │   ├── prediction_routes.py
│   │   ├── booking_routes.py
│   │   └── analytics_routes.py
│   └── static/                      # Frontend files
│
├── 📂 src/                          # Source Code
│   ├── models/                      # ML Models
│   │   ├── arima_model.py
│   │   ├── lightgbm_model.py
│   │   └── model_trainer.py
│   ├── preprocessing/               # Data Processing
│   │   ├── data_loader.py
│   │   ├── time_series_processor.py
│   │   └── feature_engineer.py
│   ├── prediction/                  # Forecasting
│   │   ├── forecaster.py
│   │   ├── direct_predictor.py
│   │   └── recursive_predictor.py
│   ├── queueing/                    # Queueing Theory
│   │   ├── erlang_c.py
│   │   ├── queue_estimator.py
│   │   └── booking_probability.py
│   └── utils/                       # Utilities
│       ├── metrics.py
│       ├── visualization.py
│       └── config.py
│
├── 📂 notebooks/                    # Jupyter Notebooks
│   └── arima.ipynb                  # Main analysis
│
├── 📂 data/                         # Data Storage
│   ├── raw/                         # Original data
│   ├── processed/                   # Cleaned data
│   └── models/                      # Saved models
│
├── 📂 config/                       # Configuration
│   ├── app_config.yaml
│   ├── model_config.yaml
│   └── queueing_config.yaml
│
├── 📂 tests/                        # Unit Tests
├── 📂 scripts/                      # Utility Scripts
├── 📂 docs/                         # Documentation
│
├── requirements.txt                 # Dependencies
├── README.md                        # This file
└── DIRECTORY_STRUCTURE.md           # Detailed structure
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd parking-predictor

# Create virtual environment
python -m venv venv

# Activate venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example config
cp .env.example .env

# Edit configuration files in config/
# - config/app_config.yaml
# - config/model_config.yaml
# - config/queueing_config.yaml
```

### 3. Train Models

```bash
# Train models using notebook analysis
python scripts/train_models.py --lot BHMBCCMKT01
```

### 4. Run Application

```bash
# Start FastAPI server
python app/main.py

# Or using uvicorn
uvicorn app.main:app --reload --port 8000
```

### 5. Access Application

- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

---

## 📖 Usage Examples

### API Endpoints

#### 1. Get Predictions
```bash
# Predict next 24 hours
curl http://localhost:8000/api/predict?hours=24

# Response:
{
  "predictions": [
    {"timestamp": "2025-10-20T15:00:00", "predicted_occupancy": 450},
    {"timestamp": "2025-10-20T16:00:00", "predicted_occupancy": 520},
    ...
  ]
}
```

#### 2. Check Booking Probability
```bash
# Check availability at specific time
curl -X POST http://localhost:8000/api/booking/probability \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2025-10-20T15:00:00",
    "lot_id": "BHMBCCMKT01"
  }'

# Response:
{
  "prob_get_spot": 0.85,
  "available_slots": 150,
  "expected_wait_minutes": 5.2,
  "recommendation": "High probability - Book now!"
}
```

#### 3. Get Analytics
```bash
# Get lot statistics
curl http://localhost:8000/api/analytics/stats?lot_id=BHMBCCMKT01

# Response:
{
  "avg_occupancy": 450.5,
  "peak_hours": [8, 9, 17, 18],
  "avg_availability": 149.5,
  "utilization_rate": 0.75
}
```

### Python SDK

```python
from src.models.lightgbm_model import LightGBMPredictor
from src.queueing.booking_probability import calculate_booking_probability

# Load model
model = LightGBMPredictor.load('data/models/lightgbm_model.pkl')

# Make predictions
predictions = model.predict_next_hours(hours=24)

# Calculate booking probability
prob = calculate_booking_probability(
    predicted_occupancy=450,
    capacity=600,
    hour_of_day=15
)

print(f"Probability of getting a spot: {prob['prob_get_spot']*100:.1f}%")
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

---

## 📊 Model Performance

### LightGBM Model (Best Performer)
- **MAE**: ~8.5 cars
- **RMSE**: ~12.3 cars
- **Horizons**: 1-48 hours
- **Features**: 8 (lags + time features)

### ARIMA Model (Baseline)
- **MAE**: ~15.2 cars
- **RMSE**: ~22.1 cars
- **Order**: (1, 0, 0)

### Queueing Theory Accuracy
- **Erlang-C Formula**: Numerically stable implementation
- **Parameter Estimation**: Data-driven λ and μ from historical patterns
- **Validation**: Matches real-world booking success rates

---

## 🎯 Use Cases

### 1. **User Mobile App**
- Show real-time availability
- Predict best times to park
- Book spots in advance
- Get wait time estimates

### 2. **Dynamic Pricing**
- Charge more during peak hours (>80% predicted occupancy)
- Discounts during low-demand periods
- Revenue optimization

### 3. **City Planning**
- Identify underutilized lots
- Optimize resource allocation
- Plan new infrastructure
- Traffic flow management

### 4. **Fleet Management**
- Route drivers to available lots
- Reduce search time
- Improve efficiency

---

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**
- **FastAPI**: Modern web framework
- **LightGBM**: Gradient boosting
- **Statsmodels**: ARIMA modeling
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: ML utilities

### Frontend
- **HTML/CSS/JavaScript**
- **Chart.js**: Visualizations
- **Bootstrap**: UI framework

### Infrastructure
- **Docker**: Containerization
- **Uvicorn**: ASGI server
- **Pytest**: Testing framework

---

## 📚 Documentation

- **[Directory Structure](DIRECTORY_STRUCTURE.md)**: Detailed project organization
- **[Migration Guide](MIGRATION_GUIDE.md)**: File reorganization steps
- **[Analysis Summary](summary.md)**: Notebook analysis explanation
- **[API Documentation](http://localhost:8000/docs)**: Interactive API docs (when running)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Ayush** - Initial work and analysis

---

## 🙏 Acknowledgments

- Birmingham City Council for parking data
- LightGBM and Statsmodels teams
- FastAPI framework
- Queueing theory research (Erlang-C formula)

---

## 📧 Contact

For questions or support, please open an issue in the repository.

---

## 🗺️ Roadmap

### Version 1.1 (Q4 2025)
- [ ] Add weather data integration
- [ ] Implement real-time data updates
- [ ] Add email/SMS notifications
- [ ] Mobile app (React Native)

### Version 1.2 (Q1 2026)
- [ ] Deep learning models (LSTM/GRU)
- [ ] Multi-city support
- [ ] Advanced analytics dashboard
- [ ] Payment integration

### Version 2.0 (Q2 2026)
- [ ] Computer vision for real-time occupancy
- [ ] IoT sensor integration
- [ ] Blockchain-based booking
- [ ] AI-powered pricing optimization

---

**⭐ Star this repository if you find it helpful!**
