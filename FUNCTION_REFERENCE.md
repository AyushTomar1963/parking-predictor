# 📖 Complete Function Reference Guide

## Quick Navigation
- [Models](#models)
- [Queueing Theory](#queueing-theory)
- [Prediction](#prediction)
- [Preprocessing](#preprocessing)
- [Scripts](#scripts)

---

## Models

### LightGBMModel Class

#### `__init__(params=None, num_boost_round=200)`
**Purpose:** Initialize LightGBM model with hyperparameters  
**Parameters:**
- `params` (dict): LightGBM parameters (objective, learning_rate, max_depth, etc.)
- `num_boost_round` (int): Number of boosting iterations (default: 200)

**Example:**
```python
model = LightGBMModel(
    params={'objective': 'regression', 'learning_rate': 0.05},
    num_boost_round=200
)
```

#### `fit(X_train, y_train, X_val=None, y_val=None)`
**Purpose:** Train the model on training data  
**Parameters:**
- `X_train`: Training features (DataFrame or array)
- `y_train`: Training target values
- `X_val`: Validation features (optional)
- `y_val`: Validation target (optional)

**Returns:** Trained model object

**Example:**
```python
model.fit(X_train, y_train, X_val, y_val)
```

#### `predict(X)`
**Purpose:** Generate predictions for new data  
**Parameters:**
- `X`: Features to predict on

**Returns:** Array of predictions

**Example:**
```python
predictions = model.predict(X_test)
```

#### `get_feature_importance(importance_type='gain')`
**Purpose:** Get feature importance scores  
**Parameters:**
- `importance_type` (str): Type of importance ('gain', 'split', 'weight')

**Returns:** Dictionary mapping feature names to importance scores

**Example:**
```python
importance = model.get_feature_importance()
# {'lag_24': 450.2, 'lag_1': 320.5, ...}
```

---

### XGBoostModel Class

#### `__init__(params=None, num_boost_round=200)`
**Purpose:** Initialize XGBoost model  
**Parameters:**
- `params` (dict): XGBoost parameters
- `num_boost_round` (int): Boosting iterations

#### `fit(X_train, y_train, X_val=None, y_val=None)`
**Purpose:** Train XGBoost model  
**Returns:** Trained model

#### `predict(X)`
**Purpose:** Generate predictions  
**Returns:** Predictions array

---

### ARIMAXModel Class

#### `__init__(order=(1,1,1))`
**Purpose:** Initialize ARIMA model  
**Parameters:**
- `order` (tuple): ARIMA order (p, d, q)

#### `fit(train_data, exog=None)`
**Purpose:** Fit ARIMA model on time series  
**Parameters:**
- `train_data`: Time series data
- `exog`: Exogenous variables (optional)

**Returns:** Fitted model

#### `predict(steps, exog=None)`
**Purpose:** Generate multi-step forecast  
**Parameters:**
- `steps` (int): Number of steps to forecast
- `exog`: Future exogenous variables

**Returns:** Forecast array

---

### Utility Functions

#### `evaluate_model(y_true, y_pred)`
**Purpose:** Calculate evaluation metrics  
**Parameters:**
- `y_true`: Actual values
- `y_pred`: Predicted values

**Returns:** Dictionary with metrics
```python
{
    'MAE': 8.5,
    'MSE': 151.29,
    'RMSE': 12.3,
    'MAPE': 2.8
}
```

**Example:**
```python
metrics = evaluate_model(y_test, predictions)
print(f"MAE: {metrics['MAE']:.2f}")
```

#### `compare_models(models_dict, X_test, y_test)`
**Purpose:** Compare multiple models on same test set  
**Parameters:**
- `models_dict` (dict): {'model_name': model_instance}
- `X_test`: Test features
- `y_test`: Test target

**Returns:** Dictionary with results for each model

**Example:**
```python
results = compare_models(
    {'LightGBM': lgb_model, 'XGBoost': xgb_model},
    X_test, y_test
)
```

#### `get_best_model(comparison_results, metric='RMSE')`
**Purpose:** Identify best performing model  
**Parameters:**
- `comparison_results`: Output from compare_models()
- `metric` (str): Metric to use ('MAE', 'RMSE', 'MAPE')

**Returns:** Tuple (best_model_name, best_score)

**Example:**
```python
best_name, best_score = get_best_model(results, metric='RMSE')
print(f"Best model: {best_name} (RMSE: {best_score:.2f})")
```

---

## Queueing Theory

### Erlang-C Functions

#### `calculate_erlang_c(arrival_rate_lambda, service_rate_mu, num_servers_c)`
**Purpose:** Calculate Erlang-C queueing metrics  
**Parameters:**
- `arrival_rate_lambda` (float): Arrival rate (cars/hour)
- `service_rate_mu` (float): Service rate per spot (per hour)
- `num_servers_c` (int): Number of available spots

**Returns:** Tuple (prob_wait, details_dict)

**Details Dictionary:**
```python
{
    'P0': 0.15,           # Probability system empty
    'a': 20.0,            # Offered load
    'rho': 0.8,           # Utilization per server
    'Lq': 3.2,            # Expected queue length
    'Wq': 0.16,           # Expected wait time (hours)
    'W': 2.16             # Total time in system (hours)
}
```

**Example:**
```python
prob_wait, details = calculate_erlang_c(
    arrival_rate_lambda=10.0,
    service_rate_mu=0.5,
    num_servers_c=25
)
print(f"Probability of waiting: {prob_wait:.2%}")
print(f"Expected wait: {details['Wq']*60:.1f} minutes")
```

#### `calculate_probability_immediate_service(arrival_rate_lambda, service_rate_mu, num_servers_c)`
**Purpose:** Calculate probability of getting spot immediately  
**Returns:** Float (0-1) representing probability of no wait

**Example:**
```python
prob_immediate = calculate_probability_immediate_service(10.0, 0.5, 25)
print(f"Immediate service probability: {prob_immediate:.1%}")
```

#### `get_expected_wait_time_minutes(arrival_rate_lambda, service_rate_mu, num_servers_c)`
**Purpose:** Get expected wait time in minutes  
**Returns:** Float (minutes) or inf if system unstable

**Example:**
```python
wait_minutes = get_expected_wait_time_minutes(10.0, 0.5, 25)
print(f"Expected wait: {wait_minutes:.1f} minutes")
```

---

### Queue Parameter Estimation

#### `get_queueing_inputs(df, capacity, timestamp_col='LastUpdated', occupancy_col='Occupancy')`
**Purpose:** Estimate arrival and service rates from historical data  
**Parameters:**
- `df` (DataFrame): Historical parking data
- `capacity` (int): Total lot capacity
- `timestamp_col` (str): Name of timestamp column
- `occupancy_col` (str): Name of occupancy column

**Returns:** Tuple (hourly_arrival_rates, service_rate_mu)

**Hourly Arrival Rates:**
```python
{
    0: 2.5,   # Midnight
    1: 1.8,
    ...
    9: 15.2,  # 9 AM (peak)
    ...
    23: 3.1
}
```

**Example:**
```python
df = pd.read_csv('dataset.csv')
df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])

hourly_rates, mu = get_queueing_inputs(df, capacity=600)

print(f"Service rate: {mu:.4f} per hour")
print(f"Avg parking duration: {1/mu:.2f} hours")
print(f"Peak hour arrival rate: {max(hourly_rates.values()):.2f} cars/hour")
```

#### `estimate_arrival_rate_for_hour(hourly_arrival_rates, hour_of_day)`
**Purpose:** Get arrival rate for specific hour  
**Parameters:**
- `hourly_arrival_rates` (dict): Output from get_queueing_inputs()
- `hour_of_day` (int): Hour (0-23)

**Returns:** Float (arrival rate for that hour)

**Example:**
```python
rate_at_3pm = estimate_arrival_rate_for_hour(hourly_rates, 15)
```

#### `validate_queueing_parameters(arrival_rate_lambda, service_rate_mu, num_servers_c, capacity)`
**Purpose:** Validate queueing parameters and check stability  
**Returns:** Dictionary with validation results

**Validation Dictionary:**
```python
{
    'is_valid': True,
    'warnings': [],
    'offered_load': 20.0,
    'utilization': 0.8,
    'is_stable': True
}
```

**Example:**
```python
validation = validate_queueing_parameters(15.0, 0.5, 25, 100)
if not validation['is_stable']:
    print("WARNING: System is overloaded!")
```

---

### Booking Probability

#### `get_booking_confirmation(predicted_occupancy, capacity, hour_of_day, hourly_arrival_rates, service_rate_mu)`
**Purpose:** Calculate booking success probability and generate recommendation  
**Parameters:**
- `predicted_occupancy` (float): ML model's prediction
- `capacity` (int): Total lot capacity
- `hour_of_day` (int): Hour (0-23)
- `hourly_arrival_rates` (dict): From get_queueing_inputs()
- `service_rate_mu` (float): From get_queueing_inputs()

**Returns:** Dictionary with booking details

**Result Dictionary:**
```python
{
    'prob_get_spot': 0.873,
    'prob_wait': 0.127,
    'expected_wait_minutes': 8.5,
    'available_slots': 115,
    'arrival_lambda': 12.5,
    'service_rate_mu': 0.5,
    'recommendation': '🟡 GOOD - 87.3% chance...',
    'confidence_level': 'high',
    'utilization': 0.809
}
```

**Example:**
```python
result = get_booking_confirmation(
    predicted_occupancy=485,
    capacity=600,
    hour_of_day=15,
    hourly_arrival_rates=hourly_rates,
    service_rate_mu=mu
)

print(result['recommendation'])
print(f"Success probability: {result['prob_get_spot']:.1%}")
```

#### `calculate_booking_success_probability(predicted_occupancy, capacity, hour_of_day, hourly_arrival_rates, service_rate_mu)`
**Purpose:** Simplified function returning only probability  
**Returns:** Float (0-1) probability of success

**Example:**
```python
prob = calculate_booking_success_probability(485, 600, 15, hourly_rates, mu)
```

#### `batch_booking_analysis(predictions, capacity, hourly_arrival_rates, service_rate_mu)`
**Purpose:** Analyze booking probabilities for multiple hours  
**Parameters:**
- `predictions` (dict): {hour: predicted_occupancy}
- Other parameters same as above

**Returns:** Dictionary {hour: booking_result_dict}

**Example:**
```python
predictions = {9: 450, 10: 520, 11: 580, 12: 590}
results = batch_booking_analysis(predictions, 600, hourly_rates, mu)

for hour, result in results.items():
    print(f"{hour}:00 - {result['recommendation']}")
```

---

## Prediction

### TimeSeriesForecaster Class

#### `__init__(strategy='recursive', features=None, target='Occupancy', max_horizon=24, model_params=None)`
**Purpose:** Initialize forecaster with strategy  
**Parameters:**
- `strategy` (str): 'recursive' or 'direct'
- `features` (list): Feature names
- `target` (str): Target variable name
- `max_horizon` (int): Maximum forecast horizon
- `model_params` (dict): LightGBM parameters

**Example:**
```python
forecaster = TimeSeriesForecaster(
    strategy='recursive',
    features=['lag_1', 'lag_2', 'lag_24', 'hour_of_day'],
    target='Occupancy',
    max_horizon=24
)
```

#### `fit(df, num_boost_round=200, **kwargs)`
**Purpose:** Train forecasting model(s)  
**Parameters:**
- `df` (DataFrame): Training data with features and target
- `num_boost_round` (int): Boosting iterations

**Returns:** self (for method chaining)

**Example:**
```python
forecaster.fit(train_df, num_boost_round=200)
```

#### `predict(df, steps=None)`
**Purpose:** Generate multi-step forecasts  
**Parameters:**
- `df` (DataFrame): Input data
  - Recursive: Use last row (df.iloc[-1:])
  - Direct: Use entire dataframe
- `steps` (int): Number of steps (for recursive)

**Returns:**
- Recursive: pd.Series with predictions
- Direct: Dict {horizon: predictions}

**Example:**
```python
# Recursive
predictions = forecaster.predict(test_df.iloc[-1:], steps=24)

# Direct
predictions_dict = forecaster.predict(test_df)
```

#### `predict_single_horizon(df, horizon)`
**Purpose:** Predict for specific horizon only  
**Parameters:**
- `df` (DataFrame): Input data
- `horizon` (int): Specific horizon (1 to max_horizon)

**Returns:** Single prediction or Series

**Example:**
```python
pred_6h = forecaster.predict_single_horizon(test_df, horizon=6)
```

#### `save(path)`
**Purpose:** Save trained model(s) to disk  
**Parameters:**
- `path` (str/Path): Directory to save models

**Example:**
```python
forecaster.save('models/forecaster_recursive')
```

#### `load(path)`
**Purpose:** Load trained model(s) from disk  
**Parameters:**
- `path` (str/Path): Directory with saved models

**Returns:** self

**Example:**
```python
forecaster = TimeSeriesForecaster()
forecaster.load('models/forecaster_recursive')
```

#### `get_feature_importance()`
**Purpose:** Get feature importance from trained model(s)  
**Returns:** DataFrame with features and importance scores

**Example:**
```python
importance = forecaster.get_feature_importance()
print(importance.head())
```

---

### Helper Functions

#### `multi_step_forecast(model, data, features, steps=24, strategy='recursive')`
**Purpose:** Convenience function for forecasting  
**Parameters:**
- `model`: Trained model or dict of models
- `data` (DataFrame): Input data
- `features` (list): Feature names
- `steps` (int): Forecast horizon
- `strategy` (str): 'recursive' or 'direct'

**Returns:** Predictions (Series or dict)

**Example:**
```python
predictions = multi_step_forecast(
    model=trained_model,
    data=test_df.iloc[-1:],
    features=['lag_1', 'lag_2', 'lag_24'],
    steps=24,
    strategy='recursive'
)
```

#### `create_forecaster(strategy, train_df, features, target='Occupancy', max_horizon=24, **kwargs)`
**Purpose:** Factory function to create and train forecaster  
**Returns:** Trained TimeSeriesForecaster

**Example:**
```python
forecaster = create_forecaster(
    strategy='recursive',
    train_df=train_data,
    features=['lag_1', 'lag_2', 'lag_24', 'hour_of_day'],
    max_horizon=24,
    num_boost_round=200
)
```

---

## Preprocessing

### Time Series Processing

#### `process_lot_data(df_lot)`
**Purpose:** Process raw parking lot data into clean time series  
**Parameters:**
- `df_lot` (DataFrame): Raw data with datetime index

**Returns:** Processed DataFrame with time features

**Processing Steps:**
1. Resample to hourly frequency
2. Interpolate missing values
3. Create time features (hour_of_day, day_of_week, is_weekend)

**Example:**
```python
processed_df = process_lot_data(raw_lot_df)
print(processed_df.columns)
# ['Occupancy', 'hour_of_day', 'day_of_week', 'is_weekend']
```

---

### Feature Engineering

#### `create_lag_features(df, target_col='Occupancy', lags=[1,2,3,24,48])`
**Purpose:** Create lagged versions of target variable  
**Parameters:**
- `df` (DataFrame): Input data
- `target_col` (str): Column to create lags from
- `lags` (list): List of lag values

**Returns:** DataFrame with lag columns added

**Example:**
```python
df_with_lags = create_lag_features(
    df,
    target_col='Occupancy',
    lags=[1, 2, 3, 24, 48]
)
# Adds columns: lag_1, lag_2, lag_3, lag_24, lag_48
```

---

## Scripts

### Demo Pipeline

#### `run_demo_pipeline(csv_path='data/raw/dataset.csv', lot_id='BHMBCCMKT01', capacity=600)`
**Purpose:** Run complete end-to-end demo  
**Steps:**
1. Load data
2. Preprocess
3. Create features
4. Train model
5. Evaluate
6. Calculate booking probabilities

**Example:**
```python
run_demo_pipeline(
    csv_path='data/raw/dataset.csv',
    lot_id='BHMBCCMKT01',
    capacity=600
)
```

**CLI:**
```bash
python scripts/demo_pipeline.py
python scripts/demo_pipeline.py --lot BHMBCCMKT01 --capacity 600
```

---

### Train Models

#### CLI Usage:
```bash
# Train specific models
python scripts/train_models.py --models lightgbm,xgboost

# Train all models
python scripts/train_models.py --models all

# Specify save directory
python scripts/train_models.py --models lightgbm --save-dir data/models/
```

---

### Evaluate Models

#### CLI Usage:
```bash
# Evaluate all models
python scripts/evaluate_models.py --model all

# Evaluate specific model
python scripts/evaluate_models.py --model lightgbm

# Generate plots
python scripts/evaluate_models.py --model all --plot
```

---

### Batch Predict

#### CLI Usage:
```bash
# Predict next 24 hours
python scripts/batch_predict.py --hours 24

# Include booking probabilities
python scripts/batch_predict.py --hours 24 --booking-prob

# Save to file
python scripts/batch_predict.py --hours 48 --output predictions.csv
```

---

### Data Pipeline

#### CLI Usage:
```bash
# Process all lots
python scripts/data_pipeline.py

# Save individual lot files
python scripts/data_pipeline.py --save-individual

# Show summary
python scripts/data_pipeline.py --summary
```

---

## Main Runner (run.py)

### Train Command
```bash
python run.py train --data data/processed/lot.csv --out models/model.joblib --rounds 200
```

### Demo Command
```bash
python run.py demo --data data/raw/dataset.csv --lot BHMBCCMKT01 --capacity 600
```

### Evaluate Command
```bash
python run.py eval --model all --plot
```

### Serve Command
```bash
python run.py serve --host 0.0.0.0 --port 8000 --no-reload
```

### Check Command
```bash
python run.py check data/raw/dataset.csv
```

---

## Common Workflows

### 1. Train and Evaluate New Model
```python
# Load and process data
df = pd.read_csv('data/raw/dataset.csv')
df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])
processed = process_lot_data(df)

# Create features
df_model = processed.copy()
for lag in [1, 2, 3, 24, 48]:
    df_model[f'lag_{lag}'] = df_model['Occupancy'].shift(lag)
df_model = df_model.dropna()

# Train
features = ['lag_1', 'lag_2', 'lag_3', 'lag_24', 'lag_48', 'hour_of_day']
model = LightGBMModel()
model.fit(df_model[features], df_model['Occupancy'])

# Evaluate
predictions = model.predict(test_data[features])
metrics = evaluate_model(test_data['Occupancy'], predictions)
print(metrics)
```

### 2. Generate Booking Recommendations
```python
# Estimate queueing parameters
hourly_rates, mu = get_queueing_inputs(historical_df, capacity=600)

# Make prediction
predicted_occ = model.predict(current_features)

# Calculate booking probability
result = get_booking_confirmation(
    predicted_occupancy=predicted_occ,
    capacity=600,
    hour_of_day=15,
    hourly_arrival_rates=hourly_rates,
    service_rate_mu=mu
)

print(result['recommendation'])
```

### 3. Multi-Hour Forecast with Booking Analysis
```python
# Create forecaster
forecaster = TimeSeriesForecaster(
    strategy='recursive',
    features=features,
    max_horizon=24
)
forecaster.fit(train_df)

# Generate 24-hour forecast
predictions = forecaster.predict(test_df.iloc[-1:], steps=24)

# Batch booking analysis
predictions_dict = {h: pred for h, pred in enumerate(predictions, 1)}
booking_results = batch_booking_analysis(
    predictions_dict,
    capacity=600,
    hourly_arrival_rates=hourly_rates,
    service_rate_mu=mu
)

# Display results
for hour, result in booking_results.items():
    print(f"{hour}:00 - {result['prob_get_spot']:.1%} - {result['recommendation']}")
```

---

*Complete function reference for Parking Predictor v1.0.0*
