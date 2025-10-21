# Parking Occupancy Prediction - Analysis Summary

## Project Overview
This notebook implements a comprehensive parking occupancy prediction system using time series forecasting techniques, specifically ARIMA and LightGBM models, combined with queueing theory for booking probability estimation.

---

## 1. Data Loading and Initial Setup

### Cell 1: Library Imports
- **What we did**: Imported essential libraries for time series analysis
- **Key libraries**:
  - `pandas` & `numpy`: Data manipulation
  - `matplotlib`: Visualization
  - `statsmodels`: ARIMA modeling and statistical tests
  - Configured inline plotting for Jupyter

### Cell 2-3: Data Loading and Preprocessing
- **What we did**: 
  - Loaded the parking dataset from `dataset.csv`
  - Converted `LastUpdated` column to datetime format
- **Purpose**: Prepare data for time series analysis with proper temporal indexing

### Cell 4-6: Data Segregation by Parking Lot
- **What we did**: 
  - Segregated data by individual parking lots using `SystemCodeNumber`
  - Selected one specific lot (`BHMBCCMKT01`) for detailed analysis
  - Set datetime as index for time series operations
- **Reason**: Each parking lot has unique patterns; analyzing them separately improves accuracy

---

## 2. Exploratory Data Analysis (EDA)

### Cell 7: Occupancy Over Time Visualization
- **What we did**: Created a time series plot showing occupancy patterns
- **Insights**: Visualized trends, seasonality, and patterns in parking usage over time

### Cell 8: Data Processing Function
- **What we did**: Implemented `process_lot_data()` function that:
  1. **Resamples** data to regular hourly intervals (standardizes irregular timestamps)
  2. **Interpolates** missing values using time-based interpolation
  3. **Feature Engineering**: Creates time-based features:
     - `hour_of_day` (0-23)
     - `day_of_week` (0=Monday, 6=Sunday)
     - `is_weekend` (binary: 1 for Sat/Sun, 0 for weekdays)
- **Purpose**: Transform raw data into a clean, regular time series suitable for modeling

### Cell 9: Day-of-Week Analysis
- **What we did**: 
  - Aggregated total occupancy by day of week
  - Created bar chart showing usage patterns across different days
- **Insights**: Identifies which days have highest/lowest parking demand (e.g., weekdays vs weekends)

---

## 3. Time Series Stationarity Analysis

### Cell 10: Augmented Dickey-Fuller (ADF) Test
- **What we did**: Tested for stationarity using ADF test
- **Result**: Data is already stationary (p-value check)
- **Why it matters**: Stationary data is required for ARIMA modeling; stationary means statistical properties (mean, variance) don't change over time

### Cell 11: ACF and PACF Plots
- **What we did**: Plotted Autocorrelation (ACF) and Partial Autocorrelation (PACF) functions
- **Purpose**: 
  - **ACF**: Helps determine the Moving Average (q) parameter
  - **PACF**: Helps determine the Autoregressive (p) parameter
  - These guide ARIMA model order selection

### Cell 12: Seasonal Decomposition
- **What we did**: Decomposed time series into:
  - **Trend**: Long-term direction
  - **Seasonal**: Repeating patterns (24-hour cycle)
  - **Residual**: Random noise
- **Purpose**: Understand different components affecting parking occupancy

---

## 4. Model Development

### Cell 13: Train-Test Split
- **What we did**: 
  - Split data into 80% training and 20% testing
  - Prepared features (time variables) and target (Occupancy)
- **Purpose**: Evaluate model performance on unseen data

### Cell 14: Basic ARIMA Model
- **What we did**: 
  - Fitted ARIMA(1,0,0) model on training data
  - Made predictions on test set
  - Calculated MAE and RMSE metrics
- **Results**: Baseline performance using traditional statistical approach

### Cell 15-17: LightGBM Model (Initial)
- **What we did**: 
  - Created lag features (lag_1, lag_2, lag_3, lag_24, lag_48)
  - Trained LightGBM gradient boosting model
  - Evaluated performance
- **Why lag features**: Previous occupancy values are strong predictors of future values
- **Note**: Cell 16 had a variable name typo that was fixed in later cells

### Cell 18: Corrected LightGBM Implementation
- **What we did**: 
  - Properly implemented LightGBM with aligned indices
  - Created predictions as pandas Series for proper alignment
  - Calculated metrics (MAE, RMSE)
  - Visualized actual vs predicted occupancy
- **Improvement**: Better handling of time series indices prevents misalignment errors

### Cell 19: Overfitting Check
- **What we did**: 
  - Compared training MAE vs test MAE
- **Purpose**: Detect if model is overfitting (memorizing training data vs generalizing)

---

## 5. Advanced Multi-Step Forecasting

### Cell 20: Comprehensive Multi-Step Forecasting Framework
This is the most complex cell - implements production-ready forecasting:

#### Key Components:

**A. Helper Functions**
- `make_lag_features()`: Creates lagged versions of occupancy

**B. Direct Multi-Step Strategy**
- `train_direct_models()`: Trains **separate models** for each future hour (h=1, h=2, ..., h=H)
- Each model directly predicts occupancy H hours ahead
- **Advantage**: More accurate for specific horizons

**C. Recursive Multi-Step Strategy**
- `train_recursive_model()`: Trains **single model** for 1-step ahead
- `predict_recursive()`: Iteratively predicts next hour, feeds prediction back as input
- **Advantage**: Only need one model; good for long horizons

**D. Rolling-Origin Evaluation**
- `rolling_multi_eval()`: Simulates real-world deployment
- Repeatedly trains on expanding window, tests on next H hours
- Calculates average MAE and RMSE across multiple forecast origins
- **Purpose**: Robust performance estimation

**E. Practical Examples**
- Demonstrates both direct and recursive forecasting for 6-hour and 24-hour horizons
- Visualizes predictions vs actual values
- Shows how to use the framework in production

---

## 6. Queueing Theory Integration

### Cell 21: Parking Queue Probability Calculator
Implements **Erlang-C queueing theory** to estimate booking success probability:

#### Key Components:

**A. Erlang-C Formula Implementation**
- `calculate_erlang_c()`: Calculates probability a customer must wait in an M/M/c queue
- **Inputs**: 
  - `λ` (lambda): Arrival rate (cars per hour)
  - `μ` (mu): Service rate (parking duration rate)
  - `c`: Number of available spots (servers)
- **Outputs**:
  - Probability of waiting
  - Expected queue length (Lq)
  - Expected waiting time (Wq)

**B. Data-Driven Parameter Estimation**
- `get_queueing_inputs()`: Estimates λ and μ from historical data
  - Analyzes occupancy changes to infer arrivals
  - Uses Little's Law: L = λW to derive service rate
  - Computes hourly arrival rate patterns

**C. Booking Confirmation Calculator**
- `get_booking_confirmation()`: Given predicted occupancy, calculates:
  - Probability of getting a spot immediately
  - Expected wait time if spots are full
  - Number of available slots
- **Real-world application**: Powers booking recommendations

**D. Simulation Runner**
- `run_parking_simulation()`: End-to-end pipeline
  - Loads data
  - Estimates queueing parameters
  - Tests multiple occupancy scenarios
  - Outputs actionable probabilities

---

## 7. Data Preparation for Integration

### Cell 22-23: Forecast DataFrame Creation
- **What we did**: Converted prediction Series into DataFrame format
- **Purpose**: Prepare forecasts for integration with queueing probability system

---

## Key Achievements

### 1. **Accurate Predictions**
- LightGBM model with lag features achieves low MAE/RMSE
- Multi-step forecasting handles both short-term (6h) and long-term (24h) predictions

### 2. **Feature Engineering**
- Time-based features (hour, day, weekend) capture temporal patterns
- Lag features (1, 2, 3, 24, 48 hours) capture autoregressive behavior
- Regular hourly resampling ensures model stability

### 3. **Robust Evaluation**
- Rolling-origin evaluation simulates real deployment
- Train/test split prevents overfitting
- Multiple metrics (MAE, RMSE) provide comprehensive performance view

### 4. **Practical Application**
- Queueing theory integration transforms predictions into actionable insights
- Booking probability helps users make informed decisions
- Hourly arrival patterns enable dynamic pricing strategies

### 5. **Production-Ready Code**
- Modular functions enable easy reuse
- Handles edge cases (full lots, unstable queues)
- Numerically stable implementations prevent overflow

---

## Technical Highlights

### Models Used:
1. **ARIMA(1,0,0)**: Traditional statistical baseline
2. **LightGBM**: Gradient boosting with lag features (best performer)

### Forecasting Strategies:
1. **Direct**: Separate models per horizon (more accurate)
2. **Recursive**: Single model iterated (more flexible)

### Mathematical Foundations:
1. **Time Series**: Stationarity, ACF/PACF, seasonal decomposition
2. **Queueing Theory**: Erlang-C formula, M/M/c queues, Little's Law
3. **Machine Learning**: Gradient boosting, lag features, rolling validation

---

## Real-World Use Cases

### 1. **User Booking App**
- Show probability of getting spot at desired time
- Recommend alternative times with higher availability
- Display expected wait times

### 2. **Dynamic Pricing**
- Charge more during high-demand hours (>80% predicted occupancy)
- Offer discounts during low-demand periods

### 3. **Resource Allocation**
- Predict when additional lots needed
- Optimize staff schedules based on predicted demand

### 4. **City Planning**
- Identify chronically underutilized lots
- Inform decisions on new parking infrastructure

---

## Next Steps & Improvements

### Potential Enhancements:
1. **External Features**: Weather, events, holidays
2. **Deep Learning**: LSTM/GRU for capturing complex patterns
3. **Ensemble Methods**: Combine ARIMA + LightGBM predictions
4. **Real-time Updates**: Online learning as new data arrives
5. **Multi-lot Coordination**: Recommend nearest available lot
6. **API Development**: Deploy as REST API for mobile app integration

---

## Summary Statistics

- **Dataset**: Birmingham parking data with multiple lots
- **Main Lot Analyzed**: BHMBCCMKT01
- **Features**: 7 (lag_1, lag_2, lag_3, lag_24, lag_48, hour_of_day, day_of_week, is_weekend)
- **Model**: LightGBM with 200 boosting rounds
- **Forecast Horizons**: 6 hours (short-term), 24 hours (long-term)
- **Evaluation**: Rolling-origin with expanding window
- **Application**: Queueing theory for booking probability estimation

---

## Conclusion

This project successfully combines **machine learning forecasting** with **queueing theory** to create a comprehensive parking prediction and booking system. The LightGBM model provides accurate occupancy predictions, while Erlang-C formulas convert these predictions into user-friendly booking probabilities. The system is ready for deployment in real-world parking management applications.
