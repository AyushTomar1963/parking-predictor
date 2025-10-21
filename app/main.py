# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from functools import lru_cache
import logging
import os

# Import models and utilities
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import LightGBMModel, XGBoostModel, ARIMAXModel
from src.models.model_loader import load_model, load_models_from_directory
from src.prediction import TimeSeriesForecaster
from src.queueing.queue_estimator import get_queueing_inputs
from src.queueing.booking_probability import get_booking_confirmation

# Config loader fallback
def load_config(path: str):
    """Simple config loader fallback."""
    return {
        "app": {"host": "127.0.0.1", "port": 8000, "debug": True},
        "data": {"models_dir": "data/models", "processed_dir": "data/processed"},
    }

# Setup logging
logger = logging.getLogger("parking_predictor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Load configuration
CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "app_config.yaml")
try:
    cfg = load_config(CFG_PATH)
except Exception:
    cfg = load_config(None)

app = FastAPI(title="Parking Predictor API", version="1.0")

# Allow CORS for local dev / frontend demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Pydantic request models
# -----------------------
class PredictRequest(BaseModel):
    lot_id: str
    horizon: Optional[int] = 6  # hours ahead
    model: Optional[str] = "lightgbm"  # "lightgbm" or "sarima" etc.


class BookingRequest(BaseModel):
    lot_id: str
    predicted_occupancy: float
    hour_of_day: Optional[int] = None  # if not provided, will use timestamp -> hour


# -----------------------
# Simple model cache
# -----------------------
MODEL_CACHE: Dict[str, Any] = {}

def find_models_dir() -> str:
    # Read models_dir from config if available
    try:
        return cfg.get("data", {}).get("models_dir", "data/models")
    except Exception:
        return "data/models"

@lru_cache(maxsize=8)
def load_model_by_name(name: str):
    """
    Load model from models directory using the new load_model function.
    Supports lightgbm, xgboost, and arima models.
    """
    models_dir = find_models_dir()
    try:
        # Use the new load_model function that handles different file extensions
        model = load_model(name, model_dir=models_dir)
        logger.info("Loaded %s model from %s", name, models_dir)
        return model
    except FileNotFoundError as e:
        logger.warning("Model not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error loading %s model: %s", name, e)
        raise

# -----------------------
# Endpoints
# -----------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0"}


@app.get("/api/lots")
def list_lots():
    """
    Return available processed lots (by scanning data/processed/).
    If you have a master lots file, you can return rich metadata instead.
    """
    processed_dir = cfg.get("data", {}).get("processed_dir", "data/processed")
    if not os.path.isdir(processed_dir):
        return {"lots": [], "message": f"Processed directory '{processed_dir}' not found."}
    # list files and infer lot ids from filenames (assumes one file per lot)
    files = [f for f in os.listdir(processed_dir) if f.endswith(".csv") or f.endswith(".parquet")]
    lots = []
    for f in files:
        name = os.path.splitext(f)[0]
        lots.append({"lot_id": name, "file": f})
    return {"lots": lots}


@app.post("/api/predict")
def predict(payload: PredictRequest):
    """
    High-level prediction endpoint.
    - Loads model and generates predictions for specified horizon
    - Returns list of predictions with timestamps
    """
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Load the requested model
        model = load_model_by_name(payload.model)
        logger.info(f"Loaded model: {payload.model}")
        
        # Load processed data for the lot
        processed_dir = cfg.get("data", {}).get("processed_dir", "data/processed")
        
        # Map lot_id to file - for now we only have one_lot_data.csv
        data_file = os.path.join(processed_dir, "one_lot_data.csv")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Processed data not found at {data_file}")
        
        # Load recent data
        df = pd.read_csv(data_file)
        if 'LastUpdated' in df.columns:
            df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])
            df = df.set_index('LastUpdated')
        
        # Get the most recent data point with all required features
        features = ['lag_1', 'lag_2', 'lag_3', 'lag_24', 'lag_48', 
                   'hour_of_day', 'day_of_week', 'is_weekend']
        
        # Ensure all features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
        
        # Get last complete row with no NaN values
        df_clean = df[features + ['Occupancy']].dropna()
        if df_clean.empty:
            raise ValueError("No clean data available for prediction")
        
        last_row = df_clean.iloc[-1:]
        X_pred = last_row[features]
        
        # Generate predictions for the requested horizon
        predictions = []
        current_time = df_clean.index[-1]
        
        if payload.model in ['lightgbm', 'xgboost']:
            # For ML models, generate recursive predictions
            for h in range(1, payload.horizon + 1):
                # Predict next step
                pred = model.predict(X_pred)[0]
                
                # Store prediction with timestamp
                pred_time = current_time + timedelta(hours=h)
                predictions.append({
                    "horizon": h,
                    "timestamp": pred_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "predicted_occupancy": float(round(pred, 2))
                })
                
                # Update features for next prediction (recursive approach)
                if h < payload.horizon:
                    # Shift lags: lag_1 = pred, lag_2 = old lag_1, etc.
                    new_row = X_pred.copy()
                    new_row['lag_48'] = X_pred['lag_24'].values[0]
                    new_row['lag_24'] = X_pred['lag_3'].values[0]
                    new_row['lag_3'] = X_pred['lag_2'].values[0]
                    new_row['lag_2'] = X_pred['lag_1'].values[0]
                    new_row['lag_1'] = pred
                    
                    # Update calendar features
                    pred_time = current_time + timedelta(hours=h)
                    new_row['hour_of_day'] = pred_time.hour
                    new_row['day_of_week'] = pred_time.dayofweek
                    new_row['is_weekend'] = int(pred_time.dayofweek >= 5)
                    
                    X_pred = new_row
        
        elif payload.model == 'arima':
            # For ARIMA, generate direct multi-step forecast
            preds = model.predict(steps=payload.horizon)
            for h in range(1, payload.horizon + 1):
                pred_time = current_time + timedelta(hours=h)
                predictions.append({
                    "horizon": h,
                    "timestamp": pred_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "predicted_occupancy": float(round(preds[h-1], 2))
                })
        
        return {
            "lot_id": payload.lot_id,
            "model": payload.model,
            "base_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "horizon": payload.horizon,
            "predictions": predictions,
            "status": "success"
        }
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Data not found: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/booking")
def booking(payload: BookingRequest):
    """
    Compute booking probability given predicted occupancy and a lot id.
    This endpoint will:
      1) load processed lot data (for estimating lambda/mu) -- uses queue_estimator.get_queueing_inputs
      2) call get_booking_confirmation(predicted_occupancy, capacity, hour, hourly_map, mu)
    """
    # Basic checks
    if get_queueing_inputs is None or get_booking_confirmation is None:
        raise HTTPException(status_code=500, detail="Queueing functions not implemented or import failed.")

    # find processed lot file and load its capacity & data
    processed_dir = cfg.get("data", {}).get("processed_dir", "data/processed")
    
    # Try multiple file naming patterns
    possible_files = [
        os.path.join(processed_dir, f"{payload.lot_id}.csv"),
        os.path.join(processed_dir, "one_lot_data.csv"),
        os.path.join(processed_dir, "processed_data.csv"),
    ]
    
    processed_file_csv = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            processed_file_csv = file_path
            break
    
    if processed_file_csv is None:
        raise HTTPException(
            status_code=404, 
            detail=f"Processed data for lot {payload.lot_id} not found. Tried: {possible_files}"
        )

    import pandas as pd
    df_lot = pd.read_csv(processed_file_csv)
    
    # Ensure LastUpdated is a column (not just index)
    if "LastUpdated" not in df_lot.columns and df_lot.index.name == "LastUpdated":
        df_lot = df_lot.reset_index()
    
    if "LastUpdated" in df_lot.columns:
        df_lot["LastUpdated"] = pd.to_datetime(df_lot["LastUpdated"])
    
    logger.info(f"Loaded data with columns: {list(df_lot.columns)}")
    logger.info(f"Data shape: {df_lot.shape}")

    # capacity detection: prefer Capacity column if present
    capacity = int(df_lot["Capacity"].max()) if "Capacity" in df_lot.columns else None
    if capacity is None:
        raise HTTPException(status_code=400, detail="Lot capacity not found in processed data.")
    
    logger.info(f"Capacity: {capacity}")

    # estimate hourly map & mu
    hourly_map, service_rate_mu = None, None
    try:
        logger.info("Calling get_queueing_inputs...")
        q_inputs = get_queueing_inputs(df_lot, capacity)
        logger.info(f"Queueing inputs returned: type={type(q_inputs)}")
        
        # get_queueing_inputs might return a tuple or dict depending on implementation
        if isinstance(q_inputs, tuple) and len(q_inputs) >= 2:
            hourly_map, service_rate_mu = q_inputs[0], q_inputs[1]
            logger.info(f"Got tuple: hourly_map has {len(hourly_map)} hours, mu={service_rate_mu}")
        elif isinstance(q_inputs, dict):
            # support the dict output used in earlier suggestions
            hourly_map = q_inputs.get("chosen_map", q_inputs.get("hourly_arrival_rates", {}))
            service_rate_mu = q_inputs.get("service_rate_mu", 1.0 / q_inputs.get("avg_service_time_hours", 1.0))
            logger.info(f"Got dict: hourly_map has {len(hourly_map)} hours, mu={service_rate_mu}")
        else:
            raise ValueError("Unsupported return type from get_queueing_inputs")
    except Exception as e:
        logger.error(f"Failed to estimate queueing inputs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to estimate queueing inputs: {str(e)}")

    # compute hour
    hour = payload.hour_of_day
    if hour is None:
        # if no hour given, assume current hour (UTC); you may want to convert to local timezone
        from datetime import datetime
        hour = datetime.utcnow().hour

    try:
        logger.info(f"Calling get_booking_confirmation with: occ={payload.predicted_occupancy}, cap={capacity}, hour={hour}, mu={service_rate_mu}")
        res = get_booking_confirmation(payload.predicted_occupancy, capacity, hour, hourly_map, service_rate_mu)
        logger.info(f"Booking result: {res}")
        return res
    except Exception as e:
        logger.error(f"Booking confirmation calculation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Booking confirmation calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Booking computation failed: {e}")

    # Expect res to be a dict: {prob_get_spot, prob_wait, expected_wait_minutes, available_slots, ...}
    return {"lot_id": payload.lot_id, "hour_of_day": hour, "booking": res}


# -----------------------
# Optional startup: preload models or caches
# -----------------------
@app.on_event("startup")
def startup_event():
    logger.info("Starting Parking Predictor API")
    # Example: try to pre-load LightGBM model to warm cache
    try:
        mdl = load_model_by_name("lightgbm")
        MODEL_CACHE["lightgbm"] = mdl
        logger.info("Preloaded lightgbm model into cache")
    except Exception as e:
        logger.warning("Could not preload lightgbm model: %s", e)

