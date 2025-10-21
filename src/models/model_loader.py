# src/models/model_loader.py
import joblib
import os
from pathlib import Path

# Model filename mappings - check multiple possible filenames
MODEL_FILES = {
    'lightgbm': ['lightgbm_model.txt', 'lightgbm_model'],
    'xgboost': ['xgboost_model.json', 'xgboost_model'],
    'arima': ['arima_model.pkl']
}

def load_model(model_name, model_dir='data/models'):
    """
    Load a trained model by name from the models directory.
    Supports lightgbm, xgboost, and arima models.
    """
    from src.models import LightGBMModel, XGBoostModel, ARIMAXModel
    
    model_dir = Path(model_dir)
    
    # Try different possible filenames
    possible_files = MODEL_FILES.get(model_name, [])
    if isinstance(possible_files, str):
        possible_files = [possible_files]
    
    model_path = None
    for filename in possible_files:
        potential_path = model_dir / filename
        if potential_path.exists():
            model_path = potential_path
            break
    
    if model_path is None:
        raise FileNotFoundError(f"No model found for name '{model_name}' in {model_dir}")
    
    # Load based on model type
    if model_name == 'lightgbm':
        model = LightGBMModel()
        model.load_model(str(model_path))
        return model
    elif model_name == 'xgboost':
        model = XGBoostModel()
        model.load_model(str(model_path))
        return model
    elif model_name == 'arima':
        model = ARIMAXModel()
        model.load_model(str(model_path))
        return model
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def load_models_from_directory(directory, extension=".joblib"):
    """
    Load all models with the given extension from a directory.
    Returns a dict {filename: model}.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    models = {}
    for fname in os.listdir(directory):
        if fname.endswith(extension):
            path = os.path.join(directory, fname)
            models[fname] = joblib.load(path)

    if not models:
        print(f"[Warning] No models with extension '{extension}' found in {directory}")

    return models
