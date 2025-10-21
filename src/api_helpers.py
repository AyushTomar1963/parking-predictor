"""
FastAPI endpoint helpers and utility functions.
"""
from datetime import datetime
import pickle


def load_model(model_path):
    """Load a trained model from disk."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def format_prediction_response(predictions, timestamps):
    """Format prediction data for API response."""
    response = {
        'predictions': [],
        'generated_at': datetime.now().isoformat()
    }
    
    for pred, ts in zip(predictions, timestamps):
        response['predictions'].append({
            'timestamp': ts.isoformat(),
            'predicted_occupancy': float(pred)
        })
    
    return response


def validate_datetime_input(datetime_str):
    """Validate and parse datetime input."""
    try:
        dt = datetime.fromisoformat(datetime_str)
        return dt, None
    except ValueError as e:
        return None, f"Invalid datetime format: {str(e)}"


def calculate_parking_metrics(predictions, capacity):
    """Calculate useful metrics from predictions."""
    occupancy_rate = predictions / capacity
    available_spots = capacity - predictions
    
    metrics = {
        'average_occupancy': float(predictions.mean()),
        'peak_occupancy': float(predictions.max()),
        'average_availability': float(available_spots.mean()),
        'occupancy_rate': float(occupancy_rate.mean())
    }
    
    return metrics
