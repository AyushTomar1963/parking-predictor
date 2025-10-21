/**
 * API client for Parking Predictor backend
 * Handles all HTTP communication with FastAPI server
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

/**
 * Fetch list of available parking lots
 * @returns {Promise<Array>} List of lot objects
 */
export async function listLots() {
  try {
    const response = await fetch(`${BASE_URL}/api/lots`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    return data.lots || [];
  } catch (error) {
    console.error('Error fetching lots:', error);
    throw error;
  }
}

/**
 * Get occupancy predictions for a parking lot
 * @param {string} lot_id - Parking lot identifier
 * @param {number} horizon - Number of hours to predict
 * @param {string} model - Model name (lightgbm, xgboost, arima)
 * @returns {Promise<Object>} Predictions object
 */
export async function predict(lot_id, horizon, model = 'lightgbm') {
  try {
    const response = await fetch(`${BASE_URL}/api/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        lot_id,
        horizon,
        model,
      }),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching predictions:', error);
    throw error;
  }
}

/**
 * Calculate booking probability for predicted occupancy
 * @param {string} lot_id - Parking lot identifier
 * @param {number} predicted_occupancy - Predicted occupancy value
 * @param {number} hour_of_day - Hour (0-23)
 * @param {number|null} capacity - Lot capacity (optional)
 * @returns {Promise<Object>} Booking probability results
 */
export async function booking(lot_id, predicted_occupancy, hour_of_day, capacity = null) {
  try {
    const payload = {
      lot_id,
      predicted_occupancy,
      hour_of_day,
    };
    
    if (capacity !== null) {
      payload.capacity = capacity;
    }
    
    const response = await fetch(`${BASE_URL}/api/booking`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching booking probability:', error);
    throw error;
  }
}
