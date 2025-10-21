/**
 * PredictForm component
 * Form inputs for model selection and prediction horizon
 */
export default function PredictForm({ model, horizon, onModelChange, onHorizonChange, onSubmit, loading }) {
  return (
    <div className="predict-form">
      <div className="form-row">
        <div className="form-group">
          <label htmlFor="model-select">Model:</label>
          <select
            id="model-select"
            value={model}
            onChange={(e) => onModelChange(e.target.value)}
            className="form-select"
            disabled={loading}
          >
            <option value="lightgbm">LightGBM</option>
            <option value="xgboost">XGBoost</option>
            <option value="arima">ARIMA</option>
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="horizon-input">Horizon (hours):</label>
          <input
            id="horizon-input"
            type="number"
            min="1"
            max="48"
            value={horizon}
            onChange={(e) => onHorizonChange(parseInt(e.target.value) || 1)}
            className="form-input"
            disabled={loading}
          />
        </div>
      </div>

      <button
        onClick={onSubmit}
        className="btn-primary"
        disabled={loading}
      >
        {loading ? 'Predicting...' : '🔮 Predict'}
      </button>
    </div>
  )
}
