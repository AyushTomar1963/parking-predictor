/**
 * BookingCard component
 * Displays booking probability analysis for a specific prediction
 */
import { useState } from 'react'
import { booking } from '../api'

export default function BookingCard({ lotId, prediction, onClose }) {
  const [bookingResult, setBookingResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [capacity, setCapacity] = useState(600) // Default capacity

  // Extract hour from timestamp
  const getHourFromTimestamp = (timestamp) => {
    const date = new Date(timestamp)
    return date.getHours()
  }

  const handleCheckBooking = async () => {
    setLoading(true)
    setError(null)
    setBookingResult(null)

    const hour = getHourFromTimestamp(prediction.timestamp)

    try {
      const result = await booking(
        lotId,
        prediction.predicted_occupancy,
        hour,
        capacity
      )
      setBookingResult(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="booking-card">
      <h3>🎟️ Booking Probability Analysis</h3>
      
      <div className="booking-info">
        <p><strong>Time:</strong> {prediction.timestamp}</p>
        <p><strong>Predicted Occupancy:</strong> {prediction.predicted_occupancy}</p>
        <p><strong>Hour of Day:</strong> {getHourFromTimestamp(prediction.timestamp)}</p>
      </div>

      <div className="form-group">
        <label htmlFor="capacity-input">
          Lot Capacity:
          <small> (adjust if needed)</small>
        </label>
        <input
          id="capacity-input"
          type="number"
          min="1"
          value={capacity}
          onChange={(e) => setCapacity(parseInt(e.target.value) || 600)}
          className="form-input"
        />
      </div>

      {!bookingResult && (
        <button
          onClick={handleCheckBooking}
          className="btn-primary"
          disabled={loading}
        >
          {loading ? 'Calculating...' : 'Calculate Booking Probability'}
        </button>
      )}

      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {bookingResult && (
        <div className="booking-results">
          <h4>Results</h4>
          
          <div className="result-grid">
            <div className="result-card">
              <span className="result-label">Probability of Getting Spot</span>
              <span className="result-value success">
                {(bookingResult.prob_get_spot * 100).toFixed(1)}%
              </span>
            </div>

            {bookingResult.prob_wait !== undefined && (
              <div className="result-card">
                <span className="result-label">Probability of Waiting</span>
                <span className="result-value warning">
                  {(bookingResult.prob_wait * 100).toFixed(1)}%
                </span>
              </div>
            )}

            {bookingResult.expected_wait_minutes !== undefined && (
              <div className="result-card">
                <span className="result-label">Expected Wait Time</span>
                <span className="result-value">
                  {bookingResult.expected_wait_minutes.toFixed(1)} min
                </span>
              </div>
            )}

            {bookingResult.confidence_level && (
              <div className="result-card">
                <span className="result-label">Confidence Level</span>
                <span className={`result-value ${bookingResult.confidence_level}`}>
                  {bookingResult.confidence_level}
                </span>
              </div>
            )}
          </div>

          {bookingResult.recommendation && (
            <div className="recommendation">
              <strong>💡 Recommendation:</strong>
              <p>{bookingResult.recommendation}</p>
            </div>
          )}

          <button onClick={onClose} className="btn-secondary">
            Close
          </button>
        </div>
      )}
    </div>
  )
}
