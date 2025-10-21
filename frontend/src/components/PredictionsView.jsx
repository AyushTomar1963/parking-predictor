/**
 * PredictionsView component
 * Displays predictions in table and chart format
 */
import { useState } from 'react'
import BookingCard from './BookingCard'
import { formatTimestamp } from '../utils/date'

// Try to import recharts, but degrade gracefully if not available
let LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer;
try {
  const recharts = await import('recharts');
  LineChart = recharts.LineChart;
  Line = recharts.Line;
  XAxis = recharts.XAxis;
  YAxis = recharts.YAxis;
  Tooltip = recharts.Tooltip;
  CartesianGrid = recharts.CartesianGrid;
  ResponsiveContainer = recharts.ResponsiveContainer;
} catch (err) {
  console.log('Recharts not available, chart will be hidden');
}

export default function PredictionsView({ predictions, lotId }) {
  const [selectedPrediction, setSelectedPrediction] = useState(null)
  const [showBooking, setShowBooking] = useState(false)

  const handleCheckBooking = (pred) => {
    setSelectedPrediction(pred)
    setShowBooking(true)
  }

  const closeBooking = () => {
    setShowBooking(false)
    setSelectedPrediction(null)
  }

  if (!predictions || !predictions.predictions) {
    return null
  }

  const predList = predictions.predictions

  return (
    <div className="predictions-container">
      <h2>Predictions for {predictions.lot_id}</h2>
      <p className="predictions-info">
        Model: <strong>{predictions.model}</strong> | 
        Base time: <strong>{predictions.base_time}</strong> | 
        Status: <span className="status-success">{predictions.status}</span>
      </p>

      {/* Chart */}
      {LineChart ? (
        <div className="chart-container">
          <h3>Occupancy Forecast</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={predList}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="horizon" 
                label={{ value: 'Hours Ahead', position: 'insideBottom', offset: -5 }}
              />
              <YAxis 
                label={{ value: 'Occupancy', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                labelFormatter={(value) => `${value} hours ahead`}
                formatter={(value) => [value.toFixed(2), 'Occupancy']}
              />
              <Line 
                type="monotone" 
                dataKey="predicted_occupancy" 
                stroke="#2563eb" 
                strokeWidth={2}
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="chart-fallback">
          📊 Install recharts for visualization: <code>npm install recharts</code>
        </div>
      )}

      {/* Table */}
      <div className="table-container">
        <h3>Predictions Details</h3>
        <table className="predictions-table">
          <thead>
            <tr>
              <th>Horizon</th>
              <th>Timestamp</th>
              <th>Predicted Occupancy</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {predList.map((pred) => (
              <tr key={pred.horizon}>
                <td>{pred.horizon}h</td>
                <td>{formatTimestamp(pred.timestamp)}</td>
                <td className="occupancy-value">{pred.predicted_occupancy}</td>
                <td>
                  <button
                    className="btn-secondary"
                    onClick={() => handleCheckBooking(pred)}
                  >
                    Check Booking
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Booking Modal */}
      {showBooking && selectedPrediction && (
        <div className="modal-overlay" onClick={closeBooking}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={closeBooking}>×</button>
            <BookingCard
              lotId={lotId}
              prediction={selectedPrediction}
              onClose={closeBooking}
            />
          </div>
        </div>
      )}
    </div>
  )
}
