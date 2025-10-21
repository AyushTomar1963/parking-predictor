/**
 * Main App component
 * Manages state and coordinates child components
 */
import { useState, useEffect } from 'react'
import Header from './components/Header'
import LotsList from './components/LotsList'
import PredictForm from './components/PredictForm'
import PredictionsView from './components/PredictionsView'
import { listLots, predict } from './api'

function App() {
  const [lots, setLots] = useState([])
  const [selectedLot, setSelectedLot] = useState('')
  const [model, setModel] = useState('lightgbm')
  const [horizon, setHorizon] = useState(6)
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Load available lots on mount
  useEffect(() => {
    async function loadLots() {
      try {
        const lotsList = await listLots()
        setLots(lotsList)
        if (lotsList.length > 0) {
          setSelectedLot(lotsList[0].lot_id || lotsList[0])
        }
      } catch (err) {
        setError('Failed to load parking lots: ' + err.message)
      }
    }
    loadLots()
  }, [])

  // Handle prediction request
  async function handlePredict() {
    if (!selectedLot) {
      setError('Please select a parking lot')
      return
    }

    setLoading(true)
    setError(null)
    setPredictions(null)

    try {
      const result = await predict(selectedLot, horizon, model)
      setPredictions(result)
    } catch (err) {
      setError('Prediction failed: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <Header />
      
      {error && (
        <div className="error-banner">
          ⚠️ {error}
        </div>
      )}

      <div className="controls-section">
        <LotsList
          lots={lots}
          selected={selectedLot}
          onChange={setSelectedLot}
        />

        <PredictForm
          model={model}
          horizon={horizon}
          onModelChange={setModel}
          onHorizonChange={setHorizon}
          onSubmit={handlePredict}
          loading={loading}
        />
      </div>

      {predictions && (
        <PredictionsView
          predictions={predictions}
          lotId={selectedLot}
        />
      )}

      {loading && (
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Generating predictions...</p>
        </div>
      )}
    </div>
  )
}

export default App
