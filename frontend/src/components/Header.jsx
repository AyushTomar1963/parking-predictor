/**
 * Header component
 * Displays app title and brief instructions
 */
export default function Header() {
  return (
    <header className="app-header">
      <h1>🅿️ Parking Predictor</h1>
      <p className="subtitle">
        Predict parking lot occupancy using machine learning models
      </p>
      <div className="instructions">
        <ol>
          <li>Select a parking lot</li>
          <li>Choose prediction model and time horizon</li>
          <li>Click "Predict" to see future occupancy</li>
          <li>Check booking probability for each time slot</li>
        </ol>
      </div>
    </header>
  )
}
