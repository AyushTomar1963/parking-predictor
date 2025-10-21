# Parking Predictor Frontend

Modern React-based web interface for the Parking Predictor API.

## Features

- 🎯 **Real-time Predictions**: Get parking occupancy forecasts using ML models
- 📊 **Visual Analytics**: Interactive charts showing occupancy trends
- 🎟️ **Booking Probability**: Calculate likelihood of finding a parking spot
- 🔮 **Multiple Models**: Compare LightGBM, XGBoost, and ARIMA predictions
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Prerequisites

- Node.js (v18 or higher)
- Backend API running on `http://127.0.0.1:8000`

## Installation

```bash
cd frontend
npm install
```

## Configuration

Create or edit `.env` file in the `frontend` directory:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Running the App

### Development Mode

```bash
npm run dev
```

The app will open automatically at `http://localhost:3000`

### Production Build

```bash
npm run build
npm run preview
```

## API Endpoints Used

The frontend communicates with these backend endpoints:

- `GET /api/lots` - List available parking lots
- `POST /api/predict` - Get occupancy predictions
- `POST /api/booking` - Calculate booking probability

## Project Structure

```
frontend/
├── src/
│   ├── components/         # React components
│   │   ├── Header.jsx      # App title and instructions
│   │   ├── LotsList.jsx    # Parking lot selector
│   │   ├── PredictForm.jsx # Model and horizon inputs
│   │   ├── PredictionsView.jsx  # Results table and chart
│   │   └── BookingCard.jsx # Booking probability modal
│   ├── utils/
│   │   └── date.js         # Date formatting helpers
│   ├── api.js              # API client functions
│   ├── App.jsx             # Main app component
│   ├── main.jsx            # React entry point
│   └── styles.css          # Global styles
├── index.html              # HTML template
├── vite.config.js          # Vite configuration
├── package.json            # Dependencies
└── .env                    # Environment variables
```

## Usage

1. **Select a parking lot** from the dropdown
2. **Choose a prediction model**:
   - **LightGBM**: Fast gradient boosting (recommended)
   - **XGBoost**: Alternative gradient boosting
   - **ARIMA**: Time series model
3. **Set time horizon** (1-48 hours ahead)
4. **Click "Predict"** to generate forecasts
5. **View results** in table and chart format
6. **Click "Check Booking"** on any prediction to see booking probability

## Features Explained

### Predictions View

- **Table**: Detailed list of predictions with timestamps
- **Chart**: Visual representation of occupancy trends (requires recharts)
- **Booking Analysis**: Click any row to check booking probability

### Booking Probability

- **Probability of Getting Spot**: Likelihood of finding available parking
- **Expected Wait Time**: Average wait time in minutes
- **Confidence Level**: Reliability of the prediction
- **Recommendation**: Suggested action based on analysis

## Troubleshooting

### Backend Connection Issues

Make sure:
1. Backend API is running: `uvicorn app.main:app --reload --port 8000`
2. CORS is enabled in the backend
3. `.env` has correct `VITE_API_BASE_URL`

### Chart Not Showing

Install recharts if missing:
```bash
npm install recharts
```

### Port Already in Use

Change port in `vite.config.js`:
```js
server: {
  port: 3001  // Use different port
}
```

## Development

### Adding New Components

Create components in `src/components/`:
```jsx
export default function MyComponent({ props }) {
  return <div>Content</div>
}
```

### API Integration

Add new API calls in `src/api.js`:
```javascript
export async function myApiCall(params) {
  const response = await fetch(`${BASE_URL}/api/endpoint`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params)
  })
  return await response.json()
}
```

## Technologies

- **React 18** - UI framework
- **Vite** - Build tool
- **Recharts** - Charting library
- **Native Fetch API** - HTTP client

## License

MIT
