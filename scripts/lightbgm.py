# scripts/train_lightgbm.py
import argparse
import pandas as pd
import joblib
from pathlib import Path
import sys, os

# Add src to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import LightGBMModel, evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Train LightGBM model on processed one-lot dataset")
    parser.add_argument("--data", type=str, required=True, help="Path to processed dataset (CSV)")
    parser.add_argument("--out", type=str, required=True, help="Output model path (.joblib)")
    args = parser.parse_args()

    # === 1. Load the processed data ===
    print(f"📂 Loading processed dataset: {args.data}")
    df = pd.read_csv(args.data)
    print(f"✅ Loaded {len(df)} rows")

    # === 2. Define features and target ===
    features = ['lag_1', 'lag_2', 'lag_3', 'lag_24', 'lag_48', 
                'hour_of_day', 'day_of_week', 'is_weekend']
    target = 'Occupancy'

    # === 3. Split train-test ===
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    # === 4. Train LightGBM ===
    print("⚙ Training LightGBM model...")
    model = LightGBMModel(
        params={"objective": "regression", "metric": "l2", "verbosity": -1, "seed": 42},
        num_boost_round=200
    )
    model.fit(X_train, y_train)
    print("✅ Training complete")

    # === 5. Evaluate ===
    preds = model.predict(X_test)
    metrics = evaluate_model(y_test, preds)
    print(f"📊 MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")

    # === 6. Save model ===
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)
    print(f"💾 Model saved to {output_path}")

if __name__ == "__main__":
    main()