"""
Example script demonstrating diverse model implementations.
Shows ARIMAX, LSTM, LightGBM, and XGBoost models.
"""

import numpy as np
import pandas as pd
from src.models import (
    ARIMAXModel, 
    LSTMModel, 
    LightGBMModel, 
    XGBoostModel,
    evaluate_model,
    compare_models,
    get_best_model
)

def generate_sample_data(n_samples=1000):
    """Generate sample time series data for testing."""
    np.random.seed(42)
    
    # Create time-based features
    time = np.arange(n_samples)
    
    # Generate synthetic occupancy data with trend and seasonality
    trend = 0.05 * time
    seasonality = 50 * np.sin(2 * np.pi * time / 24)  # Daily pattern
    noise = np.random.normal(0, 10, n_samples)
    occupancy = 300 + trend + seasonality + noise
    occupancy = np.clip(occupancy, 0, 600)  # Keep within capacity
    
    # Create DataFrame
    df = pd.DataFrame({
        'occupancy': occupancy,
        'hour': time % 24,
        'day_of_week': (time // 24) % 7,
        'is_weekend': ((time // 24) % 7) >= 5
    })
    
    return df


def prepare_features(df, n_lags=5):
    """Prepare lag features for ML models."""
    df_feat = df.copy()
    
    # Add lag features
    for lag in range(1, n_lags + 1):
        df_feat[f'lag_{lag}'] = df_feat['occupancy'].shift(lag)
    
    # Drop NaN rows
    df_feat = df_feat.dropna()
    
    return df_feat


def demo_traditional_models():
    """Demonstrate ARIMAX model."""
    print("\n" + "="*60)
    print("📊 TRADITIONAL TIME SERIES MODEL - ARIMAX")
    print("="*60)
    
    # Generate data
    df = generate_sample_data(500)
    train_size = int(len(df) * 0.8)
    train_data = df['occupancy'][:train_size]
    test_data = df['occupancy'][train_size:]
    
    # Train ARIMAX
    print("\n🔧 Training ARIMAX(1,1,1)...")
    arimax = ARIMAXModel(order=(1, 1, 1))
    arimax.fit(train_data)
    
    # Predict
    predictions = arimax.predict(steps=len(test_data))
    
    # Evaluate
    metrics = evaluate_model(test_data.values, predictions)
    print(f"✅ ARIMAX Results:")
    print(f"   MAE:  {metrics['MAE']:.2f}")
    print(f"   RMSE: {metrics['RMSE']:.2f}")
    
    return arimax, metrics


def demo_gradient_boosting_models():
    """Demonstrate LightGBM and XGBoost models."""
    print("\n" + "="*60)
    print("🚀 GRADIENT BOOSTING MODELS - LightGBM & XGBoost")
    print("="*60)
    
    # Generate and prepare data
    df = generate_sample_data(1000)
    df_feat = prepare_features(df, n_lags=5)
    
    # Split data
    train_size = int(len(df_feat) * 0.8)
    train = df_feat[:train_size]
    test = df_feat[train_size:]
    
    feature_cols = ['hour', 'day_of_week', 'is_weekend', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
    X_train = train[feature_cols]
    y_train = train['occupancy']
    X_test = test[feature_cols]
    y_test = test['occupancy']
    
    models_dict = {}
    
    # Train LightGBM
    print("\n🔧 Training LightGBM...")
    lgbm = LightGBMModel(num_boost_round=100)
    lgbm.fit(X_train, y_train)
    models_dict['LightGBM'] = lgbm
    print("✅ LightGBM training complete!")
    
    # Train XGBoost
    print("\n🔧 Training XGBoost...")
    xgbm = XGBoostModel(num_boost_round=100)
    xgbm.fit(X_train, y_train)
    models_dict['XGBoost'] = xgbm
    print("✅ XGBoost training complete!")
    
    # Compare models
    print("\n📊 Comparing Models...")
    results = compare_models(models_dict, X_test, y_test)
    
    for model_name, metrics in results.items():
        print(f"\n✨ {model_name} Results:")
        print(f"   MAE:  {metrics['MAE']:.2f}")
        print(f"   RMSE: {metrics['RMSE']:.2f}")
        if 'MAPE' in metrics:
            print(f"   MAPE: {metrics['MAPE']:.2f}%")
    
    # Get best model
    best_name, best_score = get_best_model(results, metric='RMSE')
    print(f"\n🏆 Best Model: {best_name} (RMSE: {best_score:.2f})")
    
    return models_dict, results


def demo_model_diversity():
    """Main demo showing diverse model implementations."""
    print("\n" + "🎯"*30)
    print("PARKING PREDICTOR - MODEL DIVERSITY DEMONSTRATION")
    print("🎯"*30)
    
    print("\n📚 Available Models:")
    print("   1. ARIMAX     - Statistical time series model")
    print("   2. LSTM       - Deep learning recurrent neural network")
    print("   3. LightGBM   - Gradient boosting (Microsoft)")
    print("   4. XGBoost    - Gradient boosting (DMLC)")
    
    # Demo traditional models
    arimax_model, arimax_metrics = demo_traditional_models()
    
    # Demo gradient boosting models
    gb_models, gb_results = demo_gradient_boosting_models()
    
    # Summary
    print("\n" + "="*60)
    print("📈 MODEL DIVERSITY SUMMARY")
    print("="*60)
    print("\n✅ Successfully demonstrated 4 different model types:")
    print("   • Statistical Models: ARIMAX")
    print("   • Deep Learning: LSTM (architecture defined)")
    print("   • Gradient Boosting: LightGBM, XGBoost")
    
    print("\n🎯 Model Selection Flexibility:")
    print("   • Fast predictions: LightGBM, XGBoost")
    print("   • Interpretability: ARIMAX")
    print("   • Complex patterns: LSTM")
    print("   • Best accuracy: Use model comparison utility")
    
    print("\n" + "🎉"*30)
    print("DEMONSTRATION COMPLETE!")
    print("🎉"*30 + "\n")


if __name__ == "__main__":
    demo_model_diversity()
