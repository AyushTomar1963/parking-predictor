"""
Integrated RL Demo: ML Prediction + Queueing Theory + RL Agent

This script demonstrates the complete integration of:
1. ML model predictions (LightGBM)
2. Queueing theory parameters (Erlang-C)
3. RL agent decisions (Q-Learning)

Shows how RL agent can optimize booking decisions using predicted occupancy
and queueing metrics from the existing parking predictor system.

Usage:
    python scripts/rl_integrated_demo.py
    python scripts/rl_integrated_demo.py --data data/raw/dataset.csv --lot BHMBCCMKT01
"""

import sys
import os
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import process_lot_data
from src.models import LightGBMModel, evaluate_model
from src.queueing import get_queueing_inputs, get_booking_confirmation
from src.rl import ParkingEnv, QLearningAgent, train_agent, evaluate_agent


def run_integrated_demo(
    csv_path: str = 'data/raw/dataset.csv',
    lot_id: str = 'BHMBCCMKT01',
    capacity: int = 600,
    train_rl: bool = True
):
    """
    Run integrated demo showing ML + Queueing + RL pipeline.
    
    Args:
        csv_path: Path to dataset CSV
        lot_id: Parking lot system code
        capacity: Lot capacity
        train_rl: Whether to train RL agent (or use random policy)
    """
    
    print("=" * 80)
    print("INTEGRATED DEMO: ML + QUEUEING + RL")
    print("=" * 80)
    
    # ========== STEP 1: LOAD AND PREPROCESS DATA ==========
    print("\n[1/6] Loading and preprocessing data...")
    df = pd.read_csv(csv_path)
    df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])
    
    # Segregate by lot
    segregated_lots = {lot_id: group_df for lot_id, group_df in df.groupby('SystemCodeNumber')}
    one_lot_df = segregated_lots[lot_id]
    one_lot_df = one_lot_df.set_index('LastUpdated')
    
    processed_lot_data = process_lot_data(one_lot_df)
    
    print(f"   ✓ Loaded {len(df)} records")
    print(f"   ✓ Selected lot: {lot_id} ({len(one_lot_df)} records)")
    print(f"   ✓ Processed to {len(processed_lot_data)} hourly records")
    
    # ========== STEP 2: ESTIMATE QUEUEING PARAMETERS ==========
    print("\n[2/6] Estimating queueing parameters...")
    hourly_arrival_rates, service_rate_mu = get_queueing_inputs(
        one_lot_df.reset_index(),
        capacity=capacity
    )
    
    print(f"   ✓ Estimated hourly arrival rates for 24 hours")
    print(f"   ✓ Service rate (μ): {service_rate_mu:.4f} per hour")
    print(f"   ✓ Avg parking duration: {1/service_rate_mu:.2f} hours")
    
    # ========== STEP 3: TRAIN ML MODEL ==========
    print("\n[3/6] Training ML prediction model...")
    df_model = processed_lot_data.copy()
    for lag in [1, 2, 3, 24, 48]:
        df_model[f'lag_{lag}'] = df_model['Occupancy'].shift(lag)
    df_model = df_model.dropna()
    
    features = ['lag_1', 'lag_2', 'lag_3', 'lag_24', 'lag_48', 'hour_of_day', 'day_of_week', 'is_weekend']
    target = 'Occupancy'
    
    split = int(len(df_model) * 0.8)
    train, test = df_model.iloc[:split], df_model.iloc[split:]
    
    ml_model = LightGBMModel(
        params={'objective': 'regression', 'metric': 'l2', 'verbosity': -1, 'seed': 42},
        num_boost_round=200
    )
    ml_model.fit(train[features], train[target])
    
    test_predictions = ml_model.predict(test[features])
    metrics = evaluate_model(test[target], test_predictions)
    
    print(f"   ✓ ML Model trained (LightGBM)")
    print(f"   ✓ Test MAE: {metrics['MAE']:.2f} cars")
    print(f"   ✓ Test RMSE: {metrics['RMSE']:.2f} cars")
    
    # ========== STEP 4: INITIALIZE RL ENVIRONMENT ==========
    print("\n[4/6] Initializing RL environment...")
    env = ParkingEnv(
        capacity=capacity,
        hourly_arrival_rates=hourly_arrival_rates,
        service_rate_mu=service_rate_mu,
        max_steps=24
    )
    
    print(f"   ✓ Environment created with real queueing parameters")
    print(f"   ✓ State space: {env.n_occupancy_levels} × {env.n_hours} = {env.state_space_size} states")
    print(f"   ✓ Action space: {env.n_actions} actions (Accept/Defer)")
    
    # ========== STEP 5: TRAIN RL AGENT ==========
    if train_rl:
        print("\n[5/6] Training RL agent...")
        agent = QLearningAgent(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995
        )
        
        training_metrics = train_agent(
            env=env,
            agent=agent,
            n_episodes=500,
            max_steps_per_episode=24,
            verbose=True,
            log_interval=100
        )
        
        print(f"   ✓ Agent trained for 500 episodes")
        print(f"   ✓ Final epsilon: {agent.epsilon:.4f}")
    else:
        print("\n[5/6] Using random policy (no training)...")
        agent = None
    
    # ========== STEP 6: DEMONSTRATE INTEGRATED DECISION MAKING ==========
    print("\n[6/6] Demonstrating integrated decision making...")
    print("\n" + "=" * 80)
    print("SCENARIO: Customer wants to book parking for next 6 hours")
    print("=" * 80)
    
    # Get current state from test data
    current_row = test.iloc[-1]
    current_hour = int(current_row['hour_of_day'])
    
    print(f"\nCurrent time: {current_hour}:00")
    print(f"Current occupancy: {int(current_row['Occupancy'])}/{capacity}")
    
    # Make predictions for next 6 hours
    print("\n" + "-" * 80)
    print("HOUR | ML PRED | QUEUEING PROB | RL DECISION | RECOMMENDATION")
    print("-" * 80)
    
    for h in range(6):
        future_hour = (current_hour + h) % 24
        
        # ML Prediction (simplified - using last prediction)
        predicted_occ = test_predictions[-1] + np.random.randint(-20, 20)
        predicted_occ = max(0, min(capacity, predicted_occ))
        
        # Queueing Theory Probability
        booking_result = get_booking_confirmation(
            predicted_occupancy=predicted_occ,
            capacity=capacity,
            hour_of_day=future_hour,
            hourly_arrival_rates=hourly_arrival_rates,
            service_rate_mu=service_rate_mu
        )
        
        # RL Agent Decision
        if agent:
            occupancy_level = min(10, int((predicted_occ / capacity) * 10))
            state = (occupancy_level, future_hour)
            rl_action = agent.get_action(state, training=False)
            rl_decision = "ACCEPT" if rl_action == 0 else "DEFER"
        else:
            rl_decision = "RANDOM"
        
        # Recommendation
        prob = booking_result['prob_get_spot']
        if prob >= 0.9:
            recommendation = "🟢 BOOK NOW"
        elif prob >= 0.7:
            recommendation = "🟡 MODERATE"
        else:
            recommendation = "🔴 RISKY"
        
        print(
            f"{future_hour:02d}:00 | "
            f"{int(predicted_occ):>3}/{capacity:<3} | "
            f"{prob:>5.1%}         | "
            f"{rl_decision:<11} | "
            f"{recommendation}"
        )
    
    print("-" * 80)
    
    # Show how systems work together
    print("\n" + "=" * 80)
    print("HOW THE SYSTEMS WORK TOGETHER:")
    print("=" * 80)
    print("""
1. ML MODEL (LightGBM):
   - Predicts occupancy for future hours
   - Uses historical patterns and lag features
   - Provides base forecast
   
2. QUEUEING THEORY (Erlang-C):
   - Converts occupancy to booking probability
   - Accounts for arrival rates and service times
   - Provides wait time estimates
   
3. RL AGENT (Q-Learning):
   - Learns optimal accept/defer policy
   - Considers both immediate and future rewards
   - Adapts to changing conditions
   
INTEGRATION:
   ML Prediction → Available Spots → Queueing Metrics → RL Decision
   
ADVANTAGE OF RL:
   - Learns from experience (not just static rules)
   - Optimizes long-term revenue and customer satisfaction
   - Adapts to different times of day and occupancy levels
   - Can handle complex trade-offs (accept now vs defer for better slot)
    """)
    
    print("=" * 80)
    print("✅ INTEGRATED DEMO COMPLETE!")
    print("=" * 80)
    
    # Return components for further use
    return {
        'ml_model': ml_model,
        'queueing_params': (hourly_arrival_rates, service_rate_mu),
        'rl_agent': agent,
        'env': env,
        'test_data': test
    }


def main():
    parser = argparse.ArgumentParser(
        description="Integrated demo of ML + Queueing + RL"
    )
    
    parser.add_argument('--data', type=str, default='data/raw/dataset.csv',
                       help='Path to dataset CSV')
    parser.add_argument('--lot', type=str, default='BHMBCCMKT01',
                       help='Parking lot system code')
    parser.add_argument('--capacity', type=int, default=600,
                       help='Parking lot capacity')
    parser.add_argument('--no-train-rl', action='store_true',
                       help='Skip RL training (use random policy)')
    
    args = parser.parse_args()
    
    components = run_integrated_demo(
        csv_path=args.data,
        lot_id=args.lot,
        capacity=args.capacity,
        train_rl=not args.no_train_rl
    )
    
    print("\nComponents returned:")
    print(f"  - ml_model: {type(components['ml_model']).__name__}")
    print(f"  - queueing_params: hourly_rates + service_rate")
    print(f"  - rl_agent: {type(components['rl_agent']).__name__ if components['rl_agent'] else 'None'}")
    print(f"  - env: {type(components['env']).__name__}")


if __name__ == "__main__":
    main()
