"""
Train Q-Learning Agent for Parking Allocation

This script trains a Q-learning agent to learn optimal parking booking policies.
The agent learns when to accept or defer bookings based on current occupancy
and time of day, optimizing for successful parking experiences and revenue.

Usage:
    python scripts/train_rl_agent.py
    python scripts/train_rl_agent.py --episodes 2000 --capacity 600
    python scripts/train_rl_agent.py --load-data data/processed/lot.csv
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

from src.rl import (
    ParkingEnv,
    QLearningAgent,
    train_agent,
    evaluate_agent,
    plot_learning_curve,
    save_q_table,
    load_q_table
)
from src.rl.trainer import compare_policies
from src.rl.utils import (
    plot_policy_heatmap,
    plot_value_heatmap,
    plot_policy_comparison,
    print_performance_summary
)


def load_queueing_params_from_data(csv_path: str, capacity: int):
    """
    Load historical data and estimate queueing parameters.
    
    Args:
        csv_path: Path to processed parking data CSV
        capacity: Parking lot capacity
        
    Returns:
        Tuple of (hourly_arrival_rates, service_rate_mu)
    """
    try:
        from src.queueing import get_queueing_inputs
        
        df = pd.read_csv(csv_path)
        df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])
        
        hourly_rates, mu = get_queueing_inputs(df, capacity)
        
        return hourly_rates, mu
    
    except Exception as e:
        print(f"Warning: Could not load data from {csv_path}: {e}")
        print("Using default parameters...")
        return None, 0.5


def main():
    parser = argparse.ArgumentParser(
        description="Train Q-Learning agent for parking allocation"
    )
    
    # Environment parameters
    parser.add_argument('--capacity', type=int, default=600,
                       help='Parking lot capacity (default: 600)')
    parser.add_argument('--load-data', type=str, default=None,
                       help='Path to historical data CSV for parameter estimation')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--max-steps', type=int, default=24,
                       help='Maximum steps per episode (default: 24)')
    
    # Agent parameters
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate alpha (default: 0.1)')
    parser.add_argument('--discount', type=float, default=0.95,
                       help='Discount factor gamma (default: 0.95)')
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='Initial exploration rate (default: 1.0)')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                       help='Epsilon decay rate (default: 0.995)')
    
    # Evaluation parameters
    parser.add_argument('--eval-episodes', type=int, default=100,
                       help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--compare-policies', action='store_true',
                       help='Compare with baseline policies')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='results/rl',
                       help='Output directory for results (default: results/rl)')
    parser.add_argument('--save-agent', action='store_true',
                       help='Save trained agent')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Q-LEARNING AGENT TRAINING FOR PARKING ALLOCATION")
    print("=" * 80)
    
    # Load queueing parameters from data if provided
    if args.load_data:
        print(f"\nLoading historical data from: {args.load_data}")
        hourly_rates, mu = load_queueing_params_from_data(args.load_data, args.capacity)
    else:
        hourly_rates = None
        mu = 0.5
    
    # Initialize environment
    print("\n📍 Initializing environment...")
    env = ParkingEnv(
        capacity=args.capacity,
        hourly_arrival_rates=hourly_rates,
        service_rate_mu=mu,
        max_steps=args.max_steps
    )
    
    print(f"  Capacity: {args.capacity}")
    print(f"  Service rate (μ): {mu:.4f} per hour")
    print(f"  Avg parking duration: {1/mu:.2f} hours")
    
    # Initialize agent
    print("\n🤖 Initializing Q-Learning agent...")
    agent = QLearningAgent(
        learning_rate=args.learning_rate,
        discount_factor=args.discount,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay
    )
    
    print(agent)
    
    # Train agent
    print(f"\n🎓 Training for {args.episodes} episodes...")
    training_metrics = train_agent(
        env=env,
        agent=agent,
        n_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        verbose=True,
        log_interval=max(1, args.episodes // 10)
    )
    
    # Evaluate agent
    print(f"\n📊 Evaluating agent over {args.eval_episodes} episodes...")
    eval_metrics = evaluate_agent(
        env=env,
        agent=agent,
        n_episodes=args.eval_episodes,
        verbose=True
    )
    
    # Compare with baseline policies
    comparison_results = None
    if args.compare_policies:
        print("\n🔍 Comparing with baseline policies...")
        comparison_results = compare_policies(
            env=env,
            agent=agent,
            n_episodes=args.eval_episodes,
            verbose=True
        )
    
    # Print comprehensive summary
    print_performance_summary(training_metrics, eval_metrics, comparison_results)
    
    # Save results
    print("\n💾 Saving results...")
    
    # Save Q-table
    q_table_path = output_dir / 'q_table.npy'
    save_q_table(agent, str(q_table_path))
    
    # Save complete agent
    if args.save_agent:
        agent_path = output_dir / 'agent.pkl'
        agent.save(str(agent_path))
        print(f"Complete agent saved to: {agent_path}")
    
    # Save metrics
    metrics_path = output_dir / 'training_metrics.npz'
    np.savez(
        metrics_path,
        episode_rewards=training_metrics['episode_rewards'],
        success_rates=training_metrics['success_rates'],
        epsilon_history=training_metrics['epsilon_history'],
        avg_td_errors=training_metrics['avg_td_errors']
    )
    print(f"Training metrics saved to: {metrics_path}")
    
    # Generate plots
    if not args.no_plots:
        print("\n📈 Generating visualizations...")
        
        # Learning curve
        learning_curve_path = output_dir / 'rl_learning_curve.png'
        plot_learning_curve(
            training_metrics,
            save_path=str(learning_curve_path),
            show=False
        )
        
        # Policy heatmap
        policy_path = output_dir / 'policy_heatmap.png'
        plot_policy_heatmap(
            agent,
            save_path=str(policy_path),
            show=False
        )
        
        # Value heatmap
        value_path = output_dir / 'value_heatmap.png'
        plot_value_heatmap(
            agent,
            save_path=str(value_path),
            show=False
        )
        
        # Policy comparison
        if comparison_results:
            comparison_path = output_dir / 'policy_comparison.png'
            plot_policy_comparison(
                comparison_results,
                save_path=str(comparison_path),
                show=False
            )
    
    # Save summary report
    report_path = output_dir / 'training_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Q-LEARNING AGENT TRAINING REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Parking capacity: {args.capacity}\n")
        f.write(f"Training episodes: {args.episodes}\n")
        f.write(f"Max steps per episode: {args.max_steps}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Discount factor: {args.discount}\n")
        f.write(f"Initial epsilon: {args.epsilon}\n")
        f.write(f"Epsilon decay: {args.epsilon_decay}\n\n")
        
        f.write("TRAINING RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Training time: {training_metrics['training_time']:.2f} seconds\n")
        f.write(f"Final epsilon: {training_metrics['epsilon_history'][-1]:.4f}\n")
        
        last_100_rewards = training_metrics['episode_rewards'][-100:]
        last_100_success = training_metrics['success_rates'][-100:]
        f.write(f"\nLast 100 episodes:\n")
        f.write(f"  Mean reward: {np.mean(last_100_rewards):.2f} ± {np.std(last_100_rewards):.2f}\n")
        f.write(f"  Mean success rate: {np.mean(last_100_success):.2%}\n\n")
        
        f.write("EVALUATION RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}\n")
        f.write(f"Success rate: {eval_metrics['mean_success_rate']:.2%}\n")
        f.write(f"Avg utilization: {eval_metrics['mean_utilization']:.2%}\n")
        f.write(f"Avg bookings accepted: {eval_metrics['mean_accepted']:.1f}\n")
        f.write(f"Avg bookings deferred: {eval_metrics['mean_deferred']:.1f}\n\n")
        
        if comparison_results:
            f.write("POLICY COMPARISON:\n")
            f.write("-" * 80 + "\n")
            for policy_name, metrics in comparison_results.items():
                f.write(f"\n{policy_name}:\n")
                f.write(f"  Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}\n")
                f.write(f"  Success rate: {metrics['mean_success_rate']:.2%}\n")
                f.write(f"  Utilization: {metrics['mean_utilization']:.2%}\n")
    
    print(f"Training report saved to: {report_path}")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - q_table.npy: Trained Q-table")
    print(f"  - training_metrics.npz: Training metrics")
    print(f"  - training_report.txt: Summary report")
    if not args.no_plots:
        print(f"  - rl_learning_curve.png: Learning progress")
        print(f"  - policy_heatmap.png: Learned policy visualization")
        print(f"  - value_heatmap.png: State value function")
        if comparison_results:
            print(f"  - policy_comparison.png: Policy comparison")
    if args.save_agent:
        print(f"  - agent.pkl: Complete agent state")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
