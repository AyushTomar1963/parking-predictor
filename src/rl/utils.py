"""
Utility Functions for RL Module

Includes:
- Plotting learning curves
- Visualizing Q-tables and policies
- Saving/loading Q-tables
- Performance analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path


def plot_learning_curve(
    metrics: Dict[str, List],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot learning curves from training metrics.
    
    Creates a multi-panel figure showing:
    - Episode rewards over time
    - Success rate over time
    - Epsilon decay
    - TD error over time
    
    Args:
        metrics: Dictionary from train_agent() with training metrics
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Q-Learning Training Progress', fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards
    ax = axes[0, 0]
    rewards = metrics['episode_rewards']
    episodes = range(1, len(rewards) + 1)
    
    ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
    
    # Moving average
    window = min(100, len(rewards) // 10)
    if window > 1:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(rewards) + 1), moving_avg, 
                color='red', linewidth=2, label=f'{window}-episode MA')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Success Rate
    ax = axes[0, 1]
    success_rates = metrics['success_rates']
    
    ax.plot(episodes, success_rates, alpha=0.3, color='green', label='Raw')
    
    if window > 1:
        moving_avg = np.convolve(success_rates, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(success_rates) + 1), moving_avg,
                color='darkgreen', linewidth=2, label=f'{window}-episode MA')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Booking Success Rate')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Epsilon Decay
    ax = axes[1, 0]
    epsilon = metrics['epsilon_history']
    
    ax.plot(episodes, epsilon, color='purple', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon (ε)')
    ax.set_title('Exploration Rate Decay')
    ax.grid(True, alpha=0.3)
    
    # 4. TD Error
    ax = axes[1, 1]
    td_errors = metrics['avg_td_errors']
    
    ax.plot(episodes, td_errors, alpha=0.3, color='orange', label='Raw')
    
    if window > 1:
        moving_avg = np.convolve(td_errors, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(td_errors) + 1), moving_avg,
                color='darkorange', linewidth=2, label=f'{window}-episode MA')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg |TD Error|')
    ax.set_title('Temporal Difference Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_policy_heatmap(
    agent,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot learned policy as heatmap.
    
    Shows which action (accept/defer) the agent chooses for each
    (occupancy, hour) state combination.
    
    Args:
        agent: Trained QLearningAgent
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    policy = agent.get_policy_heatmap()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap
    sns.heatmap(
        policy,
        cmap=['lightgreen', 'lightcoral'],
        cbar_kws={'label': 'Action', 'ticks': [0.25, 0.75]},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )
    
    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(['Accept (0)', 'Defer (1)'])
    
    # Labels
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Occupancy Level (0=0%, 10=100%)', fontsize=12)
    ax.set_title('Learned Policy: Accept vs Defer Decisions', fontsize=14, fontweight='bold')
    
    # Set ticks
    ax.set_xticks(np.arange(0, 24, 2) + 0.5)
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_yticks(np.arange(0, 11) + 0.5)
    ax.set_yticklabels([f'{i*10}%' for i in range(11)])
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy heatmap saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_value_heatmap(
    agent,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot state values as heatmap.
    
    Shows V(s) = max_a Q(s,a) for each state.
    
    Args:
        agent: Trained QLearningAgent
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    values = agent.get_value_heatmap()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap
    sns.heatmap(
        values,
        cmap='YlOrRd',
        cbar_kws={'label': 'State Value V(s)'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        fmt='.1f'
    )
    
    # Labels
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Occupancy Level (0=0%, 10=100%)', fontsize=12)
    ax.set_title('State Value Function V(s)', fontsize=14, fontweight='bold')
    
    # Set ticks
    ax.set_xticks(np.arange(0, 24, 2) + 0.5)
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_yticks(np.arange(0, 11) + 0.5)
    ax.set_yticklabels([f'{i*10}%' for i in range(11)])
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Value heatmap saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_policy_comparison(
    comparison_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot comparison of different policies.
    
    Args:
        comparison_results: Output from compare_policies()
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    """
    policies = list(comparison_results.keys())
    
    # Extract metrics
    rewards = [comparison_results[p]['mean_reward'] for p in policies]
    reward_stds = [comparison_results[p]['std_reward'] for p in policies]
    success_rates = [comparison_results[p]['mean_success_rate'] for p in policies]
    utilizations = [comparison_results[p]['mean_utilization'] for p in policies]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Policy Comparison', fontsize=16, fontweight='bold')
    
    # 1. Rewards
    ax = axes[0]
    bars = ax.bar(range(len(policies)), rewards, yerr=reward_stds, 
                   capsize=5, color='skyblue', edgecolor='black')
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=45, ha='right')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Average Reward per Episode')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight best
    best_idx = np.argmax(rewards)
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkgoldenrod')
    bars[best_idx].set_linewidth(2)
    
    # 2. Success Rates
    ax = axes[1]
    bars = ax.bar(range(len(policies)), success_rates, 
                   color='lightgreen', edgecolor='black')
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=45, ha='right')
    ax.set_ylabel('Success Rate')
    ax.set_title('Booking Success Rate')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight best
    best_idx = np.argmax(success_rates)
    bars[best_idx].set_color('darkgreen')
    
    # 3. Utilization
    ax = axes[2]
    bars = ax.bar(range(len(policies)), utilizations,
                   color='lightcoral', edgecolor='black')
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=45, ha='right')
    ax.set_ylabel('Utilization')
    ax.set_title('Average Lot Utilization')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy comparison saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_q_table(agent, filepath: str) -> None:
    """
    Save Q-table to .npy file.
    
    Args:
        agent: QLearningAgent instance
        filepath: Path to save Q-table
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, agent.q_table)
    print(f"Q-table saved to: {filepath}")


def load_q_table(filepath: str) -> np.ndarray:
    """
    Load Q-table from .npy file.
    
    Args:
        filepath: Path to Q-table file
        
    Returns:
        Q-table array
    """
    q_table = np.load(filepath)
    print(f"Q-table loaded from: {filepath}")
    return q_table


def print_performance_summary(
    training_metrics: Dict,
    eval_metrics: Dict,
    comparison_results: Optional[Dict] = None
) -> None:
    """
    Print comprehensive performance summary.
    
    Args:
        training_metrics: Output from train_agent()
        eval_metrics: Output from evaluate_agent()
        comparison_results: Output from compare_policies() (optional)
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    # Training stats
    print("\n📊 TRAINING STATISTICS:")
    print("-" * 80)
    print(f"Total episodes: {len(training_metrics['episode_rewards'])}")
    print(f"Training time: {training_metrics['training_time']:.2f} seconds")
    print(f"Final epsilon: {training_metrics['epsilon_history'][-1]:.4f}")
    
    # Last 100 episodes
    last_100_rewards = training_metrics['episode_rewards'][-100:]
    last_100_success = training_metrics['success_rates'][-100:]
    print(f"\nLast 100 episodes:")
    print(f"  Mean reward: {np.mean(last_100_rewards):.2f} ± {np.std(last_100_rewards):.2f}")
    print(f"  Mean success rate: {np.mean(last_100_success):.2%}")
    
    # Evaluation stats
    print("\n🎯 EVALUATION RESULTS:")
    print("-" * 80)
    print(f"Mean reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
    print(f"Success rate: {eval_metrics['mean_success_rate']:.2%}")
    print(f"Avg utilization: {eval_metrics['mean_utilization']:.2%}")
    print(f"Avg bookings accepted: {eval_metrics['mean_accepted']:.1f}")
    print(f"Avg bookings deferred: {eval_metrics['mean_deferred']:.1f}")
    
    # Policy comparison
    if comparison_results:
        print("\n🏆 POLICY COMPARISON:")
        print("-" * 80)
        
        # Find best policy
        best_policy = max(comparison_results.items(), 
                         key=lambda x: x[1]['mean_reward'])
        
        print(f"Best policy: {best_policy[0]}")
        print(f"  Reward: {best_policy[1]['mean_reward']:.2f}")
        print(f"  Success rate: {best_policy[1]['mean_success_rate']:.2%}")
        
        # Show improvement over baselines
        if 'RL Agent' in comparison_results:
            rl_reward = comparison_results['RL Agent']['mean_reward']
            
            print("\nRL Agent vs Baselines:")
            for policy_name, metrics in comparison_results.items():
                if policy_name != 'RL Agent':
                    improvement = ((rl_reward - metrics['mean_reward']) / 
                                 abs(metrics['mean_reward']) * 100)
                    print(f"  vs {policy_name}: {improvement:+.1f}% reward improvement")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Test utilities
    print("Testing RL utilities...")
    
    # Create dummy metrics
    n_episodes = 500
    metrics = {
        'episode_rewards': np.cumsum(np.random.randn(n_episodes)) + np.arange(n_episodes) * 0.1,
        'success_rates': np.clip(np.random.rand(n_episodes) * 0.5 + 0.5, 0, 1),
        'epsilon_history': [max(0.01, 1.0 * 0.995**i) for i in range(n_episodes)],
        'avg_td_errors': np.abs(np.random.randn(n_episodes)) * np.exp(-np.arange(n_episodes)/100),
        'training_time': 45.2
    }
    
    # Plot learning curve
    plot_learning_curve(metrics, save_path='test_learning_curve.png', show=False)
    print("✅ Learning curve plotted")
    
    # Clean up
    import os
    if os.path.exists('test_learning_curve.png'):
        os.remove('test_learning_curve.png')
    
    print("\n✅ Utilities test complete!")
