"""
Training and Evaluation Utilities for RL Agents

Functions for:
- Training Q-learning agents over multiple episodes
- Evaluating agent performance
- Comparing RL policy with baseline policies
- Tracking and logging metrics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time

from .parking_env import ParkingEnv
from .q_learning_agent import QLearningAgent


def train_agent(
    env: ParkingEnv,
    agent: QLearningAgent,
    n_episodes: int = 1000,
    max_steps_per_episode: int = 24,
    verbose: bool = True,
    log_interval: int = 100
) -> Dict[str, List]:
    """
    Train Q-learning agent on parking environment.
    
    Args:
        env: ParkingEnv instance
        agent: QLearningAgent instance
        n_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        verbose: Print progress
        log_interval: Episodes between progress logs
        
    Returns:
        Dictionary with training metrics:
            - episode_rewards: List of total rewards per episode
            - episode_lengths: List of episode lengths
            - epsilon_history: Epsilon values over time
            - success_rates: Successful parking rate per episode
    """
    # Metrics storage
    episode_rewards = []
    episode_lengths = []
    epsilon_history = []
    success_rates = []
    avg_td_errors = []
    
    if verbose:
        print("=" * 60)
        print("TRAINING Q-LEARNING AGENT")
        print("=" * 60)
        print(f"Episodes: {n_episodes}")
        print(f"Max steps per episode: {max_steps_per_episode}")
        print(f"Initial epsilon: {agent.epsilon:.4f}")
        print(f"Learning rate: {agent.learning_rate}")
        print(f"Discount factor: {agent.discount_factor}")
        print("=" * 60)
    
    start_time = time.time()
    
    # Training loop
    iterator = tqdm(range(n_episodes), desc="Training") if verbose else range(n_episodes)
    
    for episode in iterator:
        # Reset environment
        state = env.reset(start_hour=np.random.randint(0, 24))
        
        episode_reward = 0
        episode_td_errors = []
        steps = 0
        
        # Episode loop
        for step in range(max_steps_per_episode):
            # Select action
            action = agent.get_action(state, training=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Update Q-table
            td_error = agent.update(state, action, reward, next_state, done)
            episode_td_errors.append(abs(td_error))
            
            # Accumulate reward
            episode_reward += reward
            steps += 1
            
            # Move to next state
            state = next_state
            
            if done:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        epsilon_history.append(agent.epsilon)
        
        # Calculate success rate
        if info['bookings_accepted'] > 0:
            success_rate = info['successful_parks'] / info['bookings_accepted']
        else:
            success_rate = 0.0
        success_rates.append(success_rate)
        
        avg_td_errors.append(np.mean(episode_td_errors) if episode_td_errors else 0.0)
        
        # Periodic logging
        if verbose and (episode + 1) % log_interval == 0:
            recent_rewards = episode_rewards[-log_interval:]
            recent_success = success_rates[-log_interval:]
            
            tqdm.write(
                f"\nEpisode {episode + 1}/{n_episodes} | "
                f"Avg Reward: {np.mean(recent_rewards):.2f} | "
                f"Success Rate: {np.mean(recent_success):.2%} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Episodes per second: {n_episodes/elapsed_time:.2f}")
        print(f"Final epsilon: {agent.epsilon:.4f}")
        print(f"Total Q-table updates: {agent.total_updates}")
        print(f"\nFinal 100 episodes:")
        print(f"  Avg reward: {np.mean(episode_rewards[-100:]):.2f}")
        print(f"  Avg success rate: {np.mean(success_rates[-100:]):.2%}")
        print("=" * 60)
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'epsilon_history': epsilon_history,
        'success_rates': success_rates,
        'avg_td_errors': avg_td_errors,
        'training_time': elapsed_time
    }


def evaluate_agent(
    env: ParkingEnv,
    agent: QLearningAgent,
    n_episodes: int = 100,
    max_steps_per_episode: int = 24,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate trained agent performance.
    
    Args:
        env: ParkingEnv instance
        agent: Trained QLearningAgent
        n_episodes: Number of evaluation episodes
        max_steps_per_episode: Maximum steps per episode
        verbose: Print results
        
    Returns:
        Dictionary with evaluation metrics
    """
    total_rewards = []
    success_rates = []
    utilizations = []
    accepted_counts = []
    deferred_counts = []
    
    for episode in range(n_episodes):
        state = env.reset(start_hour=np.random.randint(0, 24))
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            # Use greedy policy (no exploration)
            action = agent.get_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        
        if info['bookings_accepted'] > 0:
            success_rate = info['successful_parks'] / info['bookings_accepted']
        else:
            success_rate = 1.0  # No bookings = no failures
        
        success_rates.append(success_rate)
        utilizations.append(info['utilization'])
        accepted_counts.append(info['bookings_accepted'])
        deferred_counts.append(info['bookings_deferred'])
    
    metrics = {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_success_rate': np.mean(success_rates),
        'mean_utilization': np.mean(utilizations),
        'mean_accepted': np.mean(accepted_counts),
        'mean_deferred': np.mean(deferred_counts),
        'total_decisions': np.mean(accepted_counts) + np.mean(deferred_counts)
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Episodes: {n_episodes}")
        print(f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Success rate: {metrics['mean_success_rate']:.2%}")
        print(f"Avg utilization: {metrics['mean_utilization']:.2%}")
        print(f"Avg bookings accepted: {metrics['mean_accepted']:.1f}")
        print(f"Avg bookings deferred: {metrics['mean_deferred']:.1f}")
        print("=" * 60)
    
    return metrics


def compare_policies(
    env: ParkingEnv,
    agent: QLearningAgent,
    n_episodes: int = 100,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compare RL agent with baseline policies.
    
    Baseline policies:
        1. Always Accept: Always accept bookings
        2. Always Defer: Always defer bookings
        3. Threshold: Accept if utilization < 80%
        4. Erlang-C: Accept if prob(immediate) > 70%
    
    Args:
        env: ParkingEnv instance
        agent: Trained QLearningAgent
        n_episodes: Number of episodes per policy
        verbose: Print comparison
        
    Returns:
        Dictionary mapping policy name to metrics
    """
    policies = {}
    
    # 1. RL Agent
    print("\nEvaluating RL Agent...")
    policies['RL Agent'] = evaluate_agent(env, agent, n_episodes, verbose=False)
    
    # 2. Always Accept
    print("Evaluating Always Accept...")
    policies['Always Accept'] = _evaluate_baseline(env, 'always_accept', n_episodes)
    
    # 3. Always Defer
    print("Evaluating Always Defer...")
    policies['Always Defer'] = _evaluate_baseline(env, 'always_defer', n_episodes)
    
    # 4. Threshold (80%)
    print("Evaluating Threshold Policy...")
    policies['Threshold (80%)'] = _evaluate_baseline(env, 'threshold', n_episodes)
    
    # 5. Erlang-C based
    print("Evaluating Erlang-C Policy...")
    policies['Erlang-C'] = _evaluate_baseline(env, 'erlang', n_episodes)
    
    if verbose:
        print("\n" + "=" * 80)
        print("POLICY COMPARISON")
        print("=" * 80)
        print(f"{'Policy':<20} {'Reward':<15} {'Success Rate':<15} {'Utilization':<15}")
        print("-" * 80)
        
        for policy_name, metrics in policies.items():
            print(
                f"{policy_name:<20} "
                f"{metrics['mean_reward']:>7.2f} ± {metrics['std_reward']:<5.2f} "
                f"{metrics['mean_success_rate']:>13.2%} "
                f"{metrics['mean_utilization']:>14.2%}"
            )
        
        print("=" * 80)
        
        # Find best policy
        best_policy = max(policies.items(), key=lambda x: x[1]['mean_reward'])
        print(f"\n🏆 Best Policy: {best_policy[0]} (Reward: {best_policy[1]['mean_reward']:.2f})")
        print("=" * 80)
    
    return policies


def _evaluate_baseline(
    env: ParkingEnv,
    policy_type: str,
    n_episodes: int
) -> Dict[str, float]:
    """
    Evaluate baseline policy.
    
    Args:
        env: ParkingEnv instance
        policy_type: Type of baseline ('always_accept', 'always_defer', 'threshold', 'erlang')
        n_episodes: Number of episodes
        
    Returns:
        Metrics dictionary
    """
    total_rewards = []
    success_rates = []
    utilizations = []
    accepted_counts = []
    deferred_counts = []
    
    for episode in range(n_episodes):
        state = env.reset(start_hour=np.random.randint(0, 24))
        episode_reward = 0
        
        for step in range(24):
            # Select action based on policy type
            if policy_type == 'always_accept':
                action = 0
            elif policy_type == 'always_defer':
                action = 1
            elif policy_type == 'threshold':
                # Accept if utilization < 80%
                utilization = env.current_occupancy / env.capacity
                action = 0 if utilization < 0.8 else 1
            elif policy_type == 'erlang':
                # Accept if prob(immediate) > 70%
                metrics = env.get_queueing_metrics()
                action = 0 if metrics['prob_immediate'] > 0.7 else 1
            else:
                action = 0
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        
        if info['bookings_accepted'] > 0:
            success_rate = info['successful_parks'] / info['bookings_accepted']
        else:
            success_rate = 1.0
        
        success_rates.append(success_rate)
        utilizations.append(info['utilization'])
        accepted_counts.append(info['bookings_accepted'])
        deferred_counts.append(info['bookings_deferred'])
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_success_rate': np.mean(success_rates),
        'mean_utilization': np.mean(utilizations),
        'mean_accepted': np.mean(accepted_counts),
        'mean_deferred': np.mean(deferred_counts)
    }


if __name__ == "__main__":
    # Test training
    print("Testing trainer...")
    
    env = ParkingEnv(capacity=600)
    agent = QLearningAgent()
    
    # Short training run
    metrics = train_agent(env, agent, n_episodes=100, verbose=True, log_interval=50)
    
    # Evaluate
    eval_metrics = evaluate_agent(env, agent, n_episodes=20, verbose=True)
    
    # Compare policies
    comparison = compare_policies(env, agent, n_episodes=20, verbose=True)
    
    print("\n✅ Trainer test complete!")
