"""
Q-Learning Agent for Parking Allocation

Implements tabular Q-learning with:
- Q-table for state-action values
- ε-greedy exploration strategy
- Learning rate decay
- Experience replay (optional)
"""

import numpy as np
from typing import Tuple, Optional
import pickle


class QLearningAgent:
    """
    Q-Learning agent for parking booking decisions.
    
    Uses tabular Q-learning to learn optimal policy for accepting/deferring
    parking bookings based on current occupancy and time of day.
    
    Algorithm:
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    
    Where:
        - s: current state (occupancy_level, hour)
        - a: action (accept/defer)
        - r: reward
        - s': next state
        - α: learning rate
        - γ: discount factor
    
    Attributes:
        n_occupancy_levels: Number of occupancy discretization levels
        n_hours: Number of hours (24)
        n_actions: Number of actions (2: accept, defer)
        q_table: Q-value table of shape (n_occupancy, n_hours, n_actions)
        epsilon: Exploration rate
        learning_rate: Step size for Q-value updates
        discount_factor: Future reward discount
    """
    
    def __init__(
        self,
        n_occupancy_levels: int = 11,
        n_hours: int = 24,
        n_actions: int = 2,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            n_occupancy_levels: Number of occupancy levels (default: 11 for 0-100%)
            n_hours: Number of hours in a day (default: 24)
            n_actions: Number of actions (default: 2)
            learning_rate: Learning rate α (default: 0.1)
            discount_factor: Discount factor γ (default: 0.95)
            epsilon: Initial exploration rate (default: 1.0)
            epsilon_min: Minimum exploration rate (default: 0.01)
            epsilon_decay: Epsilon decay rate per episode (default: 0.995)
        """
        self.n_occupancy_levels = n_occupancy_levels
        self.n_hours = n_hours
        self.n_actions = n_actions
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table with zeros
        # Shape: (occupancy_levels, hours, actions)
        self.q_table = np.zeros((n_occupancy_levels, n_hours, n_actions))
        
        # Statistics
        self.total_updates = 0
        self.episode_count = 0
        
    def get_action(self, state: Tuple[int, int], training: bool = True) -> int:
        """
        Select action using ε-greedy policy.
        
        During training:
            - With probability ε: random action (exploration)
            - With probability 1-ε: best action (exploitation)
        
        During evaluation:
            - Always select best action
        
        Args:
            state: Tuple of (occupancy_level, hour)
            training: If True, use ε-greedy; if False, always exploit
            
        Returns:
            Action (0 = accept, 1 = defer)
        """
        occupancy_level, hour = state
        
        # Ensure state is within bounds
        occupancy_level = min(occupancy_level, self.n_occupancy_levels - 1)
        hour = hour % self.n_hours
        
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.n_actions)
        else:
            # Exploit: best action according to Q-table
            return int(np.argmax(self.q_table[occupancy_level, hour, :]))
    
    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        done: bool
    ) -> float:
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state (occupancy_level, hour)
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            TD error (for monitoring learning progress)
        """
        occupancy_level, hour = state
        next_occupancy_level, next_hour = next_state
        
        # Ensure states are within bounds
        occupancy_level = min(occupancy_level, self.n_occupancy_levels - 1)
        hour = hour % self.n_hours
        next_occupancy_level = min(next_occupancy_level, self.n_occupancy_levels - 1)
        next_hour = next_hour % self.n_hours
        
        # Current Q-value
        current_q = self.q_table[occupancy_level, hour, action]
        
        # Maximum Q-value for next state (0 if terminal)
        if done:
            max_next_q = 0.0
        else:
            max_next_q = np.max(self.q_table[next_occupancy_level, next_hour, :])
        
        # TD target
        target = reward + self.discount_factor * max_next_q
        
        # TD error
        td_error = target - current_q
        
        # Q-learning update
        self.q_table[occupancy_level, hour, action] += self.learning_rate * td_error
        
        self.total_updates += 1
        
        return td_error
    
    def decay_epsilon(self) -> None:
        """
        Decay exploration rate after each episode.
        
        Epsilon decays exponentially but never goes below epsilon_min.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1
    
    def get_q_values(self, state: Tuple[int, int]) -> np.ndarray:
        """
        Get Q-values for all actions in given state.
        
        Args:
            state: Tuple of (occupancy_level, hour)
            
        Returns:
            Array of Q-values for each action
        """
        occupancy_level, hour = state
        occupancy_level = min(occupancy_level, self.n_occupancy_levels - 1)
        hour = hour % self.n_hours
        
        return self.q_table[occupancy_level, hour, :].copy()
    
    def get_value(self, state: Tuple[int, int]) -> float:
        """
        Get state value (max Q-value over actions).
        
        Args:
            state: Tuple of (occupancy_level, hour)
            
        Returns:
            State value V(s) = max_a Q(s,a)
        """
        return np.max(self.get_q_values(state))
    
    def get_policy(self, state: Tuple[int, int]) -> int:
        """
        Get greedy policy action for state.
        
        Args:
            state: Tuple of (occupancy_level, hour)
            
        Returns:
            Best action according to current Q-table
        """
        return int(np.argmax(self.get_q_values(state)))
    
    def save(self, filepath: str) -> None:
        """
        Save Q-table and agent parameters to file.
        
        Args:
            filepath: Path to save file (.pkl or .npy)
        """
        if filepath.endswith('.npy'):
            # Save only Q-table as numpy array
            np.save(filepath, self.q_table)
        else:
            # Save complete agent state as pickle
            agent_state = {
                'q_table': self.q_table,
                'n_occupancy_levels': self.n_occupancy_levels,
                'n_hours': self.n_hours,
                'n_actions': self.n_actions,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'total_updates': self.total_updates,
                'episode_count': self.episode_count
            }
            with open(filepath, 'wb') as f:
                pickle.dump(agent_state, f)
    
    def load(self, filepath: str) -> None:
        """
        Load Q-table and agent parameters from file.
        
        Args:
            filepath: Path to saved file (.pkl or .npy)
        """
        if filepath.endswith('.npy'):
            # Load only Q-table
            self.q_table = np.load(filepath)
        else:
            # Load complete agent state
            with open(filepath, 'rb') as f:
                agent_state = pickle.load(f)
            
            self.q_table = agent_state['q_table']
            self.n_occupancy_levels = agent_state['n_occupancy_levels']
            self.n_hours = agent_state['n_hours']
            self.n_actions = agent_state['n_actions']
            self.learning_rate = agent_state['learning_rate']
            self.discount_factor = agent_state['discount_factor']
            self.epsilon = agent_state['epsilon']
            self.epsilon_min = agent_state['epsilon_min']
            self.epsilon_decay = agent_state['epsilon_decay']
            self.total_updates = agent_state['total_updates']
            self.episode_count = agent_state['episode_count']
    
    def get_policy_heatmap(self) -> np.ndarray:
        """
        Get policy as 2D heatmap (occupancy x hour).
        
        Returns:
            2D array where each cell contains the best action for that state
        """
        policy = np.zeros((self.n_occupancy_levels, self.n_hours), dtype=int)
        
        for occ in range(self.n_occupancy_levels):
            for hour in range(self.n_hours):
                policy[occ, hour] = self.get_policy((occ, hour))
        
        return policy
    
    def get_value_heatmap(self) -> np.ndarray:
        """
        Get state values as 2D heatmap (occupancy x hour).
        
        Returns:
            2D array where each cell contains V(s) for that state
        """
        values = np.zeros((self.n_occupancy_levels, self.n_hours))
        
        for occ in range(self.n_occupancy_levels):
            for hour in range(self.n_hours):
                values[occ, hour] = self.get_value((occ, hour))
        
        return values
    
    def reset_epsilon(self, epsilon: Optional[float] = None) -> None:
        """
        Reset epsilon to initial value or specified value.
        
        Args:
            epsilon: New epsilon value (if None, reset to 1.0)
        """
        self.epsilon = epsilon if epsilon is not None else 1.0
    
    def __repr__(self) -> str:
        """String representation of agent."""
        return (
            f"QLearningAgent(\n"
            f"  State space: {self.n_occupancy_levels} × {self.n_hours} = "
            f"{self.n_occupancy_levels * self.n_hours} states\n"
            f"  Action space: {self.n_actions} actions\n"
            f"  Learning rate: {self.learning_rate}\n"
            f"  Discount factor: {self.discount_factor}\n"
            f"  Epsilon: {self.epsilon:.4f}\n"
            f"  Episodes trained: {self.episode_count}\n"
            f"  Total updates: {self.total_updates}\n"
            f")"
        )


if __name__ == "__main__":
    # Test the agent
    print("Testing QLearningAgent...")
    
    agent = QLearningAgent()
    print(agent)
    
    # Test action selection
    state = (5, 12)  # 50% occupancy at noon
    print(f"\nState: {state}")
    print(f"Q-values: {agent.get_q_values(state)}")
    print(f"Best action: {agent.get_policy(state)}")
    
    # Test update
    next_state = (6, 13)
    reward = 10.0
    td_error = agent.update(state, action=0, reward=reward, next_state=next_state, done=False)
    print(f"\nAfter update:")
    print(f"Q-values: {agent.get_q_values(state)}")
    print(f"TD error: {td_error:.4f}")
    
    # Test save/load
    agent.save('test_agent.pkl')
    agent2 = QLearningAgent()
    agent2.load('test_agent.pkl')
    print(f"\nLoaded agent Q-values: {agent2.get_q_values(state)}")
    
    import os
    os.remove('test_agent.pkl')
    
    print("\n✅ Agent test complete!")
