"""
ParkingEnv: Simulated Parking Environment for RL

This environment simulates parking lot dynamics using:
- Predicted occupancy from ML models
- Queueing parameters (λ, μ, c) from historical data
- Stochastic arrivals and departures

State: (occupancy_level, hour_of_day)
Actions: 0 = Accept booking, 1 = Defer booking
Rewards: Based on successful parking, wait times, and capacity utilization
"""

import numpy as np
from typing import Tuple, Dict, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.queueing import calculate_erlang_c


class ParkingEnv:
    """
    Parking lot environment for reinforcement learning.
    
    The environment simulates a parking lot with:
    - Dynamic occupancy based on arrivals and departures
    - Time-varying arrival rates (hourly patterns)
    - Stochastic transitions
    - Reward structure encouraging optimal booking decisions
    
    State Space:
        - occupancy_level: Discretized occupancy (0-10 representing 0-100%)
        - hour_of_day: Hour (0-23)
    
    Action Space:
        - 0: Accept booking (allow customer to park)
        - 1: Defer booking (suggest alternative time/lot)
    
    Reward Structure:
        - Accept when space available: +10
        - Accept when crowded (>80%): -5 (causes congestion)
        - Accept when full: -20 (customer can't park)
        - Defer when space available: -2 (lost revenue)
        - Defer when crowded: +5 (good decision)
        - Defer when full: +3 (avoided bad experience)
    """
    
    def __init__(
        self,
        capacity: int = 600,
        hourly_arrival_rates: Optional[Dict[int, float]] = None,
        service_rate_mu: float = 0.5,
        time_step_minutes: int = 60,
        max_steps: int = 24
    ):
        """
        Initialize parking environment.
        
        Args:
            capacity: Total parking lot capacity
            hourly_arrival_rates: Dict mapping hour to arrival rate (cars/hour)
            service_rate_mu: Service rate per spot (per hour)
            time_step_minutes: Minutes per time step (default: 60)
            max_steps: Maximum steps per episode (default: 24 hours)
        """
        self.capacity = capacity
        self.service_rate_mu = service_rate_mu
        self.time_step_minutes = time_step_minutes
        self.max_steps = max_steps
        
        # Default hourly arrival rates (if not provided)
        if hourly_arrival_rates is None:
            # Realistic pattern: low at night, high during day
            self.hourly_arrival_rates = {
                0: 2.0, 1: 1.5, 2: 1.0, 3: 0.8, 4: 1.0, 5: 2.5,
                6: 5.0, 7: 10.0, 8: 15.0, 9: 18.0, 10: 20.0, 11: 22.0,
                12: 20.0, 13: 18.0, 14: 16.0, 15: 14.0, 16: 12.0, 17: 15.0,
                18: 10.0, 19: 8.0, 20: 6.0, 21: 5.0, 22: 4.0, 23: 3.0
            }
        else:
            self.hourly_arrival_rates = hourly_arrival_rates
        
        # State space
        self.n_occupancy_levels = 11  # 0-10 (0%, 10%, ..., 100%)
        self.n_hours = 24
        self.state_space_size = self.n_occupancy_levels * self.n_hours
        
        # Action space
        self.n_actions = 2  # 0 = accept, 1 = defer
        
        # Current state
        self.current_occupancy = 0
        self.current_hour = 0
        self.step_count = 0
        
        # Episode statistics
        self.total_bookings_accepted = 0
        self.total_bookings_deferred = 0
        self.total_successful_parks = 0
        self.total_failed_parks = 0
        
    def reset(self, start_hour: int = 0, start_occupancy: Optional[int] = None) -> Tuple[int, int]:
        """
        Reset environment to initial state.
        
        Args:
            start_hour: Starting hour (0-23)
            start_occupancy: Starting occupancy (if None, random 20-50%)
            
        Returns:
            Tuple of (occupancy_level, hour_of_day)
        """
        self.current_hour = start_hour
        
        if start_occupancy is None:
            # Random starting occupancy between 20-50% of capacity
            self.current_occupancy = int(np.random.uniform(0.2, 0.5) * self.capacity)
        else:
            self.current_occupancy = min(start_occupancy, self.capacity)
        
        self.step_count = 0
        self.total_bookings_accepted = 0
        self.total_bookings_deferred = 0
        self.total_successful_parks = 0
        self.total_failed_parks = 0
        
        return self._get_state()
    
    def _get_state(self) -> Tuple[int, int]:
        """
        Get current state as (occupancy_level, hour).
        
        Returns:
            Tuple of (occupancy_level, hour_of_day)
        """
        # Discretize occupancy to 0-10 scale
        occupancy_level = min(10, int((self.current_occupancy / self.capacity) * 10))
        return (occupancy_level, self.current_hour)
    
    def _get_arrival_rate(self) -> float:
        """Get arrival rate for current hour."""
        return self.hourly_arrival_rates.get(self.current_hour, 10.0)
    
    def _simulate_arrivals_departures(self) -> None:
        """
        Simulate stochastic arrivals and departures for one time step.
        
        Uses Poisson arrivals and exponential service times.
        """
        # Time step in hours
        dt = self.time_step_minutes / 60.0
        
        # Arrivals (Poisson process)
        arrival_rate = self._get_arrival_rate()
        expected_arrivals = arrival_rate * dt
        actual_arrivals = np.random.poisson(expected_arrivals)
        
        # Departures (exponential service)
        # Each occupied spot has probability μ*dt of departing
        departure_prob = self.service_rate_mu * dt
        actual_departures = 0
        for _ in range(self.current_occupancy):
            if np.random.random() < departure_prob:
                actual_departures += 1
        
        # Update occupancy (bounded by capacity)
        self.current_occupancy = max(0, self.current_occupancy + actual_arrivals - actual_departures)
        self.current_occupancy = min(self.capacity, self.current_occupancy)
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        Execute one time step in the environment.
        
        Args:
            action: 0 = accept booking, 1 = defer booking
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Calculate reward based on action and current state
        reward = self._calculate_reward(action)
        
        # Update statistics
        if action == 0:
            self.total_bookings_accepted += 1
            # Check if booking was successful
            if self.current_occupancy < self.capacity:
                self.total_successful_parks += 1
                # Add one car if accepted
                self.current_occupancy = min(self.capacity, self.current_occupancy + 1)
            else:
                self.total_failed_parks += 1
        else:
            self.total_bookings_deferred += 1
        
        # Simulate natural arrivals and departures
        self._simulate_arrivals_departures()
        
        # Advance time
        self.current_hour = (self.current_hour + 1) % 24
        self.step_count += 1
        
        # Check if episode is done
        done = self.step_count >= self.max_steps
        
        # Get next state
        next_state = self._get_state()
        
        # Info dictionary
        info = {
            'occupancy': self.current_occupancy,
            'utilization': self.current_occupancy / self.capacity,
            'hour': self.current_hour,
            'bookings_accepted': self.total_bookings_accepted,
            'bookings_deferred': self.total_bookings_deferred,
            'successful_parks': self.total_successful_parks,
            'failed_parks': self.total_failed_parks
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, action: int) -> float:
        """
        Calculate reward for taking action in current state.
        
        Reward structure encourages:
        - Accepting bookings when space is available
        - Deferring bookings when lot is crowded/full
        - Maximizing successful parking experiences
        - Avoiding overcapacity situations
        
        Args:
            action: 0 = accept, 1 = defer
            
        Returns:
            Reward value
        """
        utilization = self.current_occupancy / self.capacity
        available_spots = self.capacity - self.current_occupancy
        
        if action == 0:  # Accept booking
            if available_spots > 0.2 * self.capacity:  # Plenty of space (>20%)
                reward = 10.0
            elif available_spots > 0:  # Some space but crowded
                # Penalize based on how crowded
                reward = 10.0 - (utilization - 0.8) * 50
            else:  # Full - very bad
                reward = -20.0
        
        else:  # Defer booking (action == 1)
            if available_spots > 0.2 * self.capacity:  # Plenty of space - lost revenue
                reward = -2.0
            elif available_spots > 0:  # Crowded - good decision
                reward = 5.0
            else:  # Full - good decision to defer
                reward = 3.0
        
        return reward
    
    def get_queueing_metrics(self) -> Dict[str, float]:
        """
        Get current queueing theory metrics using Erlang-C.
        
        Returns:
            Dictionary with queueing metrics
        """
        arrival_rate = self._get_arrival_rate()
        available_spots = max(1, self.capacity - self.current_occupancy)
        
        prob_wait, details = calculate_erlang_c(
            arrival_rate_lambda=arrival_rate,
            service_rate_mu=self.service_rate_mu,
            num_servers_c=available_spots
        )
        
        return {
            'prob_wait': prob_wait,
            'prob_immediate': 1.0 - prob_wait,
            'expected_wait_minutes': details['Wq'] * 60,
            'utilization': details['rho'],
            'queue_length': details['Lq']
        }
    
    def render(self, mode: str = 'human') -> None:
        """
        Render current environment state.
        
        Args:
            mode: Rendering mode ('human' for text output)
        """
        if mode == 'human':
            utilization = self.current_occupancy / self.capacity
            bar_length = 50
            filled = int(utilization * bar_length)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\n{'='*60}")
            print(f"Hour: {self.current_hour:02d}:00 | Step: {self.step_count}/{self.max_steps}")
            print(f"Occupancy: {self.current_occupancy}/{self.capacity} ({utilization:.1%})")
            print(f"[{bar}]")
            print(f"Arrival Rate: {self._get_arrival_rate():.1f} cars/hour")
            print(f"Accepted: {self.total_bookings_accepted} | Deferred: {self.total_bookings_deferred}")
            print(f"Successful: {self.total_successful_parks} | Failed: {self.total_failed_parks}")
            
            # Show queueing metrics
            metrics = self.get_queueing_metrics()
            print(f"\nQueueing Metrics:")
            print(f"  Prob(immediate spot): {metrics['prob_immediate']:.1%}")
            print(f"  Expected wait: {metrics['expected_wait_minutes']:.1f} min")
            print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test the environment
    print("Testing ParkingEnv...")
    
    env = ParkingEnv(capacity=600)
    state = env.reset(start_hour=9)
    
    print(f"Initial state: occupancy_level={state[0]}, hour={state[1]}")
    env.render()
    
    # Run a few steps
    for i in range(5):
        action = np.random.choice([0, 1])  # Random action
        action_name = "ACCEPT" if action == 0 else "DEFER"
        
        next_state, reward, done, info = env.step(action)
        
        print(f"\nStep {i+1}: Action={action_name}, Reward={reward:.1f}")
        print(f"Next state: occupancy_level={next_state[0]}, hour={next_state[1]}")
        env.render()
        
        if done:
            break
    
    print("\n✅ Environment test complete!")
