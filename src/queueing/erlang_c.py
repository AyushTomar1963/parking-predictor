"""
Erlang-C (M/M/c Queue) Implementation

This module implements the Erlang-C formula for calculating waiting probabilities
in a multi-server queueing system (M/M/c queue).

Based on: Main.ipynb Cell #15 (parking_queue_tools.py section)
"""

import math
from typing import Tuple, Dict


def _safe_pow_div(a: float, n: int) -> float:
    """
    Compute a**n / n! in a numerically stable way using log-gamma.
    
    This prevents overflow for large values of a or n by using logarithms:
    a**n / n! = exp(n*log(a) - log(n!))
    
    Args:
        a: Base value (traffic intensity parameter)
        n: Exponent and factorial denominator
        
    Returns:
        float: The computed value a**n / n!
    """
    if a == 0.0 and n == 0:
        return 1.0
    if a == 0.0:
        return 0.0
    
    # Use logarithms to avoid overflow: exp(n*log(a) - log(n!))
    # math.lgamma(n+1) = log(n!)
    val = math.exp(n * math.log(a) - math.lgamma(n + 1))
    return val


def calculate_erlang_c(
    arrival_rate_lambda: float,
    service_rate_mu: float,
    num_servers_c: int
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate Erlang-C results for an M/M/c queueing system.
    
    The M/M/c queue model represents:
    - M: Markovian (Poisson) arrivals with rate λ (lambda)
    - M: Markovian (exponential) service times with rate μ (mu)
    - c: Number of identical servers (parking spots)
    
    Returns the probability that an arriving customer must wait (Pw),
    along with other queueing metrics.
    
    Formulas:
        a = λ / μ           (offered load)
        ρ = a / c           (utilization per server)
        P0 = probability of empty system
        Pw = Erlang-C formula (probability of waiting)
        Lq = expected number in queue
        Wq = expected waiting time in queue
        W = expected total time in system
    
    Args:
        arrival_rate_lambda: Arrival rate (λ) in customers per hour
        service_rate_mu: Service rate per server (μ) in customers per hour
        num_servers_c: Number of servers (c) - available parking spots
        
    Returns:
        Tuple containing:
            - prob_wait (float): Probability that arriving customer waits (Pw)
            - details (dict): Dictionary with keys:
                - 'P0': Probability system is empty
                - 'a': Offered load (λ/μ)
                - 'rho': Utilization per server (a/c)
                - 'Lq': Expected number in queue
                - 'Wq': Expected waiting time in queue (hours)
                - 'W': Expected total time in system (hours)
    
    Example:
        >>> prob_wait, details = calculate_erlang_c(10.0, 0.5, 25)
        >>> print(f"Probability of waiting: {prob_wait:.2%}")
        >>> print(f"Expected wait time: {details['Wq']*60:.1f} minutes")
    
    References:
        - Standard M/M/c / Erlang-C formulas
        - Dahiya et al. queueing theory papers
    """
    # Defensive checks: ensure valid inputs
    try:
        c = int(max(1, int(num_servers_c)))
    except Exception:
        c = 1
    
    # Handle edge case: no arrivals
    if arrival_rate_lambda <= 0:
        return 0.0, {
            'P0': 1.0,
            'a': 0.0,
            'rho': 0.0,
            'Lq': 0.0,
            'Wq': 0.0,
            'W': 1.0 / service_rate_mu if service_rate_mu > 0 else float('inf')
        }
    
    # Handle edge case: no service capacity
    if service_rate_mu <= 0:
        return 1.0, {
            'P0': 0.0,
            'a': float('inf'),
            'rho': 1.0,
            'Lq': float('inf'),
            'Wq': float('inf'),
            'W': float('inf')
        }
    
    # Calculate traffic intensity parameter (offered load)
    a = arrival_rate_lambda / service_rate_mu
    
    # Calculate utilization per server
    rho = a / c
    
    # Check for system instability (arrival rate exceeds service capacity)
    # If ρ >= 1, the queue grows unbounded
    if rho >= 1.0 - 1e-12:
        return 1.0, {
            'P0': 0.0,
            'a': a,
            'rho': rho,
            'Lq': float('inf'),
            'Wq': float('inf'),
            'W': float('inf')
        }
    
    # Compute P0 (probability of empty system) using numerically stable method
    # P0 = 1 / [sum_{n=0}^{c-1} (a^n / n!) + (a^c / c!) * (1 / (1 - ρ))]
    
    sum_terms = 0.0
    for n in range(0, c):
        sum_terms += _safe_pow_div(a, n)
    
    # Last term: a^c / (c! * (1 - ρ))
    last_term = _safe_pow_div(a, c) / (1.0 - rho)
    
    denom = sum_terms + last_term
    if denom == 0:
        P0 = 0.0
    else:
        P0 = 1.0 / denom
    
    # Erlang-C formula: Probability that arriving customer waits
    # Pw = (a^c / (c! * (1 - ρ))) * P0
    Pw = last_term * P0
    
    # Expected number of customers in queue (Little's Law)
    # Lq = Pw * ρ / (1 - ρ)
    Lq = Pw * rho / (1.0 - rho)
    
    # Expected waiting time in queue (hours)
    # Wq = Lq / λ
    Wq = Lq / arrival_rate_lambda if arrival_rate_lambda > 0 else 0.0
    
    # Expected total time in system (hours)
    # W = Wq + 1/μ (waiting time + service time)
    W = Wq + 1.0 / service_rate_mu
    
    # Package all details
    details = {
        'P0': P0,           # Probability system is empty
        'a': a,             # Offered load
        'rho': rho,         # Utilization per server
        'Lq': Lq,           # Expected queue length
        'Wq': Wq,           # Expected waiting time (hours)
        'W': W              # Expected total time in system (hours)
    }
    
    # Clamp probability to [0, 1] range for safety
    prob_wait = float(min(1.0, max(0.0, Pw)))
    
    return prob_wait, details


def calculate_probability_immediate_service(
    arrival_rate_lambda: float,
    service_rate_mu: float,
    num_servers_c: int
) -> float:
    """
    Calculate the probability of getting immediate service (no waiting).
    
    This is simply 1 - Pw (Erlang-C probability).
    
    Args:
        arrival_rate_lambda: Arrival rate (λ) in customers per hour
        service_rate_mu: Service rate per server (μ) in customers per hour
        num_servers_c: Number of servers (c) - available parking spots
        
    Returns:
        float: Probability of immediate service (getting a spot without waiting)
    
    Example:
        >>> prob_immediate = calculate_probability_immediate_service(10.0, 0.5, 25)
        >>> print(f"Probability of immediate parking: {prob_immediate:.2%}")
    """
    prob_wait, _ = calculate_erlang_c(arrival_rate_lambda, service_rate_mu, num_servers_c)
    return 1.0 - prob_wait


def get_expected_wait_time_minutes(
    arrival_rate_lambda: float,
    service_rate_mu: float,
    num_servers_c: int
) -> float:
    """
    Get the expected waiting time in minutes for arriving customers who must wait.
    
    Args:
        arrival_rate_lambda: Arrival rate (λ) in customers per hour
        service_rate_mu: Service rate per server (μ) in customers per hour
        num_servers_c: Number of servers (c) - available parking spots
        
    Returns:
        float: Expected waiting time in minutes (inf if queue is unstable)
    
    Example:
        >>> wait_minutes = get_expected_wait_time_minutes(10.0, 0.5, 25)
        >>> print(f"Expected wait time: {wait_minutes:.1f} minutes")
    """
    _, details = calculate_erlang_c(arrival_rate_lambda, service_rate_mu, num_servers_c)
    Wq_hours = details['Wq']
    
    if math.isinf(Wq_hours):
        return float('inf')
    
    return Wq_hours * 60.0  # Convert hours to minutes


if __name__ == "__main__":
    # Test example from notebook
    print("=" * 60)
    print("Erlang-C Calculator - Test Examples")
    print("=" * 60)
    
    # Example 1: Medium load
    print("\nExample 1: Medium Load")
    print("-" * 40)
    arrival_rate = 10.0  # 10 cars per hour
    service_rate = 0.5   # Average parking duration: 2 hours (1/0.5)
    available_spots = 25
    
    prob_wait, details = calculate_erlang_c(arrival_rate, service_rate, available_spots)
    
    print(f"Arrival rate (λ): {arrival_rate} cars/hour")
    print(f"Service rate (μ): {service_rate} per hour (avg duration: {1/service_rate:.1f} hours)")
    print(f"Available spots (c): {available_spots}")
    print(f"Utilization (ρ): {details['rho']:.2%}")
    print(f"Probability of waiting: {prob_wait:.2%}")
    print(f"Probability of immediate spot: {(1-prob_wait):.2%}")
    print(f"Expected wait time: {details['Wq']*60:.1f} minutes")
    
    # Example 2: High load (near capacity)
    print("\nExample 2: High Load (Near Capacity)")
    print("-" * 40)
    arrival_rate = 15.0
    available_spots = 18  # Close to capacity (15/0.5 = 30 needed)
    
    prob_wait, details = calculate_erlang_c(arrival_rate, service_rate, available_spots)
    
    print(f"Arrival rate (λ): {arrival_rate} cars/hour")
    print(f"Service rate (μ): {service_rate} per hour")
    print(f"Available spots (c): {available_spots}")
    print(f"Utilization (ρ): {details['rho']:.2%}")
    print(f"Probability of waiting: {prob_wait:.2%}")
    print(f"Probability of immediate spot: {(1-prob_wait):.2%}")
    if math.isfinite(details['Wq']):
        print(f"Expected wait time: {details['Wq']*60:.1f} minutes")
    else:
        print("Expected wait time: INFINITE (system overloaded)")
    
    print("\n" + "=" * 60)
