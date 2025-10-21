"""
Queueing Theory Module for Parking Prediction

This package implements queueing theory (Erlang-C) to calculate booking probabilities
and wait times for parking spots based on predicted occupancy.

Based on: Main.ipynb Cell #15 (complete queueing logic implementation)

Modules:
    - erlang_c: Core Erlang-C (M/M/c) queueing formula implementation
    - queue_estimator: Estimate λ and μ from historical parking data
    - booking_probability: Calculate booking success probabilities

Usage Example:
    >>> from src.queueing import get_booking_confirmation, get_queueing_inputs
    >>> 
    >>> # Step 1: Estimate queueing parameters from historical data
    >>> import pandas as pd
    >>> df = pd.read_csv('data/raw/dataset.csv')
    >>> df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])
    >>> 
    >>> hourly_rates, service_rate = get_queueing_inputs(
    ...     df, capacity=600, timestamp_col='LastUpdated', occupancy_col='Occupancy'
    ... )
    >>> 
    >>> # Step 2: Get booking probability for predicted occupancy
    >>> result = get_booking_confirmation(
    ...     predicted_occupancy=540,
    ...     capacity=600,
    ...     hour_of_day=15,
    ...     hourly_arrival_rates=hourly_rates,
    ...     service_rate_mu=service_rate
    ... )
    >>> 
    >>> print(f"Booking success probability: {result['prob_get_spot']:.1%}")
    >>> print(f"Expected wait: {result['expected_wait_minutes']:.1f} minutes")
    >>> print(f"Recommendation: {result['recommendation']}")

Theory:
    M/M/c Queue Model:
        - M: Markovian (Poisson) arrivals with rate λ
        - M: Markovian (exponential) service with rate μ
        - c: Number of parallel servers (available parking spots)
    
    Key Formulas:
        - Offered load: a = λ / μ
        - Utilization: ρ = a / c
        - Erlang-C: Pw = probability of waiting
        - Expected queue length: Lq = Pw * ρ / (1 - ρ)
        - Expected wait time: Wq = Lq / λ
    
    Little's Law:
        L = λ * W
        (avg number in system = arrival rate × avg time in system)

References:
    - Dahiya et al. queueing theory papers
    - Standard M/M/c queueing theory
    - Main.ipynb parking_queue_tools.py section
"""

from .erlang_c import (
    calculate_erlang_c,
    calculate_probability_immediate_service,
    get_expected_wait_time_minutes
)

from .queue_estimator import (
    get_queueing_inputs,
    estimate_arrival_rate_for_hour,
    validate_queueing_parameters
)

from .booking_probability import (
    get_booking_confirmation,
    calculate_booking_success_probability,
    batch_booking_analysis
)

__all__ = [
    # Erlang-C functions
    'calculate_erlang_c',
    'calculate_probability_immediate_service',
    'get_expected_wait_time_minutes',
    
    # Queue parameter estimation
    'get_queueing_inputs',
    'estimate_arrival_rate_for_hour',
    'validate_queueing_parameters',
    
    # Booking probability
    'get_booking_confirmation',
    'calculate_booking_success_probability',
    'batch_booking_analysis',
]
"""
Queueing theory implementation for booking probability estimation.
Includes Erlang-C formula and parameter estimation.
"""
