"""
Queue Parameter Estimator

This module estimates queueing theory parameters (λ, μ) from historical parking data.

Based on: Main.ipynb Cell #15 (get_queueing_inputs function)
"""

import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict


def get_queueing_inputs(
    df: pd.DataFrame,
    capacity: int,
    timestamp_col: str = 'LastUpdated',
    occupancy_col: str = 'Occupancy'
) -> Tuple[Dict[int, float], float]:
    """
    Estimate hourly arrival rates and empirical service rate from observed parking data.
    
    This function analyzes historical parking occupancy data to derive:
    1. Hourly arrival rates (λ) - how many cars arrive per hour at different times
    2. Service rate (μ) - how long cars typically stay (using Little's Law)
    
    Method:
        - Computes occupancy changes over time
        - Positive changes indicate arrivals during that interval
        - Calculates arrival rate per hour for each time period
        - Groups by hour of day to get average hourly patterns
        - Uses Little's Law (L = λ * W) to estimate service rate:
          W = L / λ  (avg time in system)
          μ = 1 / W  (service rate)
    
    Args:
        df: DataFrame containing parking lot data with columns:
            - timestamp_col: DateTime column with observation timestamps
            - occupancy_col: Integer column with number of occupied spots
        capacity: Total parking lot capacity (number of spots)
        timestamp_col: Name of timestamp column (default: 'LastUpdated')
        occupancy_col: Name of occupancy column (default: 'Occupancy')
        
    Returns:
        Tuple containing:
            - hourly_arrival_rates (dict): {hour_of_day (0-23) -> avg arrivals per hour}
            - service_rate_mu (float): Estimated service rate per slot (per hour)
    
    Example:
        >>> df = pd.read_csv('dataset.csv')
        >>> df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])
        >>> hourly_rates, mu = get_queueing_inputs(df, capacity=600)
        >>> print(f"Peak hour arrivals: {max(hourly_rates.values()):.2f} cars/hour")
        >>> print(f"Average parking duration: {1/mu:.2f} hours")
    
    References:
        - Little's Law: L = λW
        - Based on notebook analysis in Main.ipynb
    """
    # Validate inputs
    df_proc = df.copy()
    if timestamp_col not in df_proc.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame")
    if occupancy_col not in df_proc.columns:
        raise ValueError(f"Occupancy column '{occupancy_col}' not found in DataFrame")
    
    # Ensure timestamp is datetime type
    df_proc[timestamp_col] = pd.to_datetime(df_proc[timestamp_col])
    
    # Sort by timestamp to ensure chronological order
    df_proc = df_proc.sort_values(by=timestamp_col).reset_index(drop=True)
    
    # --- Step 1: Compute occupancy changes and time deltas ---
    
    # Calculate change in occupancy between consecutive observations
    df_proc['OccupancyChange'] = df_proc[occupancy_col].diff()
    
    # Calculate time difference between observations (in hours)
    df_proc['DeltaHours'] = df_proc[timestamp_col].diff().dt.total_seconds() / 3600.0
    
    # Avoid division by zero: replace zero or negative deltas with NaN
    df_proc['DeltaHours'] = df_proc['DeltaHours'].replace(0, np.nan)
    
    # --- Step 2: Estimate arrivals from positive occupancy changes ---
    
    # Arrivals are approximated by positive occupancy increases
    # (negative changes represent departures)
    df_proc['Arrivals'] = df_proc['OccupancyChange'].clip(lower=0).fillna(0.0)
    
    # Calculate arrival rate per interval (arrivals per hour)
    df_proc['IntervalArrivalRate'] = df_proc['Arrivals'] / df_proc['DeltaHours']
    
    # Extract hour of day for grouping
    df_proc['hour_of_day'] = df_proc[timestamp_col].dt.hour
    
    # --- Step 3: Group by hour to get average hourly arrival patterns ---
    
    # Average arrival rate for each hour of the day (0-23)
    hourly_arrival_rates = (
        df_proc.groupby('hour_of_day')['IntervalArrivalRate']
        .mean()
        .fillna(0.0)
        .to_dict()
    )
    
    # --- Step 4: Calculate overall arrival rate (λ) ---
    
    # Overall average arrival rate across all time periods
    # Remove infinities and NaNs for robust calculation
    overall_lambda = (
        df_proc['IntervalArrivalRate']
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    overall_lambda = overall_lambda.mean() if len(overall_lambda) > 0 else 0.0
    
    # --- Step 5: Estimate service rate (μ) using Little's Law ---
    
    # Average occupancy (L) - average number of cars in system
    avg_occupancy = df_proc[occupancy_col].mean()
    
    # Little's Law: L = λ * W
    # Therefore: W = L / λ (average time in system)
    # And: μ = 1 / W (service rate per slot)
    
    if overall_lambda > 1e-8:
        avg_service_time_hours = max(1e-6, avg_occupancy / overall_lambda)
    else:
        # Fallback: assume reasonable parking duration (1 hour) if no arrival data
        avg_service_time_hours = 1.0
    
    service_rate_mu = 1.0 / avg_service_time_hours
    
    # --- Print summary for debugging/verification ---
    
    print("=" * 60)
    print("Queueing Parameter Estimation Summary")
    print("=" * 60)
    print(f"Total observations: {len(df_proc)}")
    print(f"Time span: {df_proc[timestamp_col].min()} to {df_proc[timestamp_col].max()}")
    print(f"Parking lot capacity: {capacity} spots")
    print("-" * 60)
    print(f"Estimated overall arrival rate (λ): {overall_lambda:.4f} cars/hour")
    print(f"Average occupancy (L): {avg_occupancy:.2f} cars")
    print(f"Estimated avg parking duration (W): {avg_service_time_hours:.2f} hours")
    print(f"Estimated service rate per slot (μ): {service_rate_mu:.4f} per hour")
    print("-" * 60)
    print("Hourly arrival rate patterns:")
    for hour in sorted(hourly_arrival_rates.keys()):
        rate = hourly_arrival_rates[hour]
        print(f"  Hour {hour:2d}:00 - {rate:6.3f} cars/hour")
    print("=" * 60)
    
    return hourly_arrival_rates, float(service_rate_mu)


def estimate_arrival_rate_for_hour(
    hourly_arrival_rates: Dict[int, float],
    hour_of_day: int
) -> float:
    """
    Get the estimated arrival rate for a specific hour of day.
    
    If the specific hour is not in the dictionary, returns the average
    of all hourly rates as a fallback.
    
    Args:
        hourly_arrival_rates: Dictionary mapping hour (0-23) to arrival rate
        hour_of_day: Hour to query (0-23)
        
    Returns:
        float: Estimated arrival rate for that hour (cars per hour)
    
    Example:
        >>> rates = {9: 15.0, 10: 18.0, 11: 20.0}
        >>> rate = estimate_arrival_rate_for_hour(rates, 10)
        >>> print(f"10 AM arrival rate: {rate:.1f} cars/hour")
    """
    if not hourly_arrival_rates:
        return 0.0
    
    # Return rate for specific hour if available
    if hour_of_day in hourly_arrival_rates:
        return float(hourly_arrival_rates[hour_of_day])
    
    # Fallback: return average of all hours
    return float(np.mean(list(hourly_arrival_rates.values())))


def validate_queueing_parameters(
    arrival_rate_lambda: float,
    service_rate_mu: float,
    num_servers_c: int,
    capacity: int
) -> Dict[str, any]:
    """
    Validate estimated queueing parameters and check for potential issues.
    
    Args:
        arrival_rate_lambda: Estimated arrival rate (cars/hour)
        service_rate_mu: Estimated service rate per slot (per hour)
        num_servers_c: Number of available servers (parking spots)
        capacity: Total lot capacity
        
    Returns:
        dict: Validation results with keys:
            - 'is_valid': bool
            - 'warnings': list of warning messages
            - 'offered_load': float (a = λ/μ)
            - 'utilization': float (ρ = a/c)
            - 'is_stable': bool (ρ < 1)
    
    Example:
        >>> validation = validate_queueing_parameters(15.0, 0.5, 25, 100)
        >>> if not validation['is_stable']:
        >>>     print("WARNING: System is overloaded!")
    """
    warnings = []
    
    # Check for non-positive parameters
    if arrival_rate_lambda <= 0:
        warnings.append("Arrival rate is zero or negative")
    if service_rate_mu <= 0:
        warnings.append("Service rate is zero or negative")
    if num_servers_c <= 0:
        warnings.append("Number of servers is zero or negative")
    
    # Calculate offered load and utilization
    if service_rate_mu > 0:
        offered_load = arrival_rate_lambda / service_rate_mu
    else:
        offered_load = float('inf')
        warnings.append("Cannot calculate offered load (service rate is zero)")
    
    if num_servers_c > 0:
        utilization = offered_load / num_servers_c
    else:
        utilization = float('inf')
        warnings.append("Cannot calculate utilization (no servers)")
    
    # Check for system stability
    is_stable = utilization < 1.0
    if not is_stable:
        warnings.append(
            f"System is UNSTABLE (ρ={utilization:.2f} >= 1.0). "
            f"Arrival rate exceeds service capacity!"
        )
    
    # Check if utilization is very high (>90%)
    if 0.9 <= utilization < 1.0:
        warnings.append(
            f"System utilization is very high (ρ={utilization:.2%}). "
            f"Expect long wait times."
        )
    
    # Check if available servers exceed capacity
    if num_servers_c > capacity:
        warnings.append(
            f"Available servers ({num_servers_c}) exceeds capacity ({capacity})"
        )
    
    is_valid = len(warnings) == 0 or (is_stable and arrival_rate_lambda > 0 and service_rate_mu > 0)
    
    return {
        'is_valid': is_valid,
        'warnings': warnings,
        'offered_load': offered_load,
        'utilization': utilization,
        'is_stable': is_stable
    }


if __name__ == "__main__":
    # Test example with synthetic data
    print("\n" + "=" * 60)
    print("Queue Estimator - Test Example")
    print("=" * 60 + "\n")
    
    # Create synthetic parking data
    dates = pd.date_range('2024-01-01', periods=24*7, freq='H')  # 1 week hourly
    
    # Simulate occupancy pattern (higher during business hours)
    np.random.seed(42)
    base_occupancy = 50
    hourly_pattern = [
        20, 15, 10, 8, 8, 10, 25, 45, 65, 75, 80, 85,  # Midnight to Noon
        85, 80, 75, 70, 65, 60, 55, 50, 45, 35, 30, 25   # Noon to Midnight
    ]
    
    occupancy = []
    for date in dates:
        hour = date.hour
        base = hourly_pattern[hour]
        noise = np.random.normal(0, 5)
        occ = max(0, int(base + noise))
        occupancy.append(occ)
    
    df_test = pd.DataFrame({
        'LastUpdated': dates,
        'Occupancy': occupancy
    })
    
    # Test the estimator
    capacity = 100
    hourly_rates, mu = get_queueing_inputs(df_test, capacity)
    
    print("\n" + "=" * 60)
    print("Validation Check")
    print("=" * 60)
    
    # Validate for peak hour
    peak_hour = max(hourly_rates.keys(), key=lambda h: hourly_rates[h])
    peak_lambda = hourly_rates[peak_hour]
    available_spots = capacity - int(max(occupancy))
    
    validation = validate_queueing_parameters(peak_lambda, mu, available_spots, capacity)
    
    print(f"\nPeak hour: {peak_hour}:00")
    print(f"Peak arrival rate: {peak_lambda:.2f} cars/hour")
    print(f"Available spots: {available_spots}")
    print(f"Offered load: {validation['offered_load']:.2f}")
    print(f"Utilization: {validation['utilization']:.2%}")
    print(f"System stable: {validation['is_stable']}")
    
    if validation['warnings']:
        print("\nWarnings:")
        for warning in validation['warnings']:
            print(f"  ⚠️  {warning}")
    else:
        print("\n✅ All validation checks passed!")
    
    print("\n" + "=" * 60)
