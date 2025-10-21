"""
Booking Probability Calculator

This module calculates the probability of successfully getting a parking spot
based on predicted occupancy and queueing theory.

Based on: Main.ipynb Cell #15 (get_booking_confirmation function)
"""

import math
import numpy as np
from typing import Dict
from .erlang_c import calculate_erlang_c


def get_booking_confirmation(
    predicted_occupancy: float,
    capacity: int,
    hour_of_day: int,
    hourly_arrival_rates: Dict[int, float],
    service_rate_mu: float
) -> Dict[str, float]:
    """
    Estimate the probability of acquiring a parking spot at a given hour.
    
    This function combines:
    1. Predicted occupancy from ML models
    2. Queueing theory (Erlang-C) for wait probability
    3. Hourly arrival patterns
    
    to provide booking success probability and expected wait time.
    
    Algorithm:
        1. Calculate available spots: capacity - predicted_occupancy
        2. Get arrival rate (λ) for the specific hour
        3. Apply Erlang-C formula with available spots as servers
        4. Return probability of immediate service and wait time
    
    Args:
        predicted_occupancy: ML model's predicted occupancy for this hour
        capacity: Total parking lot capacity (number of spots)
        hour_of_day: Hour of day (0-23) for arrival rate lookup
        hourly_arrival_rates: Dict mapping hour to average arrival rate
        service_rate_mu: Service rate per slot (per hour)
        
    Returns:
        Dictionary with keys:
            - 'prob_get_spot' (float): Probability of getting a spot immediately (0-1)
            - 'prob_wait' (float): Probability of having to wait (0-1)
            - 'expected_wait_minutes' (float): Expected wait time in minutes
            - 'available_slots' (int): Number of available parking spots
            - 'arrival_lambda' (float): Arrival rate used for this hour
            - 'service_rate_mu' (float): Service rate used
            - 'recommendation' (str): User-friendly recommendation
    
    Example:
        >>> result = get_booking_confirmation(
        ...     predicted_occupancy=540,
        ...     capacity=600,
        ...     hour_of_day=15,
        ...     hourly_arrival_rates={15: 12.5},
        ...     service_rate_mu=0.5
        ... )
        >>> print(f"Booking success probability: {result['prob_get_spot']:.1%}")
        >>> print(f"Recommendation: {result['recommendation']}")
    
    References:
        - Based on Main.ipynb Cell #15
        - Combines ML prediction with Erlang-C queueing model
    """
    # Calculate available parking spots
    available = int(max(0, capacity - int(round(predicted_occupancy))))
    
    # Handle case where lot is full or over capacity
    if available <= 0:
        return {
            'prob_get_spot': 0.0,
            'prob_wait': 1.0,
            'expected_wait_minutes': float('inf'),
            'available_slots': 0,
            'arrival_lambda': 0.0,
            'service_rate_mu': service_rate_mu,
            'recommendation': '🔴 LOT FULL - Do not book. No spots available.',
            'confidence_level': 'high'
        }
    
    # Get arrival rate for this specific hour (fallback to mean if not found)
    if len(hourly_arrival_rates) == 0:
        arrival_lambda = 0.0
    else:
        arrival_lambda = float(
            hourly_arrival_rates.get(
                hour_of_day,
                np.mean(list(hourly_arrival_rates.values()))
            )
        )
    
    # Apply Erlang-C formula to calculate waiting probability
    prob_wait, details = calculate_erlang_c(arrival_lambda, service_rate_mu, available)
    
    # Calculate expected wait time in minutes
    Wq_hours = details.get('Wq', 0.0)
    if math.isinf(Wq_hours):
        expected_wait_minutes = float('inf')
    else:
        expected_wait_minutes = Wq_hours * 60.0
    
    # Probability of getting a spot immediately
    prob_get_spot = 1.0 - prob_wait
    
    # Generate user-friendly recommendation
    recommendation = _generate_recommendation(
        prob_get_spot, expected_wait_minutes, available, capacity
    )
    
    # Determine confidence level based on available data
    confidence_level = _determine_confidence(available, arrival_lambda, service_rate_mu)
    
    return {
        'prob_get_spot': float(prob_get_spot),
        'prob_wait': float(prob_wait),
        'expected_wait_minutes': float(expected_wait_minutes),
        'available_slots': available,
        'arrival_lambda': float(arrival_lambda),
        'service_rate_mu': float(service_rate_mu),
        'recommendation': recommendation,
        'confidence_level': confidence_level,
        'utilization': details.get('rho', 0.0)
    }


def _generate_recommendation(
    prob_get_spot: float,
    expected_wait_minutes: float,
    available_slots: int,
    capacity: int
) -> str:
    """
    Generate user-friendly booking recommendation based on probability.
    
    Recommendation buckets:
        - Very High (≥95%): Strong recommendation to book
        - High (70-95%): Good chance, recommend booking
        - Medium (40-70%): Moderate chance, consider alternatives
        - Low (<40%): Poor chance, find alternative
    
    Args:
        prob_get_spot: Probability of immediate service (0-1)
        expected_wait_minutes: Expected wait time in minutes
        available_slots: Number of available spots
        capacity: Total lot capacity
        
    Returns:
        str: User-friendly recommendation message with emoji
    """
    percent_available = (available_slots / capacity) * 100 if capacity > 0 else 0
    
    # Very High Probability (≥95%)
    if prob_get_spot >= 0.95:
        return (
            f"🟢 EXCELLENT - {prob_get_spot:.1%} chance of immediate spot. "
            f"{available_slots} spots available ({percent_available:.0f}% of capacity). "
            f"Highly recommended to book!"
        )
    
    # High Probability (70-95%)
    elif prob_get_spot >= 0.70:
        wait_msg = f"~{expected_wait_minutes:.0f} min wait" if math.isfinite(expected_wait_minutes) else "possible wait"
        return (
            f"🟡 GOOD - {prob_get_spot:.1%} chance of immediate spot. "
            f"{available_slots} spots available. "
            f"If wait occurs: {wait_msg}. Recommended to book."
        )
    
    # Medium Probability (40-70%)
    elif prob_get_spot >= 0.40:
        wait_msg = f"~{expected_wait_minutes:.0f} min" if math.isfinite(expected_wait_minutes) else "significant"
        return (
            f"🟠 MODERATE - {prob_get_spot:.1%} chance of immediate spot. "
            f"Only {available_slots} spots left. "
            f"Expected wait: {wait_msg}. Consider alternatives."
        )
    
    # Low Probability (<40%)
    else:
        return (
            f"🔴 POOR - Only {prob_get_spot:.1%} chance of immediate spot. "
            f"Very limited availability ({available_slots} spots). "
            f"Likely long wait. Recommend finding alternative parking."
        )


def _determine_confidence(
    available_slots: int,
    arrival_lambda: float,
    service_rate_mu: float
) -> str:
    """
    Determine confidence level in the prediction based on data quality.
    
    Args:
        available_slots: Number of available parking spots
        arrival_lambda: Estimated arrival rate
        service_rate_mu: Estimated service rate
        
    Returns:
        str: Confidence level ('high', 'medium', 'low')
    """
    # Low confidence if parameters are questionable
    if arrival_lambda <= 0 or service_rate_mu <= 0:
        return 'low'
    
    # Low confidence if very few spots (high uncertainty)
    if available_slots < 5:
        return 'medium'
    
    # High confidence otherwise
    return 'high'


def calculate_booking_success_probability(
    predicted_occupancy: float,
    capacity: int,
    hour_of_day: int,
    hourly_arrival_rates: Dict[int, float],
    service_rate_mu: float
) -> float:
    """
    Simplified function that returns just the booking success probability.
    
    Convenience function for when only the probability is needed, not full details.
    
    Args:
        predicted_occupancy: ML model's predicted occupancy
        capacity: Total parking lot capacity
        hour_of_day: Hour of day (0-23)
        hourly_arrival_rates: Dict mapping hour to arrival rate
        service_rate_mu: Service rate per slot (per hour)
        
    Returns:
        float: Probability of successfully getting a spot (0-1)
    
    Example:
        >>> prob = calculate_booking_success_probability(540, 600, 15, {15: 12.5}, 0.5)
        >>> print(f"Success probability: {prob:.1%}")
    """
    result = get_booking_confirmation(
        predicted_occupancy, capacity, hour_of_day,
        hourly_arrival_rates, service_rate_mu
    )
    return result['prob_get_spot']


def batch_booking_analysis(
    predictions: Dict[int, float],
    capacity: int,
    hourly_arrival_rates: Dict[int, float],
    service_rate_mu: float
) -> Dict[int, Dict[str, float]]:
    """
    Analyze booking probabilities for multiple hours at once.
    
    Useful for showing a full day's forecast with booking recommendations.
    
    Args:
        predictions: Dict mapping hour_of_day to predicted occupancy
        capacity: Total parking lot capacity
        hourly_arrival_rates: Dict mapping hour to arrival rate
        service_rate_mu: Service rate per slot (per hour)
        
    Returns:
        Dict mapping hour_of_day to booking confirmation result dict
    
    Example:
        >>> predictions = {9: 450, 10: 520, 11: 580, 12: 590}
        >>> results = batch_booking_analysis(predictions, 600, {9:10, 10:12, 11:15, 12:18}, 0.5)
        >>> for hour, result in results.items():
        ...     print(f"{hour}:00 - {result['recommendation']}")
    """
    results = {}
    
    for hour_of_day, predicted_occ in predictions.items():
        results[hour_of_day] = get_booking_confirmation(
            predicted_occ, capacity, hour_of_day,
            hourly_arrival_rates, service_rate_mu
        )
    
    return results


if __name__ == "__main__":
    # Test example
    print("=" * 70)
    print("Booking Probability Calculator - Test Examples")
    print("=" * 70)
    
    # Test parameters
    capacity = 600
    hourly_rates = {
        9: 10.0, 10: 12.0, 11: 15.0, 12: 18.0,
        13: 16.0, 14: 14.0, 15: 12.5, 16: 10.0
    }
    mu = 0.5  # 2-hour average parking duration
    
    # Test scenarios
    scenarios = [
        {'hour': 9, 'pred_occ': 360, 'desc': 'Morning - Low occupancy'},
        {'hour': 12, 'pred_occ': 540, 'desc': 'Noon - High occupancy'},
        {'hour': 15, 'pred_occ': 588, 'desc': 'Afternoon - Very high occupancy'},
        {'hour': 16, 'pred_occ': 600, 'desc': 'Evening - Full capacity'},
    ]
    
    for scenario in scenarios:
        print(f"\n{'-' * 70}")
        print(f"Scenario: {scenario['desc']}")
        print(f"Time: {scenario['hour']}:00")
        print(f"Predicted Occupancy: {scenario['pred_occ']} / {capacity}")
        print('-' * 70)
        
        result = get_booking_confirmation(
            predicted_occupancy=scenario['pred_occ'],
            capacity=capacity,
            hour_of_day=scenario['hour'],
            hourly_arrival_rates=hourly_rates,
            service_rate_mu=mu
        )
        
        print(f"Available Slots: {result['available_slots']}")
        print(f"Arrival Rate (λ): {result['arrival_lambda']:.2f} cars/hour")
        print(f"Service Rate (μ): {result['service_rate_mu']:.2f} per hour")
        print(f"Utilization: {result['utilization']:.1%}")
        print(f"Probability of Immediate Spot: {result['prob_get_spot']:.1%}")
        print(f"Probability of Waiting: {result['prob_wait']:.1%}")
        
        if math.isfinite(result['expected_wait_minutes']):
            print(f"Expected Wait Time: {result['expected_wait_minutes']:.1f} minutes")
        else:
            print("Expected Wait Time: INFINITE (overloaded)")
        
        print(f"Confidence: {result['confidence_level'].upper()}")
        print(f"\n{result['recommendation']}")
    
    print("\n" + "=" * 70)
    print("Batch Analysis Example - Full Day Forecast")
    print("=" * 70)
    
    # Full day predictions
    daily_predictions = {
        9: 360, 10: 420, 11: 480, 12: 540,
        13: 560, 14: 550, 15: 588, 16: 600
    }
    
    batch_results = batch_booking_analysis(daily_predictions, capacity, hourly_rates, mu)
    
    print(f"\n{'Hour':<6} {'Occupancy':<12} {'Available':<10} {'Prob':<8} {'Status':<20}")
    print("-" * 70)
    
    for hour in sorted(batch_results.keys()):
        result = batch_results[hour]
        occ = daily_predictions[hour]
        available = result['available_slots']
        prob = result['prob_get_spot']
        
        # Simple status indicator
        if prob >= 0.95:
            status = "🟢 EXCELLENT"
        elif prob >= 0.70:
            status = "🟡 GOOD"
        elif prob >= 0.40:
            status = "🟠 MODERATE"
        else:
            status = "🔴 POOR"
        
        print(f"{hour:02d}:00  {occ:>4}/{capacity:<5} {available:<10} {prob:>6.1%}  {status}")
    
    print("\n" + "=" * 70)
