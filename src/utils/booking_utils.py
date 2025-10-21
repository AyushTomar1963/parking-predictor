"""
Booking logic, availability checking, dynamic pricing.
"""
from datetime import datetime, timedelta


class ParkingBookingSystem:
    """Manage parking bookings and availability."""
    
    def __init__(self, total_spots, pricing_base=10):
        self.total_spots = total_spots
        self.pricing_base = pricing_base
        self.bookings = []
    
    def check_availability(self, start_time, end_time, predicted_occupancy):
        """Check if spots are available for the given time range."""
        available_spots = self.total_spots - predicted_occupancy
        return max(0, available_spots)
    
    def calculate_dynamic_price(self, predicted_occupancy, time_slot):
        """Calculate dynamic pricing based on predicted occupancy."""
        occupancy_rate = predicted_occupancy / self.total_spots
        
        # Higher occupancy → higher price
        price_multiplier = 1 + (occupancy_rate * 1.5)
        
        # Peak hours (8-10 AM, 5-7 PM) → higher price
        hour = time_slot.hour
        if (8 <= hour <= 10) or (17 <= hour <= 19):
            price_multiplier *= 1.3
        
        # Weekend discount
        if time_slot.weekday() >= 5:
            price_multiplier *= 0.9
        
        final_price = self.pricing_base * price_multiplier
        return round(final_price, 2)
    
    def create_booking(self, user_id, start_time, end_time, spot_id):
        """Create a new parking booking."""
        booking = {
            'booking_id': len(self.bookings) + 1,
            'user_id': user_id,
            'start_time': start_time,
            'end_time': end_time,
            'spot_id': spot_id,
            'status': 'confirmed'
        }
        self.bookings.append(booking)
        return booking
    
    def get_recommendations(self, desired_time, predictions_df):
        """Recommend best times to park based on predictions."""
        # Find time slots with lower occupancy around desired time
        window_start = desired_time - timedelta(hours=2)
        window_end = desired_time + timedelta(hours=2)
        
        window_predictions = predictions_df[
            (predictions_df['timestamp'] >= window_start) &
            (predictions_df['timestamp'] <= window_end)
        ]
        
        # Sort by lowest occupancy
        recommendations = window_predictions.sort_values('predicted_occupancy').head(3)
        return recommendations
