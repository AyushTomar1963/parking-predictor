"""
Unit tests for booking utilities.
"""
import unittest
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from booking_utils import ParkingBookingSystem


class TestBookingSystem(unittest.TestCase):
    
    def setUp(self):
        self.booking_system = ParkingBookingSystem(total_spots=100, pricing_base=10)
    
    def test_check_availability(self):
        """Test availability checking."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=2)
        predicted_occupancy = 60
        
        available = self.booking_system.check_availability(start_time, end_time, predicted_occupancy)
        self.assertEqual(available, 40)
    
    def test_dynamic_pricing(self):
        """Test dynamic pricing calculation."""
        # Test peak hour pricing
        peak_time = datetime.now().replace(hour=9, minute=0)
        price = self.booking_system.calculate_dynamic_price(80, peak_time)
        self.assertGreater(price, 10)  # Should be higher than base price
    
    def test_create_booking(self):
        """Test booking creation."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=2)
        
        booking = self.booking_system.create_booking(
            user_id="user123",
            start_time=start_time,
            end_time=end_time,
            spot_id=1
        )
        
        self.assertEqual(booking['user_id'], "user123")
        self.assertEqual(booking['status'], "confirmed")
        self.assertEqual(len(self.booking_system.bookings), 1)


if __name__ == '__main__':
    unittest.main()
