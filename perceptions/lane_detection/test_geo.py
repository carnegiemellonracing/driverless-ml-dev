import unittest
import numpy as np
from perceptions.lane_detection.geo import (
    check_c_seg, 
    check_c_poly, 
    OnlineLaneWidthCalculator,
    get_segment_angle
)

class TestGeometricConstraints(unittest.TestCase):
    
    def test_c_seg(self):
        # Straight line - Pass
        p1 = np.array([0, 0])
        p2 = np.array([1, 0])
        p3 = np.array([2, 0])
        self.assertTrue(check_c_seg([p1, p2, p3]))
        
        # 90 deg turn - Pass
        p3_90 = np.array([1, 1])
        self.assertTrue(check_c_seg([p1, p2, p3_90]))
        
        # 100 deg turn (angle > 90) - Fail
        # Vector (1,0) to (-0.2, 1) -> angle > 90
        p3_sharp = np.array([0.8, 1]) 
        # v1 = (1,0), v2 = (-0.2, 1). dot = -0.2. acos(-0.2) > 90 deg.
        self.assertFalse(check_c_seg([p1, p2, p3_sharp]))

    def test_c_poly(self):
        # Simple box - Pass
        left = [np.array([0, 1]), np.array([2, 1])]
        right = [np.array([0, -1]), np.array([2, -1])]
        self.assertTrue(check_c_poly(left, right))
        
        # Bowtie (self intersection) - Fail
        # Left crosses right
        left_cross = [np.array([0, 1]), np.array([2, -1])]
        right_cross = [np.array([0, -1]), np.array([2, 1])]
        self.assertFalse(check_c_poly(left_cross, right_cross))

    def test_online_lane_width(self):
        calc = OnlineLaneWidthCalculator(w_min=2.0, w_max=4.0)
        
        # Parallel lines dist=3.0 - Pass
        left = [np.array([0, 1.5]), np.array([10, 1.5])]
        right = [np.array([0, -1.5]), np.array([10, -1.5])]
        
        calc.update(left, right)
        self.assertTrue(calc.check_constraints())
        
        # Parallel lines dist=1.0 - Fail (too narrow)
        left_narrow = [np.array([0, 0.5]), np.array([10, 0.5])]
        right_narrow = [np.array([0, -0.5]), np.array([10, -0.5])]
        
        calc.update(left_narrow, right_narrow)
        self.assertFalse(calc.check_constraints())
        
        # Parallel lines dist=5.0 - Fail (too wide)
        left_wide = [np.array([0, 2.5]), np.array([10, 2.5])]
        right_wide = [np.array([0, -2.5]), np.array([10, -2.5])]
        
        calc.update(left_wide, right_wide)
        self.assertFalse(calc.check_constraints())

if __name__ == '__main__':
    unittest.main()
