import unittest
import numpy as np
from geo import (
    C_seg, 
    C_poly, 
    get_segment_angle
)

class TestGeometricConstraints(unittest.TestCase):
    
    def test_c_seg(self):
        # Straight line - Pass
        p1 = np.array([0, 0])
        p2 = np.array([1, 0])
        p3 = np.array([2, 0])
        self.assertTrue(C_seg([p1, p2, p3]), "Failed straight line - should pass")
        
        # 90 deg turn - Pass
        p3_90 = np.array([1, 1])
        self.assertTrue(C_seg([p1, p2, p3_90]), "Failed 90 deg turn - should pass")
        
        # 100 deg turn (angle > 90) - Fail
        # Vector (1,0) to (-0.2, 1) -> angle > 90
        p3_sharp = np.array([0.8, 1]) 
        # v1 = (1,0), v2 = (-0.2, 1). dot = -0.2. acos(-0.2) > 90 deg.
        self.assertFalse(C_seg([p1, p2, p3_sharp]), "Failed 100 deg turn - should fail")

    def test_c_poly(self):
        # Simple box - Pass
        left = [np.array([0, 1]), np.array([2, 1])]
        right = [np.array([0, -1]), np.array([2, -1])]
        self.assertTrue(C_poly(left, right), "Failed simple box - should pass")
        
        # Bowtie (self intersection) - Fail
        # Left crosses right
        left_cross = [np.array([0, 1]), np.array([2, -1])]
        right_cross = [np.array([0, -1]), np.array([2, 1])]
        self.assertFalse(C_poly(left_cross, right_cross), "Failed bowtie - should fail")

if __name__ == '__main__':
    unittest.main()
