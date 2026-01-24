import unittest
import numpy as np
import math
from geo import (
    C_seg, 
    C_poly, 
    get_segment_angle,
    find_starting_vertices
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


class TestFindStartingVertices(unittest.TestCase):
    def test_symmetric_pair(self):
        car_pos = np.array([0.0, 0.0])
        car_heading_rad = 0.0
        cone_map = np.array([
            [1.0, 1.0],
            [1.0, -1.0],
        ])
        graph = {0: [1], 1: [0]}
        
        left_pt, right_pt = find_starting_vertices(graph, cone_map, car_pos, car_heading_rad, max_range=5.0)
        
        # Both should be found
        self.assertIsNotNone(left_pt, "Failed: left starting point not found")
        self.assertIsNotNone(right_pt, "Failed: right starting point not found")
        
        # Verify the selected points
        self.assertEqual(left_pt, 0, "Failed: left starting point wrong")
        self.assertEqual(right_pt, 1, "Failed: right starting point wrong")

    def test_candidates_outside_range(self):
        car_pos = np.array([0.0, 0.0])
        car_heading_rad = 0.0
        cone_map = np.array([
            [100.0, 100.0],
            [100.0, -100.0],
        ])
        graph = {0: [1], 1: [0]}
        
        left_pt, right_pt = find_starting_vertices(graph, cone_map, car_pos, car_heading_rad, max_range=1.0)
        
        self.assertIsNone(left_pt, "Failed: found left point outside range")
        self.assertIsNone(right_pt, "Failed: found right point outside range")

    def test_no_left_candidates(self):
        """Test when only right candidates exist (all points negative angle)."""
        car_pos = np.array([0.0, 0.0])
        car_heading_rad = 0.0
        
        # Both points to the right
        cone_map = np.array([
            [1.0, -0.5],
            [1.0, -1.0],
        ])
        graph = {0: [1], 1: [0]}
        
        left_pt, right_pt = find_starting_vertices(graph, cone_map, car_pos, car_heading_rad, max_range=5.0)
        
        self.assertIsNone(left_pt, "Failed: found left point when no valid pair exists")
        self.assertIsNone(right_pt, "Failed: found right point when no valid pair exists")

    def test_no_right_candidates(self):
        """Test when only left candidates exist (all points positive angle)."""
        car_pos = np.array([0.0, 0.0])
        car_heading_rad = 0.0
        
        # Both points to the left
        cone_map = np.array([
            [1.0, 0.5],
            [1.0, 1.0],
        ])
        graph = {0: [1], 1: [0]}
        
        left_pt, right_pt = find_starting_vertices(graph, cone_map, car_pos, car_heading_rad, max_range=5.0)
        
        self.assertIsNone(left_pt, "Failed: found left point when no valid pair exists")
        self.assertIsNone(right_pt, "Failed: found right point when no valid pair exists")

    def test_multiple_candidates_symmetry_selection(self):
        """Test that the most symmetric pair is selected when multiple pairs exist."""
        car_pos = np.array([0.0, 0.0])
        car_heading_rad = 0.0
        
        cone_map = np.array([
            [1.0, 2.0],      # Left, steep
            [1.0, 1.0],      # Left, 45 deg
            [1.0, -0.5],     # Right, shallow
            [1.0, -1.0],     # Right, 45 deg
        ])
        graph = {0: [], 1: [], 2: [], 3: []}
        
        left_pt, right_pt = find_starting_vertices(graph, cone_map, car_pos, car_heading_rad, max_range=5.0)
        
        self.assertEqual(left_pt, 1, "Failed: left starting point not most symmetric")
        self.assertEqual(right_pt, 3, "Failed: right starting point not most symmetric")

    def test_different_heading(self):
        """Test with car heading in a different direction."""
        car_pos = np.array([0.0, 0.0])
        car_heading_rad = math.pi / 2  # Pointing up (90 deg)
        
        # Points relative to upward heading:
        # (1, 1) is to the right (negative angle)
        # (-1, 1) is to the left (positive angle)
        cone_map = np.array([
            [-1.0, 1.0],     # Left of upward heading
            [1.0, 1.0],      # Right of upward heading
        ])
        graph = {0: [1], 1: [0]}
        
        left_pt, right_pt = find_starting_vertices(graph, cone_map, car_pos, car_heading_rad, max_range=5.0)
        
        self.assertIsNotNone(left_pt)
        self.assertIsNotNone(right_pt)
        
        self.assertEqual(left_pt, 0, "Failed: left starting point wrong")
        self.assertEqual(right_pt, 1, "Failed: left starting point wrong")

if __name__ == '__main__':
    unittest.main()
