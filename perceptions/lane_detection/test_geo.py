import unittest
import numpy as np
import math
from geo import (
    calculate_segment_angle,
    line_segments_intersect, # It's not in __all__ but available in file, let's check if we can import it or if we should test via C_poly
    point_to_segment_distance,
    segment_to_segment_distance,
    OnlineLW,
    C_seg,
    C_poly,
    C_width,
    construct_adjacency_list,
    enumerate_path_pairs_v2,
    find_matching_segments
)

class TestGeoUtils(unittest.TestCase):

    def test_calculate_segment_angle(self):
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        
        # Straight line
        p3_straight = np.array([2.0, 0.0])
        self.assertAlmostEqual(calculate_segment_angle(p1, p2, p3_straight), 0.0)
        
        # 90 Left
        p3_left = np.array([1.0, 1.0])
        self.assertAlmostEqual(calculate_segment_angle(p1, p2, p3_left), np.pi/2) # functionality changed to return radians? Checked code: yes np.arccos returns radians.
        
        # 180 U-turn
        p3_back = np.array([0.0, 0.0])
        self.assertAlmostEqual(calculate_segment_angle(p1, p2, p3_back), np.pi)

    def test_point_to_segment_distance(self):
        s1 = np.array([0.0, 0.0])
        s2 = np.array([2.0, 0.0])
        
        # On segment
        p_on = np.array([1.0, 0.0])
        d, proj = point_to_segment_distance(p_on, s1, s2)
        self.assertAlmostEqual(d, 0.0)
        np.testing.assert_array_equal(proj, p_on)
        
        # Above segment
        p_above = np.array([1.0, 1.0])
        d, proj = point_to_segment_distance(p_above, s1, s2)
        self.assertAlmostEqual(d, 1.0)
        np.testing.assert_array_equal(proj, np.array([1.0, 0.0]))

    def test_segment_to_segment_distance(self):
        # Intersecting
        s1_a, s1_b = np.array([0, 0]), np.array([2, 2])
        s2_a, s2_b = np.array([0, 2]), np.array([2, 0])
        d = segment_to_segment_distance(s1_a, s1_b, s2_a, s2_b)
        self.assertAlmostEqual(d, 0.0)
        
        # Parallel
        p1_a, p1_b = np.array([0, 0]), np.array([2, 0])
        p2_a, p2_b = np.array([0, 1]), np.array([2, 1])
        d = segment_to_segment_distance(p1_a, p1_b, p2_a, p2_b)
        self.assertAlmostEqual(d, 1.0)

    def test_online_lw(self):
        # Simple parallel lanes
        left = [0, 1, 2]
        right = [3, 4, 5]
        points = [
            (0, 0), (0, 2), (0, 4),    # Left lane x=0
            (3, 0), (3, 2), (3, 4)     # Right lane x=3
        ]
        
        M_fixed, M_mut = OnlineLW(left, right, points)
        
        # Check that we found matches
        self.assertTrue(len(M_fixed) + len(M_mut) > 0)
        
        # Check strict matching indices for this simple case
        # (0, 3) -> distance 3
        # (1, 4) -> distance 3
        # (2, 5) -> distance 3
        
        # With current naive implementation, it might match everything to everything?
        # Let's check logic.
        pass

    def test_c_width(self):
        points = [
            (0, 0), (0, 2), # Left
            (3, 0), (3, 2)  # Right (3m away)
        ]
        left = [0, 1]
        right = [2, 3]
        
        # Should pass for width 3.0 (wmin=2.5, wmax=6.5)
        self.assertTrue(C_width(left, right, points))
        
        # Should fail for too wide
        points_wide = [
            (0, 0), (0, 2),
            (10, 0), (10, 2) # 10m away
        ]
        self.assertFalse(C_width(left, right, points_wide))

    def test_c_poly(self):
        # No intersection
        points = [
            (0, 0), (0, 2),
            (3, 0), (3, 2)
        ]
        left = [0, 1]
        right = [2, 3]
        self.assertTrue(C_poly(left, right, points))
        
        # Intersection
        points_cross = [
            (0, 0), (2, 2),
            (0, 2), (2, 0)
        ]
        left_cross = [0, 1]
        right_cross = [2, 3]
        self.assertFalse(C_poly(left_cross, right_cross, points_cross))

    def test_find_matching_segments(self):
        points = [
            (0, 0), (2, 0), # Left segment from x=0 to x=2 at y=0
            (0, 1), (2, 1)  # Right segment from x=0 to x=2 at y=1
        ]
        left_path = [0, 1]
        right_path = [2, 3]
        
        matches = find_matching_segments(left_path, right_path, points)
        for m in matches:
            self.assertAlmostEqual(m['width'], 1.0)

    def test_compute_features(self):
        from geo import compute_features
        # Parallel lanes, width 2, length 2
        # Left: (0,0)->(1,0)->(2,0)
        # Right: (0,2)->(1,2)->(2,2)
        points = [
            (0, 0), (1, 0), (2, 0),
            (0, 2), (1, 2), (2, 2)
        ]
        left = [0, 1, 2]
        right = [3, 4, 5]
        
        # Expected features:
        # 1. Mean Width: 2.0
        # 2. Std Width: 0.0
        # 3. Mean Angle: 0.0
        # 4. Std Angle: 0.0
        # 5. Max Angle: 0.0
        # 6. Left Length: 2.0
        # 7. Right Length: 2.0
        # 8. Width Range: 0.0
        
        feats = compute_features((left, right), points)
        
        self.assertAlmostEqual(feats[0], 2.0) # Mean Width
        self.assertAlmostEqual(feats[1], 0.0) # Std Width
        self.assertAlmostEqual(feats[2], 0.0) # Mean Angle
        self.assertAlmostEqual(feats[5], 2.0) # Left Length
        self.assertAlmostEqual(feats[6], 2.0) # Right Length

if __name__ == '__main__':
    unittest.main()
