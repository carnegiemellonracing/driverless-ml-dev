import unittest
import numpy as np
from geo import C_seg, C_poly, C_width, find_starting_vertices
from models import LaneCandidate, PerceptualFieldContext
from config import W_MIN, W_MAX
import math


def create_test_context(points: np.ndarray) -> PerceptualFieldContext:
    """Helper to create PerceptualFieldContext for tests.

    Creates a simple context where all points are visible (indices 0, 1, 2, ...).
    """
    n = len(points)
    visible_indices = set(range(n))
    adj_list = {i: [] for i in range(n)}  # Empty adjacency for constraint tests

    return PerceptualFieldContext(
        cone_map=points,
        visible_indices=visible_indices,
        adj_list=adj_list,
        car_pos=np.array([0.0, 0.0]),
        car_heading=0.0,
    )


class TestGeometricConstraints(unittest.TestCase):

    def test_c_seg(self):
        # Straight line - Pass
        map_points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        context = create_test_context(map_points)
        candidate = LaneCandidate(
            left_path=[0, 1, 2],
            right_path=[],
            left_visited=set(),
            right_visited=set(),
        )
        self.assertTrue(
            C_seg(candidate, context, side="left"),
            "Failed straight line - should pass",
        )

        # 90 deg turn - Pass
        map_points_90 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        context_90 = create_test_context(map_points_90)
        candidate_90 = LaneCandidate(
            left_path=[0, 1, 2],
            right_path=[],
            left_visited=set(),
            right_visited=set(),
        )
        self.assertTrue(
            C_seg(candidate_90, context_90, side="left"),
            "Failed 90 deg turn - should pass",
        )

        # 100 deg turn (angle > 90) - Fail
        # Vector (1,0) to (-0.2, 1) -> angle > 90
        map_points_sharp = np.array([[0.0, 0.0], [1.0, 0.0], [0.8, 1.0]])
        context_sharp = create_test_context(map_points_sharp)
        candidate_sharp = LaneCandidate(
            left_path=[0, 1, 2],
            right_path=[],
            left_visited=set(),
            right_visited=set(),
        )
        self.assertFalse(
            C_seg(candidate_sharp, context_sharp, side="left"),
            "Failed 100 deg turn - should fail",
        )

    def test_c_poly(self):
        # Simple box - Pass
        # left: [(0, 1), (2, 1)], right: [(0, -1), (2, -1)]
        map_points_box = np.array([[0.0, 1.0], [2.0, 1.0], [0.0, -1.0], [2.0, -1.0]])
        context_box = create_test_context(map_points_box)
        candidate_box = LaneCandidate(
            left_path=[0, 1], right_path=[2, 3], left_visited=set(), right_visited=set()
        )
        self.assertTrue(
            C_poly(candidate_box, context_box), "Failed simple box - should pass"
        )

        # Bowtie (self intersection) - Fail
        # Left crosses right
        # left: [(0, 1), (2, -1)], right: [(0, -1), (2, 1)]
        map_points_cross = np.array([[0.0, 1.0], [2.0, -1.0], [0.0, -1.0], [2.0, 1.0]])
        context_cross = create_test_context(map_points_cross)
        candidate_cross = LaneCandidate(
            left_path=[0, 1], right_path=[2, 3], left_visited=set(), right_visited=set()
        )
        self.assertFalse(
            C_poly(candidate_cross, context_cross), "Failed bowtie - should fail"
        )

    def test_c_width(self):
        # Test 1: Width within bounds (4.0m, between W_MIN=2.5 and W_MAX=6.5) - Pass
        # Left boundary at y=2, right boundary at y=-2 -> width = 4.0m
        map_points_good = np.array(
            [
                [0.0, 2.0],  # left[0]
                [5.0, 2.0],  # left[1]
                [10.0, 2.0],  # left[2]
                [0.0, -2.0],  # right[0]
                [5.0, -2.0],  # right[1]
                [10.0, -2.0],  # right[2]
            ]
        )
        context_good = create_test_context(map_points_good)
        candidate_good = LaneCandidate(
            left_path=[0, 1, 2],
            right_path=[3, 4, 5],
            left_visited=set(),
            right_visited=set(),
        )
        self.assertTrue(
            C_width(candidate_good, context_good),
            f"Failed: width=4.0m should pass (W_MIN={W_MIN}, W_MAX={W_MAX})",
        )

        # Test 2: Width too narrow (1.0m, below W_MIN=2.5) - Fail
        # Left boundary at y=0.5, right boundary at y=-0.5 -> width = 1.0m
        map_points_narrow = np.array(
            [
                [0.0, 0.5],  # left[0]
                [5.0, 0.5],  # left[1]
                [0.0, -0.5],  # right[0]
                [5.0, -0.5],  # right[1]
            ]
        )
        context_narrow = create_test_context(map_points_narrow)
        candidate_narrow = LaneCandidate(
            left_path=[0, 1],
            right_path=[2, 3],
            left_visited=set(),
            right_visited=set(),
        )
        self.assertFalse(
            C_width(candidate_narrow, context_narrow),
            f"Failed: width=1.0m should fail (below W_MIN={W_MIN})",
        )

        # Test 3: Width too wide (10.0m, above W_MAX=6.5) - Fail
        # Left boundary at y=5, right boundary at y=-5 -> width = 10.0m
        map_points_wide = np.array(
            [
                [0.0, 5.0],  # left[0]
                [5.0, 5.0],  # left[1]
                [0.0, -5.0],  # right[0]
                [5.0, -5.0],  # right[1]
            ]
        )
        context_wide = create_test_context(map_points_wide)
        candidate_wide = LaneCandidate(
            left_path=[0, 1],
            right_path=[2, 3],
            left_visited=set(),
            right_visited=set(),
        )
        self.assertFalse(
            C_width(candidate_wide, context_wide),
            f"Failed: width=10.0m should fail (above W_MAX={W_MAX})",
        )

        # Test 4: Width at lower boundary (W_MIN=2.5 exactly) - Pass
        # Left at y=1.25, right at y=-1.25 -> width = 2.5m
        map_points_min = np.array(
            [
                [0.0, 1.25],  # left[0]
                [5.0, 1.25],  # left[1]
                [0.0, -1.25],  # right[0]
                [5.0, -1.25],  # right[1]
            ]
        )
        context_min = create_test_context(map_points_min)
        candidate_min = LaneCandidate(
            left_path=[0, 1],
            right_path=[2, 3],
            left_visited=set(),
            right_visited=set(),
        )
        self.assertTrue(
            C_width(candidate_min, context_min),
            f"Failed: width=W_MIN={W_MIN} should pass (edge case)",
        )

        # Test 5: Width at upper boundary (W_MAX=6.5 exactly) - Pass
        # Left at y=3.25, right at y=-3.25 -> width = 6.5m
        map_points_max = np.array(
            [
                [0.0, 3.25],  # left[0]
                [5.0, 3.25],  # left[1]
                [0.0, -3.25],  # right[0]
                [5.0, -3.25],  # right[1]
            ]
        )
        context_max = create_test_context(map_points_max)
        candidate_max = LaneCandidate(
            left_path=[0, 1],
            right_path=[2, 3],
            left_visited=set(),
            right_visited=set(),
        )
        self.assertTrue(
            C_width(candidate_max, context_max),
            f"Failed: width=W_MAX={W_MAX} should pass (edge case)",
        )

        # Test 6: Variable width - narrowest part too narrow - Fail
        # Lane that narrows from 4m to 1m
        map_points_var_narrow = np.array(
            [
                [0.0, 2.0],  # left[0] - width 4m here
                [5.0, 0.5],  # left[1] - width 1m here (too narrow)
                [0.0, -2.0],  # right[0]
                [5.0, -0.5],  # right[1]
            ]
        )
        context_var_narrow = create_test_context(map_points_var_narrow)
        candidate_var_narrow = LaneCandidate(
            left_path=[0, 1],
            right_path=[2, 3],
            left_visited=set(),
            right_visited=set(),
        )
        self.assertFalse(
            C_width(candidate_var_narrow, context_var_narrow),
            "Failed: variable width with narrow section should fail",
        )


class TestFindStartingVertices(unittest.TestCase):
    def test_symmetric_pair(self):
        car_pos = np.array([0.0, 0.0])
        car_heading_rad = 0.0
        cone_map = np.array(
            [
                [1.0, 1.0],
                [1.0, -1.0],
            ]
        )
        graph = {0: [1], 1: [0]}
        ctx = PerceptualFieldContext(cone_map, set(range(len(cone_map))), graph, car_pos, car_heading_rad)

        left_pt, right_pt = find_starting_vertices(
            ctx, max_range=5.0
        )

        # Both should be found
        self.assertIsNotNone(left_pt, "Failed: left starting point not found")
        self.assertIsNotNone(right_pt, "Failed: right starting point not found")

        # Verify the selected points
        self.assertEqual(left_pt, 0, "Failed: left starting point wrong")
        self.assertEqual(right_pt, 1, "Failed: right starting point wrong")

    def test_candidates_outside_range(self):
        car_pos = np.array([0.0, 0.0])
        car_heading_rad = 0.0
        cone_map = np.array(
            [
                [100.0, 100.0],
                [100.0, -100.0],
            ]
        )
        graph = {0: [1], 1: [0]}

        ctx = PerceptualFieldContext(cone_map, set(range(len(cone_map))), graph, car_pos, car_heading_rad)

        left_pt, right_pt = find_starting_vertices(
            ctx, max_range=5.0
        )

        self.assertIsNone(left_pt, "Failed: found left point outside range")
        self.assertIsNone(right_pt, "Failed: found right point outside range")

    def test_no_left_candidates(self):
        """Test when only right candidates exist (all points negative angle)."""
        car_pos = np.array([0.0, 0.0])
        car_heading_rad = 0.0

        # Both points to the right
        cone_map = np.array(
            [
                [1.0, -0.5],
                [1.0, -1.0],
            ]
        )
        graph = {0: [1], 1: [0]}

        ctx = PerceptualFieldContext(cone_map, set(range(len(cone_map))), graph, car_pos, car_heading_rad)

        left_pt, right_pt = find_starting_vertices(
            ctx, max_range=5.0
        )

        self.assertIsNone(left_pt, "Failed: found left point when no valid pair exists")
        self.assertIsNone(
            right_pt, "Failed: found right point when no valid pair exists"
        )

    def test_no_right_candidates(self):
        """Test when only left candidates exist (all points positive angle)."""
        car_pos = np.array([0.0, 0.0])
        car_heading_rad = 0.0

        # Both points to the left
        cone_map = np.array(
            [
                [1.0, 0.5],
                [1.0, 1.0],
            ]
        )
        graph = {0: [1], 1: [0]}

        ctx = PerceptualFieldContext(cone_map, set(range(len(cone_map))), graph, car_pos, car_heading_rad)

        left_pt, right_pt = find_starting_vertices(
            ctx, max_range=5.0
        )

        self.assertIsNone(left_pt, "Failed: found left point when no valid pair exists")
        self.assertIsNone(
            right_pt, "Failed: found right point when no valid pair exists"
        )

    def test_multiple_candidates_symmetry_selection(self):
        """Test that the most symmetric pair is selected when multiple pairs exist."""
        car_pos = np.array([0.0, 0.0])
        car_heading_rad = 0.0

        cone_map = np.array(
            [
                [1.0, 2.0],  # Left, steep
                [1.0, 1.0],  # Left, 45 deg
                [1.0, -0.5],  # Right, shallow
                [1.0, -1.0],  # Right, 45 deg
            ]
        )
        graph = {0: [], 1: [], 2: [], 3: []}

        ctx = PerceptualFieldContext(cone_map, set(range(len(cone_map))), graph, car_pos, car_heading_rad)

        left_pt, right_pt = find_starting_vertices(
            ctx, max_range=5.0
        )

        self.assertEqual(left_pt, 1, "Failed: left starting point not most symmetric")
        self.assertEqual(right_pt, 3, "Failed: right starting point not most symmetric")

    def test_different_heading(self):
        """Test with car heading in a different direction."""
        car_pos = np.array([0.0, 0.0])
        car_heading_rad = math.pi / 2  # Pointing up (90 deg)

        # Points relative to upward heading:
        # (1, 1) is to the right (negative angle)
        # (-1, 1) is to the left (positive angle)
        cone_map = np.array(
            [
                [-1.0, 1.0],  # Left of upward heading
                [1.0, 1.0],  # Right of upward heading
            ]
        )
        graph = {0: [1], 1: [0]}

        ctx = PerceptualFieldContext(cone_map, set(range(len(cone_map))), graph, car_pos, car_heading_rad)

        left_pt, right_pt = find_starting_vertices(
            ctx, max_range=5.0
        )

        self.assertIsNotNone(left_pt)
        self.assertIsNotNone(right_pt)

        self.assertEqual(left_pt, 0, "Failed: left starting point wrong")
        self.assertEqual(right_pt, 1, "Failed: left starting point wrong")


if __name__ == "__main__":
    unittest.main()
