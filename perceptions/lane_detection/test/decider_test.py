import array
from operator import rshift
from re import M
from turtle import right
import unittest
import numpy as np
from perceptions.lane_detection.geo import (
    C_seg,
    C_poly,
    C_width,
    find_starting_vertices,
)
from perceptions.lane_detection.deciders import (
    enumerate_path_pairs,
    next_vertex_decider,
    left_right_decider,
)
from perceptions.lane_detection.models import LaneCandidate, PerceptualFieldContext
from perceptions.lane_detection.config import W_MIN, W_MAX
import math


class TestNextVertexDecider(unittest.TestCase):
    """Tests for next_vertex_decider (NVD) function."""

    def test_neighbors_sorted_by_angle(self):
        """Test that neighbors are sorted by ascending angle deviation."""
        # Setup: current at origin, previous at (-1, 0), so direction is +x
        # Neighbors at various angles from +x direction
        cone_map = np.array(
            [
                [-1.0, 0.0],  # idx 0: previous point
                [0.0, 0.0],  # idx 1: current point
                [1.0, 0.0],  # idx 2: neighbor straight ahead (0 deg)
                [1.0, 1.0],  # idx 3: neighbor at 45 deg
                [0.0, 1.0],  # idx 4: neighbor at 90 deg
            ]
        )
        adj_list = {
            0: [1],
            1: [0, 2, 3, 4],  # current has 4 neighbors
            2: [1],
            3: [1],
            4: [1],
        }
        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices=set(range(5)),
            adj_list=adj_list,
            car_pos=np.array([0.0, 0.0]),
            car_heading=0.0,
        )

        path = [0, 1]  # prev=0, curr=1
        result = next_vertex_decider(ctx, path, car_heading=0.0)

        # Should be sorted: idx 2 (0 deg), idx 3 (45 deg), idx 4 (90 deg), idx 0 (180 deg)
        self.assertEqual(
            result[0], 2, "First neighbor should be straight ahead (0 deg)"
        )
        self.assertEqual(result[1], 3, "Second neighbor should be at 45 deg")
        self.assertEqual(result[2], 4, "Third neighbor should be at 90 deg")
        self.assertEqual(result[3], 0, "Fourth neighbor should be behind (180 deg)")

    def test_single_point_path_uses_car_heading(self):
        """Test that single-point path uses car_heading as direction."""
        # Car heading is +x (0 radians), starting at point 0
        cone_map = np.array(
            [
                [0.0, 0.0],  # idx 0: current (start) point
                [1.0, 0.0],  # idx 1: straight ahead
                [0.0, 1.0],  # idx 2: 90 deg left
            ]
        )
        adj_list = {
            0: [1, 2],
            1: [0],
            2: [0],
        }
        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices=set(range(3)),
            adj_list=adj_list,
            car_pos=np.array([0.0, 0.0]),
            car_heading=0.0,
        )

        path = [0]  # Only current point
        result = next_vertex_decider(ctx, path, car_heading=0.0)

        # With car_heading=0 (pointing +x), idx 1 is closer (0 deg) than idx 2 (90 deg)
        self.assertEqual(
            result[0], 1, "Should prefer neighbor aligned with car heading"
        )
        self.assertEqual(result[1], 2, "Second neighbor at 90 deg")

    def test_cache_hit_returns_same_result(self):
        """Test that cached results are returned on subsequent calls."""
        cone_map = np.array(
            [
                [0.0, 0.0],  # idx 0
                [1.0, 0.0],  # idx 1
                [2.0, 0.0],  # idx 2
            ]
        )
        adj_list = {0: [1], 1: [0, 2], 2: [1]}
        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices=set(range(3)),
            adj_list=adj_list,
            car_pos=np.array([0.0, 0.0]),
            car_heading=0.0,
        )

        path = [0, 1]
        result1 = next_vertex_decider(ctx, path, car_heading=0.0)

        # Verify cache was populated
        self.assertIn((0, 1), ctx.nvd_cache, "Cache should contain the key")

        # Call again - should return cached result
        result2 = next_vertex_decider(ctx, path, car_heading=0.0)
        self.assertEqual(result1, result2, "Cached result should match original")

    def test_empty_neighbors_returns_empty_list(self):
        """Test that empty neighbor list returns empty result."""
        cone_map = np.array(
            [
                [0.0, 0.0],  # idx 0: isolated point
            ]
        )
        adj_list = {0: []}  # No neighbors
        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices={0},
            adj_list=adj_list,
            car_pos=np.array([0.0, 0.0]),
            car_heading=0.0,
        )

        path = [0]
        result = next_vertex_decider(ctx, path, car_heading=0.0)

        self.assertEqual(result, [], "Should return empty list for no neighbors")

    def test_different_car_heading(self):
        """Test with car heading pointing in different direction."""
        # Car heading is +y (pi/2 radians)
        cone_map = np.array(
            [
                [0.0, 0.0],  # idx 0: current point
                [0.0, 1.0],  # idx 1: straight ahead (+y)
                [1.0, 0.0],  # idx 2: 90 deg right
            ]
        )
        adj_list = {0: [1, 2], 1: [0], 2: [0]}
        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices=set(range(3)),
            adj_list=adj_list,
            car_pos=np.array([0.0, 0.0]),
            car_heading=math.pi / 2,  # Pointing +y
        )

        path = [0]  # Single point, uses car_heading
        result = next_vertex_decider(ctx, path, car_heading=math.pi / 2)

        # With car_heading=pi/2 (pointing +y), idx 1 is closer (0 deg) than idx 2 (90 deg)
        self.assertEqual(
            result[0], 1, "Should prefer neighbor aligned with car heading (+y)"
        )
        self.assertEqual(result[1], 2, "Second neighbor at 90 deg")


class TestLeftRightDecider(unittest.TestCase):
    """Tests for left_right_decider (LRD) function."""

    def test_insufficient_lane_history(self):
        """Test that function defaults to left (0) when lanes have < 2 points."""
        cone_map = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ]
        )
        adj_list = {0: [1], 1: [0, 2], 2: [1]}
        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices=set(range(3)),
            adj_list=adj_list,
            car_pos=np.array([0.0, 0.0]),
            car_heading=0.0,
        )

        left_lane = [0]  # Only 1 point
        right_lane = [2]  # Only 1 point
        left_candidate = 1
        right_candidate = 2

        result = left_right_decider(
            ctx, left_lane, right_lane, left_candidate, right_candidate
        )
        self.assertEqual(
            result, 0, "Should default to left (0) with insufficient history"
        )

    def test_straight_lane_prefer_left_smooth(self):
        """Test that adding to a straight lane is preferred when it's smoother."""
        # Lanes going straight along x-axis, left adds smoothly
        cone_map = np.array(
            [
                [0.0, 1.0],  # left[0]
                [1.0, 1.0],  # left[1]
                [0.0, -1.0],  # right[0]
                [1.0, -1.0],  # right[1]
                [2.0, 1.0],  # left_candidate (straight continuation)
                [2.0, -1.0],  # right_candidate (straight continuation)
            ]
        )
        adj_list = {i: [] for i in range(6)}
        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices=set(range(6)),
            adj_list=adj_list,
            car_pos=np.array([0.0, 0.0]),
            car_heading=0.0,
        )

        left_lane = [0, 1]
        right_lane = [2, 3]
        left_candidate = 4
        right_candidate = 5

        result = left_right_decider(
            ctx, left_lane, right_lane, left_candidate, right_candidate
        )
        self.assertIn(
            result, [0, 1], "Result should be 0 (left) or 1 (right) for straight case"
        )

    def test_left_has_tighter_angle(self):
        """Test preferring left when left candidate creates smaller angle deviation."""
        # Setup: left lane going +x, right lane going +x
        # Left candidate creates small angle (0 deg turn)
        # Right candidate creates large angle (90 deg turn)
        cone_map = np.array(
            [
                [0.0, 1.0],  # left[0]
                [1.0, 1.0],  # left[1]
                [0.0, -1.0],  # right[0]
                [1.0, -1.0],  # right[1]
                [2.0, 1.0],  # left_candidate (0 deg continuation)
                [1.0, 0.0],  # right_candidate (90 deg turn right)
            ]
        )
        adj_list = {i: [] for i in range(6)}
        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices=set(range(6)),
            adj_list=adj_list,
            car_pos=np.array([0.0, 0.0]),
            car_heading=0.0,
        )

        left_lane = [0, 1]
        right_lane = [2, 3]
        left_candidate = 4
        right_candidate = 5

        result = left_right_decider(
            ctx, left_lane, right_lane, left_candidate, right_candidate
        )
        self.assertEqual(result, 0, "Should prefer left candidate with tighter angle")

    def test_right_has_tighter_angle(self):
        """Test preferring right when right candidate creates smaller angle deviation."""
        cone_map = np.array(
            [
                [0.0, 1.0],  # left[0]
                [1.0, 1.0],  # left[1]
                [0.0, -1.0],  # right[0]
                [1.0, -1.0],  # right[1]
                [1.0, 2.0],  # left_candidate (90 deg turn left)
                [2.0, -1.0],  # right_candidate (0 deg continuation)
            ]
        )
        adj_list = {i: [] for i in range(6)}
        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices=set(range(6)),
            adj_list=adj_list,
            car_pos=np.array([0.0, 0.0]),
            car_heading=0.0,
        )

        left_lane = [0, 1]
        right_lane = [2, 3]
        left_candidate = 4
        right_candidate = 5

        result = left_right_decider(
            ctx, left_lane, right_lane, left_candidate, right_candidate
        )
        self.assertEqual(result, 1, "Should prefer right candidate with tighter angle")

    def test_complex_curved_lanes(self):
        """Test with more complex curved lane geometry."""
        # Left lane curves right, right lane curves right
        cone_map = np.array(
            [
                [0.0, 2.0],  # left[0]
                [1.0, 2.5],  # left[1] (curving)
                [0.0, -2.0],  # right[0]
                [1.0, -2.0],  # right[1]
                [2.0, 2.7],  # left_candidate (continues left curve)
                [2.0, -1.5],  # right_candidate (sharper curve)
            ]
        )
        adj_list = {i: [] for i in range(6)}
        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices=set(range(6)),
            adj_list=adj_list,
            car_pos=np.array([0.0, 0.0]),
            car_heading=0.0,
        )

        left_lane = [0, 1]
        right_lane = [2, 3]
        left_candidate = 4
        right_candidate = 5

        result = left_right_decider(
            ctx, left_lane, right_lane, left_candidate, right_candidate
        )

        self.assertIn(result, [0, 1], "Should return a valid decision")

    def test_three_point_lanes(self):
        """Test with longer lane history (3 points each)."""
        cone_map = np.array(
            [
                [0.0, 2.0],  # left[0]
                [1.0, 2.0],  # left[1]
                [2.0, 2.0],  # left[2]
                [0.0, -2.0],  # right[0]
                [1.0, -2.0],  # right[1]
                [2.0, -2.0],  # right[2]
                [3.0, 2.0],  # left_candidate
                [3.0, -2.0],  # right_candidate
            ]
        )
        adj_list = {i: [] for i in range(8)}
        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices=set(range(8)),
            adj_list=adj_list,
            car_pos=np.array([0.0, 0.0]),
            car_heading=0.0,
        )

        left_lane = [0, 1, 2]
        right_lane = [3, 4, 5]
        left_candidate = 6
        right_candidate = 7

        result = left_right_decider(
            ctx, left_lane, right_lane, left_candidate, right_candidate
        )

        self.assertIn(result, [0, 1], "Should work with longer lane history")

    def test_lane_with_sharp_left_turn(self):
        """Test lane with a sharp left turn in the candidate."""
        cone_map = np.array(
            [
                [0.0, 0.0],  # left[0]
                [1.0, 0.0],  # left[1] (going +x)
                [0.0, -2.0],  # right[0]
                [1.0, -2.0],  # right[1] (going +x)
                [1.0, 1.0],  # left_candidate (sharp left, ~90 deg)
                [2.0, -2.0],  # right_candidate (straight, ~0 deg)
            ]
        )
        adj_list = {i: [] for i in range(6)}
        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices=set(range(6)),
            adj_list=adj_list,
            car_pos=np.array([0.0, 0.0]),
            car_heading=0.0,
        )

        left_lane = [0, 1]
        right_lane = [2, 3]
        left_candidate = 4
        right_candidate = 5

        result = left_right_decider(
            ctx, left_lane, right_lane, left_candidate, right_candidate
        )

        self.assertEqual(result, 1, "Should prefer right candidate with smoother lane")


class TestEnumeratePathPairs(unittest.TestCase):
    def test_straight_line(self):
        """Test 1: Straight lane with valid width.

        Expected to find one valid pair
        """

        cone_map = np.array(
            [
                [0.0, 2.0],  # left[0]
                [5.0, 2.0],  # left[1]
                [10.0, 2.0],  # left[2]
                [0.0, -2.0],  # right[0]
                [5.0, -2.0],  # right[1]
                [10.0, -2.0],  # right[2]
            ]
        )

        adj_list = {
            0: [1],
            1: [0, 2],
            2: [1],
            3: [4],
            4: [3, 5],
            5: [4],
        }

        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices=set(range(6)),
            adj_list=adj_list,
            car_pos=np.array([0.0, 0.0]),
            car_heading=0.0,
        )

        l_start, r_start = find_starting_vertices(ctx, max_range=5.0)
        self.assertIsNotNone(l_start)
        self.assertIsNotNone(r_start)

        results = enumerate_path_pairs(ctx, l_start, r_start)

        self.assertGreater(len(results), 0, "Should find valid path pair")

        candidate = results[0]
        self.assertGreater(len(candidate.left_path), 0)
        self.assertGreater(len(candidate.right_path), 0)
        self.assertTrue(candidate.is_valid)

    def test_lane_too_narrow_bt(self):
        """Test 2: Lane that becomes too narrow.

        Expected: Invalidate paths where width becomes to narrow.
        """
        cone_map = np.array(
            [
                [0.0, 2.0],  # left[0] - width 4m
                [5.0, 1.5],  # left[1] - width 3m
                [10.0, 0.5],  # left[2] - width 1m (too narrow!)
                [0.0, -2.0],  # right[0]
                [5.0, -1.5],  # right[1]
                [10.0, -0.5],  # right[2]
            ]
        )

        adj_list = {
            0: [1],
            1: [0, 2],
            2: [1],
            3: [4],
            4: [3, 5],
            5: [4],
        }

        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices=set(range(6)),
            adj_list=adj_list,
            car_pos=np.array([0.0, 0.0]),
            car_heading=0.0,
        )

        l_start, r_start = find_starting_vertices(ctx, max_range=5.0)
        self.assertIsNotNone(l_start)
        self.assertIsNotNone(r_start)

        results = enumerate_path_pairs(ctx, l_start, r_start)

        self.assertGreater(len(results), 0, "Should find valid path pair")

        for candidate in results:
            if 1 in candidate.left_path and 3 in candidate.right_path:
                self.assertFalse(candidate.is_valid),
                "Path with narrow section should fail width constraint and be marked invalid"

    def test_sharp_turn_bt(self):
        """Test 3: Sharp turn (>90º) should bt due to C_seg

        Expected: Paths with >90º turns should be marked invalid
        """
        # Left: (0,0), (5,0), (4,4)
        # Right: (0,-2.5), (5,-2.5), (10,-2.5), (8,3)
        cone_map = np.array(
            [
                [0.0, 0.0],  # left[0]
                [5.0, 0.0],  # left[1]
                [4.0, 4.0],  # left[2] - creates >90° turn
                [0.0, -2.5],  # right[0]
                [5.0, -2.5],  # right[1]
                [10.0, -2.5],  # right[2]
                [8.0, 3.0],  # right[3]
            ]
        )

        adj_list = {
            0: [1],
            1: [0, 2],
            2: [1],  # left path
            3: [4],
            4: [3, 5],
            5: [4, 6],
            6: [5],  # right path
        }

        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices=set(range(7)),
            adj_list=adj_list,
            car_pos=np.array([0.0, 0.0]),
            car_heading=0.0,
        )

        l_start, r_start = find_starting_vertices(ctx, max_range=5.0)
        self.assertIsNotNone(l_start)
        self.assertIsNotNone(r_start)

        results = enumerate_path_pairs(ctx, l_start, r_start)

        # Check that paths with sharp turns are filtered
        for candidate in results:
            if len(candidate.left_path) >= 3:
                # Left path should not have sharp turn
                self.assertTrue(
                    C_seg(candidate, ctx, side="left"),
                    "Valid paths should satisfy C_seg constraint on left side",
                )
            if len(candidate.right_path) >= 3:
                # Right path should not have sharp turn
                self.assertTrue(
                    C_seg(candidate, ctx, side="right"),
                    "Valid paths should satisfy C_seg constraint on right side",
                )

    def test_empty_graph(self):
        """Test 4: Empty graph with no valid starting points

        Expected: Returns empty list of candidates
        """

        cone_map = np.array([[60.0, 60.0]])
        adj_list = {0: []}

        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices={0},
            adj_list=adj_list,
            car_pos=np.array([0.0, 0.0]),
            car_heading=0.0,
        )

        l_start, r_start = find_starting_vertices(ctx, max_range=5.0)

        results = enumerate_path_pairs(ctx, left_start=l_start, right_start=r_start)
        self.assertEqual(
            len(results), 0, "Should return empty for no starting vertices"
        )


if __name__ == "__main__":
    unittest.main()
