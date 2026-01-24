import unittest
import numpy as np
from geo import C_seg, C_poly, get_segment_angle
from models import LaneCandidate, GlobalContext


class TestGeometricConstraints(unittest.TestCase):

    def test_c_seg(self):
        # Straight line - Pass
        map_points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        context = GlobalContext(map_points)
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
        context_90 = GlobalContext(map_points_90)
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
        context_sharp = GlobalContext(map_points_sharp)
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
        context_box = GlobalContext(map_points_box)
        candidate_box = LaneCandidate(
            left_path=[0, 1], right_path=[2, 3], left_visited=set(), right_visited=set()
        )
        self.assertTrue(
            C_poly(candidate_box, context_box), "Failed simple box - should pass"
        )

        # Bowtie (self intersection) - Fail
        # Left crosses right
        # left: [(0, 1), (2, -1)], right: [(0, -1), (2, 1)]
        map_points_cross = np.array(
            [[0.0, 1.0], [2.0, -1.0], [0.0, -1.0], [2.0, 1.0]]
        )
        context_cross = GlobalContext(map_points_cross)
        candidate_cross = LaneCandidate(
            left_path=[0, 1], right_path=[2, 3], left_visited=set(), right_visited=set()
        )
        self.assertFalse(
            C_poly(candidate_cross, context_cross), "Failed bowtie - should fail"
        )


if __name__ == "__main__":
    unittest.main()
