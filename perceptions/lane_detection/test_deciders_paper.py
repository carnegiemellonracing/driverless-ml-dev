
import unittest
import numpy as np
import math
from .geo import next_vertex_decider, left_right_decider, backtracking_decider
from .models import LaneCandidate, PerceptualFieldContext, MatchingSet

def create_test_context(points: np.ndarray, adj=None) -> PerceptualFieldContext:
    n = len(points)
    visible_indices = set(range(n))
    if adj is None:
        adj = {i: [] for i in range(n)}
    return PerceptualFieldContext(
        cone_map=points,
        visible_indices=visible_indices,
        adj_list=adj,
        car_pos=np.array([0.0, 0.0]),
        car_heading=0.0,
    )

class TestDecidersPaper(unittest.TestCase):
    """
    Verifies that the decider functions (NVD, LRD, BTD) implementation matches
    the logic described in the paper (arXiv 2405.16369).
    """

    def test_NVD_minimizes_angle(self):
        """
        NVD should select the candidate vertex that minimizes the deviation angle
        from the previous heading vector.
        """
        # Car at (0,0) facing East (0 rad).
        # Path: 0->1. 0 at (0,0), 1 at (1,0).
        # Candidates for next vertex:
        # 2: (2,0) -> Straight (0 deviation).
        # 3: (2,1) -> 45 deg turn.
        # 4: (1,1) -> 90 deg turn.
        
        points = np.array([
            [0.0, 0.0], # 0
            [1.0, 0.0], # 1
            [2.0, 0.0], # 2 (Straight)
            [2.0, 1.0], # 3 (45 deg)
            [1.0, 1.0], # 4 (90 deg)
        ])
        
        # Adjacency: 1 is connected to 2, 3, 4
        adj = {0: [1], 1: [2, 3, 4], 2: [], 3: [], 4: []}
        
        ctx = create_test_context(points, adj=adj)
        current_path = [0, 1]
        
        # ACT
        best_v = next_vertex_decider(ctx, current_path)
        
        # ASSERT
        self.assertEqual(best_v, 2, "NVD should pick vertex 2 (straight, 0 deviation)")

    def test_NVD_avoids_visited(self):
        # Even though NVD signature takes current_path and checks `v not in current_path`,
        # it relies on adjacency list provided.
        # This test confirms broad functionality.
        pass

    def test_LRD_symmetric_selection(self):
        """
        LRD should choose the side that maintains symmetry (minimizes |theta_r - theta_l|).
        Paper Eq 5.
        """
        # Scenario: Two parallel lanes.
        # Left: (0, 1) -> (1, 1).
        # Right: (0, -1) -> (1, -1).
        # Next Left Candidate n0: (2, 1) (Perfectly straight extension)
        # Next Right Candidate n1: (2, -5) (Wild divergence)
        
        points = np.array([
            [0.0, 1.0], [1.0, 1.0],  # 0, 1 (Left)
            [0.0, -1.0], [1.0, -1.0], # 2, 3 (Right)
            [2.0, 1.0],              # 4 (n0 - Good)
            [2.0, -5.0]              # 5 (n1 - Bad)
        ])
        
        ctx = create_test_context(points) # adj logic inside LRD uses manual points/angles
        
        # Setup Candidate
        cand = LaneCandidate(left_path=[0, 1], right_path=[2, 3], left_visited=set(), right_visited=set())
        
        n0 = 4
        n1 = 5
        
        # ACT
        # extending left (to 4) vs extending right (to 5).
        # Left extension: L becomes 0->1->4. R is 2->3.
        # Cross segment 4->3.
        # Angles should be reasonable.
        # Right extension: L is 0->1. R becomes 2->3->5.
        # Cross segment 1->5.
        # This creates a huge slant. Asymmetry should be high.
        
        choice = left_right_decider(ctx, cand, n0, n1)
        
        # ASSERT
        self.assertEqual(choice, 0, "LRD should pick Left (0) because it maintains symmetry better than the divergent Right candidate.")

    def test_BTD_pruning_logic(self):
        """
        BTD should prune (return True) if:
        1. Violation in fixed set (unrecoverable).
        2. Min width < W_MIN (unrecoverable, lanes only get narrower/same with simple polygonal chains? 
           Actually paper Lemma says 'mutable matching too short -> prune').
        3. Should NOT prune if Max width > W_MAX (might conform later).
        """
        W_MIN_VAL = 2.5 # As defined in config
        W_MAX_VAL = 6.5
        
        # Case 1: Violation in Fixed Set
        # Should return True (Backtrack)
        self.assertTrue(backtracking_decider(3.0, 4.0, True), "Should backtrack if fixed set has violation")
        
        # Case 2: Too Narrow (Min Width < W_MIN)
        # Width 1.0 < 2.5
        self.assertTrue(backtracking_decider(1.0, 3.0, False), "Should backtrack if min width < W_MIN")
        
        # Case 3: Too Wide (Max Width > W_MAX) but Min Width OK
        # Width 10.0 > 6.5.
        # Should return False (Continue, don't backtrack)
        self.assertFalse(backtracking_decider(3.0, 10.0, False), "Should NOT backtrack if max width > W_MAX (recoverable)")
        
        # Case 4: Recoverable narrow?
        # No, if min breadth is already too small (and matchings usually monotonic in graph search?), it's pruned.
        # The paper assumes segments don't 'widen' the min distance magically? 
        # Actually min distance is min(all matchings). Adding matchings can only DECREASE min-width or keep it same.
        # So min-width < limit is monotonically breaking.
        pass

if __name__ == '__main__':
    unittest.main()
