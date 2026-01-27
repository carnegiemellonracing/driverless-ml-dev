from typing import List

import numpy as np
from perceptions.lane_detection.config import W_MAX, W_MIN
from perceptions.lane_detection.models import PerceptualFieldContext, Lane, Point
from perceptions.lane_detection.geo import get_segment_angle

def backtracking_decider(
    min_width: float, max_width: float, violation_in_fixed_set: bool
) -> bool:
    """Implements BTD
    Returns True if we have to backtrack (unfixable error)

    Args:
        min_width: Minimum width computed by online_lane_width
        max_width: Maximum width computed by online_lane_width
        violation_in_fixed: True if a width violation occurred in the 'fixed'
                            portion of matchings (before the boundary endpoints)

    Returns:
        True if must Backtrack (unrecoverable violation)
        False if can Continue (valid or recoverable)
    """
    # 1. Violation in Fixed Set → Backtrack
    #    Fixed matchings won't change as we extend the path.
    #    If they already violate constraints, this branch is dead.
    if violation_in_fixed_set:
        return True

        # 2. Too Narrow (min_width < W_MIN) → Backtrack
    #    Lane is too narrow. Extending the path can only make it
    #    narrower or keep it the same - never wider at this point.
    #    This is unrecoverable.
    if min_width < W_MIN:
        return True

    # 3. Too Wide (max_width > W_MAX) → Continue (Don't Backtrack)
    #    Lane is currently too wide, BUT this is recoverable.
    #    As we extend the path, the boundaries may converge and
    #    the width could decrease to acceptable levels.
    if max_width > W_MAX:
        return False

    # 4. Valid - all constraints satisfied
    return False


def next_vertex_decider(
    ctx: PerceptualFieldContext, path: Lane, car_heading: float
) -> List[int]:
    """Implements NVD (Eq 3) with Memoization.
    Returns neighbors sorted by smallest angle deviation.

    Args:
        ctx: Perceptual field context with cone_map, adj_list, and nvd_cache
        path: Current path as list of vertex indices
        car_heading: Car heading in radians

    Returns:
        List of neighbor indices sorted by ascending angle deviation
    """
    curr_idx = path[-1]

    # Determine the 'current vector'
    if len(path) == 1:
        # Special case: Use car heading if only 1 point
        vec_curr = np.array([np.cos(car_heading), np.sin(car_heading)])
        cache_key = (-1, curr_idx)
    else:
        prev_idx = path[-2]
        vec_curr = ctx.cone_map[curr_idx] - ctx.cone_map[prev_idx]
        cache_key = (prev_idx, curr_idx)

    # Check Cache
    if cache_key in ctx.nvd_cache:
        return ctx.nvd_cache[cache_key]

    neighbors = ctx.adj_list.get(curr_idx, [])
    if not neighbors:
        return []

    # Vectorized angle calculation
    neighbors_arr = np.array(neighbors)
    vecs_next = ctx.cone_map[neighbors_arr] - ctx.cone_map[curr_idx]

    norm_curr = np.linalg.norm(vec_curr)
    norms_next = np.linalg.norm(vecs_next, axis=1)

    # Avoid division by zero
    norms_next = np.where(norms_next == 0, 1e-6, norms_next)
    if norm_curr == 0:
        norm_curr = 1e-6

    dot_products = np.dot(vecs_next, vec_curr)
    cos_angles = np.clip(dot_products / (norm_curr * norms_next), -1.0, 1.0)
    abs_angles = np.abs(np.arccos(cos_angles))

    # Sort neighbors by ascending angle
    sorted_indices = np.argsort(abs_angles)
    sorted_neighbors = [neighbors[i] for i in sorted_indices]

    ctx.nvd_cache[cache_key] = sorted_neighbors
    return sorted_neighbors

def left_right_decider(ctx: PerceptualFieldContext, left_lane: Lane, right_lane: Lane,
                       left_candidate: int, right_candidate: int) -> int:
    """Left-Right decider
    Greedy Heuristic that decides whether it is better to add left or right candidate to
    their respective paths. Does so by creating two new path pairs, P_1' being 
    left_lane.append(left_candidate) and right_lane
    and P_2' being 
    left_lane and right_lane.append(right_candidate)

    Then for each path it computes 
    theta_l = angle between the segments:
         - left_lane[-2] to left_lane[-1]
         - left_lane[-1] to right_lane[-1]
    and 
    theta_l = angle between the segments:
         - right_lane[-2] to right_lane[-1]
          -right_lane[-1] to left_lane[-1]

    those values being named theta_l^1, theta_r^1 and theta_l^2, theta_r^2 respectively

    and outputs 0 (left) if |theta_r^1, theta_l^1| < |theta_r^2, theta_l^2|
    and 1 (right) otherwise


    Args:
        ctx: Perceptual field context with cone_map
        left_lane: List of point indices in left lane candidate
        right_lane:List of point indices in right lane candidate
        left_candidate: candidate left point index from NVD
        right_candidate: candidate right point index from NVD

    Returns:
        Integer 0,1 depending on whether it is better to add the left or right point
    """
    if len(left_lane) < 2 or len(right_lane) < 2:
        return 0 # Left default bias
    
    cone_map = ctx.cone_map
    
    left_lane_p1 = left_lane + [left_candidate]
    
    # theta_l^1: angle at the junction in the left lane
    # Segments: left_lane[-2]->left_lane[-1] and left_lane[-1]->left_candidate
    p1_left_prev = cone_map[left_lane[-2]]
    p1_left_curr = cone_map[left_lane[-1]]
    p1_left_next = cone_map[left_candidate]
    theta_l_1 = get_segment_angle(p1_left_prev, p1_left_curr, p1_left_next)
    
    # theta_r^1: angle in the cross connection
    # Segments: right_lane[-2]->right_lane[-1] and right_lane[-1]->left_lane[-1]
    p1_right_prev = cone_map[right_lane[-2]]
    p1_right_curr = cone_map[right_lane[-1]]
    p1_right_next = cone_map[left_lane[-1]]
    theta_r_1 = get_segment_angle(p1_right_prev, p1_right_curr, p1_right_next)
    
    # Scenario 2: Add right_candidate to right_lane
    # P_2' = left_lane, right_lane + [right_candidate]
    right_lane_p2 = right_lane + [right_candidate]
    
    # theta_l^2: angle in the cross connection
    # Segments: left_lane[-2]->left_lane[-1] and left_lane[-1]->right_lane[-1]
    p2_left_prev = cone_map[left_lane[-2]]
    p2_left_curr = cone_map[left_lane[-1]]
    p2_left_next = cone_map[right_lane[-1]]
    theta_l_2 = get_segment_angle(p2_left_prev, p2_left_curr, p2_left_next)
    
    # theta_r^2: angle at the junction in the right lane
    # Segments: right_lane[-2]->right_lane[-1] and right_lane[-1]->right_candidate
    p2_right_prev = cone_map[right_lane[-2]]
    p2_right_curr = cone_map[right_lane[-1]]
    p2_right_next = cone_map[right_candidate]
    theta_r_2 = get_segment_angle(p2_right_prev, p2_right_curr, p2_right_next)
    
    # Compare: choose based on which scenario has smaller angular deviation
    scenario_1_deviation = abs(theta_r_1) + abs(theta_l_1)
    scenario_2_deviation = abs(theta_r_2) + abs(theta_l_2)
    
    # Return 0 for left if scenario 1 is better, 1 for right if scenario 2 is better
    if scenario_1_deviation < scenario_2_deviation:
        return 0  # Prefer left
    else:
        return 1  # Prefer right


