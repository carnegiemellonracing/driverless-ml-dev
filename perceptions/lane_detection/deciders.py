from typing import List

import numpy as np

from perceptions.lane_detection.config import W_MAX, W_MIN
from perceptions.lane_detection.models import PerceptualFieldContext


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
    ctx: PerceptualFieldContext, path: List[int], car_heading: float
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
