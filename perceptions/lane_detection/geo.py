import numpy as np
from typing import List, Tuple
from models import Point, Lane
import math


"""
UTILS
"""
def within_range(point, car_pos, perceptual_field):
    if np.linalg.norm(car_pos - point) <= perceptual_field: # TODO for this and similar inequalities, do we want strict or equal
        return True
    return False
    
def within_cone(point: np.ndarray, car_pos: np.ndarray, heading_rad: float, cone_angle_rad: float) -> bool:
    """
    Checks if the given coordinates are within the "cone" around car heading with angle cone_angle and starting at (mid_x, mid_y)
    Compares angle formed by the slope of coordinates (relative to (mid_x, mid_y)) to car heading
    """
    relative_pt = point - car_pos
    car = np.array([np.cos(heading_rad), math.sin(heading_rad)])
    ip = np.dot(relative_pt, car)

    rounded = min(1, max(-1, ip / (np.linalg.norm(relative_pt) * np.linalg.norm(car))))
    
    theta = math.acos(rounded)
    
    if theta <= cone_angle_rad / 2:
        return True
    return False

def get_segment_angle(p1: Point, p2: Point, p3: Point) -> float:
    """Calculates the absolute deflection angle between two consecutive segments.

    Args:
        p1: Start point of the first segment.
        p2: End point of the first segment / Start point of the second segment.
        p3: End point of the second segment.

    Returns:
        The angle in degrees between the two segments. 0.0 indicates a straight line,
        90.0 indicates a right angle turn.
    """
    v1 = p2 - p1
    v2 = p3 - p2

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    # Calculate angle using dot product
    # We compute the angle between v1 and v2.
    # A straight line (v1 parallel to v2) gives 0 degrees.
    v1_u = v1 / norm_v1
    v2_u = v2 / norm_v2

    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)

    return np.degrees(angle_rad)


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Checks if segment (p1, p2) intersects with segment (p3, p4).

    Uses the Counter-Clockwise (CCW) method.
    """

    def ccw(A: Point, B: Point, C: Point) -> bool:
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def point_to_segment_distance(p: Point, s1: Point, s2: Point) -> Tuple[float, float]:
    """Calculates the shortest distance from point p to segment (s1, s2).

    Args:
        p: The query point.
        s1: Start point of the segment.
        s2: End point of the segment.

    Returns:
        A tuple containing:
        - distance: The Euclidean distance.
        - t: The projection parameter [0, 1] where proj = s1 + t * (s2 - s1).
    """
    segment_vec = s2 - s1
    l2 = np.sum(segment_vec**2)

    if l2 == 0:
        return np.linalg.norm(p - s1), 0.0

    t = np.dot(p - s1, segment_vec) / l2
    t = max(0.0, min(1.0, t))

    projection = s1 + t * segment_vec
    distance = np.linalg.norm(p - projection)

    return distance, t


def segment_to_segment_distance(
    s1_start: Point, s1_end: Point, s2_start: Point, s2_end: Point
) -> Tuple[float, float, float]:
    """Calculates the shortest distance between two segments.

    Returns:
        (distance, t1, t2):
        - distance: Euclidean distance
        - t1: param on segment 1 [0, 1]
        - t2: param on segment 2 [0, 1]
    """
    # 1. Check for intersection (Distance = 0)
    if segments_intersect(s1_start, s1_end, s2_start, s2_end):
        pass

    # 2. If no intersection, the closest pair MUST involve at least one endpoint.
    # We test all 4 endpoint-to-segment projections.

    # s1 start to s2
    d1, t_s2_1 = point_to_segment_distance(s1_start, s2_start, s2_end)
    # s1 end to s2
    d2, t_s2_2 = point_to_segment_distance(s1_end, s2_start, s2_end)

    # s2 start to s1
    d3, t_s1_1 = point_to_segment_distance(s2_start, s1_start, s1_end)
    # s2 end to s1
    d4, t_s1_2 = point_to_segment_distance(s2_end, s1_start, s1_end)

    # Find minimum
    min_d = min(d1, d2, d3, d4)

    if min_d == d1:
        return d1, 0.0, t_s2_1
    elif min_d == d2:
        return d2, 1.0, t_s2_2
    elif min_d == d3:
        return d3, t_s1_1, 0.0
    else:
        return d4, t_s1_2, 1.0


def get_point_at_param(lane: Lane, t: float) -> Point:
    """Interpolates a point on the lane at parameter t (Eq 7/8)."""
    i = int(np.floor(t))
    lam = t - i

    # Clamp to end
    if i >= len(lane) - 1:
        return lane[-1]

    p_i = lane[i]
    p_next = lane[i + 1]

    # P(i + lambda) = (1 - lambda)p_i + lambda * p_next
    return (1.0 - lam) * p_i + lam * p_next


"""
MAIN FUNCTIONS
"""


def C_seg(boundary: Lane, max_angle: float = 90.0) -> bool:
    """Verifies the Segment Consistency constraint (C_seg).

    Ensures that the absolute angle between any two consecutive line segments
    does not exceed `max_angle`.

    Args:
        boundary: List of points defining the lane boundary.
        max_angle: Maximum allowable angle in degrees.

    Returns:
        True if the constraint is satisfied, False otherwise.
    """
    if len(boundary) < 3:
        return True

    for i in range(len(boundary) - 2):
        if get_segment_angle(boundary[i], boundary[i + 1], boundary[i + 2]) > max_angle:
            return False

    return True


def C_poly(left: Lane, right: Lane) -> bool:
    """Verifies the Polynomial Consistency constraint (C_poly).

    Ensures that the polygon formed by the left and right boundaries does not
    intersect itself. The polygon is constructed by concatenating the left
    boundary with the reversed right boundary.

    Args:
        left: List of points defining the left boundary.
        right: List of points defining the right boundary.

    Returns:
        True if the polygon is simple (no self-intersections), False otherwise.
    """
    poly_points = left + right[::-1]
    n = len(poly_points)

    if n < 4:
        return True

    # Check for self-intersections between non-adjacent segments
    for i in range(n):
        p1 = poly_points[i]
        p2 = poly_points[(i + 1) % n]

        # Check against all other segments, skipping adjacent ones
        # Adjacent segments: (i-1, i) and (i+1, i+2)
        # We start checking from i+2.
        # We stop at n-1 (to avoid checking last segment against first if they are adjacent,
        # but here (n-1, 0) is adjacent to (0, 1) so we stop at n-2 effectively for i=0).

        for j in range(i + 2, n):
            # If we are at the last segment (n-1, 0), we shouldn't check against (0, 1)
            if i == 0 and j == n - 1:
                continue

            p3 = poly_points[j]
            p4 = poly_points[(j + 1) % n]

            if segments_intersect(p1, p2, p3, p4):
                return False

    return True


# def compute_matchings(
#     left: Lane, right: Lane, start_u: float, start_v: float
# ) -> List[MatchingPoint]:
#     """Computes matching points using Eq 9 and 11 logic.

#     Iterates through the lane starting from start_u/start_v to the end.
#     """
#     new_matches = []

#     start_idx_L = int(np.floor(start_u))
#     start_idx_R = int(np.floor(start_v))

#     len_L = len(left)
#     len_R = len(right)

#     # 1. Left Vertices (k=0, s=0) against full Right lane
#     # Iterate left vertices starting from start_u
#     for i in range(start_idx_L, len_L):
#         p_query = left[i]

#         # Search against all Right segments (Eq 8: target is complete polygonal chain)
#         best_dist = float("inf")
#         best_v = 0.0

#         for j in range(len_R - 1):
#             dist, t = point_to_segment_distance(p_query, right[j], right[j + 1])
#             if dist < best_dist:
#                 best_dist = dist
#                 best_v = float(j) + t

#         new_matches.append((float(i), best_v))

#     # 2. Left Segments (k=1, s=0) against full Right lane
#     for i in range(start_idx_L, len_L - 1):
#         # Search against all Right segments
#         best_dist = float("inf")
#         best_params = (0.0, 0.0)  # (t_on_left_seg, t_on_right_seg)

#         for j in range(len_R - 1):
#             d, t_l, t_r = segment_to_segment_distance(
#                 left[i], left[i + 1], right[j], right[j + 1]
#             )
#             if d < best_dist:
#                 best_dist = d
#                 best_params = (t_l, t_r)

#         u_global = float(i) + best_params[0]
#         v_global = float(j) + best_params[1]
#         new_matches.append((u_global, v_global))

#     # 3. Right Vertices (k=0, s=1) against full Left lane
#     for j in range(start_idx_R, len_R):
#         p_query = right[j]

#         best_dist = float("inf")
#         best_u = 0.0

#         for i in range(len_L - 1):
#             dist, t = point_to_segment_distance(p_query, left[i], left[i + 1])
#             if dist < best_dist:
#                 best_dist = dist
#                 best_u = float(i) + t

#         new_matches.append((best_u, float(j)))

#     # 4. Right Segments (k=1, s=1)
#     for i in range(start_idx_L, len_L):
#         p_query = left[i]

#         best_dist = float("inf")
#         best_v = 0.0

#         for j in range(len_R - 1):
#             dist, t = point_to_segment_distance(p_query, right[j], right[j + 1])
#             if dist < best_dist:
#                 best_dist = dist
#                 best_v = float(j) + t

#         new_matches.append((float(i), best_v))

#     return new_matches


# def OnlineLW(
#     L: Lane, R: Lane, M_fixed_prev: List[MatchingPoint]
# ) -> Tuple[List[MatchingPoint], List[MatchingPoint]]:
#     """
#     Algorithm 3: Online algorithm for lane width calculation.

#     Implements the pseudo-code strictly:
#     1. Determine start points (u_s, v_s) from M_fixed_prev.
#     2. Compute new matchings M' starting from (u_s, v_s).
#     3. Sort M' and split it into M'_fixed and M_mut.
#     4. Update M_fixed by uniting M_fixed_prev and M'_fixed.

#     Args:
#         L: Left lane boundary (polygonal chain).
#         R: Right lane boundary (polygonal chain).
#         M_fixed_prev: Fixed matching points from the last iteration (sorted).

#     Returns:
#         (M_fixed_t, M_mut_t): Updated fixed and mutable matching points.
#     """

#     # Line 4-5: Check if previous fixed set is empty
#     if not M_fixed_prev:
#         u_s, v_s = 0.0, 0.0
#     else:
#         # Line 7: Start from the last fixed point
#         u_s, v_s = M_fixed_prev[-1]

#     # Line 8: Compute matching for L, R starting from u_s, v_s (Eq. 11)
#     # This function (provided in the previous step) calculates the union of the 4 sub-searches.
#     M_prime = compute_matchings(L, R, u_s, v_s)

#     # Line 9: Sort M' and split into M'_fixed and M_mut^t
#     # The paper notes that Eq 11 doesn't guarantee monotonicity, so we must sort lexicographically.
#     M_prime.sort(key=lambda p: (p[0], p[1]))

#     # Logic for splitting M' (defined in text section VI.A.2 ):
#     # "The set... is split... based on the first matching point which matches at least one end of either boundary"
#     # i.e., u == |L|-1 OR v == |R|-1.

#     max_u = float(len(L) - 1)
#     max_v = float(len(R) - 1)

#     # Default split index is the end (all points are fixed) if no boundary end is reached
#     split_idx = len(M_prime)

#     for k, (u, v) in enumerate(M_prime):
#         # Use epsilon for float comparison
#         if u >= max_u - 1e-5 or v >= max_v - 1e-5:
#             split_idx = k
#             break

#     M_prime_fixed = M_prime[:split_idx]
#     M_mut_t = M_prime[split_idx:]

#     # Line 10: M_fixed^t <- M_fixed^{t-1} U M'_fixed
#     # Since M_prime started searching from the end of M_fixed_prev, we append M'_fixed.
#     # Note: If M_prime re-found the start point (u_s, v_s), strictly ensure no duplicates if necessary,
#     # though typically the search starts *after* or inclusive. Assuming list concatenation preserves order here.
#     M_fixed_t = M_fixed_prev + M_prime_fixed

#     # Line 11: return M_fixed^t, M_mut^t
#     return M_fixed_t, M_mut_t
