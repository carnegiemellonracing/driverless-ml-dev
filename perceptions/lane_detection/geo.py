import numpy as np
from typing import List, Tuple
from models import Point, Lane, LaneCandidate, GlobalContext
import math


"""
UTILS
"""


def within_range(point, car_pos, perceptual_field):
    if (
        np.linalg.norm(car_pos - point) <= perceptual_field
    ):  # TODO for this and similar inequalities, do we want strict or equal
from typing import List, Tuple, Dict, Set, Optional

# Import dataclasses and config from models
from perceptions.lane_detection.models import MatchingSet, LaneCandidate, GlobalContext
from perceptions.lane_detection.config import D_MAX, W_MIN, W_MAX, PHI_MAX

def deprecated(reason):
    def decorator(func):
        return func
    return decorator

# Export list for clean imports
__all__ = [
    "point_to_segment_distance",
    "construct_adjacency_list",
    "find_matching_segments",
    "find_matching_points",
    "enumerate_path_pairs",
    "enumerate_path_pairs_v2",
    "next_vertex_decider",
    "left_right_decider",
    "constraint_decider",
    "compute_features",
    "generate_feature_pairs",
    "OnlineLW",
    "calculate_segment_angle",
    "matching_to_distance",
    "point_to_polygonal_chain_distance",
    "segment_to_polygonal_chain_distance",
    "C_seg",
    "C_poly",
    "C_width",
    "within_range",
    "within_cone"
]


def within_range(point, car_pos, max_range):
    """Checks if a point is within a certain range of the car."""
    return np.linalg.norm(point - car_pos) <= max_range


def within_cone(point, car_pos, heading, cone_angle):
    """Checks if a point is within the car's field of view cone."""
    if np.array_equal(point, car_pos):
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
    # Calculate angle between heading vector and point vector
    dot = np.clip(np.dot(v_point / norm_point, v_heading), -1.0, 1.0)
    angle = np.arccos(dot)
    return angle <= (cone_angle / 2)


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

    v1_u = v1 / norm_v1
    v2_u = v2 / norm_v2

    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)

    return angle_rad


def point_to_segment_distance(point, seg_start, seg_end):
    """
    Calculate perpendicular distance from a point to a line segment.
    Uses projection with clamping to segment bounds.
    """
    point = np.array(point)
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)

    # Vector from segment start to end
    seg_vec = seg_end - seg_start
    # Vector from segment start to point
    point_vec = point - seg_start

    # Compute projection parameter t
    seg_length_sq = np.dot(seg_vec, seg_vec)

    # Handle degenerate segment (start == end)
    if seg_length_sq < 1e-8:
        distance = np.linalg.norm(point_vec)
        return distance, seg_start

    # Project point onto line containing segment
    t = np.dot(point_vec, seg_vec) / seg_length_sq

    # Clamp t to [0, 1] to keep projection on segment
    t_clamped = np.clip(t, 0.0, 1.0)

    # Compute projection point on segment
    projection = seg_start + t_clamped * seg_vec

    # Compute distance from point to projection
    distance = np.linalg.norm(point - projection)

    return distance, projection


def segment_to_segment_distance(s1_start, s1_end, s2_start, s2_end):
    """Calculates the shortest distance between two segments."""
    if line_segments_intersect(s1_start, s1_end, s2_start, s2_end):
        return 0.0

    # Test all 4 endpoint-to-segment projections
    d1, _ = point_to_segment_distance(s1_start, s2_start, s2_end)
    d2, _ = point_to_segment_distance(s1_end, s2_start, s2_end)
    d3, _ = point_to_segment_distance(s2_start, s1_start, s1_end)
    d4, _ = point_to_segment_distance(s2_end, s1_start, s1_end)

    return min(d1, d2, d3, d4)


def point_to_polygonal_chain_distance(point, chain):
    """Calculates minimum distance from a point to a polygonal chain."""
    min_dist = float("inf")
    if len(chain) < 2:
        return min_dist
        
    for i in range(len(chain) - 1):
        dist, _ = point_to_segment_distance(point, chain[i], chain[i+1])
        if dist < min_dist:
            min_dist = dist
    return min_dist


def segment_to_polygonal_chain_distance(seg_start, seg_end, chain):
    """Calculates minimum distance from a segment to a polygonal chain."""
    min_dist = float("inf")
    if len(chain) < 2:
        return min_dist
        
    for i in range(len(chain) - 1):
        dist = segment_to_segment_distance(seg_start, seg_end, chain[i], chain[i+1])
        if dist < min_dist:
            min_dist = dist
    return min_dist


def matching_to_distance(u, v, left_path, right_path, points):
    """Calculates distance between matched points u (on left) and v (on right)."""
    # Simply distance between points u and v? 
    # The paper implies matches (u, v) can be indices or fractional indices.
    # Assuming integer indices for fixed matchings based on usage in Cwidth_bt.
    p_u = np.array(points[left_path[u]])
    p_v = np.array(points[right_path[v]])
    return np.linalg.norm(p_u - p_v)


def construct_adjacency_list(points, dmax):
    adjacency_list = {i: [] for i in range(len(points))}

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if np.linalg.norm(np.array(points[i]) - np.array(points[j])) <= dmax:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)

    return adjacency_list


def find_matching_segments(left_path, right_path, points, fixed_matches=None):
    """
    Hybrid approach: Uses nearest neighbor search for point matching combined
    with perpendicular distance to segments for accurate width calculation.
    """
    if fixed_matches is None:
        fixed_matches = set()

    left_coords = np.array([points[i] for i in left_path])
    right_coords = np.array([points[i] for i in right_path])

    matching_lines = []

    # Need at least 2 points to form segments
    if len(right_path) < 2:
        # Fallback to point-to-point for edge case
        for i, left_point in enumerate(left_coords):
            if len(right_coords) > 0:
                distances = np.linalg.norm(right_coords - left_point, axis=1)
                nearest_idx = np.argmin(distances)
                matching_lines.append(
                    {
                        "left_idx": i,
                        "right_seg": (nearest_idx, nearest_idx),
                        "left_point": left_point,
                        "projection_point": right_coords[nearest_idx],
                        "width": distances[nearest_idx],
                        "is_fixed": False,
                    }
                )
        return matching_lines

    # For each point on the left boundary, find nearest segment on right boundary
    for i, left_point in enumerate(left_coords):
        min_distance = float("inf")
        best_seg_idx = 0
        best_projection = None

        # Check all segments on the right boundary
        for j in range(len(right_path) - 1):
            seg_start = right_coords[j]
            seg_end = right_coords[j + 1]

            # Calculate perpendicular distance to this segment
            distance, projection = point_to_segment_distance(
                left_point, seg_start, seg_end
            )

            if distance < min_distance:
                min_distance = distance
                best_seg_idx = j
                best_projection = projection

        # Check if this is a fixed match
        seg_tuple = (best_seg_idx, best_seg_idx + 1)
        # Simplify checking: is this left_idx involved in any fixed match?
        # The exact structure of fixed_matches needs to be consistent. 
        # Assuming fixed_matches is set of (u, v) indices for now.
        is_fixed = False
        
        matching_lines.append(
            {
                "left_idx": i,
                "right_seg": seg_tuple,
                "left_point": left_point,
                "projection_point": best_projection,
                "width": min_distance,
                "is_fixed": is_fixed,
            }
        )

    return matching_lines


def find_matching_points(left_path, right_path, points, fixed_matches=None):
    """Old point-to-point matching using simple Euclidean distance."""
    if fixed_matches is None:
        fixed_matches = set()

    left_coords = np.array([points[i] for i in left_path])
    right_coords = np.array([points[i] for i in right_path])

    matching_lines = []

    for i, left_point in enumerate(left_coords):
        distances = np.linalg.norm(right_coords - left_point, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_right_point = right_coords[nearest_idx]

        is_fixed = (i, nearest_idx) in fixed_matches

        matching_lines.append(
            {
                "left_idx": i,
                "right_idx": nearest_idx,
                "left_point": left_point,
                "right_point": nearest_right_point,
                "width": distances[nearest_idx],
                "is_fixed": is_fixed,
            }
        )

    return matching_lines


def line_segments_intersect(p1, p2, p3, p4):
    """Check if two line segments intersect using cross product method"""
    p1, p2, p3, p4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def OnlineLW(left_path: List[int], right_path: List[int], points, 
             prev_matching: Optional[MatchingSet] = None) -> MatchingSet:
    """
    Algorithm 3: Online algorithm for lane width calculation.
    Returns a MatchingSet with fixed and mutable matchings.
    
    Args:
        left_path: List of point indices for left boundary
        right_path: List of point indices for right boundary  
        points: List/array of (x, y) coordinates
        prev_matching: Previous MatchingSet state (for incremental update)
    
    Returns:
        MatchingSet with updated fixed/mutable matchings
    """
    if prev_matching is None:
        prev_matching = MatchingSet()
        u_s, v_s = 0, 0
    else:
        return d4, t_s1_2, 1.0


def get_point_at_param(
    lane_candidate: LaneCandidate, context: GlobalContext, t: float, side: str = "left"
) -> Point:
    """Interpolates a point on the lane at parameter t (Eq 7/8).

    Args:
        lane_candidate: The LaneCandidate containing left and right paths
        context: GlobalContext with the map points
        t: Parameter value for interpolation
        side: Which boundary to interpolate ("left" or "right")

    Returns:
        Interpolated point on the specified boundary
    """
    # Select the appropriate path based on side
    path = lane_candidate.left_path if side == "left" else lane_candidate.right_path

    i = int(np.floor(t))
    lam = t - i

    # Clamp to end
    if i >= len(path) - 1:
        return context.map_points[path[-1]]

    # Get the actual points from the global map using indices
    p_i = context.map_points[path[i]]
    p_next = context.map_points[path[i + 1]]

    # P(i + lambda) = (1 - lambda)p_i + lambda * p_next
    return (1.0 - lam) * p_i + lam * p_next


"""
MAIN FUNCTIONS
"""


def C_seg(
    lane_candidate: LaneCandidate,
    context: GlobalContext,
    side: str = "left",
    max_angle: float = 90.0,
) -> bool:
    """Verifies the Segment Consistency constraint (C_seg) for a LaneCandidate.

    Ensures that the absolute angle between any two consecutive line segments
    does not exceed `max_angle`.

    Args:
        lane_candidate: The LaneCandidate containing left and right paths.
        context: GlobalContext with the map points.
        side: Which boundary to check ("left" or "right").
        max_angle: Maximum allowable angle in degrees.

    Returns:
        True if the constraint is satisfied, False otherwise.
    """
    # Get the path indices based on side
    path = lane_candidate.left_path if side == "left" else lane_candidate.right_path

    if len(path) < 3:
        return True

    # Check angles between consecutive segments
    for i in range(len(path) - 2):
        p1 = context.map_points[path[i]]
        p2 = context.map_points[path[i + 1]]
        p3 = context.map_points[path[i + 2]]

        if get_segment_angle(p1, p2, p3) > max_angle:
            return False
    return True


def C_poly(lane_candidate: LaneCandidate, context: GlobalContext) -> bool:
    """Verifies the Polynomial Consistency constraint (C_poly) for a LaneCandidate.

    Ensures that the polygon formed by the left and right boundaries does not
    intersect itself. The polygon is constructed by concatenating the left
    boundary with the reversed right boundary.

    Args:
        lane_candidate: The LaneCandidate containing left and right paths.
        context: GlobalContext with the map points.

    Returns:
        True if the polygon is simple (no self-intersections), False otherwise.
    """
    # Convert paths to actual points
    left_points = [context.map_points[idx] for idx in lane_candidate.left_path]
    right_points = [context.map_points[idx] for idx in lane_candidate.right_path]

    # Construct polygon by concatenating left with reversed right
    poly_points = left_points + right_points[::-1]
    n = len(poly_points)

    if n < 4:
def C_poly(left_path, right_path, points):
    """Polygon Consistency Constraint."""
    if len(left_path) < 2 or len(right_path) < 2:
        return True
        
    polygon_indices = list(left_path) + list(reversed(right_path))
    polygon_points = [np.array(points[idx]) for idx in polygon_indices]
    n = len(polygon_points)

    for i in range(n):
        p1 = poly_points[i]
        p2 = poly_points[(i + 1) % n]

        # Check against all other segments, skipping adjacent ones
        for j in range(i + 2, n):
            if j == (i + 1) % n or i == (j + 1) % n:
                continue
            if line_segments_intersect(polygon_points[i], polygon_points[(i+1)%n], 
                                     polygon_points[j], polygon_points[(j+1)%n]):
                return False
    return True


def C_width(left_path, right_path, points, wmin=W_MIN, wmax=W_MAX):
    """Width Consistency Constraint."""
    if len(left_path) < 1 or len(right_path) < 1:
        return True
        
    left_coords = [np.array(points[i]) for i in left_path]
    right_coords = [np.array(points[i]) for i in right_path]
    
    # Check simple point-to-chain distances
    for p in left_coords:
        if not (wmin < point_to_polygonal_chain_distance(p, right_coords) < wmax):
            return False
            
    for p in right_coords:
        if not (wmin < point_to_polygonal_chain_distance(p, left_coords) < wmax):
            return False
            
    return True


def constraint_decider(path_pair, points):
    left, right = path_pair
    return C_seg(left, points) and C_seg(right, points) and \
           C_width(left, right, points) and C_poly(left, right, points)


def next_vertex_decider(current_path, adjacent_vertices, points, heading_vector=None):
    if not adjacent_vertices:
        return None
        
    current_idx = current_path[-1]
    current_point = np.array(points[current_idx])
    
    best_v = adjacent_vertices[0]
    min_angle = float("inf")
    
    # If starting, use heading
    if len(current_path) == 1:
        if heading_vector is None:
             return best_v
        v_prev = heading_vector
    else:
        prev_idx = current_path[-2]
        prev_point = np.array(points[prev_idx])
        v_prev = current_point - prev_point
        
    for v_idx in adjacent_vertices:
        next_point = np.array(points[v_idx])
        v_next = next_point - current_point
        
        # Calculate angle
        norm_prev = np.linalg.norm(v_prev)
        norm_next = np.linalg.norm(v_next)
        
        if norm_prev == 0 or norm_next == 0:
            angle = 0
        else:
            dot = np.clip(np.dot(v_prev, v_next) / (norm_prev * norm_next), -1.0, 1.0)
            angle = np.arccos(dot)
            
        if angle < min_angle:
            min_angle = angle
            best_v = v_idx
            
    return best_v


def left_right_decider(left_path, right_path, points, left_candidate, right_candidate):
    """Returns 0 for left, 1 for right based on smaller deviation."""
    l_curr = np.array(points[left_path[-1]])
    r_curr = np.array(points[right_path[-1]])
    l_next = np.array(points[left_candidate])
    r_next = np.array(points[right_candidate])
    
    # Angle deviation for left extension
    # Compare (L_curr -> L_next) vs (L_curr -> R_curr)
    l_vec = l_next - l_curr
    cross_vec = r_curr - l_curr # Vector to other side
    
    # Just a heuristic placeholder roughly matching the paper's intent
    # Real implementation needs exact vectors
    return 0 


def enumerate_path_pairs(graph, sl, sr, itmax=100):
   """Placeholder for older enumerate function"""
   return []


def enumerate_path_pairs_v2(graph, points, paths, visited, heading_vector, it, itmax=2500):
    """Recursive enumeration of path pairs."""
    if it > itmax:
        return []
        
    left_path, right_path = paths
    l_last, r_last = left_path[-1], right_path[-1]
    
    l_adj = [v for v in graph[l_last] if v not in visited and v not in left_path]
    r_adj = [v for v in graph[r_last] if v not in visited and v not in right_path]
    
    results = []
    
    # Base case: no more extensions possible
    if not l_adj and not r_adj:
        if constraint_decider(paths, points):
            return [paths]
        return []

    # Choose side to extend (simple alternation or heuristic)
    # Just a basic DFS structure for now
    if l_adj:
        next_l = next_vertex_decider(left_path, l_adj, points, heading_vector)
        if next_l is not None:
             new_paths = (left_path + [next_l], right_path)
             if constraint_decider(new_paths, points):
                 if len(new_paths[0]) > 2 and len(new_paths[1]) > 2: # Min length check
                     results.append(new_paths)
                 results.extend(enumerate_path_pairs_v2(graph, points, new_paths, visited | {next_l}, heading_vector, it+1, itmax))

    if r_adj:
        next_r = next_vertex_decider(right_path, r_adj, points, heading_vector)
        if next_r is not None:
             new_paths = (left_path, right_path + [next_r])
             if constraint_decider(new_paths, points):
                 if len(new_paths[0]) > 2 and len(new_paths[1]) > 2:
                     results.append(new_paths)
                 results.extend(enumerate_path_pairs_v2(graph, points, new_paths, visited | {next_r}, heading_vector, it+1, itmax))
                 
    return results


def compute_features(path_pair, points):
    """
    Computes 8 geometric features for a lane candidate.
    
    Features:
    1. Mean Width
    2. Std Dev Width
    3. Mean Segment Angle (Smoothness)
    4. Std Dev Segment Angle
    5. Max Segment Angle
    6. Left Lane Length
    7. Right Lane Length
    8. Width Range (Max - Min)
    """
    left_path, right_path = path_pair
    
    # 1. Width Statistics
    matching_segments = find_matching_segments(left_path, right_path, points)
    widths = [m["width"] for m in matching_segments]
    
    if not widths:
        mu_w, sigma_w, w_range = 0.0, 0.0, 0.0
    else:
        mu_w = np.mean(widths)
        sigma_w = np.std(widths)
        w_range = np.max(widths) - np.min(widths)

    # 2. Angle Statistics (Curvature/Smoothness)
    angles = []
    
    # Left path angles
    for i in range(len(left_path) - 2):
        p1 = np.array(points[left_path[i]])
        p2 = np.array(points[left_path[i + 1]])
        p3 = np.array(points[left_path[i + 2]])
        angles.append(calculate_segment_angle(p1, p2, p3))
        
    # Right path angles
    for i in range(len(right_path) - 2):
        p1 = np.array(points[right_path[i]])
        p2 = np.array(points[right_path[i + 1]])
        p3 = np.array(points[right_path[i + 2]])
        angles.append(calculate_segment_angle(p1, p2, p3))
        
    if not angles:
        mu_alpha, sigma_alpha, max_alpha = 0.0, 0.0, 0.0
    else:
        mu_alpha = np.mean(angles)
        sigma_alpha = np.std(angles)
        max_alpha = np.max(angles)
        
    # 3. Length Statistics
    def path_length(path):
        length = 0.0
        for i in range(len(path) - 1):
            p1 = np.array(points[path[i]])
            p2 = np.array(points[path[i+1]])
            length += np.linalg.norm(p2 - p1)
        return length
        
    len_left = path_length(left_path)
    len_right = path_length(right_path)
    
    return [mu_w, sigma_w, mu_alpha, sigma_alpha, max_alpha, len_left, len_right, w_range]

def generate_feature_pairs(path_pairs, points):
    """
    Generates feature vectors for a list of path pairs.
    Returns a list of feature lists (one per path pair).
    """
    features_list = []
    for pair in path_pairs:
        features_list.append(compute_features(pair, points))
    return features_list



# NOTE: Visualization functions have been removed to keep this module pure.
# For visualization, import from a separate visualization module or use:
#   from perceptions.lane_detection.visualization import visualize_path_pairs


# =============================================================================
# MODEL-BASED FUNCTIONS (using GlobalContext, LaneCandidate, MatchingSet)
# =============================================================================

def next_vertex_decider_ctx(
    ctx: GlobalContext,
    current_path: List[int],
    heading_vector: Optional[np.ndarray] = None
) -> Optional[int]:
    """
    Selects the next vertex to extend the path using GlobalContext.
    Uses NVD cache when available.
    
    Args:
        ctx: GlobalContext with map_points, adj_list, and nvd_cache
        current_path: Current path as list of point indices
        heading_vector: Initial heading vector (used when path has only 1 point)
    
    Returns:
        Index of next vertex to visit, or None if no valid extension
    """
    if not current_path:
        return None
        
    current_idx = current_path[-1]
    adjacent = [v for v in ctx.adj_list.get(current_idx, []) if v not in current_path]
    
    if not adjacent:
        return None
    
    # Check NVD cache first
    if len(current_path) >= 2:
        prev_idx = current_path[-2]
        cache_key = (prev_idx, current_idx)
        if cache_key in ctx.nvd_cache:
            # Return first unvisited neighbor from cached sorted list
            for v in ctx.nvd_cache[cache_key]:
                if v in adjacent:
                    return v
    
    # Compute NVD manually
    current_point = ctx.map_points[current_idx]
    
    if len(current_path) == 1:
        if heading_vector is None:
            return adjacent[0]
        v_prev = heading_vector
    else:
        prev_point = ctx.map_points[current_path[-2]]
        v_prev = current_point - prev_point
    
    best_v = adjacent[0]
    min_angle = float("inf")
    
    for v_idx in adjacent:
        next_point = ctx.map_points[v_idx]
        v_next = next_point - current_point
        
        norm_prev = np.linalg.norm(v_prev)
        norm_next = np.linalg.norm(v_next)
        
        if norm_prev == 0 or norm_next == 0:
            angle = 0.0
        else:
            dot = np.clip(np.dot(v_prev, v_next) / (norm_prev * norm_next), -1.0, 1.0)
            angle = np.arccos(dot)
        
        if angle < min_angle:
            min_angle = angle
            best_v = v_idx
    
    return best_v


def constraint_decider_ctx(candidate: LaneCandidate, ctx: GlobalContext) -> bool:
    """
    Checks all geometric constraints on a LaneCandidate using GlobalContext.
    
    Args:
        candidate: LaneCandidate with left_path, right_path
        ctx: GlobalContext with map_points
    
    Returns:
        True if all constraints pass, False otherwise
    """
    points = ctx.map_points
    left = candidate.left_path
    right = candidate.right_path
    
    return (C_seg(left, points) and 
            C_seg(right, points) and 
            C_width(left, right, points) and 
            C_poly(left, right, points))


def extend_candidate(
    candidate: LaneCandidate,
    ctx: GlobalContext,
    extend_left: bool,
    next_vertex: int
) -> LaneCandidate:
    """
    Creates a new LaneCandidate by extending the current one.
    This is a pure function - returns new candidate without mutating input.
    
    Args:
        candidate: Current LaneCandidate
        ctx: GlobalContext
        extend_left: True to extend left path, False for right
        next_vertex: Vertex index to add
    
    Returns:
        New LaneCandidate with extended path
    """
    if extend_left:
        new_left = candidate.left_path + [next_vertex]
        new_right = list(candidate.right_path)
        new_left_visited = candidate.left_visited | {next_vertex}
        new_right_visited = set(candidate.right_visited)
    else:
        new_left = list(candidate.left_path)
        new_right = candidate.right_path + [next_vertex]
        new_left_visited = set(candidate.left_visited)
        new_right_visited = candidate.right_visited | {next_vertex}
    
    # Update matchings incrementally
    new_matchings = OnlineLW(new_left, new_right, ctx.map_points, candidate.matchings)
    
    return LaneCandidate(
        left_path=new_left,
        right_path=new_right,
        left_visited=new_left_visited,
        right_visited=new_right_visited,
        matchings=new_matchings,
        is_valid=True
    )


def enumerate_candidates(
    ctx: GlobalContext,
    initial_left: List[int],
    initial_right: List[int],
    heading_vector: np.ndarray,
    max_iterations: int = 2500
) -> List[LaneCandidate]:
    """
    Enumerates valid lane candidates using GlobalContext and LaneCandidate types.
    
    Args:
        ctx: GlobalContext with map, graph, and caches
        initial_left: Starting left path (list of point indices)
        initial_right: Starting right path (list of point indices)
        heading_vector: Car heading as 2D unit vector
        max_iterations: Maximum DFS iterations
    
    Returns:
        List of valid LaneCandidate objects
    """
    initial_candidate = LaneCandidate(
        left_path=initial_left,
        right_path=initial_right,
        left_visited=set(initial_left),
        right_visited=set(initial_right),
        matchings=MatchingSet(),
        is_valid=True
    )
    
    results: List[LaneCandidate] = []
    stack: List[Tuple[LaneCandidate, int]] = [(initial_candidate, 0)]
    
    while stack:
        candidate, iteration = stack.pop()
        
        if iteration > max_iterations:
            continue
        
        # Check constraints
        if not constraint_decider_ctx(candidate, ctx):
            continue
        
        l_last = candidate.left_path[-1]
        r_last = candidate.right_path[-1]
        
        # Get unvisited neighbors
        l_adj = [v for v in ctx.adj_list.get(l_last, []) 
                 if v not in candidate.left_visited and v not in candidate.right_visited]
        r_adj = [v for v in ctx.adj_list.get(r_last, []) 
                 if v not in candidate.right_visited and v not in candidate.left_visited]
        
        # Base case: no more extensions
        if not l_adj and not r_adj:
            if len(candidate.left_path) > 2 and len(candidate.right_path) > 2:
                results.append(candidate)
            continue
        
        # Try extending left
        if l_adj:
            next_l = next_vertex_decider_ctx(ctx, candidate.left_path, heading_vector)
            if next_l is not None and next_l in l_adj:
                new_candidate = extend_candidate(candidate, ctx, True, next_l)
                stack.append((new_candidate, iteration + 1))
        
        # Try extending right
        if r_adj:
            next_r = next_vertex_decider_ctx(ctx, candidate.right_path, heading_vector)
            if next_r is not None and next_r in r_adj:
                new_candidate = extend_candidate(candidate, ctx, False, next_r)
                stack.append((new_candidate, iteration + 1))
    
    return results


def compute_features_from_candidate(candidate: LaneCandidate, ctx: GlobalContext) -> List[float]:
    """
    Computes 8 geometric features from a LaneCandidate using GlobalContext.
    
    Args:
        candidate: LaneCandidate with paths and matchings
        ctx: GlobalContext with map_points
    
    Returns:
        List of 8 feature values
    """
    return compute_features((candidate.left_path, candidate.right_path), ctx.map_points)