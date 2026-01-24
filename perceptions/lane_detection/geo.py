import numpy as np
import math
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
        
    v_point = point - car_pos
    v_heading = np.array([np.cos(heading), np.sin(heading)])
    
    norm_point = np.linalg.norm(v_point)
    if norm_point == 0:
        return True
        
    # Calculate angle between heading vector and point vector
    dot = np.clip(np.dot(v_point / norm_point, v_heading), -1.0, 1.0)
    angle = np.arccos(dot)
    
    return angle <= (cone_angle / 2)


def calculate_segment_angle(p1, p2, p3):
    """Calculates the absolute deflection angle between two consecutive segments."""
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
    """Calculates minimum distance from a point to a polygonal chain.
    
    Handles edge cases:
    - Empty chain: returns inf
    - Single point chain: returns point-to-point distance
    - Multi-point chain: returns min distance to any segment
    """
    if len(chain) == 0:
        return float("inf")
    
    if len(chain) == 1:
        # Single point - just compute point-to-point distance
        return np.linalg.norm(np.array(point) - np.array(chain[0]))
    
    # Multi-point chain - find min distance to any segment
    min_dist = float("inf")
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
        u_s = prev_matching.last_fixed_l_idx
        v_s = prev_matching.last_fixed_r_idx

    # Start with previous fixed matchings
    new_fixed_indices = list(prev_matching.fixed_indices)
    new_fixed_widths = list(prev_matching.fixed_widths)

    # Compute new matches starting from last fixed position
    l_indices = range(u_s, len(left_path))
    r_indices = range(v_s, len(right_path))
    
    if not l_indices or not r_indices:
        return MatchingSet(
            fixed_indices=new_fixed_indices,
            fixed_widths=new_fixed_widths,
            last_fixed_l_idx=u_s,
            last_fixed_r_idx=v_s
        )

    new_matches = []
    
    # Greedy nearest-neighbor matching
    for i in l_indices:
        best_j = -1
        min_d = float("inf")
        for j in r_indices:
            d = np.linalg.norm(np.array(points[left_path[i]]) - np.array(points[right_path[j]]))
            if d < min_d:
                min_d = d
                best_j = j
        if best_j != -1:
            new_matches.append(((i, best_j), min_d))
            
    # Sort by left index
    new_matches.sort(key=lambda x: (x[0][0], x[0][1]))
    
    # Split into fixed and mutable
    # Matches not at endpoints are "fixed"
    max_u = len(left_path) - 1
    max_v = len(right_path) - 1
    
    last_l_idx = u_s
    last_r_idx = v_s
    
    for (u, v), width in new_matches:
        if u < max_u and v < max_v:
            # This match is fixed (not at boundary)
            new_fixed_indices.append((u, v))
            new_fixed_widths.append(width)
            last_l_idx = max(last_l_idx, u + 1)
            last_r_idx = max(last_r_idx, v + 1)
    
    # Deduplicate
    seen = set()
    deduped_indices = []
    deduped_widths = []
    for idx, w in zip(new_fixed_indices, new_fixed_widths):
        if idx not in seen:
            seen.add(idx)
            deduped_indices.append(idx)
            deduped_widths.append(w)
    
    return MatchingSet(
        fixed_indices=deduped_indices,
        fixed_widths=deduped_widths,
        last_fixed_l_idx=last_l_idx,
        last_fixed_r_idx=last_r_idx
    )


def C_seg(boundary, points, max_angle=PHI_MAX):
    """Segment Consistency Constraint."""
    for i in range(len(boundary) - 2):
        p1 = points[boundary[i]]
        p2 = points[boundary[i + 1]]
        p3 = points[boundary[i + 2]]
        if calculate_segment_angle(p1, p2, p3) > max_angle:
            return False
    return True


def C_poly(left_path, right_path, points):
    """Polygon Consistency Constraint."""
    if len(left_path) < 2 or len(right_path) < 2:
        return True
        
    polygon_indices = list(left_path) + list(reversed(right_path))
    polygon_points = [np.array(points[idx]) for idx in polygon_indices]
    n = len(polygon_points)

    for i in range(n):
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

def left_right_decider(paths, points, n0, n1):
    """LRD: Decides which side to extend (0=left, 1=right).
    
    From paper Section V-B, Equations 4-5:
    Computes θl and θr (angles between last segments and cross-lane segment)
    for both potential extensions, picks the side with smaller |θr - θl|.
    
    Args:
        paths: Tuple of (left_path, right_path)
        points: Point coordinates
        n0: Next vertex candidate for left side
        n1: Next vertex candidate for right side
    
    Returns:
        0 for left, 1 for right
    """
    left_path, right_path = paths
    
    if n0 is None:
        return 1
    if n1 is None:
        return 0
    
    # Need at least 2 points to compute angles
    if len(left_path) < 2 or len(right_path) < 2:
        # Fallback: extend shorter path
        return 0 if len(left_path) <= len(right_path) else 1
    
    def compute_angle(p1, p2, p3):
        """Compute angle between vectors (p1->p2) and (p2->p3)."""
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p2)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return np.arccos(cos_angle)
    
    # Current last points
    ln = np.array(points[left_path[-1]])    # Last left point
    ln_1 = np.array(points[left_path[-2]])  # Second-to-last left
    rm = np.array(points[right_path[-1]])   # Last right point
    rm_1 = np.array(points[right_path[-2]]) # Second-to-last right
    
    # Next points
    pn0 = np.array(points[n0])  # Next left candidate
    pn1 = np.array(points[n1])  # Next right candidate
    
    # Case 1: Extend left (add n0 to left path)
    # New left last point becomes pn0
    # θl = ∠(ln-1 -> ln, ln -> rm) -- no change since ln is now second-to-last
    # Actually after extending left: ln becomes ln-1, pn0 becomes new ln
    # θ1_l = ∠(ln -> pn0, pn0 -> rm)
    # θ1_r = ∠(rm-1 -> rm, rm -> pn0)
    theta1_l = compute_angle(ln, pn0, rm)
    theta1_r = compute_angle(rm_1, rm, pn0)
    
    # Case 2: Extend right (add n1 to right path)  
    # θ2_l = ∠(ln-1 -> ln, ln -> pn1)
    # θ2_r = ∠(rm -> pn1, pn1 -> ln)
    theta2_l = compute_angle(ln_1, ln, pn1)
    theta2_r = compute_angle(rm, pn1, ln)
    
    # Paper Eq. 5: Choose side with smaller |θr - θl|
    diff1 = abs(theta1_r - theta1_l)
    diff2 = abs(theta2_r - theta2_l)
    
    return 0 if diff1 < diff2 else 1


def backtracking_decider(paths, points, has_left_options, has_right_options):
    """BTD: Decides whether to backtrack based on Lemmas 1-3 from paper.
    
    From paper Section VI-B:
    - Lemma 1 (Cseg): Once violated, cannot be satisfied → backtrack
    - Lemma 2 (Cpoly): Can only be satisfied if violation is by ln-rm segment
    - Lemma 3 (Cwidth): 
        - Fixed matching too long/short → backtrack
        - Mutable matching too short → backtrack
        - Mutable matching too long → might be fixable, don't backtrack
    
    Returns:
        True to backtrack (stop exploring), False to continue
    """
    left_path, right_path = paths
    
    # If constraints are satisfied, don't backtrack
    if constraint_decider(paths, points):
        return False
    
    # Check Lemma 1: Cseg - once violated, cannot be satisfied
    if not C_seg(left_path, points) or not C_seg(right_path, points):
        return True  # Backtrack
    
    # Check Lemma 2: Cpoly - can only be satisfied if violation is by ln-rm segment
    if not C_poly(left_path, right_path, points):
        # Check if violation is caused by the ln-rm segment (connecting last points)
        # If so, extending might fix it. Otherwise, backtrack.
        if not _cpoly_violation_is_by_last_segment(left_path, right_path, points):
            return True  # Fixed segments intersecting → backtrack
        # Violation is by ln-rm segment, might be fixable
        return False
    
    # Check Lemma 3: Cwidth
    if not C_width(left_path, right_path, points):
        # Determine if violation is by fixed or mutable matching, and too long vs too short
        violation_type = _cwidth_violation_type(left_path, right_path, points)
        
        if violation_type == 'fixed_too_long' or violation_type == 'fixed_too_short':
            return True  # Backtrack
        elif violation_type == 'mutable_too_short':
            return True  # Backtrack
        elif violation_type == 'mutable_too_long':
            return False  # Might be fixable by extending
        else:
            return True  # Unknown violation, backtrack to be safe
    
    return False


def _cpoly_violation_is_by_last_segment(left_path, right_path, points):
    """Check if Cpoly violation is caused by the ln-rm segment.
    
    The ln-rm segment connects the last points of left and right paths.
    If this segment causes the intersection, extending might fix it.
    """
    if len(left_path) < 2 or len(right_path) < 2:
        return True  # Too short to determine, assume fixable
    
    polygon_indices = list(left_path) + list(reversed(right_path))
    polygon_points = [np.array(points[idx]) for idx in polygon_indices]
    n = len(polygon_points)
    
    # The ln-rm segment is the segment connecting left_path[-1] to right_path[-1]
    # In polygon order, this is at index len(left_path)-1 → len(left_path)
    ln_rm_idx = len(left_path) - 1
    
    for i in range(n):
        for j in range(i + 2, n):
            if j == (i + 1) % n or i == (j + 1) % n:
                continue
            if line_segments_intersect(polygon_points[i], polygon_points[(i+1)%n], 
                                       polygon_points[j], polygon_points[(j+1)%n]):
                # Found intersection - check if either segment is ln-rm
                if i == ln_rm_idx or j == ln_rm_idx:
                    return True  # Violation is by ln-rm, might be fixable
                else:
                    return False  # Violation is by other segments, not fixable
    
    return True  # No intersection found, shouldn't happen if Cpoly failed


def _cwidth_violation_type(left_path, right_path, points):
    """Determine type of Cwidth violation for Lemma 3 logic.
    
    Returns one of:
    - 'fixed_too_long': Fixed matching line is too long
    - 'fixed_too_short': Fixed matching line is too short  
    - 'mutable_too_long': Mutable matching line is too long
    - 'mutable_too_short': Mutable matching line is too short
    - None: No violation found
    """
    if len(left_path) < 1 or len(right_path) < 1:
        return None
    
    left_coords = [np.array(points[i]) for i in left_path]
    right_coords = [np.array(points[i]) for i in right_path]
    
    # Mutable indices are the last points of each path
    mutable_l_idx = len(left_path) - 1
    mutable_r_idx = len(right_path) - 1
    
    # Check each left point's distance to right chain
    for i, p in enumerate(left_coords):
        dist = point_to_polygonal_chain_distance(p, right_coords)
        is_mutable = (i == mutable_l_idx)
        
        if dist < W_MIN:
            return 'mutable_too_short' if is_mutable else 'fixed_too_short'
        if dist > W_MAX:
            return 'mutable_too_long' if is_mutable else 'fixed_too_long'
    
    # Check each right point's distance to left chain
    for i, p in enumerate(right_coords):
        dist = point_to_polygonal_chain_distance(p, left_coords)
        is_mutable = (i == mutable_r_idx)
        
        if dist < W_MIN:
            return 'mutable_too_short' if is_mutable else 'fixed_too_short'
        if dist > W_MAX:
            return 'mutable_too_long' if is_mutable else 'fixed_too_long'
    
    return None


def enumerate_path_pairs(graph, sl, sr, itmax=100):
    """Placeholder for older enumerate function"""
    return []


def EPP(graph, points, paths, visited_map, heading_vector, iteration, itmax=2500):
    """Enumerate Path Pairs - Algorithm 1 from paper (arXiv 2405.16369).
    
    Explores lane candidates using greedy DFS with backtracking.
    Uses NVD for vertex selection, LRD for side selection, BTD for pruning.
    
    Args:
        graph: Adjacency list {vertex: [neighbors]}
        points: Point coordinates array
        paths: Tuple of (left_path, right_path) - mutable lists
        visited_map: Dict of {side: {vertex: set of visited neighbors}}
        heading_vector: Car heading as 2D unit vector
        iteration: Current iteration count
        itmax: Maximum iterations
    
    Returns:
        Set of valid path pairs
    """
    results = []
    
    while True:  # Loop from paper line 3
        if iteration >= itmax:
            return results
        iteration += 1
        
        left_path, right_path = paths
        
        # cs ← P[s].back() - current vertices (line 7)
        c0, c1 = left_path[-1], right_path[-1]
        
        # vs ← V[s][cs] - visited sets for current vertices (line 8)
        v0 = visited_map[0].get(c0, set())
        v1 = visited_map[1].get(c1, set())
        
        # us ← (G[cs] \ vs) \ P[s] - unvisited adjacent vertices (line 10)
        u0 = [v for v in graph.get(c0, []) if v not in v0 and v not in left_path]
        u1 = [v for v in graph.get(c1, []) if v not in v1 and v not in right_path]
        
        # Base case: no more extensions (line 11-12)
        if not u0 and not u1:
            return results
        
        # ns ← NVD(P[s], us) - best next vertex for each side (line 14)
        n0 = next_vertex_decider(left_path, u0, points, heading_vector) if u0 else None
        n1 = next_vertex_decider(right_path, u1, points, heading_vector) if u1 else None
        
        # Choose which side to extend (lines 15-17)
        if u0 and u1:
            s = left_right_decider(paths, points, n0, n1)  # LRD
        else:
            s = 1 if not u0 else 0
        
        # Get the next vertex for chosen side
        ns = n0 if s == 0 else n1
        if ns is None:
            return results
        
        # P[s].push(ns) - add next vertex to path (line 18)
        if s == 0:
            left_path.append(ns)
        else:
            right_path.append(ns)
        
        # V[s][cs].add(ns) - mark as visited from current vertex (line 19)
        cs = c0 if s == 0 else c1
        if cs not in visited_map[s]:
            visited_map[s][cs] = set()
        visited_map[s][cs].add(ns)
        
        # CD(P) - if lane satisfies constraints, add to results (lines 20-21)
        if constraint_decider(paths, points):
            if len(left_path) > 2 and len(right_path) > 2:
                # Deep copy the paths to store result
                results.append((list(left_path), list(right_path)))
        
        # BTD - if NOT backtracking, recurse (lines 22-23)
        if not backtracking_decider(paths, points, bool(u0), bool(u1)):
            results.extend(EPP(graph, points, paths, visited_map, heading_vector, iteration, itmax))
        
        # P[s].pop() - backtrack (line 24)
        if s == 0:
            left_path.pop()
        else:
            right_path.pop()
        
        # Continue loop to try other options (loop continues until no more unvisited)


def enumerate_path_pairs_v2(graph, points, paths, visited, heading_vector, it, itmax=2500):
    """Wrapper for EPP that matches the old interface.
    
    Converts the simple visited set to the per-vertex visited map structure
    required by the paper's algorithm.
    """
    left_path, right_path = paths
    
    # Convert to mutable lists
    left_path = list(left_path)
    right_path = list(right_path)
    
    # Initialize visited_map: {side: {vertex: set of visited neighbors}}
    visited_map = {0: {}, 1: {}}
    
    # Call the paper-faithful EPP
    return EPP(graph, points, (left_path, right_path), visited_map, heading_vector, it, itmax)



def compute_features(path_pair, points):
    """
    Computes 8 geometric features for a lane candidate matching arXiv 2405.16369.
    
    Features:
    1. Lane length (mean of left and right boundary lengths in meters)
    2. Number of points (left boundary)
    3. Number of points (right boundary)
    4. Variance of lane width
    5. Variance of segment lengths (left)
    6. Variance of segment lengths (right)
    7. Variance of angles between consecutive segments (left)
    8. Variance of angles between consecutive segments (right)
    """
    left_path, right_path = path_pair
    
    # Helpers
    def get_segments(path):
        segments = []
        lengths = []
        if len(path) < 2:
            return [], []
        for i in range(len(path) - 1):
            p1 = np.array(points[path[i]])
            p2 = np.array(points[path[i+1]])
            segments.append(p2 - p1)
            lengths.append(np.linalg.norm(p2 - p1))
        return segments, lengths

    def get_angles(path):
        angles = []
        if len(path) < 3:
            return []
        for i in range(len(path) - 2):
            p1 = np.array(points[path[i]])
            p2 = np.array(points[path[i+1]])
            p3 = np.array(points[path[i+2]])
            angles.append(calculate_segment_angle(p1, p2, p3))
        return angles

    # Data extraction
    l_segs, l_lengths = get_segments(left_path)
    r_segs, r_lengths = get_segments(right_path)
    l_angles = get_angles(left_path)
    r_angles = get_angles(right_path)
    
    # 1. Lane length (mean)
    mean_len = (sum(l_lengths) + sum(r_lengths)) / 2.0
    
    # 2 & 3. Number of points
    n_left = len(left_path)
    n_right = len(right_path)
    
    # 4. Variance of width
    # Sample widths by checking distance from left points to right chain and vice versa
    widths = []
    if len(left_path) > 0 and len(right_path) > 0:
        l_coords = [np.array(points[i]) for i in left_path]
        r_coords = [np.array(points[i]) for i in right_path]
        
        for p in l_coords:
            widths.append(point_to_polygonal_chain_distance(p, r_coords))
        for p in r_coords:
            widths.append(point_to_polygonal_chain_distance(p, l_coords))
            
    var_width = np.var(widths) if widths else 0.0
    
    # 5 & 6. Variance of segment lengths
    var_len_l = np.var(l_lengths) if l_lengths else 0.0
    var_len_r = np.var(r_lengths) if r_lengths else 0.0
    
    # 7 & 8. Variance of angles
    var_ang_l = np.var(l_angles) if l_angles else 0.0
    var_ang_r = np.var(r_angles) if r_angles else 0.0
    
    return [
        mean_len, float(n_left), float(n_right), var_width,
        var_len_l, var_len_r, var_ang_l, var_ang_r
    ]

def generate_feature_pairs(path_pairs, points):
    """
    Generates feature vectors for a list of path pairs.
    Returns a list of feature lists (one per path pair).
    """
    features_list = []
    for pair in path_pairs:
        features_list.append(compute_features(pair, points))
    return features_list


def compute_lane_iou(candidate_pair, gt_pair, points, grid_res=0.5):
    """
    Computes Intersection over Union (IoU) between candidate and GT lane polygons.
    Uses grid sampling approximation since Shapely/OpenCV are unavailable.
    
    Args:
        candidate_pair: (left_path, right_path) indices
        gt_pair: (left_path, right_path) indices
        points: Cone map points
        grid_res: Resolution of sampling grid (meters)
    """
    from matplotlib.path import Path
    
    def form_polygon(pair):
        lp, rp = pair
        if not lp or not rp: 
            return None
        # Polygon: Left path -> Right path (reversed) -> Close
        poly_indices = list(lp) + list(reversed(rp)) + [lp[0]]
        return np.array([points[i] for i in poly_indices])
    
    cand_poly_pts = form_polygon(candidate_pair)
    gt_poly_pts = form_polygon(gt_pair)
    
    if cand_poly_pts is None or gt_poly_pts is None:
        return 0.0
        
    # Define bounding box covering both polygons
    all_pts = np.vstack([cand_poly_pts, gt_poly_pts])
    min_x, min_y = np.min(all_pts, axis=0)
    max_x, max_y = np.max(all_pts, axis=0)
    
    # Generate grid
    x_range = np.arange(min_x, max_x, grid_res)
    y_range = np.arange(min_y, max_y, grid_res)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    
    if len(grid_points) == 0:
        return 0.0
    
    # Check containment
    cand_path = Path(cand_poly_pts)
    gt_path = Path(gt_poly_pts)
    
    in_cand = cand_path.contains_points(grid_points)
    in_gt = gt_path.contains_points(grid_points)
    
    # Compute Intersection & Union
    intersection = np.sum(in_cand & in_gt)
    union = np.sum(in_cand | in_gt)
    
    if union == 0:
        return 0.0
        
    return float(intersection) / float(union)



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