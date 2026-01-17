import numpy as np
import math
import matplotlib.pyplot as plt

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


def OnlineLW(left_path, right_path, points, M_fixed_prev=None):
    """
    Algorithm 3: Online algorithm for lane width calculation.
    Returns (M_fixed_t, M_mut_t) where M are lists of (u, v) matching indices.
    """
    if M_fixed_prev is None:
        M_fixed_prev = []
        u_s, v_s = 0, 0
    elif not M_fixed_prev:
        u_s, v_s = 0, 0
    else:
        u_s, v_s = M_fixed_prev[-1]

    # Compute matches starting from u_s, v_s
    # Simplified: finding matches just based on nearest neighbors for vertices in current range
    new_matches = []
    
    l_indices = range(u_s, len(left_path))
    r_indices = range(v_s, len(right_path))
    
    if not l_indices or not r_indices:
        return M_fixed_prev, []

    # Naive greedy matching for demonstration of the algorithm structure
    # In reality this should follow Eq 11 more strictly
    for i in l_indices:
        best_j = -1
        min_d = float("inf")
        for j in r_indices:
            d = np.linalg.norm(np.array(points[left_path[i]]) - np.array(points[right_path[j]]))
            if d < min_d:
                min_d = d
                best_j = j
        if best_j != -1:
            new_matches.append((i, best_j))
            
    # Sort matches
    new_matches.sort(key=lambda x: (x[0], x[1]))
    
    # Split M' into M'_fixed and M_mut
    split_idx = len(new_matches)
    max_u = len(left_path) - 1
    max_v = len(right_path) - 1
    
    for k, (u, v) in enumerate(new_matches):
        if u == max_u or v == max_v:
            split_idx = k
            break
            
    M_prime_fixed = new_matches[:split_idx]
    M_mut = new_matches[split_idx:]
    
    M_fixed = M_fixed_prev + M_prime_fixed
    # Deduplicate
    M_fixed = sorted(list(set(M_fixed)), key=lambda x: (x[0], x[1]))
    
    return M_fixed, M_mut


def C_seg(boundary, points, max_angle=np.pi/2):
    """Segment Consistency Constraint"""
    for i in range(len(boundary) - 2):
        p1 = points[boundary[i]]
        p2 = points[boundary[i + 1]]
        p3 = points[boundary[i + 2]]
        if calculate_segment_angle(p1, p2, p3) > max_angle:
            return False
    return True


def C_poly(left_path, right_path, points):
    """Polynomial Consistency Constraint"""
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


def C_width(left_path, right_path, points, wmin=2.5, wmax=6.5):
    """Width Consistency Constraint"""
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





def visualize_path_pairs(path_pairs, points, title="Path Pairs"):
    """
    Visualize the found path pairs for debugging and analysis.
    """
    plt.figure(figsize=(10, 8))

    # Plot all points
    points_array = np.array(points)
    plt.scatter(
        points_array[:, 0],
        points_array[:, 1],
        c="black",
        s=100,
        label="Points",
        zorder=5,
    )

    # Plot each path pair
    colors = plt.cm.tab10(np.linspace(0, 1, len(path_pairs)))
    for i, (left_path, right_path) in enumerate(path_pairs):
        color = colors[i]

        # Plot left path
        left_coords = np.array([points[j] for j in left_path])
        plt.plot(
            left_coords[:, 0],
            left_coords[:, 1],
            "o-",
            color=color,
            linewidth=2,
            markersize=8,
            label=f"Left {i+1}",
        )

        # Plot right path
        right_coords = np.array([points[j] for j in right_path])
        plt.plot(
            right_coords[:, 0],
            right_coords[:, 1],
            "s-",
            color=color,
            linewidth=2,
            markersize=8,
            label=f"Right {i+1}",
        )

        # Plot matching lines (using paper-accurate segment-based matching)
        matching_lines = find_matching_segments(left_path, right_path, points)
        for match in matching_lines:
            plt.plot(
                [match["left_point"][0], match["projection_point"][0]],
                [match["left_point"][1], match["projection_point"][1]],
                "--",
                color=color,
                alpha=0.5,
                linewidth=1,
            )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    points = [
        (0, 0),
        (0, 3),
    (0, 6),
    (0, 9),
    (0, 12),
    (4, 0),
    (4, 3),
    (4, 6),
    (4, 9),
    (4, 12),
    ]  # Example set of 2D points

    # points = [(0, 0), (0, 3), (4, 0), (4, 3)] # Example set of 2D points
    dmax = 5
    adj_list = construct_adjacency_list(points, 4)
    print("Original points:", points)
    print("Adjacency list:", adj_list)

    # Find path pairs with improved constraints
    path_pairs = enumerate_path_pairs(adj_list, 0, 2)
    print(f"Found {len(path_pairs)} valid path pairs:")
    for i, pair in enumerate(path_pairs):
        print(f" Pair {i+1}: Left={pair[0]}, Right={pair[1]}")

    # Generate feature pairs
    feature_pairs = generate_feature_pairs(path_pairs, points)
    print(f"\nGenerated {len(feature_pairs)} feature pairs for ranking")

    # Visualize results
    if path_pairs:
        visualize_path_pairs(path_pairs, points, "Improved Lane Detection Results")