import numpy as np

def point_to_segment_distance(point, seg_start, seg_end):
    """
    Calculate minimum distance from a point to a line segment.
    
    This healper function is used for computing distances in the C_width
    constraint's nearest neighbor search (Equation 9).
    
    Args:
        point: numpy array [x, y]
        seg_start: numpy array [x, y] - start of segment
        seg_end: numpy array [x, y] - end of segment
    
    Returns:
        Minimum distance from point to segment
    """
    # Vector from seg_start to seg_end
    seg_vec = seg_end - seg_start
    
    # Vector from seg_start to point
    point_vec = point - seg_start

    # Length of segment squared
    seg_len_sq = np.dot(seg_vec, seg_vec)
    
    if seg_len_sq == 0:
        # Segment is actually a point
        return np.linalg.norm(point - seg_start)

    # Project point onto the line defined by the segment
    # t is the parameter: 0 means seg_start, 1 means seg_end
    t = np.dot(point_vec, seg_vec) / seg_len_sq

    # Clamp t to [0, 1] to stay on the segment
    t = np.clip(t, 0, 1)

    # Find the closest point on the segment
    closest_point = seg_start + t * seg_vec

    # Return distance
    return np.linalg.norm(point - closest_point)

def point_to_segment_distance_with_position(point, seg_start, seg_end):
    """
    Calculate minimum distance from a point to a line segment AND the position
    on the segment..
    
    This healper function is used for computing distances in the C_width
    constraint's nearest neighbor search (Equation 9).
    
    Args:
        point: numpy array [x, y]
        seg_start: numpy array [x, y] - start of segment
        seg_end: numpy array [x, y] - end of segment
    
    Returns:
        Minimum distance from point to segment and fractional position t ∈ [0,1]
    """
    # Vector from seg_start to seg_end
    seg_vec = seg_end - seg_start
    
    # Vector from seg_start to point
    point_vec = point - seg_start

    # Length of segment squared
    seg_len_sq = np.dot(seg_vec, seg_vec)
    
    if seg_len_sq == 0:
        # Segment is actually a point
        return np.linalg.norm(point - seg_start)

    # Project point onto the line defined by the segment
    # t is the parameter: 0 means seg_start, 1 means seg_end
    t = np.dot(point_vec, seg_vec) / seg_len_sq

    # Clamp t to [0, 1] to stay on the segment
    t = np.clip(t, 0, 1)

    # Find the closest point on the segment
    closest_point = seg_start + t * seg_vec

    # Return distance
    return np.linalg.norm(point - closest_point), t

def point_to_polygonal_chain_distance(point, chain_coords):
    """
    Calculate minimum distance from a point to a polygonal chain.
    
    This implements the nearest neighbor search to a continuous
    polygonal chain (including all points along segments).
    Used in M(L, R, 0, 0) and M(L, R, 1, 0).
    
    Args:
        point: numpy array [x, y]
        chain_coords: numpy array of shape (n, 2) - vertices of chain
    
    Returns:
        Minimum distance from point to the entire chain
    """
    min_dist = float('inf')
    
    # Check distance to all vertices
    for vertex in chain_coords:
        dist = np.linalg.norm(point - vertex)
        min_dist = min(min_dist, dist)
    
    # Check distance to all segments
    for i in range(len(chain_coords) - 1):
        seg_start = chain_coords[i]
        seg_end = chain_coords[i + 1]
        dist = point_to_segment_distance(point, seg_start, seg_end)
        min_dist = min(min_dist, dist)
    
    return min_dist

def point_to_polygonal_chain_distance_with_position(point, chain_coords):
    """
    Calculate minimum distance from a point to a polygonal chain AND the exact position.
    
    Extended version of point_to_polygonal_chain_distance that also returns
    the fractional index of the nearest point on the chain.
    
    Args:
        point: numpy array [x, y]
        chain_coords: numpy array of shape (n, 2) - vertices of chain
    
    Returns:
        tuple: (min_distance, fractional_index)
            - min_distance: Minimum distance from point to chain
            - fractional_index: Position on chain where i+t means the nearest point
              is at parameter t ∈ [0,1] along segment [i, i+1]
    """
    min_dist = float('inf')
    min_pos = 0.0
    
    # Check distance to all vertices
    for i, vertex in enumerate(chain_coords):
        dist = np.linalg.norm(point - vertex)
        if dist < min_dist:
            min_dist = dist
            min_pos = i
    
    # Check distance to all segments
    for i in range(len(chain_coords) - 1):
        seg_start = chain_coords[i]
        seg_end = chain_coords[i + 1]
        
        dist, t = point_to_segment_distance_with_position(point, seg_start, seg_end) 
        if dist < min_dist:
            min_dist = dist
            min_pos = i + t  # Exact fractional position
    
    return min_dist, min_pos

def segment_to_segment_distance(seg1_start, seg1_end, seg2_start, seg2_end):
    """
    Calculate minimum distance between two line segments.
    
    Args:
        seg1_start, seg1_end: numpy arrays [x, y] - first segment
        seg2_start, seg2_end: numpy arrays [x, y] - second segment
    
    Returns:
        Minimum distance between the two segments
    """
    # Check all four endpoint-to-segment distances
    d1 = point_to_segment_distance(seg1_start, seg2_start, seg2_end)
    d2 = point_to_segment_distance(seg1_end, seg2_start, seg2_end)
    d3 = point_to_segment_distance(seg2_start, seg1_start, seg1_end)
    d4 = point_to_segment_distance(seg2_end, seg1_start, seg1_end)
    
    return min(d1, d2, d3, d4)

def segment_to_segment_distance_with_position(seg1_start, seg1_end, seg2_start, seg2_end):
    """
    Calculate minimum distance between two line segments AND the positions.
    
    Args:
        seg1_start, seg1_end: numpy arrays [x, y] - first segment
        seg2_start, seg2_end: numpy arrays [x, y] - second segment
    
    Returns:
        tuple: (min_distance, pos1, pos2)
        - min_distance: Minimum distance between the two segments
        - pos1: Fractional position on first segment (0 to 1)
        - pos2: Fractional position on second segment (0 to 1)
    """
    min_dist = float('inf')
    pos1 = 0.0
    pos2 = 0.0

    # Check seg1_start to seg2
    d1, t1 = point_to_segment_distance_with_position(seg1_start, seg2_start, seg2_end)
    if d1 < min_dist:
        min_dist = d1
        pos1 = 0.0
        pos2 = t1

    # Check seg1_end to seg2
    d2, t2 = point_to_segment_distance_with_position(seg1_end, seg2_start, seg2_end)
    if d2 < min_dist:
        min_dist = d2
        pos1 = 1.0
        pos2 = t2

    # Check seg2_start to seg1
    d3, t3 = point_to_segment_distance_with_position(seg2_start, seg1_start, seg1_end)
    if d3 < min_dist:
        min_dist = d3
        pos1 = t3
        pos2 = 0.0

    # Check seg2_end to seg1
    d4, t4 = point_to_segment_distance_with_position(seg2_end, seg1_start, seg1_end)
    if d4 < min_dist:
        min_dist = d4
        pos1 = t4
        pos2 = 1.0

    return min_dist, pos1, pos2

def segment_to_polygonal_chain_distance(seg_start, seg_end, chain_coords):
    """
    Calculate minimum distance from a line segment to a polygonal chain.
    
    This finds the closest pair of points where one is on the query segment
    and the other is on the target chain.
    Used in M(L, R, 0, 1) and M(L, R, 1, 1).
    
    Args:
        seg_start: numpy array [x, y] - start of query segment
        seg_end: numpy array [x, y] - end of query segment
        chain_coords: numpy array of shape (n, 2) - vertices of target chain
    
    Returns:
        Minimum distance between the segment and the chain
    """
    min_dist = float('inf')
    
    # Distance from segment endpoints to chain
    min_dist = min(min_dist, point_to_polygonal_chain_distance(seg_start, chain_coords))
    min_dist = min(min_dist, point_to_polygonal_chain_distance(seg_end, chain_coords))
    
    # Distance from chain vertices to query segment
    for vertex in chain_coords:
        dist = point_to_segment_distance(vertex, seg_start, seg_end)
        min_dist = min(min_dist, dist)
    
    # Segment-to-segment distances for all chain segments
    for i in range(len(chain_coords) - 1):
        chain_seg_start = chain_coords[i]
        chain_seg_end = chain_coords[i + 1]
        
        # Use segment-to-segment distance formula
        dist = segment_to_segment_distance(seg_start, seg_end, 
                                          chain_seg_start, chain_seg_end)
        min_dist = min(min_dist, dist)
    
    return min_dist

def segment_to_polygonal_chain_distance_with_position(seg_start, seg_end, chain_coords):
    """
    Calculate minimum distance from a line segment to a polygonal chain AND positions.

    Extended version of segment_to_polygonal_chain_distance that returns the fractional
    positions on both the query segment and the target chain where minimum distance occurs.

    Args:
        seg_start: numpy array [x, y] - start of query segment
        seg_end: numpy array [x, y] - end of query segment
        chain_coords: numpy array of shape (n, 2) - vertices of target chain

    Returns:
        tuple: (min_distance, seg_pos, chain_pos)
            - min_distance: Minimum distance between segment and chain
            - seg_pos: Fractional position on query segment (0 to 1)
            - chain_pos: Fractional index on chain (i+t for segment [i, i+1])
    """
    min_dist = float('inf')
    seg_pos = 0.0
    chain_pos = 0.0

    # Case 1: Distance from segment start to chain
    dist, chain_idx = point_to_polygonal_chain_distance_with_position(seg_start, chain_coords)
    if dist < min_dist:
        min_dist = dist
        seg_pos = 0.0
        chain_pos = chain_idx

    # Case 2: Distance from segment end to chain
    dist, chain_idx = point_to_polygonal_chain_distance_with_position(seg_end, chain_coords)
    if dist < min_dist:
        min_dist = dist
        seg_pos = 1.0
        chain_pos = chain_idx

    # Case 3: Distance from chain vertices to query segment
    for i, vertex in enumerate(chain_coords):
        dist, t = point_to_segment_distance_with_position(vertex, seg_start, seg_end)
        if dist < min_dist:
            min_dist = dist
            seg_pos = t
            chain_pos = float(i)

    # Case 4: Segment-to-segment distances for all chain segments
    for i in range(len(chain_coords) - 1):
        chain_seg_start = chain_coords[i]
        chain_seg_end = chain_coords[i + 1]

        dist, t_seg, t_chain = segment_to_segment_distance_with_position(
            seg_start, seg_end, chain_seg_start, chain_seg_end)
        if dist < min_dist:
            min_dist = dist
            seg_pos = t_seg
            chain_pos = i + t_chain  # Fractional position on chain segment

    return min_dist, seg_pos, chain_pos

def compute_matching_from_position(L, R, points, u_s, v_s):
    """
    Line 6 of Algorithm 3: Compute matching for L, R starting from u_s, v_s (Eq. 11)
    
    Computes all four matching cases using helper functions.
    
    Args:
        L: List of point indices for left boundary
        R: List of point indices for right boundary
        points: List of [x, y] coordinates
        u_s: Starting index in L (inclusive)
        v_s: Starting index in R (inclusive)
    
    Returns:
        M_prime: List of (u, v) tuples representing matchings
    """
    # Get coordinates for full chains
    L_coords = np.array([points[i] for i in L])
    R_coords = np.array([points[i] for i in R])
    
    M_prime = []
    
    # M(L, R, 0, 0): New left vertices to right chain
    for i in range(u_s, len(L)):
        left_point = points[L[i]]
        dist, nearest_v = point_to_polygonal_chain_distance_with_position(left_point, R_coords)
        M_prime.append((float(i), nearest_v))
    
    # M(L, R, 0, 1): New left segments to right chain
    for i in range(u_s, len(L) - 1):
        seg_start = points[L[i]]
        seg_end = points[L[i + 1]]
        dist, seg_pos, nearest_v = segment_to_polygonal_chain_distance_with_position(seg_start, seg_end, R_coords)
        M_prime.append((i + seg_pos, nearest_v))
    
    # M(L, R, 1, 0): New right vertices to left chain
    for j in range(v_s, len(R)):
        right_point = points[R[j]]
        dist, nearest_u = point_to_polygonal_chain_distance_with_position(right_point, L_coords)
        M_prime.append((nearest_u, float(j)))

    # M(L, R, 1, 1): New right segments to left chain
    for j in range(v_s, len(R) - 1):
        seg_start = points[R[j]]
        seg_end = points[R[j + 1]]
        dist, nearest_u, seg_pos = segment_to_polygonal_chain_distance_with_position(seg_start, seg_end, L_coords)
        M_prime.append((nearest_u, j + seg_pos))
    
    return M_prime 

def OnlineLW(L, R, points, M_fixed_prev=None):
    """
    Algorithm 3: Online algorithm for lane width calculation.

    Incrementally computes matching points between left and right lane boundaries.
    Splits matchings into fixed (never recomputed) and mutable (may change) sets.

    Args:
        L: List of point indices for left boundary
        R: List of point indices for right boundary
        points: List of [x, y] coordinates for all points
        M_fixed_prev: List of fixed matching points from previous iteration (default: None)

    Returns:
        tuple: (M_fixed, M_mut)
            - M_fixed: List of fixed matching points (tuples of (u, v))
            - M_mut: List of mutable matching points (tuples of (u, v))
    """
    # Line 2-5 of Algorithm 3: Determine starting position
    if M_fixed_prev is None or len(M_fixed_prev) == 0:
        u_s, v_s = (0, 0)  # Start from beginning of both chains
    else:
        u_s, v_s = M_fixed_prev[-1]  # Last element of M_fixed

    # Line 6 of Algorithm 3: Compute matching for L, R starting from u_s, v_s (Eq. 11)
    M_prime = compute_matching_from_position(L, R, points, u_s, v_s)

    # Line 7 of Algorithm 3: Sort M_prime and split into M_fixed and M_mut
    # Sort lexicographically: first by u, then by v
    # Using explicit key for clarity (tuples compare lexicographically by default)
    M_prime_sorted = sorted(M_prime, key=lambda x: (x[0], x[1]))

    # Find the split point: first matching that touches an endpoint of either boundary
    # A matching (u, v) is at an endpoint if u == len(L)-1 OR v == len(R)-1
    split_index = None
    for i, (u, v) in enumerate(M_prime_sorted):
        # Check if this matching touches the end of left boundary (u == |L|-1)
        # or the end of right boundary (v == |R|-1)
        if u >= len(L) - 1 or v >= len(R) - 1:
            split_index = i
            break

    # Split into fixed and mutable based on split_index
    if split_index is None:
        # All matchings are fixed (no endpoint touched)
        M_prime_fixed = M_prime_sorted
        M_mut = []
    else:
        # Everything before split_index is fixed, from split_index onwards is mutable
        M_prime_fixed = M_prime_sorted[:split_index]
        M_mut = M_prime_sorted[split_index:]

    # Line 8 of Algorithm 3: Merge new fixed matchings with previous iteration
    if M_fixed_prev is None or len(M_fixed_prev) == 0:
        M_fixed = M_prime_fixed
    else:
        M_fixed = M_fixed_prev + M_prime_fixed

    # Line 9 of Algorithm 3: Return M_fixed and M_mut
    return M_fixed, M_mut

def parametrize_position(fractional_index, chain, points):
    """
    Convert a fractional index to actual 2D coordinates on a polygonal chain.

    Uses the parametrization formula from the paper:
    P(i+λ) = (1-λ)p_i + λp_{i+1}

    Args:
        fractional_index: Float value where integer part is vertex index,
                         fractional part is position along segment
                         (e.g., 2.7 means 70% along segment from vertex 2 to 3)
        chain: List of point indices forming the polygonal chain
        points: List of [x, y] coordinates for all points

    Returns:
        numpy array [x, y] - coordinates at the fractional position
    """
    # Handle exact vertex positions
    if fractional_index == int(fractional_index):
        vertex_idx = int(fractional_index)
        if vertex_idx >= len(chain):
            vertex_idx = len(chain) - 1
        return np.array(points[chain[vertex_idx]])

    # Extract integer and fractional parts
    i = int(fractional_index)
    lambda_val = fractional_index - i

    # Clamp to valid range
    if i >= len(chain) - 1:
        # At or beyond last vertex
        return np.array(points[chain[-1]])

    if i < 0:
        # Before first vertex
        return np.array(points[chain[0]])

    # Linear interpolation: P(i+λ) = (1-λ)p_i + λp_{i+1}
    p_i = np.array(points[chain[i]])
    p_i_plus_1 = np.array(points[chain[i + 1]])

    interpolated_point = (1 - lambda_val) * p_i + lambda_val * p_i_plus_1

    return interpolated_point

def matching_to_distance(u, v, L, R, points):
    """
    Convert fractional matching indices to Euclidean distance.

    This implements the distance calculation from the paper:
    w_i = ||L(u_i) - R(v_i)||_2

    where L(u_i) and R(v_i) are obtained via parametrization.

    Args:
        u: Fractional index on left boundary L (e.g., 2.7 means 70% along segment [2,3])
        v: Fractional index on right boundary R
        L: List of point indices for left boundary
        R: List of point indices for right boundary
        points: List of [x, y] coordinates

    Returns:
        Float: Euclidean distance between L(u) and R(v)
    """
    # Get actual coordinates at fractional positions
    left_point = parametrize_position(u, L, points)
    right_point = parametrize_position(v, R, points)

    # Compute Euclidean distance
    distance = np.linalg.norm(left_point - right_point)

    return distance