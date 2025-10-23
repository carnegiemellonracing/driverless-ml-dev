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