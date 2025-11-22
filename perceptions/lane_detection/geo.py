import numpy as np
from typing import List, Tuple
from models import Point, Lane


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
        if get_segment_angle(boundary[i], boundary[i+1], boundary[i+2]) > max_angle:
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

