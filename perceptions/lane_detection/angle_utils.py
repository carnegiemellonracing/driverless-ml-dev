import numpy as np

def calculate_segment_angle(p1, p2, p3):
    """
    Calculate the angle between two consecutive line segments with proper angle wrapping.
    
    Args:
        p1, p2, p3: Three consecutive points as numpy arrays or lists
        
    Returns:
        float: Angle between segments in radians (0 to π)
    """
    p1 = np.array(p1)
    p2 = np.array(p2) 
    p3 = np.array(p3)
    
    # Calculate vectors for the two segments
    v1 = p2 - p1  # First segment vector
    v2 = p3 - p2  # Second segment vector
    
    # Handle edge case where points are too close
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0  # Points are too close, consider no angle
    
    # Normalize vectors
    v1_norm = v1 / norm1
    v2_norm = v2 / norm2
    
    # Calculate dot product
    dot_product = np.dot(v1_norm, v2_norm)
    
    # Clamp dot product to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Calculate angle using arccos
    angle = np.arccos(dot_product)
    
    # Ensure angle is in [0, π] range
    return np.abs(angle)

def calculate_segment_angle_alternative(p1, p2, p3):
    """
    Alternative method using atan2 for angle calculation.
    This method calculates the turning angle between two consecutive segments.
    
    Args:
        p1, p2, p3: Three consecutive points as numpy arrays or lists
        
    Returns:
        float: Turning angle in radians (0 to π)
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    # Calculate direction vectors
    v1 = p2 - p1  # First segment direction
    v2 = p3 - p2  # Second segment direction
    
    # Handle edge case where points are too close
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    
    # Calculate angles of both vectors
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    
    # Calculate the difference and handle angle wrapping
    angle_diff = angle2 - angle1
    
    # Normalize angle to [-π, π]
    while angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    while angle_diff < -np.pi:
        angle_diff += 2 * np.pi
    
    # Return absolute angle (0 to π)
    return abs(angle_diff)

def calculate_curvature_angle(p1, p2, p3):
    """
    Calculate curvature angle using the cross product method.
    This gives the external angle at point p2.
    
    Args:
        p1, p2, p3: Three consecutive points as numpy arrays or lists
        
    Returns:
        float: Curvature angle in radians (0 to π)
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    # Calculate vectors
    v1 = p2 - p1
    v2 = p3 - p2
    
    # Calculate cross product magnitude
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    
    # Calculate dot product
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    
    # Calculate angle using atan2 for proper quadrant handling
    angle = np.arctan2(abs(cross_product), dot_product)
    
    return angle
