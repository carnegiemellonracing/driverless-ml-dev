import numpy as np
import matplotlib.pyplot as plt

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

def compute_iou(pred, gt):
    """Compute IoU between predicted and ground truth points"""
    # Convert to sets of points for IoU calculation
    pred_set = set(map(tuple, pred.reshape(-1, 2)))
    gt_set = set(map(tuple, gt.reshape(-1, 2)))
    
    # Calculate intersection and union
    intersection = len(pred_set.intersection(gt_set))
    union = len(pred_set.union(gt_set))
    
    return intersection / max(union, 1)  # Avoid division by zero

def construct_adjacency_list(points, dmax):
    adjacency_list = {i: [] for i in range(len(points))}
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if np.linalg.norm(np.array(points[i]) - np.array(points[j])) <= dmax:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)
                
    return adjacency_list

def find_matching_points(left_path, right_path, points, fixed_matches=None):
    """
    Find matching points between left and right boundaries using nearest neighbor search.
    Based on the paper's matching algorithm (Equation 9).
    
    Args:
        left_path: List of indices for left boundary points
        right_path: List of indices for right boundary points  
        points: List of 2D coordinates
        fixed_matches: Set of (left_idx, right_idx) tuples that are already fixed
    
    Returns:
        List of matching line dictionaries with width and fixed/mutable status
    """
    if fixed_matches is None:
        fixed_matches = set()
    
    left_coords = np.array([points[i] for i in left_path])
    right_coords = np.array([points[i] for i in right_path])
    
    matching_lines = []
    
    for i, left_point in enumerate(left_coords):
        # Find nearest point in right boundary using nearest neighbor search
        distances = np.linalg.norm(right_coords - left_point, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_right_point = right_coords[nearest_idx]
        
        # Check if this is a fixed or mutable match
        is_fixed = (i, nearest_idx) in fixed_matches
        
        matching_lines.append({
            'left_idx': i,
            'right_idx': nearest_idx,
            'left_point': left_point,
            'right_point': nearest_right_point,
            'width': distances[nearest_idx],
            'is_fixed': is_fixed
        })
    
    return matching_lines

def line_segments_intersect(p1, p2, p3, p4):
    """Check if two line segments intersect using cross product method"""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def constraint_decider(path_pair, debug=False):
    """
    Constraint decider implementing the paper's geometric constraints.
    Includes segment angle, width, and polygon constraints with proper backtracking.
    """
    
    def Cseg(path_pair):
        """
        Segment angle constraint - ensures path doesn't turn too sharply.
        Based on paper's C_seg constraint (Equation 11).
        """
        left_path, right_path = path_pair
        
        # Check left path
        for i in range(len(left_path) - 2):
            p1 = points[left_path[i]]
            p2 = points[left_path[i+1]]
            p3 = points[left_path[i+2]]

            angle = calculate_segment_angle(p1, p2, p3)

            if angle > np.pi / 2:  # 90 degrees
                return False
                
        # Check right path
        for i in range(len(right_path) - 2):
            p1 = points[right_path[i]]
            p2 = points[right_path[i+1]]
            p3 = points[right_path[i+2]]

            angle = calculate_segment_angle(p1, p2, p3)

            if angle > np.pi / 2:  # 90 degrees
                return False
                
        return True
    
    def Cwidth(path_pair, wmin=2.5, wmax=6.5):
        """
        Width constraint based on the paper's approach.
        Implements the three backtracking criteria from the paper:
        1. Fixed matching lines that are too long/short cannot be fixed
        2. Mutable matching lines that are too short cannot be fixed  
        3. Mutable matching lines that are too long may be fixed by extending boundaries
        """
        left_path, right_path = path_pair
        
        if len(left_path) < 2 or len(right_path) < 2:
            return True  # Not enough points to check width
        
        # Find matching points using nearest neighbor search
        matching_lines = find_matching_points(left_path, right_path, points)
        
        # Check width constraints for all matching lines
        for match in matching_lines:
            width = match['width']
            
            if not (wmin < width < wmax):
                # Apply paper's backtracking criteria
                if match['is_fixed']:
                    # Fixed matching line that's too long or too short cannot be fixed
                    return False
                elif width < wmin:
                    # Mutable matching line that's too short cannot be fixed
                    return False
                # If mutable matching line is too long, it may be fixed by extending boundaries
                # This is handled by the enumeration algorithm continuing to search
        
        return True
    
    def Cpoly(path_pair):
        """
        Polygon constraint - ensures the lane forms a simple polygon.
        Based on the paper's polygon constraint (Equation 12).
        """
        left_path, right_path = path_pair
        
        if len(left_path) < 2 or len(right_path) < 2:
            return True
        
        # Create the driving lane polygon: left boundary + right boundary (reversed)
        polygon_points = []
        
        # Add left boundary points
        for i in left_path:
            polygon_points.append(np.array(points[i]))
        
        # Add right boundary points in reverse order
        for i in reversed(right_path):
            polygon_points.append(np.array(points[i]))
        
        # Check if polygon is simple (no self-intersections)
        n = len(polygon_points)
        for i in range(n):
            for j in range(i + 2, n):
                # Skip adjacent edges
                if j == (i + 1) % n or i == (j + 1) % n:
                    continue
                
                # Check if line segments intersect
                if line_segments_intersect(polygon_points[i], polygon_points[(i+1)%n],
                                         polygon_points[j], polygon_points[(j+1)%n]):
                    return False
        
        return True
    
    if debug:
        print(f"Cseg: {Cseg(path_pair)}")
        print(f"Cwidth: {Cwidth(path_pair)}")
        print(f"Cpoly: {Cpoly(path_pair)}")
    return Cseg(path_pair) and Cwidth(path_pair) and Cpoly(path_pair)

def bt_decider(path_pair, fixed_matches=None, wmin=2.5, wmax=6.5):
    """
    Backtracking (BT) decider implementing the paper's sophisticated backtracking logic.
    
    The BT decider analyzes constraint violations and determines whether to:
    1. Stop backtracking (prune the path)
    2. Continue searching (allow path extension)
    3. Apply specific backtracking strategies based on constraint type
    
    Args:
        path_pair: Tuple of (left_path, right_path) indices
        fixed_matches: Set of (left_idx, right_idx) tuples that are fixed matches
        wmin, wmax: Width constraints for lane boundaries
    
    Returns:
        dict with keys:
        - 'continue': bool - whether to continue searching this path
        - 'reason': str - reason for the decision
        - 'violations': list - list of constraint violations found
    """
    if fixed_matches is None:
        fixed_matches = set()
    
    left_path, right_path = path_pair
    violations = []
    
    # Check segment angle constraint (C_seg)
    seg_violations = []
    for i in range(len(left_path) - 2):
        p1 = points[left_path[i]]
        p2 = points[left_path[i+1]]
        p3 = points[left_path[i+2]]
        angle = calculate_segment_angle(p1, p2, p3)
        if angle > np.pi / 2:
            seg_violations.append(('left', i, angle))
    
    for i in range(len(right_path) - 2):
        p1 = points[right_path[i]]
        p2 = points[right_path[i+1]]
        p3 = points[right_path[i+2]]
        angle = calculate_segment_angle(p1, p2, p3)
        if angle > np.pi / 2:
            seg_violations.append(('right', i, angle))
    
    if seg_violations:
        violations.append(('seg_angle', seg_violations))
    
    # Check width constraint (C_width) with detailed analysis
    width_violations = []
    if len(left_path) >= 2 and len(right_path) >= 2:
        matching_lines = find_matching_points(left_path, right_path, points, fixed_matches)
        
        for match in matching_lines:
            width = match['width']
            is_fixed = match['is_fixed']
            
            if width < wmin:
                width_violations.append(('too_narrow', match, width))
            elif width > wmax:
                width_violations.append(('too_wide', match, width))
    
    if width_violations:
        violations.append(('width', width_violations))
    
    # Check polygon constraint (C_poly)
    poly_violations = []
    if len(left_path) >= 2 and len(right_path) >= 2:
        # Create polygon and check for self-intersections
        polygon_points = []
        for i in left_path:
            polygon_points.append(np.array(points[i]))
        for i in reversed(right_path):
            polygon_points.append(np.array(points[i]))
        
        n = len(polygon_points)
        for i in range(n):
            for j in range(i + 2, n):
                if j == (i + 1) % n or i == (j + 1) % n:
                    continue
                if line_segments_intersect(polygon_points[i], polygon_points[(i+1)%n],
                                         polygon_points[j], polygon_points[(j+1)%n]):
                    poly_violations.append((i, j))
    
    if poly_violations:
        violations.append(('polygon', poly_violations))
    
    # Apply backtracking logic based on violations
    if not violations:
        return {'continue': True, 'reason': 'no_violations', 'violations': []}
    
    # Strategy 1: Segment angle violations - always stop (path is geometrically invalid)
    if any(v[0] == 'seg_angle' for v in violations):
        return {'continue': False, 'reason': 'sharp_turn_violation', 'violations': violations}
    
    # Strategy 2: Polygon violations - always stop (self-intersecting lane)
    if any(v[0] == 'polygon' for v in violations):
        return {'continue': False, 'reason': 'polygon_intersection', 'violations': violations}
    
    # Strategy 3: Width violations - apply paper's backtracking criteria
    width_violation = next((v for v in violations if v[0] == 'width'), None)
    if width_violation:
        _, width_violations_list = width_violation
        
        # Check each width violation
        for violation_type, match, width in width_violations_list:
            if violation_type == 'too_narrow':
                # Narrow lanes cannot be fixed by extending boundaries
                return {'continue': False, 'reason': 'width_too_narrow', 'violations': violations}
            
            elif violation_type == 'too_wide':
                if match['is_fixed']:
                    # Fixed matching line that's too wide cannot be fixed
                    return {'continue': False, 'reason': 'fixed_width_too_wide', 'violations': violations}
                else:
                    # Mutable matching line that's too wide may be fixed by extending boundaries
                    # Continue searching to allow boundary extension
                    pass
    
    # If we reach here, continue searching (e.g., mutable width violations that might be fixable)
    return {'continue': True, 'reason': 'fixable_violations', 'violations': violations}

def debug_bt_decider(path_pair, fixed_matches=None, wmin=2.5, wmax=6.5):
    """
    Debug version of BT decider that provides detailed information about constraint analysis.
    Useful for understanding how the backtracking logic works.
    """
    result = bt_decider(path_pair, fixed_matches, wmin, wmax)
    
    print(f"BT Decider Analysis for path pair:")
    print(f"  Left path: {path_pair[0]}")
    print(f"  Right path: {path_pair[1]}")
    print(f"  Decision: {'CONTINUE' if result['continue'] else 'STOP'}")
    print(f"  Reason: {result['reason']}")
    
    if result['violations']:
        print(f"  Violations found:")
        for violation_type, violation_data in result['violations']:
            print(f"    - {violation_type}: {violation_data}")
    else:
        print(f"  No violations found")
    
    return result

def enumerate_path_pairs(graph, sl, sr, itmax=100):
    """
    Enumerate path pairs with improved backtracking based on constraint violations.
    Implements the paper's backtracking criteria for efficient pruning using the BT decider.
    """
    def dfs(path_pair, visited, i, fixed_matches=None):
        if i >= itmax:
            return []
        
        if fixed_matches is None:
            fixed_matches = set()
        
        solutions = []
        current_left, current_right = path_pair
        
        left_adj = [v for v in graph[current_left[-1]] if v not in current_left]
        right_adj = [v for v in graph[current_right[-1]] if v not in current_right]
        
        if not left_adj and not right_adj:
            return [path_pair]
        
        for ln in left_adj:
            for rn in right_adj:
                new_pair = (current_left + [ln], current_right + [rn])
                
                # Use BT decider for sophisticated backtracking logic
                bt_result = bt_decider(new_pair, fixed_matches)
                
                if bt_result['continue']:
                    # Add valid path pair to solutions
                    solutions.append(new_pair)
                    
                    # Update fixed matches for the new path
                    new_fixed_matches = fixed_matches.copy()
                    # Add new matches as fixed for future extensions
                    for j in range(len(new_pair[0])):
                        if j < len(new_pair[1]):
                            new_fixed_matches.add((j, j))
                    
                    # Recursively search from this valid path
                    solutions.extend(dfs(new_pair, visited, i + 1, new_fixed_matches))
                else:
                    # BT decider determined this path should be pruned
                    # The decision is based on sophisticated analysis of constraint violations
                    # No further exploration from this path
                    pass
        
        return solutions
    
    return dfs(([sl], [sr]), set(), 0)

def compute_features(path_pair, points):
    """
    Compute features for a path pair using the improved matching algorithm.
    """
    left_path, right_path = path_pair
    
    left_coords = np.array([points[i] for i in left_path])
    right_coords = np.array([points[i] for i in right_path])
    
    left_length = np.sum(np.linalg.norm(np.diff(left_coords, axis=0), axis=1))
    right_length = np.sum(np.linalg.norm(np.diff(right_coords, axis=0), axis=1))
    lane_length = (left_length + right_length) / 2 # feature 1
    
    num_left_points = len(left_path) # feature 2
    num_right_points = len(right_path) # feature 3
    
    # Use improved matching for width calculation
    matching_lines = find_matching_points(left_path, right_path, points)
    widths = [match['width'] for match in matching_lines]
    width_variance = np.var(widths) if widths else 0.0 # feature 4
    
    def compute_segment_variance(coords):
        if len(coords) < 2:
            return 0.0, 0.0 
        segment_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        angles = np.abs(np.diff(np.arctan2(np.diff(coords[:, 1]), np.diff(coords[:, 0]))))
        return np.var(segment_lengths), np.var(angles)
    
    left_length_var, left_angle_var = compute_segment_variance(left_coords) # feature 5, 6
    right_length_var, right_angle_var = compute_segment_variance(right_coords) # feature 7, 8
            
    feature_vector = [
        lane_length, # feature 1
        num_left_points, # feature 2
        num_right_points, # feature 3
        width_variance, # feature 4
        left_length_var, # feature 5
        right_length_var, # feature 6
        left_angle_var, # feature 7
        right_angle_var # feature 8
    ]
    
    return feature_vector

def generate_feature_pairs(path_pairs, points):
    feature_pairs = []
    for i in range(len(path_pairs)):
        for j in range(i + 1, len(path_pairs)):
            try:
                if len(path_pairs[i][0]) != len(path_pairs[i][1]) or len(path_pairs[j][0]) != len(path_pairs[j][1]):
                    continue
                
                # Compute features for both path pairs
                x1 = compute_features(path_pairs[i], points)
                x2 = compute_features(path_pairs[j], points)
                
                # Ensure both feature vectors are valid
                if not any(np.isnan(x1)) and not any(np.isnan(x2)):
                    feature_pairs.append((x1, x2))
                else:
                    print(f"Skipping invalid feature pair: x1={x1}, x2={x2}")
            except Exception as e:
                print(f"Error processing path pairs {i} and {j}: {e}")
                continue
    return feature_pairs

def visualize_path_pairs(path_pairs, points, title="Path Pairs"):
    """
    Visualize the found path pairs for debugging and analysis.
    """
    plt.figure(figsize=(10, 8))
    
    # Plot all points
    points_array = np.array(points)
    plt.scatter(points_array[:, 0], points_array[:, 1], c='black', s=100, label='Points', zorder=5)
    
    # Plot each path pair
    colors = plt.cm.tab10(np.linspace(0, 1, len(path_pairs)))
    for i, (left_path, right_path) in enumerate(path_pairs):
        color = colors[i]
        
        # Plot left path
        left_coords = np.array([points[j] for j in left_path])
        plt.plot(left_coords[:, 0], left_coords[:, 1], 'o-', color=color, 
                linewidth=2, markersize=8, label=f'Left {i+1}')
        
        # Plot right path
        right_coords = np.array([points[j] for j in right_path])
        plt.plot(right_coords[:, 0], right_coords[:, 1], 's-', color=color, 
                linewidth=2, markersize=8, label=f'Right {i+1}')
        
        # Plot matching lines
        matching_lines = find_matching_points(left_path, right_path, points)
        for match in matching_lines:
            plt.plot([match['left_point'][0], match['right_point'][0]], 
                    [match['left_point'][1], match['right_point'][1]], 
                    '--', color=color, alpha=0.5, linewidth=1)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# Example usage:
points = [(0, 0), (0, 3), (0, 6), (0, 9), (0, 12), (4, 0), (4, 3), (4, 6), (4, 9), (4, 12)]  # Example set of 2D points
# points = [(0, 0), (0, 3), (4, 0), (4, 3)]  # Example set of 2D points
dmax = 5
adj_list = construct_adjacency_list(points, 4)
print("Original points:", points)
print("Adjacency list:", adj_list)

# Find path pairs with improved constraints
path_pairs = enumerate_path_pairs(adj_list, 0, 2)
print(f"Found {len(path_pairs)} valid path pairs:")
for i, pair in enumerate(path_pairs):
    print(f"  Pair {i+1}: Left={pair[0]}, Right={pair[1]}")

# Generate feature pairs
feature_pairs = generate_feature_pairs(path_pairs, points)
print(f"\nGenerated {len(feature_pairs)} feature pairs for ranking")

# Visualize results
if path_pairs:
    visualize_path_pairs(path_pairs, points, "Improved Lane Detection Results")