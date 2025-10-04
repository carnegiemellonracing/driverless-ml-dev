import numpy as np
import matplotlib.pyplot as plt

# Export list for clean imports
__all__ = [
    'construct_adjacency_list',
    'enumerate_path_pairs',
    'enumerate_path_pairs_v2',
    'next_vertex_decider',
    'left_right_decider',
    'constraint_decider',
    'compute_features',
    'generate_feature_pairs'
]

def construct_adjacency_list(points, dmax):
    adjacency_list = {i: [] for i in range(len(points))}
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if np.linalg.norm(np.array(points[i]) - np.array(points[j])) <= dmax:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)
                
    return adjacency_list

def enumerate_path_pairs(graph, sl, sr, itmax=100):
    def dfs(path_pair, visited, i):
        if i >= itmax:
            return []

        solutions = []
        current_left, current_right = path_pair

        left_adj = [v for v in graph[current_left[-1]] if v not in current_left]
        right_adj = [v for v in graph[current_right[-1]] if v not in current_right]

        if not left_adj and not right_adj:
            return [path_pair]

        for ln in left_adj:
            for rn in right_adj:
                new_pair = (current_left + [ln], current_right + [rn])
                if constraint_decider(new_pair):
                    solutions.append(new_pair)
                    solutions.extend(dfs(new_pair, visited, i + 1))

        return solutions

    return dfs(([sl], [sr]), set(), 0)

# ============================================================================
# CORRECT PATH PAIR ENUMERATION (Algorithm 2 from Paper)
# ============================================================================

def next_vertex_decider(current_path, adjacent_vertices, points, heading_vector=None):
    """
    Next-Vertex-Decider (NVD): Select adjacent vertex with smallest angle.

    Args:
        current_path: List of vertex indices in current path
        adjacent_vertices: List of unvisited adjacent vertex indices
        points: List of [x, y] coordinates
        heading_vector: Initial heading direction (optional, for first step)

    Returns:
        Index of selected vertex (the one with smallest angle)
    """
    if len(current_path) < 1 or len(adjacent_vertices) == 0:
        return None

    if len(current_path) == 1:
        # Use heading vector if provided, otherwise choose first available
        if heading_vector is None:
            return adjacent_vertices[0]

        current_point = np.array(points[current_path[-1]])
        min_angle = float('inf')
        best_vertex = adjacent_vertices[0]

        for vertex in adjacent_vertices:
            next_point = np.array(points[vertex])
            direction = next_point - current_point
            direction_norm = direction / (np.linalg.norm(direction) + 1e-8)

            # Angle between heading and potential direction
            angle = np.arccos(np.clip(np.dot(heading_vector, direction_norm), -1.0, 1.0))

            if angle < min_angle:
                min_angle = angle
                best_vertex = vertex

        return best_vertex

    # Calculate angle between previous segment and potential next segment
    prev_point = np.array(points[current_path[-2]])
    current_point = np.array(points[current_path[-1]])
    prev_segment = current_point - prev_point

    min_angle = float('inf')
    best_vertex = adjacent_vertices[0]

    for vertex in adjacent_vertices:
        next_point = np.array(points[vertex])
        next_segment = next_point - current_point

        # Compute angle between segments
        angle = np.arctan2(next_segment[1], next_segment[0]) - np.arctan2(prev_segment[1], prev_segment[0])
        angle = np.abs(angle)

        # Normalize to [0, pi]
        if angle > np.pi:
            angle = 2 * np.pi - angle

        if angle < min_angle:
            min_angle = angle
            best_vertex = vertex

    return best_vertex

def left_right_decider(left_path, right_path, points):
    """
    Left-Right-Decider (LRD): Choose which path to extend to keep them equally advanced.

    Args:
        left_path: Current left path (list of vertex indices)
        right_path: Current right path (list of vertex indices)
        points: List of [x, y] coordinates

    Returns:
        'left' or 'right' indicating which path to extend
    """
    if len(left_path) < 2 and len(right_path) < 2:
        # Extend both equally at start
        return 'left' if len(left_path) <= len(right_path) else 'right'

    # Compute angle of last segment for each path
    def get_last_segment_angle(path):
        if len(path) < 2:
            return 0.0
        prev_point = np.array(points[path[-2]])
        curr_point = np.array(points[path[-1]])
        segment = curr_point - prev_point
        return np.arctan2(segment[1], segment[0])

    left_angle = get_last_segment_angle(left_path)
    right_angle = get_last_segment_angle(right_path)

    # Compute distance from origin for each path
    left_dist = np.linalg.norm(np.array(points[left_path[-1]]))
    right_dist = np.linalg.norm(np.array(points[right_path[-1]]))

    # Extend the path that is less advanced (closer to origin)
    if abs(left_dist - right_dist) > 0.5:  # If distance difference is significant
        return 'left' if left_dist < right_dist else 'right'

    # Otherwise, extend the path with smaller angle difference
    return 'left' if len(left_path) <= len(right_path) else 'right'

def enumerate_path_pairs_v2(graph, points, sl, sr, itmax=2500, heading_vector=None):
    """
    Correct implementation of Algorithm 2: Enumerate Path Pairs (EPP).

    Uses Next-Vertex-Decider (NVD) and Left-Right-Decider (LRD) heuristics
    to efficiently explore valid path pairs.

    Args:
        graph: Adjacency list {vertex_idx: [adjacent_vertices]}
        points: List of [x, y] coordinates
        sl: Starting vertex index for left path
        sr: Starting vertex index for right path
        itmax: Maximum iterations (default: 2500)
        heading_vector: Initial heading direction [dx, dy] (optional)

    Returns:
        List of valid path pairs: [([left_path], [right_path]), ...]
    """
    solutions = []

    def backtrack(left_path, right_path, visited_left, visited_right, iteration):
        nonlocal solutions

        if iteration >= itmax:
            return

        # Get unvisited adjacent vertices
        left_adj = [v for v in graph[left_path[-1]] if v not in visited_left]
        right_adj = [v for v in graph[right_path[-1]] if v not in visited_right]

        # If no more vertices, this is a terminal path pair
        if not left_adj and not right_adj:
            if len(left_path) > 1 and len(right_path) > 1:  # Valid path has at least 2 vertices
                solutions.append((left_path[:], right_path[:]))
            return

        # Try extending paths
        if left_adj and right_adj:
            # Use NVD to select best next vertices
            left_next = next_vertex_decider(left_path, left_adj, points, heading_vector)
            right_next = next_vertex_decider(right_path, right_adj, points, heading_vector)

            if left_next is not None and right_next is not None:
                # Try extending both paths
                new_left = left_path + [left_next]
                new_right = right_path + [right_next]
                new_pair = (new_left, new_right)

                # Check constraints
                if constraint_decider(new_pair):
                    # Add to solutions
                    solutions.append((new_left[:], new_right[:]))

                    # Continue search
                    backtrack(new_left, new_right,
                             visited_left | {left_next},
                             visited_right | {right_next},
                             iteration + 1)

        elif left_adj:
            # Only left path can be extended
            left_next = next_vertex_decider(left_path, left_adj, points, heading_vector)
            if left_next is not None:
                new_left = left_path + [left_next]
                new_pair = (new_left, right_path)

                if constraint_decider(new_pair):
                    solutions.append((new_left[:], right_path[:]))
                    backtrack(new_left, right_path,
                             visited_left | {left_next},
                             visited_right,
                             iteration + 1)

        elif right_adj:
            # Only right path can be extended
            right_next = next_vertex_decider(right_path, right_adj, points, heading_vector)
            if right_next is not None:
                new_right = right_path + [right_next]
                new_pair = (left_path, new_right)

                if constraint_decider(new_pair):
                    solutions.append((left_path[:], new_right[:]))
                    backtrack(left_path, new_right,
                             visited_left,
                             visited_right | {right_next},
                             iteration + 1)

    # Initialize heading vector if not provided (forward direction)
    if heading_vector is None:
        heading_vector = np.array([1.0, 0.0])
    else:
        heading_vector = np.array(heading_vector)
        heading_vector = heading_vector / (np.linalg.norm(heading_vector) + 1e-8)

    # Start backtracking
    backtrack([sl], [sr], {sl}, {sr}, 0)

    return solutions

def constraint_decider(path_pair):
    def Cseg(path_pair):
        for i in range(len(path_pair[0]) - 2):
            p1 = points[path_pair[0][i]]
            p2 = points[path_pair[0][i+1]]
            p3 = points[path_pair[0][i+2]]

            angle = np.abs(np.arctan2(
                p2[1] - p1[1], p2[0] - p1[0]
            ) - np.arctan2(
                p3[1] - p2[1], p3[0] - p2[0]
            ))

            if angle > np.pi / 2:
                return False
        return True
    
    def Cwidth(path_pair, wmin=2.5, wmax=6.5):
        left = path_pair[0]
        right = path_pair[1]
        
        for i in range(min(len(left), len(right))):
            left_point = np.array(points[left[i]])
            right_point = np.array(points[right[i]])
            width = np.linalg.norm(left_point - right_point)
            
            if not (wmin < width < wmax):
                return False
        return True

    
    return Cseg(path_pair) and Cwidth(path_pair)

def compute_features(path_pair, points):
    left_path, right_path = path_pair
    
    left_coords = np.array([points[i] for i in left_path])
    right_coords = np.array([points[i] for i in right_path])
    
    left_length = np.sum(np.linalg.norm(np.diff(left_coords, axis=0), axis=1))
    right_length = np.sum(np.linalg.norm(np.diff(right_coords, axis=0), axis=1))
    lane_length = (left_length + right_length) / 2 # feature 1
    
    num_left_points = len(left_path) # feature 2
    num_right_points = len(right_path) # feature 3
    
    widths = []
    for i in range(min(len(left_coords), len(right_coords))):
        width = np.linalg.norm(left_coords[i] - right_coords[i])
        widths.append(width)
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
                    # print(f"Skipping pair {i} and {j} due to mismatched path lengths.")
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
