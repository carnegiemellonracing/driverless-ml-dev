import numpy as np
import matplotlib.pyplot as plt

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

# Example usage:
# points = [(0, 0), (0, 3), (0, 6), (0, 9), (0, 12), (4, 0), (4, 3), (4, 6), (4, 9), (4, 12)]  # Example set of 2D points
points = [(0, 0), (0, 3), (4, 0), (4, 3)]  # Example set of 2D points
dmax = 5
adj_list = construct_adjacency_list(points, 4)
import pdb; pdb.set_trace()
path_pairs = enumerate_path_pairs(adj_list, 0, 2)
print(path_pairs)
import pdb; pdb.set_trace()

feature_pairs = generate_feature_pairs(path_pairs, points)

for pair in feature_pairs:
    print(pair)
