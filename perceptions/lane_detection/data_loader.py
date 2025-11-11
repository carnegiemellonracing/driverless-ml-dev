import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import random
import torch.nn.functional as F
import math
from geo import (
    angle_diff,
    within_cone
)
from collections import deque
import random

dataset_path = f"{os.path.dirname(__file__)}/dataset"


# 1. Load the dataset (boundaries and cone maps)
def load_yaml_data(path):
    with open(path, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def pad_sequence(sequence, max_len):
    """
    Pad sequences to the maximum length in the batch.
    """
    # Padding the sequence to the required size with zeros (or another padding value)
    padded_sequence = F.pad(sequence, (0, 0, 0, max_len - sequence.size(0)), value=0)
    return padded_sequence


# Load all boundaries and cone maps
boundary_paths = [f"{dataset_path}/boundaries_{i}.yaml" for i in range(1, 10)]
cone_map_paths = [f"{dataset_path}/cone_map_{i}.yaml" for i in range(1, 10)]

boundaries = [load_yaml_data(path) for path in boundary_paths]
cone_maps = [load_yaml_data(path) for path in cone_map_paths]

def augment_points(points, rotation_angle=15, scale_range=0.1, translation_range=1.0):
    points_arr = np.array(points)
    if points_arr.shape[0] == 0:
        return []

    # Rotation
    angle = np.radians(np.random.uniform(-rotation_angle, rotation_angle))
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    points_arr = points_arr @ rotation_matrix.T

    # Scaling
    scale = np.random.uniform(1 - scale_range, 1 + scale_range)
    points_arr = points_arr * scale

    # Translation
    translation = np.random.uniform(-translation_range, translation_range, size=2)
    points_arr = points_arr + translation

    return points_arr.tolist()

# ============================================================================
# PHASE 1: Path Enumeration Integration (ADDITIONS)
# ============================================================================

def build_adjacency_graph(cone_map, dmax=5.0):
    """
    Build adjacency graph from cone_map dictionary.

    Args:
        cone_map: Dictionary mapping cone_id to [x, y] coordinates
        dmax: Maximum distance threshold for adjacency (default: 5.0m)

    Returns:
        adjacency_list: Dictionary mapping point_idx to list of adjacent point indices
        points: List of [x, y] coordinates (for reference)
        cone_ids: List of cone IDs corresponding to each point index
    """
    # Convert cone_map to list of points
    cone_ids = list(cone_map.keys())
    points = [cone_map[cone_id] for cone_id in cone_ids]

    # Build adjacency list using geo.py logic
    adjacency_list = {i: [] for i in range(len(points))}

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
            if distance <= dmax:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)

    return adjacency_list, points, cone_ids

def enumerate_valid_paths(graph, points, left_start_idx, right_start_idx, itmax=2500, heading_vector=None):
    """
    Wrapper to enumerate valid path pairs using the correct Algorithm 2.

    Args:
        graph: Adjacency list from build_adjacency_graph()
        points: List of [x, y] coordinates
        left_start_idx: Starting vertex index for left boundary
        right_start_idx: Starting vertex index for right boundary
        itmax: Maximum iterations (default: 2500)
        heading_vector: Initial heading direction [dx, dy] (optional)

    Returns:
        List of valid path pairs: [([left_path], [right_path]), ...]
    """
    # Import the correct enumeration function from geo.py
    import sys
    import importlib.util

    # Load geo.py module
    geo_path = f"{os.path.dirname(__file__)}/geo.py"
    spec = importlib.util.spec_from_file_location("geo", geo_path)
    geo = importlib.util.module_from_spec(spec)

    # Set up global points variable for constraint_decider
    geo.points = points

    # Load the module
    spec.loader.exec_module(geo)

    # Call the correct enumeration function
    path_pairs = geo.enumerate_path_pairs_v2(graph, points, left_start_idx, right_start_idx,
                                             itmax=itmax, heading_vector=heading_vector)

    return path_pairs

# ============================================================================
# PHASE 2: Fix Data Representation (ADDITIONS)
# ============================================================================

def compute_path_pair_features(path_pair, points):
    """
    Compute 8-dimensional feature vector for a path pair.

    Uses the feature computation from geo.py.

    Args:
        path_pair: Tuple of (left_path, right_path) with vertex indices
        points: List of [x, y] coordinates

    Returns:
        8-dimensional feature vector as list
    """
    import importlib.util

    # Load geo.py module
    geo_path = f"{os.path.dirname(__file__)}/geo.py"
    spec = importlib.util.spec_from_file_location("geo", geo_path)
    geo = importlib.util.module_from_spec(spec)
    geo.points = points
    spec.loader.exec_module(geo)

    # Compute features using geo.py function
    features = geo.compute_features(path_pair, points)

    return features

def rank_path_pairs(path_pairs, ground_truth_left, ground_truth_right, points):
    """
    Rank path pairs by comparing to ground truth boundaries.

    Args:
        path_pairs: List of (left_path, right_path) tuples
        ground_truth_left: List of cone indices for true left boundary
        ground_truth_right: List of cone indices for true right boundary
        points: List of [x, y] coordinates

    Returns:
        List of (path_pair, score) tuples, sorted by score (best first)
    """
    scores = []

    for path_pair in path_pairs:
        left_path, right_path = path_pair

        # Score based on overlap with ground truth
        left_overlap = len(set(left_path) & set(ground_truth_left))
        right_overlap = len(set(right_path) & set(ground_truth_right))

        # Normalize by path length
        left_precision = left_overlap / len(left_path) if len(left_path) > 0 else 0
        right_precision = right_overlap / len(right_path) if len(right_path) > 0 else 0

        # Normalize by ground truth length (recall)
        left_recall = left_overlap / len(ground_truth_left) if len(ground_truth_left) > 0 else 0
        right_recall = right_overlap / len(ground_truth_right) if len(ground_truth_right) > 0 else 0

        # F1 score for left and right
        left_f1 = 2 * (left_precision * left_recall) / (left_precision + left_recall + 1e-8)
        right_f1 = 2 * (right_precision * right_recall) / (right_precision + right_recall + 1e-8)

        # Combined score (average F1)
        score = (left_f1 + right_f1) / 2

        # Bonus for longer paths (more complete detection)
        length_bonus = (len(left_path) + len(right_path)) / 100.0

        total_score = score + length_bonus
        scores.append((path_pair, total_score))

    # Sort by score (descending)
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)

    return ranked

def create_pairwise_comparisons(ranked_path_pairs, points, num_pairs_per_sample=5):
    """
    Create pairwise comparison samples from ranked path pairs.

    Args:
        ranked_path_pairs: List of (path_pair, score) tuples sorted by score
        points: List of [x, y] coordinates
        num_pairs_per_sample: How many comparison pairs to create per path

    Returns:
        List of (features1, features2, label) tuples where:
        - features1, features2 are 8D feature vectors
        - label is 1 if path1 is better, 0 if path2 is better
    """
    pairwise_data = []

    n = len(ranked_path_pairs)
    if n < 2:
        return pairwise_data

    # Create comparisons: better paths vs worse paths
    for i in range(min(n, num_pairs_per_sample * 2)):
        path_pair_i, score_i = ranked_path_pairs[i]
        features_i = compute_path_pair_features(path_pair_i, points)

        # Compare with worse paths
        for j in range(i + 1, min(n, i + 1 + num_pairs_per_sample)):
            path_pair_j, score_j = ranked_path_pairs[j]
            features_j = compute_path_pair_features(path_pair_j, points)

            # Skip if features are invalid
            if any(np.isnan(features_i)) or any(np.isnan(features_j)):
                continue

            # Label: 1 if first path is better (has higher score)
            label = 1 if score_i > score_j else 0

            pairwise_data.append((features_i, features_j, label))

    return pairwise_data

class PairwiseRankingDataset(Dataset):
    """
    Dataset for pairwise ranking of lane candidates.

    Returns (features1, features2, label) where features are 8D vectors.
    """
    def __init__(self, pairwise_data, augment=False):
        """
        Args:
            pairwise_data: List of (features1, features2, label) tuples
            augment: Whether to apply augmentation (currently not implemented for features)
        """
        self.data = pairwise_data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features1, features2, label = self.data[idx]

        # Convert to tensors
        f1_tensor = torch.tensor(features1, dtype=torch.float32)
        f2_tensor = torch.tensor(features2, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return f1_tensor, f2_tensor, label_tensor

def collate_fn_pairwise(batch):
    """
    Collate function for pairwise ranking dataset.

    Args:
        batch: List of (features1, features2, label) tuples

    Returns:
        Stacked tensors: (batch_features1, batch_features2, batch_labels)
    """
    features1_list = [item[0] for item in batch]
    features2_list = [item[1] for item in batch]
    labels_list = [item[2] for item in batch]

    # Stack into batches (no padding needed - fixed size features)
    features1_batch = torch.stack(features1_list, dim=0)
    features2_batch = torch.stack(features2_list, dim=0)
    labels_batch = torch.stack(labels_list, dim=0)

    return features1_batch, features2_batch, labels_batch

# ============================================================================
# PHASE 3: Data Generation Pipeline (ADDITIONS)
# ============================================================================

def find_starting_points(boundary_cone_ids, cone_ids):
    """
    Convert ground truth cone IDs to point indices in the graph.

    Args:
        boundary_cone_ids: List of cone IDs from boundaries (e.g., [49, 17, 13, ...])
        cone_ids: List of cone IDs from build_adjacency_graph (ordered list)

    Returns:
        List of point indices corresponding to boundary cone IDs
    """
    # Create mapping from cone_id to point_index
    cone_id_to_idx = {cone_id: idx for idx, cone_id in enumerate(cone_ids)}

    # Convert boundary cone IDs to point indices
    point_indices = []
    for cone_id in boundary_cone_ids:
        if cone_id in cone_id_to_idx:
            point_indices.append(cone_id_to_idx[cone_id])

    return point_indices

def augment_path_pair_points(path_pair, points, rotation_angle=15, scale_range=0.1, translation_range=1.0):
    """
    Apply same transformation to both paths to preserve spatial relationship.

    Args:
        path_pair: Tuple of (left_path, right_path) with vertex indices
        points: List of [x, y] coordinates
        rotation_angle: Max rotation in degrees
        scale_range: Scale variation range
        translation_range: Translation range in meters

    Returns:
        Augmented points list (same structure as input)
    """
    # Get all points involved in the path pair
    left_path, right_path = path_pair
    all_indices = set(left_path + right_path)

    # Convert to numpy array for transformation
    points_arr = np.array(points)

    # Apply transformations to ALL points (to maintain relationships)
    # Rotation
    angle = np.radians(np.random.uniform(-rotation_angle, rotation_angle))
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    points_arr = points_arr @ rotation_matrix.T

    # Scaling
    scale = np.random.uniform(1 - scale_range, 1 + scale_range)
    points_arr = points_arr * scale

    # Translation
    translation = np.random.uniform(-translation_range, translation_range, size=2)
    points_arr = points_arr + translation

    return points_arr.tolist()

def generate_pairwise_training_data(boundaries, cone_maps,
                                   perceptual_range=30,
                                   perceptual_cone_deg=120,
                                   noise_rate=0.1, 
                                   dmax=5.0,
                                   max_paths=50,
                                   num_comparisons_per_sample=5):
    """
    Complete pipeline: Raw data → Pairwise training samples.

    Pipeline:
    1. Add noise to cone observations
    2. Build adjacency graph
    3. Enumerate valid path pairs
    4. Rank against ground truth
    5. Create pairwise comparisons
    6. Return PairwiseRankingDataset

    Args:
        boundaries: List of boundary dicts with 'left' and 'right' cone IDs
        cone_maps: List of cone_map dicts mapping cone_id to [x, y]
        perceptual_range: Range for filtering cones (meters)
        noise_rate: Rate of noise addition to points
        dmax: Maximum distance for adjacency (meters)
        max_paths: Maximum path pairs to enumerate per sample
        num_comparisons_per_sample: Number of pairwise comparisons per path

    Returns:
        PairwiseRankingDataset ready for training
    """
    all_pairwise_data = []

    for boundary, cone_map in zip(boundaries, cone_maps):
        left_boundary = boundary['left']
        right_boundary = boundary['right']

        # Build adjacency graph from cone_map
        adjacency_list, points, cone_ids = build_adjacency_graph(cone_map, dmax=dmax)

        # Convert boundary cone IDs to indices for filtering
        left_indices = find_starting_points(left_boundary, cone_ids)
        right_indices = find_starting_points(right_boundary, cone_ids)
        
        perceptual_fields = []
        prev_heading_deg = 0.0 # By convention - can change
        right_index = -1
        min_dist = float('inf')
        
        # Find closest right index
        for left_index in left_indices:
            for index in adjacency_list[left_index]:   
                left_x, left_y = points[left_index]
                right_x, right_y = points[index]
                new_dist = np.linalg.norm([left_x - right_x, left_y - right_y])
                if points[index] in right_boundary and new_dist < min_dist:
                    min_dist = new_dist
                    right_index = index
            if right_index == -1:
                continue

            left_x, left_y = points[left_index] 
            right_x, right_y = points[right_index] 
            
            # Define midpoint to be the car's position. TODO: introduce more variation (pick point bounded by 4 boundary points)
            mid_x = (left_x + right_x)/2
            mid_y = (left_y + right_y)/2
            
            # Choose the heading (+0 vs 180) most consistent with the heading of the previous perceptual field
            deg_variation = 30.0
            car_heading_deg = math.degrees(math.atan2(left_y - right_y, left_x - right_x)) + (random.random() * deg_variation)
            if angle_diff(prev_heading_deg, car_heading_deg + 180) < angle_diff(prev_heading_deg, car_heading_deg):
                car_heading_deg = (car_heading_deg + 180) % 360 
            
            # Use the adjacency list to efficiently filter out cones not in the prescribed range/cone
            subgraph = dfs_filter([mid_x, mid_y], car_heading_deg, perceptual_cone_deg, perceptual_range, adjacency_list, points, start=0, condition_fn=filter_points_within_range)
            sub_left_indices = [left_index for left_index in left_indices if left_index in subgraph.keys()]
            sub_right_indices = [right_index for right_index in right_indices if right_index in subgraph.keys()]
            prev_heading_deg = car_heading_deg
            
            # TODO: decide if we want more noise on top of current data
            perceptual_fields.append((subgraph, sub_left_indices, sub_right_indices, left_index, right_index, car_heading_deg))

        for data in perceptual_fields:
            subgraph = data[0]
            sub_left_indices = data[1]
            sub_right_indices = data[2]
            left_index = data[3]
            right_index = data[4]
            car_heading_deg = data[5]
            
            # Enumerate valid paths
            path_pairs = enumerate_valid_paths(
                subgraph, [points[key] for key in subgraph.keys()],
                left_index, right_index,
                itmax=max_paths
            )

            if len(path_pairs) == 0:
                print(f"Warning: No valid path pairs found, skipping...")
                continue

            # Step 4: Rank path pairs against ground truth
            ranked_pairs = rank_path_pairs(path_pairs, sub_left_indices, sub_right_indices, points)

            # Step 5: Create pairwise comparisons
            pairwise_data = create_pairwise_comparisons(
                ranked_pairs, points,
                num_pairs_per_sample=num_comparisons_per_sample
            )

            all_pairwise_data.extend(pairwise_data)

    # Return dataset
    if len(all_pairwise_data) == 0:
        print("Warning: No pairwise data generated!")
        return PairwiseRankingDataset([])

    print(f"Generated {len(all_pairwise_data)} pairwise training samples")
    return PairwiseRankingDataset(all_pairwise_data, augment=False)


def dfs_filter(midpt, heading_deg, cone_deg, perceptual_range, adjacency_list, points, start, condition_fn, visited=None):
    if visited is None:
        visited = {}

    # If the node is already visited, skip
    if start in visited:
        return visited

    # Initialize this node in the subgraph
    visited[start] = []

    for neighbor in adjacency_list[start]:
        if condition_fn(midpt, neighbor, points, heading_deg, cone_deg, perceptual_range):
            visited[start].append(neighbor)
            if neighbor not in visited:
                dfs_filter(adjacency_list, points, neighbor, condition_fn, visited)

    return visited

def filter_points_within_range(midpt, neighbor, points, heading_deg, cone_deg, perceptual_range):
    """
    Returns:
    - List of all points within a certain range defined on the midpoint of two left and right boundary points
    - List of only boundary points that are filtered
    """
    filtered = []
    x, y = points[neighbor]
    mid_x, mid_y = midpt
    if (within_cone(x, y, mid_x, mid_y, heading_deg, cone_deg) and (x - mid_x)**2 + (y - mid_y)**2 <= perceptual_range**2): 
        filtered.append([x, y])
            
    return filtered

def process_dataset_pairwise(boundaries_list, cone_maps_list, **kwargs):
    """
    Process entire dataset of boundaries and cone_maps.

    Args:
        boundaries_list: List of boundary dicts
        cone_maps_list: List of cone_map dicts
        **kwargs: Additional arguments for generate_pairwise_training_data

    Returns:
        PairwiseRankingDataset combining all samples
    """
    return generate_pairwise_training_data(boundaries_list, cone_maps_list, **kwargs)

