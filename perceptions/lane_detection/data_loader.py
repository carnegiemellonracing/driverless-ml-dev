
import os
import yaml
import numpy as np
import math
from typing import Dict, List, Tuple
from perceptions.lane_detection.geo import within_range, within_cone
from perceptions.lane_detection.models import PerceptualFieldContext 

"""
Reading from the dataset
"""
dataset_path = f"{os.path.dirname(__file__)}/dataset/processed"

# 1. Load the dataset (boundaries and cone maps)
def load_numpy_data(path):
    return np.load(path)
map_data = [load_numpy_data(f"{dataset_path}/data_{i}.npz") for i in range(1, 10)]

# A cone map is a nx2 numpy array that maps point indices to their [x,y] position
cone_maps = [mp['points'] for mp in map_data]
# A left boundary is an array of indices corresponding to the left track boundary
left_boundaries = [mp['left_boundary_indices'] for mp in map_data]
# Same for right
right_boundaries = [mp['right_boundary_indices'] for mp in map_data]


def build_adjacency_graph(cone_map, dmax=5.0):
    """
    Build adjacency graph from cone_map dictionary.

    Args:
        cone_map: nx2 np array mapping cone_id to [x, y] coordinates
        dmax: Maximum distance threshold for adjacency (default: 5.0m)

    Returns:
        adjacency_list: Dictionary mapping point_idx to list of adjacent point indices
    """
    # Build adjacency list using geo.py logic
    adjacency_list = {i: [] for i in range(len(cone_map))}

    for i in range(len(cone_map)):
        for j in range(i + 1, len(cone_map)):
            distance = np.linalg.norm(np.array(cone_map[i]) - np.array(cone_map[j]))
            if distance <= dmax:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)

    return adjacency_list


def subgraph_add(subgraph, point, graph):
    """
    Add a point and its neighbors to the subgraph.
    """
    if point in subgraph:
        return subgraph
    
    #Create new entry
    subgraph[point] = []

    #Populate with all neighbors that are both in full graph and subgraph
    for n in graph[point]:
        if n in subgraph:
            subgraph[point].append(n)
            subgraph[n].append(point)
    
    return subgraph


def filter_points_within_range(car_pos: np.array, car_heading_rad: float,
                                cone_map: np.ndarray, graph: Dict[int, List[int]],
                                perceptual_range: float, cone_angle_rad:float = 120.0 * np.pi/180):
    """
    Returns:
    - Subgraph perceptual field
    """
    # Store all points within the perceptual range 
    subgraph = {}
    for id, point in enumerate(cone_map):
        if (within_range(point, car_pos, perceptual_range) and within_cone(point, car_pos, car_heading_rad, cone_angle_rad)): 
            subgraph = subgraph_add(subgraph, id, graph)

    return subgraph

def get_closest(point_id, boundary, cone_map):
    min_dist = float('inf')
    closest_id = []
    pt = cone_map[point_id]
    
    for id in boundary:
        point = cone_map[id]
        dist = np.linalg.norm(pt - point)
        if dist < min_dist:
            min_dist = dist
            closest_id = id
    return closest_id

def get_car_pos(left_id, right_boundary, cone_map, noise=False):
    closest_right_id = get_closest(left_id, right_boundary, cone_map)
    closest_right_pt = cone_map[closest_right_id]
    left_pt = cone_map[left_id]

    midpt = (left_pt + closest_right_pt) / 2

    angle_noise = np.random.normal(loc=0.0, scale=10 * math.pi/ 180, size=None) if noise else 0.0

    #Perpendicular so negative reciprocal
    flip = np.random.choice([-1,1]) if noise else 1.0
    dx = left_pt[0] - closest_right_pt[0]
    dy = left_pt[1] - closest_right_pt[1]
    car_heading_rad = flip * math.atan2(dx, dy) + angle_noise
    
    return midpt, car_heading_rad

def generate_perceptual_field_data(
    left_boundary, right_boundary, cone_map, perceptual_range=30, dmax=5
) -> List[PerceptualFieldContext]:
    """
    Returns a list of PerceptualFieldContext objects representing different viewpoints.
    """
    contexts = []
    # Build adjacency graph with cone_id mapping
    adjacency_list = build_adjacency_graph(cone_map, dmax)

    # Generate a perceptual field for each left boundary point
    for left_id in left_boundary:
        car_pos, car_heading_rad = get_car_pos(left_id, right_boundary, cone_map)
        subgraph = filter_points_within_range(
            car_pos, car_heading_rad, cone_map, adjacency_list, perceptual_range
        )

        # Get the set of visible indices from the subgraph
        visible_indices = set(subgraph.keys())

        # Get GT pairs
        closest_right_id = get_closest(left_id, right_boundary, cone_map)

        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices=visible_indices,
            adj_list=subgraph,
            car_pos=car_pos,
            car_heading=car_heading_rad,
            gt_left_idx=left_id,
            gt_right_idx=closest_right_id
        )
        contexts.append(ctx)

    return contexts

def generate_noisy_perceptual_field_data(
    left_boundary, right_boundary, cone_map,
    position_noise_std=0.3, false_positive_rate=0.2,
    perceptual_range=30, dmax=5, seed=None
) -> Tuple[List[PerceptualFieldContext], np.ndarray, List[int], List[int], List[int]]:
    """
    Generates noisy perceptual field contexts.
    Returns (contexts, noisy_map, left_indices, right_indices, fp_indices)
    """
    if seed is not None: np.random.seed(seed)
    
    # 1. Add position noise
    noisy_map = cone_map + np.random.normal(0, position_noise_std, cone_map.shape)
    
    # 2. Add False Positives (clutter)
    num_cones = len(cone_map)
    num_fp = int(num_cones * false_positive_rate)
    
    if num_fp > 0:
        # Generate "HARD" noise as requested:
        # 1. Ghost cones nearby existing cones (hard to filter by distance)
        # 2. Debris between lanes (hard to filter by width if they form false segments)

        
        fp_points = []
        for _ in range(num_fp):
            mode = np.random.choice(['ghost', 'debris'])
            
            if mode == 'ghost':
                # Pick a random existing cone and spawn a ghost near it
                idx = np.random.randint(0, len(noisy_map))
                ref_pt = noisy_map[idx]
                offset = np.random.uniform(-4.0, 4.0, 2) # Within 4m
                fp_points.append(ref_pt + offset)
            else:
                # Pick random ref point and add larger offset to simulate debris or cross-track clutter
                idx = np.random.randint(0, len(noisy_map))
                ref_pt = noisy_map[idx]
                # Random direction, dist 2-8m
                angle = np.random.uniform(0, 2*np.pi)
                dist = np.random.uniform(2.0, 8.0)
                offset = np.array([dist*np.cos(angle), dist*np.sin(angle)])
                fp_points.append(ref_pt + offset)
                
        fp_points = np.array(fp_points)
        combined_map = np.vstack([noisy_map, fp_points])
    else:
        combined_map = noisy_map
        fp_points = []

    fp_indices = list(range(num_cones, num_cones + num_fp)) if num_fp > 0 else []
    
    # 3. Build Graph
    adj = build_adjacency_graph(combined_map, dmax)
    
    # 4. Generate Contexts
    contexts = []
    
    # Use bounds to drive along track
    for i in range(0, len(left_boundary), 3): # Skip some for speed
        l_idx = left_boundary[i]
        
        # Car pos derived from (noisy) track points
        # Use simple midpoint of noisy points to simulate car being on track but seeing noise
        cp, ch = get_car_pos(l_idx, right_boundary, combined_map, noise=True) # Add car noise too
        
        subgraph = filter_points_within_range(cp, ch, combined_map, adj, perceptual_range)
        
        ctx = PerceptualFieldContext(
            cone_map=combined_map,
            visible_indices=set(subgraph.keys()),
            adj_list=subgraph,
            car_pos=cp,
            car_heading=ch
        )
        contexts.append(ctx)
        
    return contexts, combined_map, left_boundary, right_boundary, fp_indices

def generate_pairwise_training_data(left_boundaries, right_boundaries, cone_maps):
    """
    Generates dataset for pairwise ranking training.
    """
    from perceptions.lane_detection.dataset import LaneDetectionDataset
    # Triple boundaries and maps
    maps_data = list(zip(left_boundaries, right_boundaries, cone_maps))
    return LaneDetectionDataset(maps_data)

def collate_fn_pairwise(batch):
    """
    Collate function for pairwise ranking.
    Batch elements are (features_pair, iou_pair).
    Returns: (features1, features2, labels)
    """
    import torch
    
    f1_list, f2_list, label_list = [], [], []
    
    for feats, ious in batch:
        # feats shape: (2, D)
        # ious shape: (2,)
        
        f1 = feats[0]
        f2 = feats[1]
        
        iou1 = ious[0]
        iou2 = ious[1]
        
        # Label: 1 if cand1 better, 0 if cand2 better
        # We skip pairs with identical IoU? Or just use 0.5?
        # BCEWithLogitsLoss expects float targets (probabilities) for mixup or smooth labels, 
        # or 0/1 for hard classification.
        # Let's use hard 0/1, filtering equal cases or randomizing.
        
        if abs(iou1 - iou2) < 1e-4:
            continue # Skip ambiguous pairs to reduce noise
            
        label = 1.0 if iou1 > iou2 else 0.0
        
        f1_list.append(f1)
        f2_list.append(f2)
        label_list.append(label)
        
    if not f1_list:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
    return torch.stack(f1_list), torch.stack(f2_list), torch.tensor(label_list, dtype=torch.float32)