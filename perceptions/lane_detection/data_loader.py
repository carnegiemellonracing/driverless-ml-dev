import os
import yaml
import numpy as np
import math
from typing import Dict, List
from geo import within_range, within_cone 
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
    
    Args:
        subgraph: Current subgraph dict
        point: Point id
        graph: Adjacency list 
    """
    if point in subgraph:
        print("Subgraph add detected duplicate point")
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
                                perceptual_range: float, cone_angle_rad:float = 120.0):
    """
    Returns:
    - Subgraph perceptual field
    
    Args:
        left_point: Cone ID of the left boundary point to use as reference
        left_boundary: List of cone IDs that are left boundary
        right_boundary: List of cone IDs that are right boundary
        cone_map: nx2 np.array mapping cone_id to [x, y] coordinates
        perceptual_range: Range in meters
        graph: Adjacency list
    """
    # Store all points within the perceptual range 
    subgraph = {}
    for id, point in enumerate(cone_map):
        if (within_range(point, car_pos, perceptual_range) and within_cone(point, car_pos, car_heading_rad, cone_angle_rad)): 
            subgraph = subgraph_add(subgraph, id, graph)

    return subgraph

def get_closest(point_id, boundary, cone_map):
    """
        Takes point id, boundary (list of indicies), and dictionary that maps ids to point locations
        Returns the point closest to point_id within boundary, returns ID
        Will return point_id if it is in the boundary, will return [] if no points in boundary
    """
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
    """
        Takes point on left boundary, entire right_boundary, cone_map, and optional noise parameters
        Returns potential car position and heading in radians
            position is midpoint between left point and closest right point
            heading is perpendicular to the line between the left point and closest right point
    """
    left_pt = cone_map[left_id]
    closest_right_id = get_closest(left_id, right_boundary, cone_map)
    closest_right_pt = cone_map[closest_right_id]
            
    midpt = left_pt + closest_right_pt / 2

    angle_noise = np.random.normal(loc=0.0, scale=10 * math.pi/ 180, size=None) if noise else 0.0

    #Perpendicular so negative reciprocal
    flip = np.random.choice([-1,1]) if noise else 1.0
    car_heading_rad = flip * math.atan2(
        left_pt[0] - closest_right_pt[0], left_pt[1] - closest_right_pt[1]) + angle_noise
    return midpt, car_heading_rad

def generate_perceptual_field_data(
    left_boundary, right_boundary, cone_map, perceptual_range=30, dmax=5
):
    perceptual_field_data = []
    # Build adjacency graph with cone_id mapping
    adjacency_list = build_adjacency_graph(cone_map, dmax)

    # Filter out points outside perceptual range. Generate a perceptual field using every left point
    for left_id in left_boundary:
        car_pos, car_heading_rad = get_car_pos(left_id, right_boundary, cone_map)
        subgraph = filter_points_within_range(
            car_pos, car_heading_rad, cone_map, adjacency_list, perceptual_range
        )
        perceptual_field_data.append((car_pos, car_heading_rad, subgraph))

    return perceptual_field_data