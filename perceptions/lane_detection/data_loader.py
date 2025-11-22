import os
import yaml
import numpy as np
import math
from typing import Dict, List
"""
Reading from the dataset

"""
dataset_path = f"{os.path.dirname(__file__)}/dataset"

# 1. Load the dataset (boundaries and cone maps)
def load_yaml_data(path):
    with open(path, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)
    
def load_cone_map(map: Dict[int, List[int]]) -> Dict[int, np.ndarray]:
    """Takes yaml dictionary and turns points into numpy arrays"""
    return {ID:np.array(point) for ID, point in map.items()}

# Load all boundaries and cone maps
boundary_paths = [f"{dataset_path}/boundaries_{i}.yaml" for i in range(1, 10)]
cone_map_paths = [f"{dataset_path}/cone_map_{i}.yaml" for i in range(1, 10)]

boundaries = [load_yaml_data(path) for path in boundary_paths]
cone_maps = [load_cone_map(load_yaml_data(path)) for path in cone_map_paths]


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


def angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)

def within_cone(x, y, mid_x, mid_y, car_heading_deg, cone_angle_deg):
    """
    Checks if the given coordinates are within the "cone" around car heading with angle cone_angle and starting at (mid_x, mid_y)
    Compares angle formed by the slope of coordinates (relative to (mid_x, mid_y)) to car heading
    """
    vec_x = x - mid_x
    vec_y = y - mid_y
    
    heading_rad = math.radians(car_heading_deg)
    hx = math.cos(heading_rad)   
    hy = math.sin(heading_rad)

    dot = vec_x * hx + vec_y * hy
    
    if dot > 0:
        point_angle = math.degrees(math.atan2(vec_y, vec_x))
        if angle_diff(point_angle, car_heading_deg) <= cone_angle_deg / 2:
            return True
    return False


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


def filter_points_within_range(car_pos: np.array, car_heading: float,
                                cone_map: Dict[int, np.ndarray], graph: Dict[int, List[int]],
                                perceptual_range: float, CONE_ANGLE_DEG:float = 120.0):
    """
    Returns:
    - Subgraph perceptual field
    
    Args:
        left_point: Cone ID of the left boundary point to use as reference
        left_boundary: List of cone IDs that are left boundary
        right_boundary: List of cone IDs that are right boundary
        cone_map: Dict mapping cone_id to [x, y] coordinates
        perceptual_range: Range in meters
        graph: Adjacency list
    """
    mid_x, mid_y = car_pos
    car_heading_deg = math.degrees(car_heading)

    # Store all points within the perceptual range 
    subgraph = {}
    for point, _ in cone_map.items():
        x, y = cone_map.get(point)
        if (within_cone(x, y, mid_x, mid_y, car_heading_deg, CONE_ANGLE_DEG) and
             (x - mid_x)**2 + (y - mid_y)**2 <= perceptual_range**2): 
            subgraph = subgraph_add(subgraph, point, graph)

    return subgraph


def get_car_pos(left_point, right_boundary, cone_map):
    left_x, left_y = cone_map.get(left_point)
    closest_right_x, closest_right_y = None, None
    closest_right = None
    min_dist_squared = float("inf")
    
    # Find right boundary point closest to left point
    for right_point in right_boundary:
        right_x, right_y = cone_map.get(right_point)
        new_dist_squared = (right_x - left_x)**2 + (right_y - left_y)**2
        if new_dist_squared < min_dist_squared:
            closest_right_x = right_x
            closest_right_y = right_y
            closest_right = right_point
            min_dist_squared = new_dist_squared
    
    # Define the midpoint
    mid_x = (left_x + closest_right_x)/2
    mid_y = (left_y + closest_right_y)/2

    # Angle convention in line with article - 0 is vertical axis, pos angle to left, neg angle to right
    angle_noise = np.random.normal(loc=0.0, scale=10.0 * math.pi/180, size=None)
    #Perpendicular so negative reciprocal
    flip = np.random.choice([-1,1])
    car_heading_rad = flip * math.atan2(left_x - closest_right_x, closest_right_y - left_y) + angle_noise
    return (mid_x, mid_y), car_heading_rad


def generate_perceptual_field_data(
    boundary, cone_map, perceptual_range=30, dmax=5
):
    perceptual_field_data = []
    # Build adjacency graph with cone_id mapping
    adjacency_list, points, cone_ids = build_adjacency_graph(cone_map, dmax)
    left_boundary = boundary["left"]
    right_boundary = boundary["right"]

    # Filter out points outside perceptual range. Generate a perceptual field using every left point
    for left_point in left_boundary:
        car_pos, car_heading_rad = get_car_pos(left_point, right_boundary, cone_map)
        subgraph = filter_points_within_range(
            car_pos, car_heading_rad, left_boundary, right_boundary, cone_map, adjacency_list, perceptual_range
        )
        perceptual_field_data.append((car_heading_rad, paths, subgraph, left_subset, right_subset))

    return perceptual_field_data