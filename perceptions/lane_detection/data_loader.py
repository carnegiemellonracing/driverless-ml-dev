import os
import yaml
import numpy as np
import math
from typing import Any, Dict, List
from perceptions.lane_detection.geo import within_range, within_cone
from perceptions.lane_detection.models import PerceptualFieldContext, Lane, Map, Graph, Point

"""
Reading from the dataset

"""
dataset_path = f"{os.path.dirname(__file__)}/dataset/processed"


# 1. Load the dataset (boundaries and cone maps)
def load_numpy_data(path):
    return np.load(path)


map_data = [load_numpy_data(f"{dataset_path}/data_{i}.npz") for i in range(1, 10)]

# A cone map is a nx2 numpy array that maps point indices to their [x,y] position
cone_maps = [mp["points"] for mp in map_data]
# A left boundary is an array of indices corresponding to the left track boundary
left_boundaries = [mp["left_boundary_indices"] for mp in map_data]
# Same for right
right_boundaries = [mp["right_boundary_indices"] for mp in map_data]


def build_adjacency_graph(cone_map: Map, dmax=5.0) -> Graph:
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


def subgraph_add(subgraph: Map, point: Point, graph: Graph):
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

    # Create new entry
    subgraph[point] = []

    # Populate with all neighbors that are both in full graph and subgraph
    for n in graph[point]:
        if n in subgraph:
            subgraph[point].append(n)
            subgraph[n].append(point)

    return subgraph


def filter_points_within_range(
    car_pos: Point,
    car_heading_rad: float,
    cone_map: Map,
    graph: Graph,
    perceptual_range: float,
    cone_angle_rad: float = 120.0 * np.pi / 180,
) -> Graph:
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
        if within_range(point, car_pos, perceptual_range) and within_cone(
            point, car_pos, car_heading_rad, cone_angle_rad
        ):
            subgraph = subgraph_add(subgraph, id, graph)

    return subgraph


def get_closest(point_id: int, boundary: Lane, cone_map: Map) -> int:
    """
    Takes point id, boundary (list of indicies), and dictionary that maps ids to point locations
    Returns the point closest to point_id within boundary, returns ID
    Will return point_id if it is in the boundary, will return [] if no points in boundary
    """
    min_dist = float("inf")
    closest_id = []
    pt = cone_map[point_id]

    for id in boundary:
        point = cone_map[id]
        dist = np.linalg.norm(pt - point)
        if dist < min_dist:
            min_dist = dist
            closest_id = id
    return closest_id


def get_car_pos(left_id, right_boundary, cone_map, noise=False) -> tuple[Point, float]:
    """
    Takes point on left boundary, entire right_boundary, cone_map, and optional noise parameters
    Returns potential car position and heading in radians
        position is midpoint between left point and closest right point
        heading is perpendicular to the line between the left point and closest right point
    """
    closest_right_id = get_closest(left_id, right_boundary, cone_map)
    closest_right_pt = cone_map[closest_right_id]
    left_pt = cone_map[left_id]

    midpt = (left_pt + closest_right_pt) / 2

    angle_noise = (
        np.random.normal(loc=0.0, scale=10 * math.pi / 180, size=None) if noise else 0.0
    )

    # Perpendicular so negative reciprocal
    flip = np.random.choice([-1, 1]) if noise else 1.0
    car_heading_rad = (
        flip
        * math.atan2(left_pt[0] - closest_right_pt[0], left_pt[1] - closest_right_pt[1])
        + angle_noise
    )
    return midpt, car_heading_rad

def augment_map(
    cone_map: Map,
    false_positive_rate: float = 0.1,
    dilatation: float = 0.2
) -> Map:
    """
    Adds false positive cone points to the cone map for data augmentation.
    
    The new points are sampled within a dilated version of the original cone map's bounding box.
    This simulates the detection of spurious cones around the actual track.
    
    Args:
        cone_map: Nx2 numpy array of cone positions
        false_positive_rate: Fraction of original cones to add as false points (e.g., 0.1 = 10%)
        dilatation: Expansion factor for bounding box (e.g., 0.2 = 20% expansion)
        
    Returns:
        Augmented cone map with false positive points appended
    """
    if len(cone_map) == 0:
        return cone_map
    
    # Calculate bounding box of original cone map
    min_point = cone_map.min(axis=0)  # (x_min, y_min)
    max_point = cone_map.max(axis=0)  # (x_max, y_max)
    
    # Calculate dimensions
    width = max_point[0] - min_point[0]
    height = max_point[1] - min_point[1]
    
    # Expand bounding box by dilatation factor in each direction
    expanded_min = min_point - np.array([width * dilatation, height * dilatation])
    expanded_max = max_point + np.array([width * dilatation, height * dilatation])
    
    # Calculate number of false points to add
    num_false_points = max(1, int(len(cone_map) * false_positive_rate))
    
    # Sample random points within the expanded region
    false_points = np.random.uniform(
        low=expanded_min,
        high=expanded_max,
        size=(num_false_points, 2)
    )
    
    # Append false points to cone map
    augmented_cone_map = np.vstack([cone_map, false_points])
    
    return augmented_cone_map


def generate_perceptual_field_data(
    left_boundary: Lane, right_boundary: Lane,
    cone_map: Map,
    perceptual_range: float = 30.0,
    dmax: float = 5.5,
    augment: bool = False,
    false_positive_rate: float = 0.1
) -> List[PerceptualFieldContext]:
    """
    Take a left and right boundary, the cone map, and some params.
    Returns a list of PerceptualFieldContext objects representing different viewpoints.

    Args:
        left_boundary: List of indices for left boundary cones
        right_boundary: List of indices for right boundary cones
        cone_map: Nx2 numpy array of cone positions (shared across all returned contexts)
        perceptual_range: Range in meters for visibility
        dmax: Maximum distance for adjacency graph

    Returns:
        List of PerceptualFieldContext objects (all sharing the same cone_map reference)
    """
    contexts = []

    if augment:
        cone_map = augment_map(cone_map, false_positive_rate)

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

        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices=visible_indices,
            adj_list=subgraph,
            car_pos=car_pos,
            car_heading=car_heading_rad,
            left_boundary=set(left_boundary) & visible_indices,
            right_boundary=set(right_boundary) & visible_indices
        )
        contexts.append(ctx)

    return contexts


def generate_all_perceptual_field_data(
    perceptual_range: int = 30,
    dmax: float = 5.5,
    augment: bool = False,
    false_positive_rate: float = 0.1,
) -> List[PerceptualFieldContext]:
    """
    Generate perceptual field data for all loaded maps.

    This is a convenience function that iterates over all loaded
    left_boundaries, right_boundaries, and cone_maps.

    Args:
        perceptual_range: Range in meters for visibility
        dmax: Maximum distance for adjacency graph

    Returns:
        List of PerceptualFieldContext objects from all maps
    """
    all_contexts = []
    for left_boundary, right_boundary, cone_map in zip(
        left_boundaries, right_boundaries, cone_maps
    ):
        contexts = generate_perceptual_field_data(
            left_boundary, right_boundary, cone_map, perceptual_range, dmax, augment, false_positive_rate
        )
        all_contexts.extend(contexts)
    return all_contexts
