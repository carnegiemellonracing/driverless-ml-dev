import os
import yaml
import numpy as np
import math
from typing import Any, Dict, List
from perceptions.lane_detection.geo import within_range, within_cone
from perceptions.lane_detection.models import (
    PerceptualFieldContext,
    Lane,
    Map,
    Graph,
    Point,
)

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

    if noise:
        # Add lateral noise (shift towards left or right boundary)
        # Vector from left to right
        vec = closest_right_pt - left_pt
        # Random shift between -20% and +20% of lane width
        shift = (np.random.random() - 0.5) * 0.4
        midpt = midpt + vec * shift

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


def generate_perceptual_field_data(
    left_boundary: Lane,
    right_boundary: Lane,
    cone_map: Map,
    perceptual_range: float = 30.0,
    dmax: float = 5,
    samples_per_point: int = 1,
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
    # Build adjacency graph with cone_id mapping
    adjacency_list = build_adjacency_graph(cone_map, dmax)

    # Generate a perceptual field for each left boundary point
    for left_id in left_boundary:
        for i in range(samples_per_point):
            # Use noise for all samples if we are oversampling,
            # but keep the first one clean if we only want 1 sample?
            # Actually, if samples_per_point > 1, let's make i=0 clean and others noisy.
            use_noise = i > 0 if samples_per_point > 1 else False

            car_pos, car_heading_rad = get_car_pos(
                left_id, right_boundary, cone_map, noise=use_noise
            )
            subgraph = filter_points_within_range(
                car_pos, car_heading_rad, cone_map, adjacency_list, perceptual_range
            )

            # Skip empty graphs
            if len(subgraph) < 3:
                continue

            # Get the set of visible indices from the subgraph
            visible_indices = set(subgraph.keys())

            ctx = PerceptualFieldContext(
                cone_map=cone_map,
                visible_indices=visible_indices,
                adj_list=subgraph,
                car_pos=car_pos,
                car_heading=car_heading_rad,
                left_boundary=set(left_boundary) & visible_indices,
                right_boundary=set(right_boundary) & visible_indices,
            )
            contexts.append(ctx)

    return contexts


def generate_data_for_maps(
    map_indices: List[int] = None,
    perceptual_range: int = 30,
    dmax: float = 5.0,
    samples_per_point: int = 1,
    augment_mirror: bool = False,
) -> List[PerceptualFieldContext]:
    """
    Generate perceptual field data for specified maps.

    Args:
        map_indices: List of indices (0-based) of maps to use. If None, uses all.
        perceptual_range: Range in meters
        dmax: Graph adjacency max distance
        samples_per_point: Number of samples per boundary point (1 = clean, >1 = noisy samples)
        augment_mirror: Whether to generate mirrored (flipped Y) versions of maps

    Returns:
        List of PerceptualFieldContext objects
    """
    all_contexts = []

    total_maps = len(cone_maps)
    if map_indices is None:
        map_indices = range(total_maps)

    for idx in map_indices:
        if idx < 0 or idx >= total_maps:
            continue

        left_boundary = left_boundaries[idx]
        right_boundary = right_boundaries[idx]
        cone_map = cone_maps[idx]

        # 1. Original Map
        contexts = generate_perceptual_field_data(
            left_boundary,
            right_boundary,
            cone_map,
            perceptual_range,
            dmax,
            samples_per_point=samples_per_point,
        )
        all_contexts.extend(contexts)

        # 2. Mirrored Map (Optional)
        if augment_mirror:
            mirrored_map = cone_map.copy()
            mirrored_map[:, 1] *= -1  # Invert Y coordinate

            # Swap left and right boundaries for the mirrored map
            contexts_mirror = generate_perceptual_field_data(
                right_boundary,
                left_boundary,
                mirrored_map,
                perceptual_range,
                dmax,
                samples_per_point=samples_per_point,
            )
            all_contexts.extend(contexts_mirror)

    return all_contexts


# Backwards compatibility wrapper if needed, or update call sites
def generate_all_perceptual_field_data(
    perceptual_range: int = 30, dmax: float = 5.0
) -> List[PerceptualFieldContext]:
    return generate_data_for_maps(
        None, perceptual_range, dmax, samples_per_point=5, augment_mirror=True
    )
