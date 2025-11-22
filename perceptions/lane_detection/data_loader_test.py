import numpy as np
import math

def angle_diff_test(): # point1, point 2
    assert angle_diff(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == math.pi / 2
    assert angle_diff(np.array([math.sqrt(2)/2.0, math.sqrt(2)/2.0]), np.array([0.0, 0.0])) == math.pi/4
    
def within_range_test(): # point1,  car_pos, perceptual_range
    car_pos = np.array([0.0, 0.0])
    perceptual_range = 5.0
    assert within_range(np.array([3.5, 3.5]), car_pos, perceptual_range) == True
    assert within_range(np.array([5.0, 5.0]), car_pos, perceptual_range) == False

def within_cone_test(): # point, car_pos, car_heading_rad, cone_angle_rad
    car_pos = np.array([0.0, 0.0])
    car_heading_rad = math.pi / 4
    cone_angle_rad = 85.0 * math.pi / 180.0
    assert within_cone(np.array([0.5, 0.0]), car_pos, car_heading_rad, cone_angle_rad) == False

def filter_points_within_range_test(): # car pos, car heading, cone rad, perceptual range, cone_map, adjacency graph
    car_pos = np.array([0.0, 0.0])
    car_heading_rad = math.pi / 4
    cone_rad = 120.0 * math.pi / 180.0
    perceptual_range = 10.0
    cone_map = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0]), 2: np.array([-1.0 ,-1.0])}
    adjacency_graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    filtered_graph = {0: [1], 1: [0]} 
    assert filter_points_within_range(car_pos, car_heading_rad, cone_rad, perceptual_range, cone_map, adjacency_graph) == filtered_graph

# loop over for indices 

# left (id]), left bound (ids]), right bound (id]), cone map (id to point]), perceptual range, graph


