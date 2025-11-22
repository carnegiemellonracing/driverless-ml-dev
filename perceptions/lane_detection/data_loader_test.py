import numpy as np
import math
from geo import within_range, within_range, within_cone
from data_loader import get_car_pos, filter_points_within_range

def test():
    within_range_test()
    within_cone_test()
    get_car_pos_no_noise_test()
    filter_points_within_range_test()

def within_range_test():
    car_pos = np.array([0, 0])
    perceptual_range = 5
    assert within_range(np.array([3.5, 3.5]), car_pos, perceptual_range) == True
    assert within_range(np.array([5, 5]), car_pos, perceptual_range) == False

def within_cone_test(): 
    car_pos = np.array([0, 0])
    car_heading_rad = math.pi / 4
    cone_angle_rad = 85 * math.pi / 180
    assert within_cone(np.array([0.5, 0]), car_pos, car_heading_rad, cone_angle_rad) == False
    assert within_cone(np.array([0, 0]), np.array([5, 0]), -math.pi/4, cone_angle_rad) == False
    assert within_cone(np.array([5 + math.sqrt(2)/2, -math.sqrt(2)/2]), np.array([5, 0]), -math.pi/4, cone_angle_rad) == True

def get_car_pos_no_noise_test(): 
    left_id = 0
    right_boundary = [1, 2]
    cone_map = {0: np.array([0, 0]), 1: np.array([1, 0]), 2: ([1, 1])}
    car_pos, heading = get_car_pos(left_id, right_boundary, cone_map)
    print(car_pos, heading)
    assert np.all(car_pos == np.array([0.5, 0])) and heading == -math.pi/2

def filter_points_within_range_test(): 
    car_pos = np.array([0, 0])
    car_heading_rad = math.pi / 4
    cone_rad = 120 * math.pi / 180
    perceptual_range = 10
    cone_map = {0: np.array([1, 0]), 1: np.array([0, 1]), 2: np.array([-1 ,-1])}
    graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    filtered_graph = {0: [1], 1: [0]} 
    assert filter_points_within_range(car_pos, car_heading_rad, cone_map, graph, perceptual_range, cone_rad) == filtered_graph
    assert filter_points_within_range(car_pos, car_heading_rad, cone_map, graph, perceptual_range, cone_rad) == filtered_graph


if __name__ == "__main__":
    test()