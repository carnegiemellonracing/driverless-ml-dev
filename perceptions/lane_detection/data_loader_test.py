import numpy as np
import math
from geo import within_range, within_range, within_cone
from data_loader import get_car_pos, filter_points_within_range, get_closest, build_adjacency_graph, generate_perceptual_field_data
import unittest

class TestDataLoader(unittest.TestCase):
    def within_range_test(self):
        car_pos = np.array([0, 0])
        perceptual_range = 5
        self.assertTrue(within_range(np.array([3.5, 3.5]), car_pos, perceptual_range), "Failed: missed point within range")
        self.assertFalse(within_range(np.array([5, 5]), car_pos, perceptual_range), "Failed: counted point outside range")

    def within_cone_test(self): 
        car_pos = np.array([0, 0])
        car_heading_rad = math.pi / 4
        cone_angle_rad = 85 * math.pi / 180
        self.assertFalse(within_cone(np.array([0.5, 0]), car_pos, car_heading_rad, cone_angle_rad), "Failed: counted point outside cone")
        self.assertFalse(within_cone(np.array([0, 0]), np.array([5, 0]), -math.pi/4, cone_angle_rad), "Failed: counted point outside cone")
        self.assertTrue(within_cone(np.array([5 + math.sqrt(2)/2, -math.sqrt(2)/2]), np.array([5, 0]), -math.pi/4, cone_angle_rad), "Failed: missed point inside cone")

    def get_closest_test(self):
        cone_map = np.array([[0, 0],
                            [1, 0],
                            [1, 1],
                            [1, 10],
                            [2, 11]])
        self.assertEqual(get_closest(0, [1,2], cone_map), 1, "missed closest")
        self.assertEqual(get_closest(0, [1,3], cone_map), 1, "missed closest")
        self.assertEqual(get_closest(4, [1,3], cone_map), 3, "missed closest")


    def get_car_pos_no_noise_test(self): 
        left_id = 0
        right_boundary = [1, 2]
        cone_map = np.array([[0, 0],
                            [1, 0],
                            [1, 1]])
        car_pos, heading = get_car_pos(left_id, right_boundary, cone_map, noise=False)
        self.assertTrue(np.all(car_pos == np.array([0.5, 0])), "Failed: wrong car position")
        self.assertEqual(heading, -math.pi/2, "Failed: wrong car heading")

    def filter_points_within_range_test(self): 
        car_pos = np.array([0, 0])
        car_heading_rad = math.pi / 4
        cone_rad = 120 * math.pi / 180
        perceptual_range = 10
        cone_map = np.array([[1, 0],
                            [0, 1],
                            [-1, -1]])
        graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        filtered_graph = {0: [1], 1: [0]} 
        self.assertEqual(filter_points_within_range(car_pos, car_heading_rad, cone_map, graph, perceptual_range, cone_rad), filtered_graph, "Failed: bad filtering")

    def build_adjacency_graph_test(self): 
        cone_map = np.array([[0, 0],
                            [1, 0],
                            [1, 1],
                            [1, 10],
                            [2, 11]])
        self.assertEqual(build_adjacency_graph(cone_map), {0: [1,2], 1: [0,2], 2: [0, 1], 3: [4], 4: [3]}, "Failed: wrong adjacency graph creation")

        cone_map = np.array([[0, 0],
                            [1, 0],
                            [1, 1]])
        
        self.assertEqual(build_adjacency_graph(cone_map), {0: [1,2], 1: [0,2], 2: [0, 1]}, "Failed: wrong adjacency graph creation")

        cone_map = np.array([[1, 0],
                            [0, 1],
                            [-1,-1]])
        self.assertEqual(build_adjacency_graph(cone_map), {0: [1,2], 1: [0,2], 2: [0, 1]}, "Failed: wrong adjacency graph creation")

    def generate_perceptual_field_data_test(self):
        cone_map = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1],
                            [2, 0],
                            [0, 2],
                            [-1, 0],
                            [0, -1],
                            [-1,-1],
                            [np.cos(62 * math.pi/180), np.sin(62 * math.pi/180) + 0.5]])
        
        left_boundary = [1, 3, 5]
        right_boundary = [0, 2, 4]

        perceptual_f = generate_perceptual_field_data(left_boundary, right_boundary, cone_map, perceptual_range=30, dmax=5)
        car_pos, car_heading_rad, subgraph = perceptual_f[0]
        self.assertTrue(np.all(car_pos == np.array([0, 0.5])), "Failed: wrong car position")
        self.assertEqual(car_heading_rad, 0.0, "Failed: wrong car heading")
        self.assertEqual(subgraph, {2: [3, 4], 3: [2, 4], 4: [2, 3]}, "Failed: wrong subgraph")

if __name__ == "__main__":
    unittest.main()