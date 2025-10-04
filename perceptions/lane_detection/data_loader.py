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
import sys

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


# 2. Preprocess the data
def generate_perceptual_field_data(
    boundaries, cone_maps, perceptual_range=30, noise_rate=0.1
):
    perceptual_field_data = []
    for boundary, cone_map in zip(boundaries, cone_maps):
        left_boundary = boundary["left"]
        right_boundary = boundary["right"]

        # Filter out points outside perceptual range
        for left_point in left_boundary:
            filtered_points, filtered_boundary = filter_points_within_range(left_point, left_boundary, right_boundary, cone_map, perceptual_range)
            noisy_points = add_noise(filtered_points, noise_rate)
            perceptual_field_data.append((noisy_points, filtered_boundary))
    
    return perceptual_field_data

def filter_points_within_range(left_point, left_boundary, right_boundary, cone_map, perceptual_range):
    all_filtered = []
    boundary_filtered = []
    left_x, left_y = cone_map.get(left_point)
    closest_right_x , closest_right_y = None
    min_dist_squared = sys.maxsize 
    
    for right_point in right_boundary:
        right_x, right_y = cone_map.get(right_point)
        new_dist_squared = (right_x - left_x)**2 + (right_y - left_y)**2
        if new_dist_squared < min_dist_squared:
            closest_right_x = right_x
            closest_right_y = right_y
            min_dist_squared = new_dist_squared
             
    mid_x = (left_x + closest_right_x)/2
    mid_y = (left_y + closest_right_y)/2
    
    for _, point in cone_map.items():
        x, y = cone_map.get(point)
        if (x - mid_x)**2 + (y - mid_y)**2 <= perceptual_range**2:  # Check if the point is within the perceptual range
            if (point in left_boundary) or (point in right_boundary):
                boundary_filtered.append([x, y])    
            all_filtered.append([x, y])
            
    return (all_filtered, boundary_filtered)


def add_noise(points, noise_rate, perceptual_range=30, false_positive_rate=0.1):
    noisy_points = []
    for point in points:
        if random.random() < noise_rate:
            # Simulate a random false positive by adding noise to the point
            noise = np.random.normal(0, 1, size=2)
            noisy_points.append([point[0] + noise[0], point[1] + noise[1]])
        else:
            noisy_points.append(point)

    # Add pure false positives
    num_false_positives = int(len(points) * false_positive_rate)
    for _ in range(num_false_positives):
        # random point within perceptual range
        r = perceptual_range * np.sqrt(random.random())
        theta = random.random() * 2 * np.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        noisy_points.append([x, y])

    return noisy_points


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


# 3. Create custom dataset class
class LaneDetectionDataset(Dataset):
    def __init__(self, perceptual_field_data, augment=False):
        self.data = perceptual_field_data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        noisy_left, noisy_right = self.data[idx]

        if self.augment:
            # Augment both left and right boundaries together to maintain their spatial relationship
            combined = np.array(noisy_left + noisy_right)
            augmented_combined = augment_points(combined)

            # Split them back
            len_left = len(noisy_left)
            noisy_left = augmented_combined[:len_left]
            noisy_right = augmented_combined[len_left:]

        left_tensor = torch.tensor(noisy_left, dtype=torch.float32)
        right_tensor = torch.tensor(noisy_right, dtype=torch.float32)
        return left_tensor, right_tensor
