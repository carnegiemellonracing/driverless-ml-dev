from torch.utils.data import Dataset
import torch
import numpy as np
from perceptions.lane_detection.geo import enumerate_path_pairs_v2
from perceptions.lane_detection.data_loader import generate_perceptual_field_data, load_yaml_data
import math
import os

def augment_feats(feats, std_dev = 0.1):
    return feats + torch.randn_like(feats) * std_dev

def augment_IoU(IoU, std_dev = 0.1):
    return IoU + torch.randn_like(IoU) * std_dev

# 3. Create custom dataset class
class LaneDetectionDataset(Dataset):
    def __init__(self, maps, augment=False):
        
        self.data = self.create_dataset(maps)
        self.augment = augment

    def create_dataset(self, maps):
        for bound, points in maps:
            perceptual_fields = generate_perceptual_field_data(bound, points)
            for car_heading_deg, paths, subgraph, left_subset, right_subset in perceptual_fields:
                h_vec = [math.cos(car_heading_deg*math.pi/180), math.sin(car_heading_deg*math.pi/180)]
                # points_numpy = {key: np.array(value_list) for key, value_list in points.items()}
                # print(points_numpy)
                enum_paths = enumerate_path_pairs_v2(subgraph, points, paths, visited=set(), heading_vector=h_vec, it=0, itmax=2500)
                print(enum_paths)
                exit()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        merged_feats, merged_IoU = self.data[idx]

        feats_tensor = torch.from_numpy(merged_feats)
        IoU_tensor   = torch.from_numpy(merged_IoU)

        if self.augment:
            # Augment both left and right boundaries together to maintain their spatial relationship
            augmented_feats = augment_feats(feats_tensor)
            augmented_IoU   = augment_IoU(IoU_tensor)

        return augmented_feats, augmented_IoU



if __name__ == "__main__":
    dataset_path = f"{os.path.dirname(__file__)}/dataset"
    boundary_paths = [f"{dataset_path}/boundaries_{i}.yaml" for i in range(1, 10)]
    cone_map_paths = [f"{dataset_path}/cone_map_{i}.yaml" for i in range(1, 10)]
    boundaries = [load_yaml_data(path) for path in boundary_paths]
    cone_maps = [load_yaml_data(path) for path in cone_map_paths]
    LaneDetectionDataset(zip(boundaries, cone_maps))