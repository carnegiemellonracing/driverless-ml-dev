from torch.utils.data import Dataset
import torch
import numpy as np
from geo import enumerate_path_pairs_v2
from data_loader import generate_perceptual_field_data, load_yaml_data
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
        from geo import compute_features, compute_lane_iou
        
        self.data = []
        
        for bound, points in maps:
            # Generate perceptual fields from ground truth
            perceptual_fields = generate_perceptual_field_data(bound, points)
            
            for car_heading_rad, paths, subgraph, left_subset, right_subset in perceptual_fields:
                h_vec = [math.cos(car_heading_rad), math.sin(car_heading_rad)]
                
                # 1. Run EPP to get candidates
                # Use current GT visible subset as start nodes is unrealistic for inference, 
                # but for training data generation we need candidates that are reachable.
                # In inference we use NVD/LRD to find start nodes. 
                # Here we follow the existing pattern: use the visible subset's first points.
                if not left_subset or not right_subset:
                    continue
                    
                sl, sr = int(left_subset[0]), int(right_subset[0])
                initial_visited = {sl, sr}
                # Run EPP
                candidates = enumerate_path_pairs_v2(
                    subgraph, points, ([sl], [sr]), initial_visited, h_vec, 0, itmax=500
                )
                
                if len(candidates) < 2:
                    continue
                
                # 2. Compute Features and IoU for all candidates
                cand_features = []
                cand_ious = []
                
                gt_pair = (left_subset, right_subset)
                
                for cand in candidates:
                    feats = compute_features(cand, points)
                    iou = compute_lane_iou(cand, gt_pair, points)
                    
                    cand_features.append(np.array(feats, dtype=np.float32))
                    cand_ious.append(iou)
                
                # 3. Generate Pairs
                # Compare every candidate with every other candidate
                num_cands = len(candidates)
                for i in range(num_cands):
                    for j in range(num_cands):
                        if i == j:
                            continue
                        
                        # Store (feat1, feat2) and (iou1, iou2)
                        # We merge them into single arrays for __getitem__ convenience
                        merged_feats = np.stack([cand_features[i], cand_features[j]])
                        merged_ious = np.array([cand_ious[i], cand_ious[j]], dtype=np.float32)
                        
                        self.data.append((merged_feats, merged_ious))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        merged_feats, merged_IoU = self.data[idx]

        feats_tensor = torch.from_numpy(merged_feats)
        IoU_tensor   = torch.from_numpy(merged_IoU)

        if self.augment:
            # Augment features with noise
            feats_tensor = augment_feats(feats_tensor)
            # IoU labels are usually kept ground truth, but user code had augment_IoU.
            # We keep it for consistency if desired, or remove for correctness.
            # Usually we don't augment regression targets with noise?
            # But here IoU is used for probability.
            # I'll convert IoU to tensor but maybe skip noise for IoU unless requested.
            # The existing code had augment_IoU. I will assume it's desired.
            IoU_tensor = augment_IoU(IoU_tensor)

        return feats_tensor, IoU_tensor



if __name__ == "__main__":
    dataset_path = f"{os.path.dirname(__file__)}/dataset"
    boundary_paths = [f"{dataset_path}/boundaries_{i}.yaml" for i in range(1, 10)]
    cone_map_paths = [f"{dataset_path}/cone_map_{i}.yaml" for i in range(1, 10)]
    boundaries = [load_yaml_data(path) for path in boundary_paths]
    cone_maps = [load_yaml_data(path) for path in cone_map_paths]
    LaneDetectionDataset(zip(boundaries, cone_maps))