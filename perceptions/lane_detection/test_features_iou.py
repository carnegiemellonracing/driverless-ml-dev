
import matplotlib
matplotlib.use('Agg')
import numpy as np

from data_loader import cone_maps, left_boundaries, right_boundaries, generate_noisy_perceptual_field_data
from geo import enumerate_path_pairs_v2, compute_features, compute_lane_iou

# Load map 0
cone_map = cone_maps[0]
left_b = list(left_boundaries[0])
right_b = list(right_boundaries[0])

# Generate noisy field
print("Generating noisy field...")
pf_data, noisy_map, noisy_l, noisy_r, _ = generate_noisy_perceptual_field_data(
    left_b, right_b, cone_map,
    position_noise_std=0.2,
    false_positive_rate=0.1,
    perceptual_range=30,
    dmax=5,
    seed=42
)

pf = pf_data[0]
car_pos, car_heading, _, subgraph, left_sub_gt, right_sub_gt = pf
# Note: left_sub_gt and right_sub_gt are the GT path indices visible in this field.

print(f"GT Path Lengths: L={len(left_sub_gt)}, R={len(right_sub_gt)}")

heading_vec = np.array([np.cos(car_heading), np.sin(car_heading)])
sl, sr = int(left_sub_gt[0]), int(right_sub_gt[0])
initial_paths = ([sl], [sr])
visited = {sl, sr}

print("Running EPP...")
path_pairs = enumerate_path_pairs_v2(subgraph, noisy_map, initial_paths, visited, heading_vec, 0, itmax=2000)
print(f"Found {len(path_pairs)} candidates")

print("\nEvaluating Candidates (Features & IoU):")
print("-" * 60)
print(f"{'ID':<3} | {'L/R Len':<10} | {'IoU':<6} | {'Features (First 4)':<30}")
print("-" * 60)

best_iou = -1.0
best_idx = -1

for i, pair in enumerate(path_pairs):
    # Compute features
    feats = compute_features(pair, noisy_map)
    
    # Compute IoU vs GT pair
    # GT pair is (left_sub_gt, right_sub_gt)
    iou = compute_lane_iou(pair, (left_sub_gt, right_sub_gt), noisy_map)
    
    if iou > best_iou:
        best_iou = iou
        best_idx = i
        
    # Format description
    feat_str = f"[{feats[0]:.1f}, {int(feats[1])}, {int(feats[2])}, {feats[3]:.2f}]"
    print(f"{i+1:<3} | {len(pair[0])}/{len(pair[1]):<10} | {iou:.4f} | {feat_str}")

print("-" * 60)
print(f"Best Candidate: #{best_idx+1} with IoU {best_iou:.4f}")

# Sanity check features
best_feats = compute_features(path_pairs[best_idx], noisy_map)
print("\nBest Candidate Features:")
feat_names = [
    "Mean Length (m)", "Points L", "Points R", "Var Width", 
    "Var Seg L", "Var Seg R", "Var Ang L", "Var Ang R"
]
for name, val in zip(feat_names, best_feats):
    print(f"  {name:<20}: {val:.4f}")
