
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from data_loader import cone_maps, left_boundaries, right_boundaries, generate_noisy_perceptual_field_data
from geo import enumerate_path_pairs_v2
from visualization import visualize_epp_results

# Load map 0
cone_map = cone_maps[0]
left_b = list(left_boundaries[0])
right_b = list(right_boundaries[0])

# Generate NOISY perceptual fields
# High noise to really test robustness: 0.3m std dev, 20% false positives
print("Generating noisy data...")
pf_data, noisy_map, noisy_l, noisy_r, fp_indices = generate_noisy_perceptual_field_data(
    left_b, right_b, cone_map,
    position_noise_std=0.3,
    false_positive_rate=0.2,
    perceptual_range=30,
    dmax=5,
    seed=42
)

print(f"Original map size: {len(cone_map)}")
print(f"Noisy map size: {len(noisy_map)} ({len(fp_indices)} false positives)")

# Pick a field with good visibility
pf = pf_data[0]
car_pos, car_heading, paths, subgraph, left_sub, right_sub = pf

heading_vec = np.array([np.cos(car_heading), np.sin(car_heading)])

print(f"Running EPP on noisy data...")
sl, sr = int(left_sub[0]), int(right_sub[0])
print(f"Starts: L={sl}, R={sr}")
initial_paths = ([sl], [sr])
visited = {sl, sr}

# Run EPP
path_pairs = enumerate_path_pairs_v2(subgraph, noisy_map, initial_paths, visited, heading_vec, 0, itmax=2000)

print(f"Found {len(path_pairs)} candidates")

# Visualize
print("Visualizing...")
fig = visualize_epp_results(
    noisy_map, car_pos, car_heading, subgraph, path_pairs,
    title=f"EPP on Noisy Data (0.3m noise, 20% FP)\nFound {len(path_pairs)} candidates",
    save_path="noisy_epp_verification.png"
)
print("Saved noisy_epp_verification.png")
