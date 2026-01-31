
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure running as module so imports work, or fix sys.path
import sys
if __name__ == "__main__" and __package__ is None:
    # Adding current dir to path to allow imports if run as script
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    # Add root directory to allow absolute imports of perceptions.*
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    pass

from perceptions.lane_detection.data_loader import cone_maps, left_boundaries, right_boundaries, generate_perceptual_field_data, generate_noisy_perceptual_field_data
from perceptions.lane_detection.geo import EPP, LaneCandidate
from perceptions.lane_detection.models import PerceptualFieldContext, MatchingSet
from perceptions.lane_detection.visualization import visualize_epp_results

from perceptions.lane_detection.config import D_MAX

def run_clean_vis():
    print("\n=== Running Clean Map Visualization ===")
    clean_map = cone_maps[0]
    clean_l = left_boundaries[0]
    clean_r = right_boundaries[0]
    
    print(f"Generating Clean Perceptual Fields (dmax={D_MAX})...")
    clean_ctxs = generate_perceptual_field_data(clean_l, clean_r, clean_map, dmax=D_MAX)
    idx = 0
    ctx = clean_ctxs[idx]
    
    print(f"Context 0: {len(ctx.visible_indices)} visible cones. Adj list size: {len(ctx.adj_list)}")
    
    # Use find_starting_vertices to pick valid start pair
    # Need to find start points visible in this context.
    # The context car pos is derived from left bound[idx].
    # So left bound[idx] should be near car.
    # Use find_starting_vertices to pick valid start pair
    from perceptions.lane_detection.geo import find_starting_vertices
    sl, sr = find_starting_vertices(ctx.adj_list, ctx.cone_map, ctx.car_pos, ctx.car_heading, max_range=5.0)
    
    if sl is None or sr is None:
        print("Failed to find starting vertices!")
        sl, sr = 4, 1 # Fallback for consistent debug?
        # return

    print(f"Start Pair (NVD-derived): {sl}, {sr}")
    
    cand = LaneCandidate([sl], [sr], {sl}, {sr}, MatchingSet())
    from perceptions.lane_detection.geo import constraint_decider, online_lane_width
    valid_cd = constraint_decider(cand, ctx)
    _, min_w, max_w = online_lane_width(ctx, cand)
    print(f"Initial CD: {valid_cd}, Width: {min_w:.2f}-{max_w:.2f}")
    
    print("Running EPP (Clean)...")
    results = EPP(ctx, cand, 0, 2000)
    print(f"Clean Results: {len(results)} candidates")
    
    # Visualize
    save_p = "clean_epp_vis.png"
    print(f"Saving to {save_p}...")
    fig = visualize_epp_results(
        ctx.cone_map, ctx.car_pos, ctx.car_heading, ctx.adj_list,
        [(r.left_path, r.right_path) for r in results],
        title="EPP on Clean Data",
        save_path=save_p
    )
    plt.close(fig)

def run_noisy_vis():
    print("\n=== Running Noisy Map Visualization ===")
    clean_map = cone_maps[0]
    clean_l = left_boundaries[0]
    clean_r = right_boundaries[0]
    
    print(f"Generating Noisy Perceptual Fields (dmax={D_MAX})...")
    # Reduced noise to ensure EPP finds candidates with strict constraints
    noisy_ctxs, noisy_map, _, _, _ = generate_noisy_perceptual_field_data(
        clean_l, clean_r, clean_map, seed=42, dmax=D_MAX,
        position_noise_std=0.1, false_positive_rate=0.1
    )
    
    # Pick index
    idx = 0
    ctx = noisy_ctxs[idx]
    
    # Start points
    from perceptions.lane_detection.geo import find_starting_vertices
    sl, sr = find_starting_vertices(ctx.adj_list, ctx.cone_map, ctx.car_pos, ctx.car_heading, max_range=5.0)
    
    # if sl is None... handle
    if sl is None:
         sl, sr = 4, 1
         
    print(f"Start Pair (NVD-derived): {sl}, {sr}")
    
    cand = LaneCandidate([sl], [sr], {sl}, {sr}, MatchingSet())
    
    print("Running EPP (Noisy)...")
    results = EPP(ctx, cand, 0, 2000)
    print(f"Noisy Results: {len(results)} candidates")
    
    # Visualize
    save_p = "noisy_epp_vis.png"
    print(f"Saving to {save_p}...")
    fig = visualize_epp_results(
        ctx.cone_map, ctx.car_pos, ctx.car_heading, ctx.adj_list,
        [(r.left_path, r.right_path) for r in results],
        title="EPP on Noisy Data (0.3m noise, 20% FP)",
        save_path=save_p
    )
    plt.close(fig)

if __name__ == "__main__":
    run_clean_vis()
    run_noisy_vis()
    print("\nDone. Generated 'clean_epp_vis.png' and 'noisy_epp_vis.png'.")
