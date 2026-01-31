
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from PIL import Image

# Ensure module path
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # perceptions/..
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # root

from perceptions.lane_detection.data_loader import cone_maps, left_boundaries, right_boundaries, generate_perceptual_field_data, get_closest
from perceptions.lane_detection.geo import EPP, LaneCandidate, find_starting_vertices
from perceptions.lane_detection.models import MatchingSet
from perceptions.lane_detection.visualization import visualize_epp_results
from perceptions.lane_detection.config import D_MAX

def create_animation():
    print("=== Generating Track Traversal Animation ===")
    
    # Configuration
    NOISE = True
    
    clean_map = cone_maps[0]
    clean_l = left_boundaries[0]
    clean_r = right_boundaries[0]
    
    if NOISE:
        print("Using NOISY data...")
        from perceptions.lane_detection.data_loader import generate_noisy_perceptual_field_data
        # Using slightly more noise as requested (std=0.2, rate=0.1)
        contexts, noisy_map, _, _, _ = generate_noisy_perceptual_field_data(
            clean_l, clean_r, clean_map, 
            position_noise_std=0.2, 
            false_positive_rate=0.1,
            perceptual_range=30.0,
            dmax=D_MAX
        )
        # For visualization, we should use the noisy map
        vis_map = noisy_map
    else:
        print("Using CLEAN data...")
        # dmax should be generous to ensure connectivity
        contexts = generate_perceptual_field_data(clean_l, clean_r, clean_map, dmax=D_MAX)
        vis_map = clean_map
    
    print(f"Total points in trajectory: {len(contexts)}")
    
    print(f"Total points in trajectory: {len(contexts)}")
    step = 1 # Process every point (dense)
    
    frame_files = []
    
    for i in range(0, len(contexts), step):
        print(f"Processing Frame {i}/{len(contexts)}...")
        ctx = contexts[i]
        
        # 0. Refine Heading using Trajectory
        # The default ctx.car_heading is based on track geometry (tangent), which can be noisy or misaligned.
        # We calculate the heading based on the car's actual motion vector for smoother visualization.
        heading_to_use = ctx.car_heading
        if i < len(contexts) - 1:
            next_pos = contexts[i+1].car_pos
            curr_pos = ctx.car_pos
            dx = next_pos[0] - curr_pos[0]
            dy = next_pos[1] - curr_pos[1]
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                heading_to_use = np.arctan2(dy, dx)
        elif i > 0:
            # Use previous motion for the last frame
            prev_pos = contexts[i-1].car_pos
            curr_pos = ctx.car_pos
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                heading_to_use = np.arctan2(dy, dx)

            if abs(dx) > 0.01 or abs(dy) > 0.01:
                heading_to_use = np.arctan2(dy, dx)

        # 0.5 Re-calculate Perceptual Field with Correct Heading
        # Since ctx was originally created with the noisy/misaligned heading, we must
        # re-filter the points using the corrected 'heading_to_use' so that the
        # "Visible Cones" actually match the 120-degree cone we draw.
        from perceptions.lane_detection.data_loader import filter_points_within_range
        
        # dmax=D_MAX must match config
        # D_MAX is already imported globally

        
        # Re-filter graph
        # Note: We need the full adjacency graph, but ctx only has the subgraph.
        # Ideally we'd have the full graph, but here we can just re-filter using the
        # shared 'cone_map' and rebuild a local subgraph or assume 'ctx.cone_map' is global.
        # ctx.cone_map IS the global map reference. We can just use data_loader.build_adjacency_graph if we needed fresh edges,
        # but reusing the static adjacency logic is likely fine or we can assume we only care about visibility.
        # To be safe and correct, we need the adjacency edges.
        # Let's import build_adjacency_graph to get full edges for the visible subset.
        from perceptions.lane_detection.data_loader import build_adjacency_graph
        
        # This is expensive to do every frame if we build full graph, but map is small (66 pts).
        # Optimization: Build full graph ONCE at start.
        if i == 0:
             full_adj_list = build_adjacency_graph(ctx.cone_map, dmax=D_MAX)
        
        new_subgraph = filter_points_within_range(
            ctx.car_pos, heading_to_use, ctx.cone_map, full_adj_list, perceptual_range=30.0
        )
        
        # Update Context with new visibility
        ctx.adj_list = new_subgraph
        ctx.visible_indices = set(new_subgraph.keys())
        ctx.car_heading = heading_to_use  # Sync heading in context
        # Using find_starting_vertices for robustness
        sl, sr = find_starting_vertices(ctx.adj_list, ctx.cone_map, ctx.car_pos, heading_to_use, max_range=5.0)
        
        if sl is None or sr is None:
            # Fallback to GT if NVD fails (e.g. sparse area)
            sl = clean_l[i]
            sr = get_closest(sl, clean_r, clean_map)
        
        # 2. Run EPP
        cand = LaneCandidate([sl], [sr], {sl}, {sr}, MatchingSet())
        results = EPP(ctx, cand, 0, 1000) # Limit iterations for speed
        
        # 3. Visualize
        fname = f"temp_frame_{i:03d}.png"
        fig = visualize_epp_results(
            ctx.cone_map, ctx.car_pos, heading_to_use, ctx.adj_list,
            [(r.left_path, r.right_path) for r in results],
            title=f"Frame {i}: {len(results)} Candidates",
            save_path=fname
        )
        plt.close(fig)
        frame_files.append(fname)
        
    print(f"Stitching {len(frame_files)} frames into GIF...")
    
    if frame_files:
        imgs = [Image.open(f) for f in frame_files]
        # Duration 200ms = 5fps (Slower)
        out_name = 'track_traversal_noisy.gif' if NOISE else 'track_traversal.gif'
        imgs[0].save(out_name, save_all=True, append_images=imgs[1:], duration=200, loop=0)
        print(f"Saved '{out_name}'")
        
        # Cleanup
        for f in frame_files:
            os.remove(f)
            
    print("Done.")

if __name__ == "__main__":
    create_animation()
