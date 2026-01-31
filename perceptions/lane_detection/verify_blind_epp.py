
import numpy as np
import random
from perceptions.lane_detection.data_loader import cone_maps, left_boundaries, right_boundaries, build_adjacency_graph, filter_points_within_range
from perceptions.lane_detection.geo import EPP, LaneCandidate, find_starting_vertices, PerceptualFieldContext
from perceptions.lane_detection.models import MatchingSet

def verify_blind_epp():
    print("=== Blind EPP Verification ===")
    print("Goal: Prove EPP works without Ground Truth indices or sequential ordering.")

    # 1. Load Data
    clean_map = cone_maps[0]
    
    # 2. SHUFFLE DATA (The "Blind" Test)
    # We create a random permutation of indices to ensure we aren't relying on 
    # the original sequential ordering (0, 1, 2...) which often correlates with lane position.
    num_points = len(clean_map)
    perm = np.random.permutation(num_points)
    shuffled_map = clean_map[perm]
    
    print(f"Map shuffled. Original Index 0 is now at {perm[0]}.")
    
    # Create index mapping to simulate "ground truth" car pos derivation
    # map_inv[new_idx] -> old_idx
    # map_fwd[old_idx] -> new_idx
    map_fwd = {old: new for new, old in enumerate(perm)}
    
    # 3. Build Graph blindly
    # We rebuild the graph purely from XY distances on the shuffled data.
    dmax = 5.5
    adj_list = build_adjacency_graph(shuffled_map, dmax)
    print(f"Graph built. Nodes: {len(adj_list)}. Edges: {sum(len(v) for v in adj_list.values())//2}")
    
    # 4. Simulate Traversal blindly
    from perceptions.lane_detection.data_loader import get_car_pos
    
    success_count = 0
    total_frames = 0
    
    # Simulate skipping through the track
    for i in range(0, len(left_boundaries[0]), 2):
        total_frames += 1
        
        # Get ground truth car pos using ORIGINAL indices (which correspond to left/right pairs)
        # We must use the original map for this calculation to ensure valid geometry for the ground truth pose
        orig_l_idx = left_boundaries[0][i] 
        
        # We need the "right" boundary corresponding to this left point to define the track center
        # This is strictly for setting up the Simulation Camera.
        from perceptions.lane_detection.data_loader import get_closest
        orig_r_idx = get_closest(orig_l_idx, right_boundaries[0], clean_map)
        
        cp, ch = get_car_pos(orig_l_idx, right_boundaries[0], clean_map)
        
        # 5. Generate Context BLINDLY
        # The algorithm gets: shuffled_map, car_pos, adj_list. 
        # It has NO idea which indices are left/right.
        
        subgraph = filter_points_within_range(cp, ch, shuffled_map, adj_list, perceptual_range=20.0)
        
        # Check if subgraph is empty
        if not subgraph:
            print(f"Frame {total_frames}: Empty subgraph!")
            continue
            
        ctx = PerceptualFieldContext(
            cone_map=shuffled_map,
            visible_indices=set(subgraph.keys()),
            adj_list=subgraph,
            car_pos=cp,
            car_heading=ch
        )
        
        # 6. Find Start Vertices (Pure Geometry)
        # CRITICAL FIX: max_range must be large enough to reach the cones! 
        # Track width is ~3-5m, so we need at least ~3m radius. Using 5.0 for safety.
        # Adjusted to 6.0 to capture boundaries on wider turns
        sl, sr = find_starting_vertices(ctx.adj_list, ctx.cone_map, ctx.car_pos, ctx.car_heading, max_range=6.0)
        
        if sl is None or sr is None:
            # Check why
            from perceptions.lane_detection.geo import within_range, within_cone
            visible = [i for i in ctx.adj_list.keys()]
            in_range = [i for i in visible if within_range(ctx.cone_map[i], ctx.car_pos, 6.0)]
            # print(f"Frame {total_frames}: Start Point failure. Visible: {len(visible)}, In Range (6m): {len(in_range)}")
            continue
            
        # 7. Run EPP
        cand = LaneCandidate([sl], [sr], {sl}, {sr}, MatchingSet())
        results = EPP(ctx, cand, 0, 1000)
        
        if len(results) > 0:
            success_count += 1
            # print(f"Frame {total_frames}: Success! Found {len(results)} candidates.")
        else:
            # print(f"Frame {total_frames}: EPP found 0 candidates.")
            pass
            
    print("-" * 30)
    print(f"Total Frames: {total_frames}")
    print(f"Successes: {success_count}")
    print(f"Success Rate: {success_count/total_frames*100:.1f}%")
    print("Note: Success means EPP found ≥1 candidate purely from geometric constraints")
    print("without access to ground truth indices or labels.")

if __name__ == "__main__":
    verify_blind_epp()
