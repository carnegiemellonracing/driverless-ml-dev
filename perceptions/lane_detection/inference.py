import torch
import numpy as np
import argparse
from typing import List, Optional

from perceptions.lane_detection.models import (
    LaneCandidate,
    PerceptualFieldContext,
    ConeClassifier,
    MatchingSet,
    Map,
)
from perceptions.lane_detection.geo import (
    build_adjacency_graph,
    find_starting_vertices,
    C_seg,
    C_width,
    C_poly,
)
from perceptions.lane_detection.deciders import (
    next_vertex_decider,
    backtracking_decider,
    enumerate_path_pairs,
)
from perceptions.lane_detection.ranker import extract_features


class Classifier:
    def __init__(self):
        print("Classifier Initialized.")

    def eval(self, cone_map: Map) -> LaneCandidate:
        print(f"Running inference on {len(cone_map)} cones...")

        # 1. Setup Context
        adj = build_adjacency_graph(cone_map, dmax=5.0)
        # Estimate car pos (mean of first few points or 0,0)
        car_pos = np.array([0.0, 0.0])  # improved estimation needed for real usage
        if len(cone_map) > 0:
            car_pos = cone_map.mean(axis=0)

        ctx = PerceptualFieldContext(
            cone_map, set(range(len(cone_map))), adj, car_pos, 0.0
        )

        # 2. Find Start
        l_start, r_start = find_starting_vertices(ctx)
        if l_start is None:
            print("No starting vertices found.")
            return None

        print(f"Starting search from L:{l_start}, R:{r_start}")

        # 3. Search
        candidates = enumerate_path_pairs(ctx, l_start, r_start)
        print(f"Generated {len(candidates)} candidates.")

        if not candidates:
            return None

        # 4. Score
        try:
            model = ConeClassifier()
            # Handle state dict loading safely
            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict = (
                checkpoint["model_state_dict"]
                if "model_state_dict" in checkpoint
                else checkpoint
            )
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            # Return longest candidate as fallback
            return max(candidates, key=lambda c: len(c.left_path) + len(c.right_path))

        best_score = -float("inf")
        best_cand = None

        with torch.no_grad():
            for i, cand in enumerate(candidates):
                feats = extract_features(cand, ctx)
                score = model(feats.unsqueeze(0)).item()

                if score > best_score:
                    best_score = score
                    best_cand = cand

        print(f"Best Lane Score: {best_score:.4f}")
        return best_cand
