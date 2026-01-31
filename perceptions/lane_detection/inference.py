import torch
import numpy as np
import argparse
from typing import List, Optional

from perceptions.lane_detection.models import (
    LaneCandidate,
    PerceptualFieldContext,
    MatchingSet,
    Map,
)
from perceptions.lane_detection.model import ConeClassifier
from perceptions.lane_detection.data_loader import build_adjacency_graph
from perceptions.lane_detection.geo import (
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
    """
    :param dmax: Maximum distance threshold for adjacency (default: 5.0m)
    """

    def __init__(self, model_path: str, dmax=5.0):
        print(f"Initializing Classifier with model: {model_path}")
        self.dmax = dmax
        self.model = ConeClassifier()
        self.load_model(model_path)

    def load_model(self, model_path: str):
        try:
            # Handle state dict loading safely
            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict = (
                checkpoint["model_state_dict"]
                if "model_state_dict" in checkpoint
                else checkpoint
            )
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            raise e

    def eval(self, cone_map: Map) -> Optional[LaneCandidate]:
        """
        Original entry point: takes a raw cone map, builds context, and runs inference.
        """
        print(f"Running inference on {len(cone_map)} cones...")

        # 1. Setup Context
        adj = build_adjacency_graph(cone_map, dmax=self.dmax)

        # Estimate car pos (mean of first few points or 0,0)
        car_pos = np.array([0.0, 0.0])
        if len(cone_map) > 0:
            car_pos = cone_map.mean(axis=0)

        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices=set(range(len(cone_map))),
            adj_list=adj,
            car_pos=car_pos,
            car_heading=0.0,
        )

        return self.eval_from_context(ctx)

    def eval_from_context(self, ctx: PerceptualFieldContext) -> Optional[LaneCandidate]:
        """
        Runs inference on a pre-built PerceptualFieldContext.
        This is useful when we already have the context (e.g. from data_loader).
        """
        # 2. Find Start
        l_start, r_start = find_starting_vertices(ctx)

        # If standard start finding fails, try to use ground truth start if available in context?
        # For now, let's stick to the heuristic.
        if l_start is None:
            # print("No starting vertices found.")
            return None

        # print(f"Starting search from L:{l_start}, R:{r_start}")

        # 3. Search
        candidates = enumerate_path_pairs(ctx, l_start, r_start)
        # print(f"Generated {len(candidates)} candidates.")

        if not candidates:
            return None

        # 4. Score
        best_score = -float("inf")
        best_cand = None

        with torch.no_grad():
            for i, cand in enumerate(candidates):
                feats = extract_features(cand, ctx)
                score = self.model(feats.unsqueeze(0)).item()

                if score > best_score:
                    best_score = score
                    best_cand = cand

        # print(f"Best Lane Score: {best_score:.4f}")
        return best_cand
