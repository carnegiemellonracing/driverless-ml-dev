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

    def eval(
        self,
        cone_map: Map,
        car_pos: np.ndarray,
        car_heading: float,
        perceptual_range: float = 30.0,
    ) -> Optional[LaneCandidate]:
        """
        Entry point: takes a raw cone map and car pose, builds context, and runs inference.

        Args:
            cone_map: Nx2 numpy array of cone positions
            car_pos: [x, y] position of the car
            car_heading: Heading angle in radians
            perceptual_range: Range in meters for visibility filtering
        """
        print(f"Running inference on {len(cone_map)} cones...")

        # 1. Build adjacency graph
        adj = build_adjacency_graph(cone_map, dmax=self.dmax)

        # 2. Filter to visible cones within perceptual range
        from perceptions.lane_detection.data_loader import filter_points_within_range

        visible_adj = filter_points_within_range(
            car_pos, car_heading, cone_map, adj, perceptual_range
        )

        ctx = PerceptualFieldContext(
            cone_map=cone_map,
            visible_indices=set(visible_adj.keys()),
            adj_list=visible_adj,
            car_pos=car_pos,
            car_heading=car_heading,
        )

        return self.eval_from_context(ctx)

    def eval_from_context(self, ctx: PerceptualFieldContext) -> Optional[LaneCandidate]:
        """
        Runs inference on a pre-built PerceptualFieldContext.
        This is useful when we already have the context (e.g. from data_loader).
        """
        # Find starting vertices (max_range=7 to handle variable lane widths)
        l_start, r_start = find_starting_vertices(ctx, max_range=5)

        if l_start is None:
            return None

        # Create initial candidate and enumerate
        initial_candidate = LaneCandidate(
            left_path=[l_start],
            right_path=[r_start],
            left_visited=set(),
            right_visited=set(),
        )
        candidates = enumerate_path_pairs(ctx, initial_candidate)

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


def detect_lane(
    cones: np.ndarray,
    car_pos: np.ndarray,
    car_heading: float,
    model_path: str = "best_model.pth",
) -> Optional[LaneCandidate]:
    """
    Convenience function for one-shot lane detection.

    Args:
        cones: Nx2 numpy array of cone [x, y] positions
        car_pos: [x, y] position of the car
        car_heading: Heading angle in radians
        model_path: Path to the trained model checkpoint

    Returns:
        Best LaneCandidate or None if detection fails
    """
    classifier = Classifier(model_path)
    return classifier.eval(cones, car_pos, car_heading)
