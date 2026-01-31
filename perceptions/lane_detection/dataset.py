"""
Lane Detection Dataset Module

Provides PyTorch Dataset classes for training the lane detection model.
Works with PerceptualFieldContext and LaneCandidate types from the refactored pipeline.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
import itertools

from perceptions.lane_detection.models import LaneCandidate, PerceptualFieldContext
from perceptions.lane_detection.geo import find_starting_vertices
from perceptions.lane_detection.deciders import enumerate_path_pairs
from perceptions.lane_detection.ranker import extract_features, IoU
from perceptions.lane_detection.data_loader import (
    generate_all_perceptual_field_data,
    left_boundaries,
    right_boundaries,
)


def augment_feats(feats: torch.Tensor, std_dev: float = 0.01) -> torch.Tensor:
    """Add Gaussian noise to features for augmentation."""
    return feats + torch.randn_like(feats) * std_dev


def augment_IoU(IoU: torch.Tensor, std_dev: float = 0.02) -> torch.Tensor:
    """Add Gaussian noise to IoU values for augmentation."""
    noisy = IoU + torch.randn_like(IoU) * std_dev
    return torch.clamp(noisy, 0.0, 1.0)  # Keep IoU in valid range


def generate_lane_candidates(
    ctx: PerceptualFieldContext, max_candidates: int = 100
) -> List[LaneCandidate]:
    """
    Generate lane candidates for a perceptual field context.

    Tries multiple starting vertex pairs to generate diverse candidates.

    Args:
        ctx: PerceptualFieldContext with cone map and visibility info
        max_candidates: Maximum number of candidates to return

    Returns:
        List of LaneCandidate objects
    """
    all_candidates = []

    # Try multiple max_range values to get different starting points
    for max_range in [2.0, 3.0, 4.0, 5.0]:
        l_start, r_start = find_starting_vertices(ctx, max_range=max_range)

        if l_start is None or r_start is None:
            continue

        # Create initial candidate
        initial_candidate = LaneCandidate(
            left_path=[l_start],
            right_path=[r_start],
            left_visited=set(),
            right_visited=set(),
        )

        # Enumerate path pairs from this starting point
        candidates = enumerate_path_pairs(ctx, initial_candidate)
        all_candidates.extend(candidates)

        # Stop early if we have enough
        if len(all_candidates) >= max_candidates:
            break

    # Return ALL candidates (both valid and invalid) for training
    # Invalid candidates will naturally have lower features/scores
    return all_candidates[:max_candidates]


class LaneDetectionDataset(Dataset):
    """
    Dataset for pairwise lane candidate ranking.

    Each sample contains:
    - A pair of feature vectors (8-dimensional each)
    - A pair of IoU scores

    The model learns to predict which candidate has higher IoU.
    """

    def __init__(
        self,
        data: List[Tuple[np.ndarray, np.ndarray]] = None,
        augment: bool = False,
        perceptual_range: int = 30,
        contexts: List[PerceptualFieldContext] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data: Optional pre-computed data. If None, generates from all maps.
            augment: Whether to apply data augmentation
            perceptual_range: Range for perceptual field generation
            contexts: Optional list of contexts to use (prevents leakage if split beforehand)
        """
        self.augment = augment

        if data is not None:
            self.data = data
        else:
            self.data = self._generate_dataset(perceptual_range, contexts)

    def _generate_dataset(
        self, perceptual_range: int, contexts: List[PerceptualFieldContext] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate training data from maps.

        Returns list of (feature_pairs, iou_pairs) tuples where:
        - feature_pairs: shape (2, 8) - features for two candidates
        - iou_pairs: shape (2,) - IoU scores for the two candidates
        """
        data = []
        if contexts is None:
            contexts = generate_all_perceptual_field_data(
                perceptual_range=perceptual_range
            )
        print(f"Generating dataset from {len(contexts)} perceptual fields...")

        for ctx_idx, ctx in enumerate(contexts):
            # Generate candidates for this context
            candidates = generate_lane_candidates(ctx, max_candidates=50)

            if len(candidates) < 2:
                continue

            # Compute features and IoU for each candidate
            candidate_data = []
            for candidate in candidates:
                features = extract_features(candidate, ctx)
                iou = IoU(ctx, candidate)
                candidate_data.append((features.numpy(), iou))

            # Create pairwise combinations
            for (feat1, iou1), (feat2, iou2) in itertools.combinations(
                candidate_data, 2
            ):
                # Randomly swap to ensure class balance (p(c1 > c2) ~= 0.5)
                if np.random.random() > 0.5:
                    feat1, feat2 = feat2, feat1
                    iou1, iou2 = iou2, iou1

                feature_pairs = np.stack([feat1, feat2], axis=0)  # (2, 8)
                iou_pairs = np.array([iou1, iou2], dtype=np.float32)  # (2,)
                data.append((feature_pairs, iou_pairs))

            if (ctx_idx + 1) % 10 == 0:
                print(
                    f"  Processed {ctx_idx + 1}/{len(contexts)} contexts, {len(data)} pairs so far"
                )

        print(f"Generated {len(data)} training pairs")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_pairs, iou_pairs = self.data[idx]

        feats_tensor = torch.from_numpy(feature_pairs).float()  # (2, 8)
        iou_tensor = torch.from_numpy(iou_pairs).float()  # (2,)

        if self.augment:
            feats_tensor = augment_feats(feats_tensor)
            # iou_tensor = augment_IoU(iou_tensor)  # Don't augment targets

        return feats_tensor, iou_tensor


if __name__ == "__main__":
    # Test dataset generation
    print("Testing LaneDetectionDataset...")
    dataset = LaneDetectionDataset(perceptual_range=30)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample feature shape: {sample[0].shape}")
        print(f"Sample IoU shape: {sample[1].shape}")
