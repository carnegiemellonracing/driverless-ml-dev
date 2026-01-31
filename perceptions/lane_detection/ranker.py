import torch
import numpy as np
from perceptions.lane_detection.models import LaneCandidate, PerceptualFieldContext
from perceptions.lane_detection.geo import (
    point_to_segment_distance,
    segment_to_segment_distance,
    get_segment_angle,
)


def _path_length(points: list) -> float:
    """Compute total length of a path defined by points."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        total += np.linalg.norm(points[i + 1] - points[i])
    return total


def _point_to_path_distance(point: np.ndarray, path_points: list) -> float:
    """Compute minimum distance from a point to a polyline path."""
    if len(path_points) == 0:
        return float("inf")
    if len(path_points) == 1:
        return np.linalg.norm(point - path_points[0])

    min_dist = float("inf")
    for i in range(len(path_points) - 1):
        dist, _ = point_to_segment_distance(point, path_points[i], path_points[i + 1])
        min_dist = min(min_dist, dist)
    return min_dist


def _compute_path_coverage(
    gt_points: list, cand_points: list, threshold: float = 1.5
) -> float:
    """
    Compute what fraction of the ground truth path is "covered" by the candidate path.

    Coverage is measured by sampling points along the GT path and checking
    what fraction are within `threshold` distance of the candidate path.

    Args:
        gt_points: List of ground truth path points
        cand_points: List of candidate path points
        threshold: Distance threshold for considering a point "covered" (meters)

    Returns:
        Coverage ratio in [0, 1]
    """
    if len(gt_points) < 2 or len(cand_points) < 2:
        # If candidate has fewer than 2 points, check vertex overlap
        if len(cand_points) == 0:
            return 0.0
        # Check if GT points are near the candidate point(s)
        covered = 0
        for gt_pt in gt_points:
            min_dist = min(np.linalg.norm(gt_pt - cp) for cp in cand_points)
            if min_dist <= threshold:
                covered += 1
        return covered / len(gt_points) if gt_points else 0.0

    # Sample points along the GT path
    gt_length = _path_length(gt_points)
    if gt_length == 0:
        return 1.0 if len(cand_points) > 0 else 0.0

    # Sample every 0.5m along GT path
    sample_interval = 0.5
    num_samples = max(int(gt_length / sample_interval), len(gt_points))

    covered_length = 0.0
    accumulated_length = 0.0

    for i in range(len(gt_points) - 1):
        seg_start = gt_points[i]
        seg_end = gt_points[i + 1]
        seg_length = np.linalg.norm(seg_end - seg_start)

        if seg_length == 0:
            continue

        # Sample along this segment
        num_seg_samples = max(2, int(seg_length / sample_interval) + 1)
        for j in range(num_seg_samples):
            t = j / (num_seg_samples - 1) if num_seg_samples > 1 else 0
            sample_point = seg_start + t * (seg_end - seg_start)

            # Check distance to candidate path
            dist = _point_to_path_distance(sample_point, cand_points)
            if dist <= threshold:
                covered_length += seg_length / num_seg_samples

        accumulated_length += seg_length

    return min(covered_length / gt_length, 1.0) if gt_length > 0 else 0.0


def IoU(
    ctx: PerceptualFieldContext, candidate: LaneCandidate, threshold: float = 1.5
) -> float:
    """
    Compute geometric IoU between candidate and ground truth boundaries.

    This measures how well the candidate paths align with the ground truth
    lane boundaries using path coverage. Unlike vertex-based IoU, this
    properly handles cases where the candidate uses different intermediate
    vertices but still traces the correct geometric path.

    Args:
        ctx: PerceptualFieldContext with ground truth boundaries
        candidate: LaneCandidate to evaluate
        threshold: Distance threshold for considering coverage (meters)

    Returns:
        IoU score in [0, 1], computed as average of left and right coverage
    """
    # Get ground truth paths as ordered point sequences
    # Sort by x-coordinate as a proxy for path ordering (assumes forward direction)
    gt_left_indices = sorted(ctx.left_boundary, key=lambda i: ctx.get_point(i)[0])
    gt_right_indices = sorted(ctx.right_boundary, key=lambda i: ctx.get_point(i)[0])

    gt_left_points = [ctx.get_point(i) for i in gt_left_indices]
    gt_right_points = [ctx.get_point(i) for i in gt_right_indices]

    # Get candidate paths
    cand_left_points = [ctx.get_point(i) for i in candidate.left_path]
    cand_right_points = [ctx.get_point(i) for i in candidate.right_path]

    # Compute coverage for each side
    left_coverage = _compute_path_coverage(gt_left_points, cand_left_points, threshold)
    right_coverage = _compute_path_coverage(
        gt_right_points, cand_right_points, threshold
    )

    # Average coverage as IoU proxy
    # Weight by path lengths if they differ significantly
    gt_left_len = _path_length(gt_left_points)
    gt_right_len = _path_length(gt_right_points)
    total_len = gt_left_len + gt_right_len

    if total_len == 0:
        return 0.0

    # Weighted average by path length
    iou = (left_coverage * gt_left_len + right_coverage * gt_right_len) / total_len
    return iou


def get_path_stats(path_indices: list[int], context: PerceptualFieldContext):
    """
    Computes path statistics required for features:
    - Total Length
    - Variance of Segment Lengths
    - Variance of Angles between segments
    """
    if len(path_indices) < 2:
        return 0.0, 0.0, 0.0

    points = [context.get_point(i) for i in path_indices]

    # 1. Segment Lengths
    segment_lengths = []
    for i in range(len(points) - 1):
        dist = np.linalg.norm(points[i + 1] - points[i])
        segment_lengths.append(dist)

    total_length = sum(segment_lengths)
    var_seg_len = np.var(segment_lengths) if segment_lengths else 0.0

    # 2. Angles between segments
    angles = []
    if len(points) >= 3:
        for i in range(len(points) - 2):
            p1 = points[i]
            p2 = points[i + 1]
            p3 = points[i + 2]
            # get_segment_angle returns degrees
            angle_deg = get_segment_angle(p1, p2, p3)
            angles.append(angle_deg)

    var_angles = np.var(angles) if angles else 0.0

    return total_length, var_seg_len, var_angles


def get_all_widths(
    ctx: PerceptualFieldContext, candidate: LaneCandidate
) -> list[float]:
    """
    Re-implements width calculation to return ALL widths for variance computation.
    Based on online_lane_width from geo.py.
    """
    l_path = candidate.left_path
    r_path = candidate.right_path
    matchings = candidate.matchings

    # Start from fixed set or beginning
    # Ideally should use matchings.fixed_widths + new ones
    # But for full variance we need the full list.
    # The fixed_widths in MatchingSet are just the widths.

    all_widths = list(matchings.fixed_widths)

    start_l = matchings.last_fixed_l_idx
    start_r = matchings.last_fixed_r_idx

    # If path hasn't grown since last fix, we might be done (optimization),
    # but simplest is to just recompute the mutable part.

    left_points = [ctx.get_point(idx) for idx in l_path[start_l:]]
    right_points = [ctx.get_point(idx) for idx in r_path[start_r:]]

    new_matchings = []

    # 2a. Left vertices to right segments
    for i, l_point in enumerate(left_points):
        l_param = float(start_l + i)
        for j in range(len(right_points) - 1):
            r_seg_start = right_points[j]
            r_seg_end = right_points[j + 1]
            dist, _ = point_to_segment_distance(l_point, r_seg_start, r_seg_end)
            new_matchings.append(dist)

    # 2b. Right vertices to left segments
    for j, r_point in enumerate(right_points):
        r_param = float(start_r + j)
        for i in range(len(left_points) - 1):
            l_seg_start = left_points[i]
            l_seg_end = left_points[i + 1]
            dist, _ = point_to_segment_distance(r_point, l_seg_start, l_seg_end)
            new_matchings.append(dist)

    # 2c. Segment to segment
    for i in range(len(left_points) - 1):
        for j in range(len(right_points) - 1):
            l_start_pt = left_points[i]
            l_end_pt = left_points[i + 1]
            r_start_pt = right_points[j]
            r_end_pt = right_points[j + 1]
            dist, _, _ = segment_to_segment_distance(
                l_start_pt, l_end_pt, r_start_pt, r_end_pt
            )
            new_matchings.append(dist)

    all_widths.extend(new_matchings)
    return all_widths


def extract_features(
    candidate: LaneCandidate, context: PerceptualFieldContext
) -> torch.Tensor:
    """
    Extracts 8 features as defined in [arXiv:2405.16369v1].

    Features:
    1. Lane Mean Length ( (L_len + R_len) / 2 )
    2. Left Point Count
    3. Right Point Count
    4. Width Variance
    5. Left Segment Length Variance
    6. Left Angle Variance
    7. Right Segment Length Variance
    8. Right Angle Variance
    """

    # 1. Path Stats
    l_len, l_var_seg, l_var_ang = get_path_stats(candidate.left_path, context)
    r_len, r_var_seg, r_var_ang = get_path_stats(candidate.right_path, context)

    # Feature 1: Mean Lane Length
    lane_mean_len = (l_len + r_len) / 2.0

    # Features 2 & 3: Point Counts
    l_count = float(len(candidate.left_path))
    r_count = float(len(candidate.right_path))

    # Feature 4: Width Variance
    widths = get_all_widths(context, candidate)
    width_var = np.var(widths) if widths else 0.0

    # Features 5-8: Variances
    # 5: Var(L_seg_len) -> l_var_seg
    # 6: Var(R_seg_len) -> r_var_seg (Note: Paper says "for both boundaries (5, 6 and 7, 8)".
    #    Order implies: 5=L_seg_var, 6=R_seg_var, 7=L_ang_var, 8=R_ang_var OR some other pair.
    #    Standard convention usually groups by type. I will group by type.

    features = [
        lane_mean_len,  # 1
        l_count,  # 2
        r_count,  # 3
        width_var,  # 4
        l_var_seg,  # 5
        r_var_seg,  # 6
        l_var_ang,  # 7
        r_var_ang,  # 8
    ]

    return torch.tensor(features, dtype=torch.float32)
