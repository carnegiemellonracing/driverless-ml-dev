import torch
import numpy as np
from perceptions.lane_detection.models import LaneCandidate, PerceptualFieldContext
from perceptions.lane_detection.geo import (
    point_to_segment_distance,
    segment_to_segment_distance,
    get_segment_angle,
)


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
