from warnings import deprecated

import numpy as np
import matplotlib.pyplot as plt

from models import Point, Lane

# Export list for clean imports
__all__ = [
    "point_to_segment_distance",
    "construct_adjacency_list",
    "find_matching_segments",
    "find_matching_points",
    "enumerate_path_pairs",
    "enumerate_path_pairs_v2",
    "next_vertex_decider",
    "left_right_decider",
    "constraint_decider",
    "compute_features",
    "generate_feature_pairs",
]


def point_to_segment_distance(point, seg_start, seg_end):
    point = np.array(point)
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)

    seg_vec = seg_end - seg_start
    point_vec = point - seg_start

    seg_length_sq = np.dot(seg_vec, seg_vec)

    if seg_length_sq < 1e-8:
        distance = np.linalg.norm(point_vec)
        return distance, seg_start

    t = np.dot(point_vec, seg_vec) / seg_length_sq
    t_clamped = np.clip(t, 0.0, 1.0)

    projection = seg_start + t_clamped * seg_vec
    distance = np.linalg.norm(point - projection)

    return distance, projection


def construct_adjacency_list(points, dmax):
    adjacency_list = {i: [] for i in range(len(points))}

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if np.linalg.norm(points[i] - points[j]) <= dmax:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)

    return adjacency_list
# ================================================================
# Analytic segment utilities
# ================================================================

def closest_point_on_segment(p, a, b):
    """
    Returns (closest_point, t) where:
      t in [0,1] is the position along segment a→b
      closest_point = a + t*(b-a)
    """
    a = np.array(a)
    b = np.array(b)
    p = np.array(p)
    ab = b - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-12:
        return a, 0.0  # degenerate segment
    t = np.dot(p - a, ab) / ab_len_sq
    t = np.clip(t, 0.0, 1.0)
    return a + t * ab, t


def segment_segment_closest(a0, a1, b0, b1):
    """
    Returns (pa, pb, ta, tb) where:
        pa = closest point on segment A
        pb = closest point on segment B
        ta in [0,1], tb in [0,1]
    This is the standard analytic closest segment–segment formula.
    """
    a0 = np.array(a0); a1 = np.array(a1)
    b0 = np.array(b0); b1 = np.array(b1)

    A = a1 - a0
    B = b1 - b0
    magA = np.dot(A, A)
    magB = np.dot(B, B)

    if magA < 1e-12:
        # A is a point → reduce to point-to-segment
        pb, tb = closest_point_on_segment(a0, b0, b1)
        return a0, pb, 0.0, tb

    if magB < 1e-12:
        # B is a point → reduce to point-to-segment
        pa, ta = closest_point_on_segment(b0, a0, a1)
        return pa, b0, ta, 0.0

    A_dot_B = np.dot(A, B)
    A_dot_A = magA
    B_dot_B = magB
    A0B0 = a0 - b0

    denom = A_dot_A * B_dot_B - A_dot_B * A_dot_B
    if abs(denom) < 1e-12:
        # Segments almost parallel
        # Fix ta = 0
        ta = 0.0
        _, tb = closest_point_on_segment(a0, b0, b1)
    else:
        ta = (np.dot(-A0B0, A) * B_dot_B - np.dot(-A0B0, B) * A_dot_B) / denom
        ta = np.clip(ta, 0.0, 1.0)
        tb = (np.dot(A0B0, B) + ta * A_dot_B) / B_dot_B
        tb = np.clip(tb, 0.0, 1.0)

    pa = a0 + ta * A
    pb = b0 + tb * B
    return pa, pb, ta, tb


# ================================================================
# Nearest neighbor search exactly per Eq. (8–10)
# ================================================================

def nearest_neighbor_search(L, R, k, s):
    """
    Implements Eq. (8–10) exactly, including continuous segment matching.

    Returns list of (u, v) where:
        u ∈ [0, |L|-1], v ∈ [0, |R|-1]
        u = i + t, t ∈ [0,1]
        v = j + t, t ∈ [0,1]
    """

    L = np.array(L)
    R = np.array(R)

    nL = len(L)
    nR = len(R)

    # Number of matching points Eq. (10)
    Nm = (nL - k) if s == 0 else (nR - k)
    if Nm <= 0:
        return []

    pairs = []

    for i in range(Nm):

        # ---------------------------------------------------------
        # Build Ωᵢ(k, s)
        # ---------------------------------------------------------

        if s == 0:
            # L is the query
            if k == 0:
                # vertex i → full polyline R
                query_type = "point"
                a0 = L[i]
                a1 = None
            else:
                # segment L[i]→L[i+1] → full polyline R
                query_type = "segment"
                a0 = L[i]
                a1 = L[i+1]
        else:
            # R is the query
            if k == 0:
                query_type = "point"
                a0 = R[i]
                a1 = None
            else:
                query_type = "segment"
                a0 = R[i]
                a1 = R[i+1]

        # ---------------------------------------------------------
        # NN search over target polyline (the entire other boundary)
        # ---------------------------------------------------------

        best_dist = float("inf")
        best_u = None
        best_v = None

        # Target is the other polyline
        if s == 0:
            target = R
            lenT = nR
        else:
            target = L
            lenT = nL

        for j in range(lenT - 1):
            b0 = target[j]
            b1 = target[j+1]

            if query_type == "point":
                # point–segment
                pb, tb = closest_point_on_segment(a0, b0, b1)
                pa = a0
                ta = 0.0
            else:
                # segment–segment
                pa, pb, ta, tb = segment_segment_closest(a0, a1, b0, b1)

            dist = np.linalg.norm(pa - pb)
            if dist < best_dist:
                best_dist = dist

                if s == 0:
                    # L-query → u comes from L, v from R
                    u = i + (ta if query_type == "segment" else 0.0)
                    v = j + tb
                else:
                    # R-query → u comes from L, v from R
                    u = j + ta
                    v = i + (tb if query_type == "segment" else 0.0)

                best_u = u
                best_v = v

        pairs.append((best_u, best_v))

    return pairs

def compute_matching_Mps(L, R):
    Mps = []
    for s in [0, 1]:
        for k in [0, 1]:
            M_sk = nearest_neighbor_search(L, R, k, s)
            Mps.extend(M_sk)
    return Mps


def onlineLW(L,R,M):
    Ustart = 0
    Vstart = 0
    
    if not M:
        Ustart,Vstart = 0,0
    else:
        Ustart,Vstart=M[-1]

    Mprime = compute_matching_Mps(L, R)
    M_prime_sorted = sorted(Mprime, key=lambda x: (x[0], x[1]))
    M_prime_fixed = []
    M_prime_mut = []

    for (u, v) in M_prime_sorted:
        if u >= Ustart and v >= Vstart:
            M_prime_fixed.append((u, v))
        else:
            M_prime_mut.append((u, v))
            
    M_fixed_new = list(M) + M_prime_fixed
    return M_fixed_new, M_prime_mut


def NVD(current_path, adjacent_vertices, points, heading_vector=None):
    if not adjacent_vertices:
        return None

    # Only one point use heading vector
    if len(current_path) == 1:
        if heading_vector is None:
            raise ValueError("Heading vector required for first step in NVD.")

        pn = points[current_path[-1]]
        heading_angle = np.arctan2(heading_vector[1], heading_vector[0])

        best = None
        best_angle = float("inf")

        for u in adjacent_vertices:
            #check against all adj vertices for best angle
            pu = points[u]
            vec = pu - pn
            angle = np.arctan2(vec[1], vec[0])
            diff = abs(angle - heading_angle)
            diff = min(diff, 2*np.pi - diff)

            if diff < best_angle:
                best_angle = diff
                best = u

        return best

    # Path length >= 2
    pn = points[current_path[-1]]
    pn_1 = points[current_path[-2]]
    prev_vec = pn - pn_1
    prev_angle = np.arctan2(prev_vec[1], prev_vec[0])

    best = None
    best_angle = float("inf")

    for u in adjacent_vertices:
        pu = points[u]
        next_vec = pu - pn
        next_angle = np.arctan2(next_vec[1], next_vec[0])

        diff = abs(next_angle - prev_angle)
        diff = min(diff, 2*np.pi - diff)

        if diff < best_angle:
            best_angle = diff
            best = u

    return best

def LRD(leftC, rightC):
    