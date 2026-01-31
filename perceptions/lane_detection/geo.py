
import numpy as np
import math
from typing import List, Tuple, Dict, Set, Optional

from perceptions.lane_detection.models import Point, Map, Graph, LaneCandidate, PerceptualFieldContext, MatchingSet, Side
from perceptions.lane_detection.config import D_MAX, W_MIN, W_MAX, PHI_MAX

# =============================================================================
# PRIMITIVES & UTILS
# =============================================================================

def within_range(point, car_pos, max_range):
    """Checks if a point is within a certain range of the car."""
    return np.linalg.norm(point - car_pos) <= max_range


def within_cone(point, car_pos, heading, cone_angle):
    """Checks if a point is within the car's field of view cone."""
    if np.array_equal(point, car_pos):
        return True
        
    v_point = point - car_pos
    v_heading = np.array([np.cos(heading), np.sin(heading)])
    
    norm_point = np.linalg.norm(v_point)
    if norm_point == 0:
        return True
        
    # Calculate angle between heading vector and point vector
    dot = np.clip(np.dot(v_point / norm_point, v_heading), -1.0, 1.0)
    angle = np.arccos(dot)
    
    return angle <= (cone_angle / 2)


def calculate_segment_angle(p1, p2, p3):
    """Calculates the absolute deflection angle between two consecutive segments."""
    v1 = p2 - p1
    v2 = p3 - p2

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    v1_u = v1 / norm_v1
    v2_u = v2 / norm_v2

    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)

    return angle_rad


def point_to_segment_distance(point, seg_start, seg_end):
    """
    Calculate perpendicular distance from a point to a line segment.
    Returns (distance, projection_point).
    """
    point = np.array(point)
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)

    seg_vec = seg_end - seg_start
    point_vec = point - seg_start

    seg_length_sq = np.dot(seg_vec, seg_vec)

    if seg_length_sq < 1e-8:
        distance = float(np.linalg.norm(point_vec))
        return distance, 0.0

    t = float(np.dot(point_vec, seg_vec) / seg_length_sq)
    t_clamped = np.clip(t, 0.0, 1.0)
    projection = seg_start + t_clamped * seg_vec
    distance = float(np.linalg.norm(point - projection))

    return distance, float(t_clamped)


def segment_to_segment_distance(s1_start, s1_end, s2_start, s2_end):
    """Calculates the shortest distance between two segments. Returns (dist, t1, t2)."""
    if line_segments_intersect(s1_start, s1_end, s2_start, s2_end):
        return 0.0, 0.5, 0.5 # t values approximate

    # Simplified distance check
    d1, _ = point_to_segment_distance(s1_start, s2_start, s2_end)
    d2, _ = point_to_segment_distance(s1_end, s2_start, s2_end)
    d3, _ = point_to_segment_distance(s2_start, s1_start, s1_end)
    d4, _ = point_to_segment_distance(s2_end, s1_start, s1_end)
    
    min_d = min(d1, d2, d3, d4)
    # Return dummy t's
    return min_d, 0.0, 0.0


def point_to_polygonal_chain_distance(point, chain):
    """Calculates minimum distance from a point to a polygonal chain."""
    if len(chain) == 0:
        return float("inf")
    if len(chain) == 1:
        return np.linalg.norm(np.array(point) - np.array(chain[0]))
    
    min_dist = float("inf")
    for i in range(len(chain) - 1):
        dist, _ = point_to_segment_distance(point, chain[i], chain[i+1])
        if dist < min_dist:
            min_dist = dist
    return min_dist


def line_segments_intersect(p1, p2, p3, p4):
    """Check if two line segments intersect using cross product method"""
    p1, p2, p3, p4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


# =============================================================================
# ALGORITHM 3: ONLINE LANE WIDTH
# =============================================================================

def online_lane_width(
    ctx: PerceptualFieldContext, candidate: LaneCandidate
) -> Tuple[MatchingSet, float, float]:
    """
    Implements Algorithm 3 (Online Lane Width Calculation).
    Returns (new_matching_set, min_width, max_width).
    """
    l_path = candidate.left_path
    r_path = candidate.right_path
    matchings = candidate.matchings

    start_l = matchings.last_fixed_l_idx
    start_r = matchings.last_fixed_r_idx

    max_l_param = float(len(l_path) - 1)
    max_r_param = float(len(r_path) - 1)

    new_matchings = []

    left_points = [ctx.get_point(idx) for idx in l_path[start_l:]]
    right_points = [ctx.get_point(idx) for idx in r_path[start_r:]]

    # 2a. Left vertices to right segments
    for i, l_point in enumerate(left_points):
        l_param = float(start_l + i)
        for j in range(len(right_points) - 1):
            r_seg_start = right_points[j]
            r_seg_end = right_points[j + 1]
            dist, t = point_to_segment_distance(l_point, r_seg_start, r_seg_end)
            r_param = float(start_r + j) + float(t)
            new_matchings.append((dist, l_param, r_param))

    # 2b. Right vertices to left segments
    for j, r_point in enumerate(right_points):
        r_param = float(start_r + j)
        for i in range(len(left_points) - 1):
            l_seg_start = left_points[i]
            l_seg_end = left_points[i + 1]
            dist, t = point_to_segment_distance(r_point, l_seg_start, l_seg_end)
            l_param = float(start_l + i) + float(t)
            new_matchings.append((dist, l_param, r_param))

    # 2c. Segment-to-segment (simplified for performance/conflict resolution)
    for i in range(len(left_points) - 1):
        for j in range(len(right_points) - 1):
             l_seg_start = left_points[i]
             l_seg_end = left_points[i + 1]
             r_seg_start = right_points[j]
             r_seg_end = right_points[j + 1]
             dist, t_l, t_r = segment_to_segment_distance(
                 l_seg_start, l_seg_end, r_seg_start, r_seg_end
             )
             if dist < 1e-6: # Intersection?
                 new_matchings.append((dist, float(start_l + i)+float(t_l), float(start_r + j)+float(t_r)))

    new_matchings.sort(key=lambda m: (float(m[1]), float(m[2])))

    split_idx = len(new_matchings)
    for k, (dist, l_param, r_param) in enumerate(new_matchings):
        if l_param >= max_l_param - 1e-9 or r_param >= max_r_param - 1e-9:
            split_idx = k
            break

    fixed_new = new_matchings[:split_idx]
    mutable_new = new_matchings[split_idx:]

    new_fixed_indices = matchings.fixed_indices + [
        (int(np.floor(m[1])), int(np.floor(m[2]))) for m in fixed_new
    ]
    new_fixed_widths = matchings.fixed_widths + [m[0] for m in fixed_new]

    if fixed_new:
        last_fixed_l = int(np.floor(fixed_new[-1][1]))
        last_fixed_r = int(np.floor(fixed_new[-1][2]))
    else:
        last_fixed_l = matchings.last_fixed_l_idx
        last_fixed_r = matchings.last_fixed_r_idx

    updated_matchings = MatchingSet(
        fixed_indices=new_fixed_indices,
        fixed_widths=new_fixed_widths,
        last_fixed_l_idx=last_fixed_l,
        last_fixed_r_idx=last_fixed_r,
    )

    all_widths = new_fixed_widths + [m[0] for m in mutable_new]
    if not all_widths:
        return updated_matchings, W_MIN + 1.0, W_MIN + 1.0

    min_w = min(all_widths)
    max_w = max(all_widths)

    return updated_matchings, min_w, max_w


# =============================================================================
# CONSTRAINTS & DECIDERS
# =============================================================================

def C_seg(
    lane_candidate: LaneCandidate,
    context: PerceptualFieldContext,
    side: str = "left",
    max_angle: float = PHI_MAX,
) -> bool:
    path = lane_candidate.left_path if side == "left" else lane_candidate.right_path
    if len(path) < 3:
        return True

    for i in range(len(path) - 2):
        p1 = context.get_point(path[i])
        p2 = context.get_point(path[i + 1])
        p3 = context.get_point(path[i + 2])

        if calculate_segment_angle(p1, p2, p3) > max_angle:
            return False
    return True


def C_poly(lane_candidate: LaneCandidate, context: PerceptualFieldContext) -> bool:
    left_points = [context.get_point(idx) for idx in lane_candidate.left_path]
    right_points = [context.get_point(idx) for idx in lane_candidate.right_path]
    
    # Construct polygon
    poly_points = left_points + right_points[::-1]
    n = len(poly_points)
    if n < 4:
        return True

    # Check self intersection
    for i in range(n):
        for j in range(i + 2, n):
            if j == (i + 1) % n or i == (j + 1) % n:
                continue
            if line_segments_intersect(poly_points[i], poly_points[(i+1)%n], 
                                     poly_points[j], poly_points[(j+1)%n]):
                return False
    return True


def C_width(lane_candidate: LaneCandidate, context: PerceptualFieldContext) -> bool:
    # Use online_lane_width to compute the min and max widths
    _, min_width, max_width = online_lane_width(context, lane_candidate)
    return W_MIN <= min_width and max_width <= W_MAX


def constraint_decider(candidate: LaneCandidate, ctx: PerceptualFieldContext) -> bool:
    return (C_seg(candidate, ctx, "left") and 
            C_seg(candidate, ctx, "right") and 
            C_poly(candidate, ctx) and 
            C_width(candidate, ctx))


def next_vertex_decider(ctx: PerceptualFieldContext, current_path: List[int]) -> Optional[int]:
    """NVD: Selects best next vertex to minimize angle."""
    if not current_path:
        return None
    current_idx = current_path[-1]
    adjacent = [v for v in ctx.adj_list.get(current_idx, []) if v not in current_path]
    if not adjacent:
        return None
        
    # Cache check
    if len(current_path) >= 2:
        prev_idx = current_path[-2]
        if (prev_idx, current_idx) in ctx.nvd_cache:
            for v in ctx.nvd_cache[(prev_idx, current_idx)]:
                if v in adjacent:
                    return v

    # Compute
    p_curr = ctx.get_point(current_idx)
    if len(current_path) == 1:
        v_prev = np.array([np.cos(ctx.car_heading), np.sin(ctx.car_heading)])
    else:
        v_prev = p_curr - ctx.get_point(current_path[-2])
        
    best_v = adjacent[0]
    min_angle = float("inf")
    
    for v_idx in adjacent:
        v_next = ctx.get_point(v_idx) - p_curr
        
        n_prev = np.linalg.norm(v_prev)
        n_next = np.linalg.norm(v_next)
        
        if n_prev == 0 or n_next == 0:
            angle = 0.0
        else:
            angle = np.arccos(np.clip(np.dot(v_prev, v_next)/(n_prev*n_next), -1, 1))
            
        if angle < min_angle:
            min_angle = angle
            best_v = v_idx
            
    return best_v


def left_right_decider(ctx: PerceptualFieldContext, candidate: LaneCandidate, n0: int, n1: int) -> int:
    """LRD: 0 for left, 1 for right."""
    if n0 is None: return 1
    if n1 is None: return 0
    
    lp = candidate.left_path
    rp = candidate.right_path
    
    if len(lp) < 2 or len(rp) < 2:
        return 0 if len(lp) <= len(rp) else 1
        
    def get_angle(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p2
        return calculate_segment_angle(p1, p2, p3) # logic reuse
        
    ln = ctx.get_point(lp[-1])
    pn0 = ctx.get_point(n0)
    rm = ctx.get_point(rp[-1])
    
    theta1_l = get_angle(ln, pn0, rm) # Approx angle check
    theta1_r = get_angle(ctx.get_point(rp[-2]), rm, pn0)
    
    diff1 = abs(theta1_r - theta1_l)
    
    # Similarly for right extension
    theta2_l = get_angle(ctx.get_point(lp[-2]), ln, ctx.get_point(n1))
    theta2_r = get_angle(rm, ctx.get_point(n1), ln)
    
    diff2 = abs(theta2_r - theta2_l)
    
    return 0 if diff1 < diff2 else 1


def backtracking_decider(min_width: float, max_width: float, violation_in_fixed: bool) -> bool:
    if violation_in_fixed: return True
    if min_width < W_MIN: return True
    return False


# =============================================================================
# EPP
# =============================================================================

def EPP(ctx: PerceptualFieldContext, candidate: LaneCandidate, iteration: int, itmax: int) -> List[LaneCandidate]:
    """Paper-faithful EPP working with new Types."""
    results = []
    stack = [(candidate, iteration)]
    
    # print(f"DEBUG: EPP Start. Stack size: {len(stack)}")
    while stack:
        cand, it = stack.pop()
        
        # Algorithm 2 adds every valid P to Phi.
        # Since we push only valid candidates (checked BTD/CD), 'cand' is valid.
        if len(cand.left_path) > 1 and len(cand.right_path) > 1: # Min length heuristic
             results.append(cand)
             
        if it > itmax: 
            # print("DEBUG: itmax reached")
            continue
        
        # Check Unvisited
        u0_idx = next_vertex_decider(ctx, cand.left_path)
        u1_idx = next_vertex_decider(ctx, cand.right_path)
        
        # print(f"DEBUG: NVD L->{u0_idx}, R->{u1_idx}")
        
        # Logic fix: NVD above filters `v not in current_path`.
        if u0_idx is not None and u0_idx in cand.left_visited: u0_idx = None
        if u1_idx is not None and u1_idx in cand.right_visited: u1_idx = None

        if u0_idx is None and u1_idx is None:
            # Dead end. Already added above.
            continue
            
        # LRD
        side = left_right_decider(ctx, cand, u0_idx, u1_idx)
        # print(f"DEBUG: LRD side={side}")
        
        next_v = u0_idx if side == 0 else u1_idx
        if next_v is None: 
            # print("DEBUG: LRD picked None side?")
            continue
        
        # Extend
        if side == 0:
            new_l = cand.left_path + [next_v]
            new_r = list(cand.right_path)
            new_lv = cand.left_visited | {next_v}
            new_rv = set(cand.right_visited)
            # print(f"DEBUG: Extending Left to {next_v}")
        else:
            new_l = list(cand.left_path)
            new_r = cand.right_path + [next_v]
            new_lv = set(cand.left_visited)
            new_rv = cand.right_visited | {next_v}
            # print(f"DEBUG: Extending Right to {next_v}")
            
        # Update matchings
        temp_cand_for_lw = LaneCandidate(new_l, new_r, new_lv, new_rv, cand.matchings)
        new_matchings, min_w, max_w = online_lane_width(ctx, temp_cand_for_lw)
        
        new_cand = LaneCandidate(new_l, new_r, new_lv, new_rv, new_matchings)
        
        # Check BTD
        # Check for violation in fixed set
        violation_in_fixed = False
        for w in new_matchings.fixed_widths:
            if w < W_MIN or w > W_MAX:
                violation_in_fixed = True
                break
                
        if backtracking_decider(min_w, max_w, violation_in_fixed): 
             # print(f"DEBUG: BTD Prune. min_w={min_w:.2f}, max_w={max_w:.2f}")
             continue # Prune
             
        # Check CD (Seg, Poly)
        if not (C_seg(new_cand, ctx, "left") and C_seg(new_cand, ctx, "right") and C_poly(new_cand, ctx)):
            # print(f"DEBUG: CD Prune.")
            continue
            
        stack.append((new_cand, it + 1))
        
    return results


def enumerate_path_pairs_v2(graph, points, paths, visited, heading_vector, it, itmax=2500):
    """Legacy wrapper."""
    ctx = PerceptualFieldContext(points, set(graph.keys()), graph, np.array([0,0]), 0.0) # Dummy car pos?
    # Actually we need car_heading for NVD.
    # heading_vector is passed.
    heading_ang = np.arctan2(heading_vector[1], heading_vector[0])
    ctx.car_heading = heading_ang
    
    sl, sr = paths[0][0], paths[1][0]
    cand = LaneCandidate([sl], [sr], {sl}, {sr})
    
    results = EPP(ctx, cand, it, itmax)
    
    return [(r.left_path, r.right_path) for r in results]

# =============================================================================
# FEATURES & IOU
# =============================================================================

def compute_features(path_pair, points):
    """Compute 8 geometric features."""
    left_path, right_path = path_pair
    
    def get_len(path):
        l = 0
        for i in range(len(path)-1):
            l += np.linalg.norm(points[path[i+1]]-points[path[i]])
        return l
        
    def get_angles(path):
        a = []
        for i in range(len(path)-2):
            a.append(calculate_segment_angle(points[path[i]], points[path[i+1]], points[path[i+2]]))
        return a
        
    l_len = get_len(left_path)
    r_len = get_len(right_path)
    mean_len = (l_len + r_len)/2
    
    l_angs = get_angles(left_path)
    r_angs = get_angles(right_path)
    
    # Width variance approximation
    widths = []
    if len(left_path) > 0 and len(right_path) > 0:
        for i in left_path:
            widths.append(point_to_polygonal_chain_distance(points[i], [points[j] for j in right_path]))
            
    return [mean_len, len(left_path), len(right_path), np.var(widths) if widths else 0,
            0, 0, np.var(l_angs) if l_angs else 0, np.var(r_angs) if r_angs else 0] # simplified

def compute_lane_iou(candidate_pair, gt_pair, points, grid_res=0.5):
    """IoU Computation."""
    from matplotlib.path import Path
    
    def get_poly(pair):
        l, r = pair
        if not l or not r: return None
        return np.array([points[i] for i in l + r[::-1]])
        
    c_poly = get_poly(candidate_pair)
    g_poly = get_poly(gt_pair)
    
    if c_poly is None or g_poly is None: return 0.0
    
    all_p = np.vstack([c_poly, g_poly])
    min_x, min_y = np.min(all_p, axis=0)
    max_x, max_y = np.max(all_p, axis=0)
    
    params = np.array(np.meshgrid(np.arange(min_x, max_x, grid_res), np.arange(min_y, max_y, grid_res))).T.reshape(-1, 2)
    if len(params) == 0: return 0.0
    
    c_in = Path(c_poly).contains_points(params)
    g_in = Path(g_poly).contains_points(params)
    
    i = np.sum(c_in & g_in)
    u = np.sum(c_in | g_in)
    return i/u if u > 0 else 0.0

def find_starting_vertices(graph: Graph, cone_map: Map, car_pos: Point, car_heading_rad: float, max_range=2) -> tuple[int, int]:
    """Selects two starting vertices from a graph close to car."""
    candidates_within_range = []
    for idx in graph.keys():
        point = cone_map[idx]
        if within_range(point, car_pos, max_range):
            candidates_within_range.append((idx, point))

    if not candidates_within_range:
        return (None, None)

    car_heading_vec = np.array([np.cos(car_heading_rad), np.sin(car_heading_rad)])

    left_candidates = []
    right_candidates = []

    for idx, point in candidates_within_range:
        vec_to_point = point - car_pos
        norm_vec = np.linalg.norm(vec_to_point)

        if norm_vec == 0:
            continue

        vec_u = vec_to_point / norm_vec
        
        cross = car_heading_vec[0] * vec_u[1] - car_heading_vec[1] * vec_u[0]
        dot = np.dot(car_heading_vec, vec_u)
        angle = np.arctan2(cross, dot)

        if angle > 0:
            left_candidates.append((idx, angle))
        elif angle < 0:
            right_candidates.append((idx, angle))

    if not left_candidates or not right_candidates:
        return (None, None)

    best_symmetry = float("inf")
    best_pair = (None, None)

    for idx_l, angle_l in left_candidates:
        for idx_r, angle_r in right_candidates:
            symmetry = abs(angle_l + angle_r)
            if symmetry < best_symmetry:
                best_symmetry = symmetry
                best_pair = (idx_l, idx_r)

    return best_pair