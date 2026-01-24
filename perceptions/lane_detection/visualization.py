"""
Visualization utilities for lane detection.
Provides functions to visualize cone maps, adjacency graphs, and lane candidates.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from typing import Dict, List, Tuple, Optional


def visualize_cone_map(
    cone_map: np.ndarray,
    left_boundary: Optional[List[int]] = None,
    right_boundary: Optional[List[int]] = None,
    adjacency_list: Optional[Dict[int, List[int]]] = None,
    title: str = "Cone Map",
    figsize: Tuple[int, int] = (12, 10),
    show_ids: bool = False,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the cone map with optional boundaries and adjacency graph.
    
    Args:
        cone_map: Nx2 array of point coordinates
        left_boundary: List of point indices for left boundary
        right_boundary: List of point indices for right boundary
        adjacency_list: Dict mapping point index to list of neighbor indices
        title: Plot title
        figsize: Figure size
        show_ids: Whether to show point IDs
        save_path: If provided, save figure to this path
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot adjacency edges first (so they're behind points)
    if adjacency_list is not None:
        for i, neighbors in adjacency_list.items():
            for j in neighbors:
                if j > i:  # Avoid drawing edges twice
                    ax.plot(
                        [cone_map[i][0], cone_map[j][0]],
                        [cone_map[i][1], cone_map[j][1]],
                        'k-', alpha=0.2, linewidth=0.5, zorder=1
                    )
    
    # Plot all points
    ax.scatter(
        cone_map[:, 0], cone_map[:, 1],
        c='gray', s=50, alpha=0.5, label='Cones', zorder=2
    )
    
    # Highlight left boundary
    if left_boundary is not None and len(left_boundary) > 0:
        left_points = cone_map[left_boundary]
        ax.scatter(
            left_points[:, 0], left_points[:, 1],
            c='blue', s=100, marker='o', label='Left Boundary', zorder=3
        )
        # Connect left boundary points
        for i in range(len(left_boundary) - 1):
            ax.plot(
                [cone_map[left_boundary[i]][0], cone_map[left_boundary[i+1]][0]],
                [cone_map[left_boundary[i]][1], cone_map[left_boundary[i+1]][1]],
                'b-', linewidth=2, alpha=0.7, zorder=2
            )
    
    # Highlight right boundary
    if right_boundary is not None and len(right_boundary) > 0:
        right_points = cone_map[right_boundary]
        ax.scatter(
            right_points[:, 0], right_points[:, 1],
            c='orange', s=100, marker='s', label='Right Boundary', zorder=3
        )
        # Connect right boundary points
        for i in range(len(right_boundary) - 1):
            ax.plot(
                [cone_map[right_boundary[i]][0], cone_map[right_boundary[i+1]][0]],
                [cone_map[right_boundary[i]][1], cone_map[right_boundary[i+1]][1]],
                'orange', linewidth=2, alpha=0.7, zorder=2
            )
    
    # Show point IDs
    if show_ids:
        for i, (x, y) in enumerate(cone_map):
            ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='bottom')
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_perceptual_field(
    cone_map: np.ndarray,
    car_pos: np.ndarray,
    car_heading_rad: float,
    subgraph: Dict[int, List[int]],
    left_subset: Optional[List[int]] = None,
    right_subset: Optional[List[int]] = None,
    perceptual_range: float = 30.0,
    title: str = "Perceptual Field",
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize a perceptual field with car position and heading.
    
    Args:
        cone_map: Nx2 array of point coordinates
        car_pos: Car position [x, y]
        car_heading_rad: Car heading in radians
        subgraph: Adjacency dict for visible points
        left_subset: Visible left boundary points
        right_subset: Visible right boundary points
        perceptual_range: Range in meters (for drawing circle)
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure to this path
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get visible point indices
    visible_ids = set(subgraph.keys())
    
    # Plot all points (dimmed)
    ax.scatter(
        cone_map[:, 0], cone_map[:, 1],
        c='lightgray', s=30, alpha=0.3, label='All Cones', zorder=1
    )
    
    # Plot subgraph edges
    for i, neighbors in subgraph.items():
        for j in neighbors:
            if j > i:
                ax.plot(
                    [cone_map[i][0], cone_map[j][0]],
                    [cone_map[i][1], cone_map[j][1]],
                    'g-', alpha=0.4, linewidth=1, zorder=2
                )
    
    # Plot visible points
    visible_coords = cone_map[list(visible_ids)]
    ax.scatter(
        visible_coords[:, 0], visible_coords[:, 1],
        c='green', s=60, alpha=0.8, label='Visible Cones', zorder=3
    )
    
    # Highlight visible boundaries
    if left_subset:
        left_coords = cone_map[left_subset]
        ax.scatter(
            left_coords[:, 0], left_coords[:, 1],
            c='blue', s=120, marker='o', label='Left Boundary', zorder=4
        )
        for i in range(len(left_subset) - 1):
            ax.plot(
                [cone_map[left_subset[i]][0], cone_map[left_subset[i+1]][0]],
                [cone_map[left_subset[i]][1], cone_map[left_subset[i+1]][1]],
                'b-', linewidth=2.5, zorder=3
            )
    
    if right_subset:
        right_coords = cone_map[right_subset]
        ax.scatter(
            right_coords[:, 0], right_coords[:, 1],
            c='orange', s=120, marker='s', label='Right Boundary', zorder=4
        )
        for i in range(len(right_subset) - 1):
            ax.plot(
                [cone_map[right_subset[i]][0], cone_map[right_subset[i+1]][0]],
                [cone_map[right_subset[i]][1], cone_map[right_subset[i+1]][1]],
                'orange', linewidth=2.5, zorder=3
            )
    
    # Draw car position and heading
    ax.scatter([car_pos[0]], [car_pos[1]], c='red', s=200, marker='^', 
               label='Car', zorder=5)
    
    # Draw heading arrow
    arrow_len = perceptual_range * 0.3
    dx = arrow_len * np.cos(car_heading_rad)
    dy = arrow_len * np.sin(car_heading_rad)
    ax.arrow(car_pos[0], car_pos[1], dx, dy, 
             head_width=1.5, head_length=1, fc='red', ec='red', zorder=5)
    
    # Draw perceptual range circle
    circle = plt.Circle(car_pos, perceptual_range, fill=False, 
                        color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.add_patch(circle)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_lane_candidates(
    cone_map: np.ndarray,
    path_pairs: List[Tuple[List[int], List[int]]],
    title: str = "Lane Candidates",
    figsize: Tuple[int, int] = (12, 10),
    max_candidates: int = 5,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize lane candidate path pairs.
    
    Args:
        cone_map: Nx2 array of point coordinates
        path_pairs: List of (left_path, right_path) tuples
        title: Plot title
        figsize: Figure size
        max_candidates: Maximum number of candidates to show
        save_path: If provided, save figure to this path
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all points
    ax.scatter(
        cone_map[:, 0], cone_map[:, 1],
        c='gray', s=30, alpha=0.3, zorder=1
    )
    
    # Color palette for candidates
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(path_pairs), max_candidates)))
    
    for idx, (left_path, right_path) in enumerate(path_pairs[:max_candidates]):
        color = colors[idx]
        
        # Draw left path
        if len(left_path) > 1:
            left_coords = cone_map[left_path]
            ax.plot(left_coords[:, 0], left_coords[:, 1], 
                   'o-', color=color, linewidth=2, markersize=8,
                   label=f'Candidate {idx+1} Left', zorder=3)
        
        # Draw right path
        if len(right_path) > 1:
            right_coords = cone_map[right_path]
            ax.plot(right_coords[:, 0], right_coords[:, 1],
                   's--', color=color, linewidth=2, markersize=8,
                   label=f'Candidate {idx+1} Right', zorder=3)
        
        # Draw width lines (matching)
        if len(left_path) > 0 and len(right_path) > 0:
            for li in range(min(len(left_path), len(right_path))):
                ax.plot(
                    [cone_map[left_path[li]][0], cone_map[right_path[li]][0]],
                    [cone_map[left_path[li]][1], cone_map[right_path[li]][1]],
                    '--', color=color, alpha=0.3, linewidth=1, zorder=2
                )
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def visualize_epp_results(
    cone_map: np.ndarray,
    car_pos: np.ndarray,
    car_heading_rad: float,
    subgraph: Dict[int, List[int]],
    path_pairs: List[Tuple[List[int], List[int]]],
    perceptual_range: float = 30.0,
    title: str = "Paper-Faithful EPP Results",
    figsize: Tuple[int, int] = (16, 8),
    max_candidates: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Comprehensive visualization of EPP results with perceptual field and candidates.
    
    Args:
        cone_map: Nx2 array of point coordinates
        car_pos: Car position [x, y]
        car_heading_rad: Car heading in radians
        subgraph: Adjacency dict for visible points
        path_pairs: List of (left_path, right_path) candidates
        perceptual_range: Range in meters
        title: Overall figure title
        figsize: Figure size
        max_candidates: Maximum candidates to show
        save_path: If provided, save figure to this path
    
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left panel: Perceptual field with graph
    ax1 = axes[0]
    ax1.scatter(cone_map[:, 0], cone_map[:, 1], c='lightgray', s=30, alpha=0.5, label='All Cones')
    
    # Get visible cones
    visible = set()
    for neighbors in subgraph.values():
        visible.update(neighbors)
    visible.update(subgraph.keys())
    visible_coords = cone_map[list(visible)]
    ax1.scatter(visible_coords[:, 0], visible_coords[:, 1], c='green', s=50, label='Visible Cones')
    
    # Draw subgraph edges
    for v, neighbors in subgraph.items():
        for u in neighbors:
            ax1.plot([cone_map[v, 0], cone_map[u, 0]], 
                    [cone_map[v, 1], cone_map[u, 1]], 'g-', alpha=0.3, linewidth=0.5)
    
    # Car position
    ax1.scatter([car_pos[0]], [car_pos[1]], c='red', s=200, marker='^', label='Car', zorder=10)
    heading_vec = np.array([np.cos(car_heading_rad), np.sin(car_heading_rad)])
    ax1.arrow(car_pos[0], car_pos[1], heading_vec[0]*3, heading_vec[1]*3, 
             head_width=1, head_length=0.5, fc='red', ec='red')
    
    # Perceptual range circle
    circle = plt.Circle(car_pos, perceptual_range, fill=False, color='red', linestyle='--', alpha=0.5)
    ax1.add_patch(circle)
    
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_title(f'Perceptual Field ({len(subgraph)} visible nodes)')
    ax1.legend(loc='upper right')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Lane candidates
    ax2 = axes[1]
    ax2.scatter(cone_map[:, 0], cone_map[:, 1], c='lightgray', s=30, alpha=0.3)
    
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(path_pairs), max_candidates)))
    for i, (lp, rp) in enumerate(path_pairs[:max_candidates]):
        color = colors[i % len(colors)]
        
        # Left path
        lx = [cone_map[v, 0] for v in lp]
        ly = [cone_map[v, 1] for v in lp]
        ax2.plot(lx, ly, 'o-', color=color, markersize=6, linewidth=2, 
                label=f'C{i+1}: L={len(lp)}, R={len(rp)}', alpha=0.8)
        
        # Right path
        rx = [cone_map[v, 0] for v in rp]
        ry = [cone_map[v, 1] for v in rp]
        ax2.plot(rx, ry, 's-', color=color, markersize=6, linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_title(f'Lane Candidates ({len(path_pairs)} from Paper-Faithful EPP)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# Quick test function
if __name__ == "__main__":
    from data_loader import cone_maps, left_boundaries, right_boundaries, build_adjacency_graph
    
    # Test with first map
    cone_map = cone_maps[0]
    left_b = left_boundaries[0]
    right_b = right_boundaries[0]
    
    print(f"Cone map shape: {cone_map.shape}")
    print(f"Left boundary size: {len(left_b)}")
    print(f"Right boundary size: {len(right_b)}")
    
    # Build graph
    adj_list = build_adjacency_graph(cone_map, dmax=5.0)
    num_edges = sum(len(v) for v in adj_list.values()) // 2
    print(f"Adjacency graph: {len(adj_list)} nodes, {num_edges} edges")
    
    # Visualize
    fig = visualize_cone_map(
        cone_map, left_b, right_b, adj_list,
        title="Full Cone Map with Adjacency Graph",
        show_ids=False
    )
    plt.show()
