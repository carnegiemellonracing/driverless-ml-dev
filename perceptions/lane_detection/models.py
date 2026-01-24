import numpy as np
from typing import Annotated, List, Literal, Tuple, Dict, Set
from dataclasses import dataclass, field

from perceptions.lane_detection.config import D_MAX
from scipy.spatial import ckdtree
import numpy.typing as npt

# Type definitions for clarity
Point = Annotated[npt.NDArray[np.float64], Literal[2, 1]]
Map = Annotated[npt.NDArray[np.float64], Literal[2, ...]]


@dataclass
class MatchingSet:
    """
    Represents the state of Algorithm 3.
    Splits width calculations into 'Fixed' (safe to cache) and 'Mutable' (volatile).
    """

    # Indices into the global map for the fixed set.
    # Shape: (N, 2) where col 0 is left_idx, col 1 is right_idx.
    fixed_indices: List[Tuple[int, int]] = field(default_factory=list)

    # Pre-calculated width values for the fixed set.
    fixed_widths: List[float] = field(default_factory=list)

    # The last index in the Left and Right paths that was successfully 'Fixed'.
    # Used to know where to start scanning for the next Mutable set.
    last_fixed_l_idx: int = 0
    last_fixed_r_idx: int = 0


@dataclass
class LaneCandidate:
    """
    Represents an individual lane candidate in the DFS tree
    """

    # 1. The Core Paths (Indices into the Global Map)
    left_path: List[int]
    right_path: List[int]

    # 2. Cycle Prevention
    # Tracks visited nodes for this specific branch to prevent loops.
    left_visited: Set[int]
    right_visited: Set[int]

    # 3. Geometric State
    # Persists the 'Fixed' matchings
    matchings: MatchingSet

    # 4. Validity Flags
    is_valid: bool = True


class GlobalContext:
    """
    Holds read-only data including array of points, graph, and NVD cache
    """

    def __init__(self, map_points: np.ndarray[(2, int)]):
        self.map_points: np.ndarray[(2, int)] = map_points  # The raw Nx2 array

        # Adjacency list: graph[i] -> [neighbor_idx_1, neighbor_idx_2, ...]
        # Edges exist if dist < D_MAX.
        self.adj_list: Dict[int, List[int]] = self._build_graph()

        # Key: (prev_idx, curr_idx) -> Representing the incoming vector.
        # Value: List[int] -> Neighbors sorted by NVD score (smallest angle deviation).
        self.nvd_cache: Dict[Tuple[int, int], List[int]] = {}

    def _build_graph(self) -> Dict[int, List[int]]:
        num_points = len(self.map_points)
        adj = {i: [] for i in range(num_points)}

        tree = ckdtree(self.map_points)

        # output is a set of tuples {(i, j), ...} where i < j
        pairs = tree.query_pairs(r=D_MAX)

        for i, j in pairs:
            adj[i].append(j)
            adj[j].append(i)

        return adj
