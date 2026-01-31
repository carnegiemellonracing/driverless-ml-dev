from enum import Enum
import numpy as np
import numpy.typing as npt
from typing import Annotated, List, Literal, Tuple, Dict, Set
from dataclasses import dataclass, field
import numpy.typing as npt

# Type definitions for clarity
Point = Annotated[npt.NDArray[np.float64], Literal[2]]
Map = Annotated[npt.NDArray[np.float64], Literal[..., 2]]
Lane = List[Point]  # A lane is a list of points
Graph = Dict[int, List[int]]

class Side(Enum):
    LEFT = "left"
    RIGHT = "right"


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
    matchings: MatchingSet = field(default_factory=MatchingSet)

    # 4. Validity Flags
    is_valid: bool = True


@dataclass
class PerceptualFieldContext:
    """
    Holds data for a single perceptual field (visible subset of map from car position).
    Used for training and inference on lane detection.
    """

    # Reference to the full cone map (shared across all contexts from same map)
    cone_map: Map  # Shape: (N, 2) where N is number of cones

    # Which global indices are visible in this perceptual field
    visible_indices: Set[int]

    # Subgraph adjacency (keys are GLOBAL indices from the original map)
    adj_list: Dict[int, List[int]]

    # Car state
    car_pos: Point
    car_heading: float

    # Optional: NVD cache for neighbor sorting during path enumeration
    nvd_cache: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)
    
    # Ground Truth Labels (for training generation)
    gt_left_idx: int = None
    gt_right_idx: int = None

    def get_point(self, global_idx: int) -> np.ndarray:
        """Get point coordinates by global index (direct access to shared map)."""
        return self.cone_map[global_idx]

    def has_point(self, global_idx: int) -> bool:
        """Check if a global index is visible in this perceptual field."""
        return global_idx in self.visible_indices
