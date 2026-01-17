#!/usr/bin/env python3
"""
Transform paired YAML files (boundaries_{i}.yaml and cone_map_{i}.yaml)
into compressed NPZ files containing cone positions and boundary indices.
"""

import yaml
import numpy as np
import os
from pathlib import Path


def load_yaml(filepath):
    """Load and parse a YAML file."""
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def transform_dataset(boundaries_file, cone_map_file, output_file):
    """
    Transform a pair of YAML files into an NPZ file.

    Args:
        boundaries_file: Path to boundaries_{i}.yaml
        cone_map_file: Path to cone_map_{i}.yaml
        output_file: Path to output data_{i}.npz
    """
    # Load the data
    boundaries = load_yaml(boundaries_file)
    cone_map = load_yaml(cone_map_file)

    # Get left and right boundary indices from the YAML
    left_boundary_orig = boundaries["left"]
    right_boundary_orig = boundaries["right"]

    # Create a mapping from original cone indices to sequential array indices
    # Get all unique cone indices and sort them for consistency
    all_cone_indices = sorted(cone_map.keys())

    # Create the points array and index mapping
    points = []
    orig_to_new_idx = {}

    for new_idx, orig_idx in enumerate(all_cone_indices):
        coords = cone_map[orig_idx]
        points.append(coords)
        orig_to_new_idx[orig_idx] = new_idx

    # Convert to numpy array
    points = np.array(points, dtype=np.float32)

    # Transform boundary indices using the mapping
    left_boundary_indices = np.array(
        [orig_to_new_idx[idx] for idx in left_boundary_orig], dtype=np.int32
    )
    right_boundary_indices = np.array(
        [orig_to_new_idx[idx] for idx in right_boundary_orig], dtype=np.int32
    )

    # Save to NPZ file
    np.savez_compressed(
        output_file,
        points=points,
        left_boundary_indices=left_boundary_indices,
        right_boundary_indices=right_boundary_indices,
    )

    print(f"Created {output_file}")
    print(f"  - Points: {points.shape}")
    print(f"  - Left boundary: {left_boundary_indices.shape[0]} points")
    print(f"  - Right boundary: {right_boundary_indices.shape[0]} points")


def main():
    """Process all dataset pairs (1-9) and create NPZ files."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Create output directory
    output_dir = script_dir / "processed"
    output_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")
    print("-" * 60)

    # Process each dataset pair
    for i in range(1, 10):
        boundaries_file = script_dir / f"boundaries_{i}.yaml"
        cone_map_file = script_dir / f"cone_map_{i}.yaml"
        output_file = output_dir / f"data_{i}.npz"

        if boundaries_file.exists() and cone_map_file.exists():
            transform_dataset(boundaries_file, cone_map_file, output_file)
            print()
        else:
            if not boundaries_file.exists():
                print(f"Warning: {boundaries_file} not found")
            if not cone_map_file.exists():
                print(f"Warning: {cone_map_file} not found")

    print("-" * 60)
    print(f"Transformation complete! Files saved to {output_dir}")


if __name__ == "__main__":
    main()
