import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geo import (
    find_matching_segments,
    find_matching_points,
    constraint_decider,
    bt_decider,
)
from angle_utils import calculate_segment_angle


class ConstraintTester:
    """
    Comprehensive tool for testing all geometric constraints together:
    - C_seg: Segment angle constraint
    - C_width: Width constraint
    - C_poly: Polygon constraint
    """

    def __init__(self, dataset_path=None):
        if dataset_path is None:
            self.dataset_path = (
                f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/dataset"
            )
        else:
            self.dataset_path = dataset_path

        self.tracks = []
        self.load_all_tracks()

    def load_yaml_data(self, path):
        """Load YAML data from file"""
        with open(path, "r") as file:
            return yaml.load(file, Loader=yaml.FullLoader)

    def load_all_tracks(self):
        """Load all boundary and cone map data"""
        boundary_paths = [
            f"{self.dataset_path}/boundaries_{i}.yaml" for i in range(1, 10)
        ]
        cone_map_paths = [
            f"{self.dataset_path}/cone_map_{i}.yaml" for i in range(1, 10)
        ]

        self.tracks = []
        for i, (boundary_path, cone_map_path) in enumerate(
            zip(boundary_paths, cone_map_paths)
        ):
            if os.path.exists(boundary_path) and os.path.exists(cone_map_path):
                boundaries = self.load_yaml_data(boundary_path)
                cone_map = self.load_yaml_data(cone_map_path)

                # Convert cone map to coordinate lookup
                coordinates = {}
                for point_id, coords in cone_map.items():
                    coordinates[int(point_id)] = coords

                track_data = {
                    "id": i + 1,
                    "boundaries": boundaries,
                    "coordinates": coordinates,
                    "left_points": [
                        coordinates[point_id]
                        for point_id in boundaries["left"]
                        if point_id in coordinates
                    ],
                    "right_points": [
                        coordinates[point_id]
                        for point_id in boundaries["right"]
                        if point_id in coordinates
                    ],
                }

                self.tracks.append(track_data)

    def test_all_constraints(self, track_id, wmin=2.5, wmax=6.5):
        """
        Test all geometric constraints together: C_seg, C_width, and C_poly
        Returns comprehensive constraint analysis
        """
        track = self.tracks[track_id - 1]
        left_points = track["left_points"]
        right_points = track["right_points"]

        # Convert to numpy arrays for easier computation
        left_coords = np.array(left_points)
        right_coords = np.array(right_points)

        constraint_results = {
            "track_id": track_id,
            "seg_angle": {"passed": True, "violations": [], "details": {}},
            "width": {"passed": True, "violations": [], "details": {}},
            "polygon": {"passed": True, "violations": [], "details": {}},
            "overall_passed": True,
        }

        # Test 1: Segment Angle Constraint (C_seg)
        seg_violations = []

        # Check left path angles
        for i in range(len(left_coords) - 2):
            p1, p2, p3 = left_coords[i], left_coords[i + 1], left_coords[i + 2]
            angle = calculate_segment_angle(p1, p2, p3)
            if angle > np.pi / 2:  # 90 degrees
                seg_violations.append(("left", i, angle))

        # Check right path angles
        for i in range(len(right_coords) - 2):
            p1, p2, p3 = right_coords[i], right_coords[i + 1], right_coords[i + 2]
            angle = calculate_segment_angle(p1, p2, p3)
            if angle > np.pi / 2:  # 90 degrees
                seg_violations.append(("right", i, angle))

        constraint_results["seg_angle"]["violations"] = seg_violations
        constraint_results["seg_angle"]["passed"] = len(seg_violations) == 0
        constraint_results["seg_angle"]["details"] = {
            "max_angle_left": max(
                [v[2] for v in seg_violations if v[0] == "left"], default=0
            ),
            "max_angle_right": max(
                [v[2] for v in seg_violations if v[0] == "right"], default=0
            ),
            "violation_count": len(seg_violations),
        }

        # Test 2: Width Constraint (C_width)
        width_analysis = self.calculate_widths_for_track(track_id, wmin, wmax)
        width_violations = width_analysis["violations"]

        constraint_results["width"]["violations"] = width_violations
        constraint_results["width"]["passed"] = len(width_violations) == 0
        constraint_results["width"]["details"] = {
            "mean_width": width_analysis["stats"]["mean"],
            "min_width": width_analysis["stats"]["min"],
            "max_width": width_analysis["stats"]["max"],
            "violation_count": len(width_violations),
            "violation_rate": width_analysis["stats"]["violation_rate"],
        }

        # Test 3: Polygon Constraint (C_poly)
        poly_violations = []

        if len(left_coords) >= 2 and len(right_coords) >= 2:
            # Create polygon: left boundary + right boundary (reversed)
            polygon_points = []
            polygon_points.extend(left_coords)
            polygon_points.extend(right_coords[::-1])  # Reverse right boundary

            # Check for self-intersections
            n = len(polygon_points)
            for i in range(n):
                for j in range(i + 2, n):
                    if j == (i + 1) % n or i == (j + 1) % n:
                        continue

                    if self._line_segments_intersect(
                        polygon_points[i],
                        polygon_points[(i + 1) % n],
                        polygon_points[j],
                        polygon_points[(j + 1) % n],
                    ):
                        poly_violations.append((i, j))

        constraint_results["polygon"]["violations"] = poly_violations
        constraint_results["polygon"]["passed"] = len(poly_violations) == 0
        constraint_results["polygon"]["details"] = {
            "polygon_points": (
                len(polygon_points) if "polygon_points" in locals() else 0
            ),
            "intersection_count": len(poly_violations),
        }

        # Overall result
        constraint_results["overall_passed"] = (
            constraint_results["seg_angle"]["passed"]
            and constraint_results["width"]["passed"]
            and constraint_results["polygon"]["passed"]
        )

        return constraint_results

    def calculate_widths_for_track(self, track_id, wmin=2.5, wmax=6.5):
        """
        Calculate widths for track using paper-accurate segment-based matching algorithm.
        Uses perpendicular distance to segments instead of point-to-point distance.
        """
        track = self.tracks[track_id - 1]

        # Convert points to the format expected by find_matching_segments
        all_points = track["left_points"] + track["right_points"]
        left_indices = list(range(len(track["left_points"])))
        right_indices = list(
            range(
                len(track["left_points"]),
                len(track["left_points"]) + len(track["right_points"]),
            )
        )

        # Calculate matching lines and widths using paper-accurate segment-based matching
        matching_lines = find_matching_segments(left_indices, right_indices, all_points)

        width_analysis = {
            "track_id": track_id,
            "matching_lines": matching_lines,
            "widths": [match["width"] for match in matching_lines],
            "violations": [],
            "stats": {},
        }

        # Analyze violations
        for i, match in enumerate(matching_lines):
            width = match["width"]
            if width < wmin:
                width_analysis["violations"].append(
                    {
                        "type": "too_narrow",
                        "index": i,
                        "width": width,
                        "threshold": wmin,
                        "severity": (wmin - width) / wmin,
                    }
                )
            elif width > wmax:
                width_analysis["violations"].append(
                    {
                        "type": "too_wide",
                        "index": i,
                        "width": width,
                        "threshold": wmax,
                        "severity": (width - wmax) / wmax,
                    }
                )

        # Calculate statistics
        widths = width_analysis["widths"]
        width_analysis["stats"] = {
            "mean": np.mean(widths),
            "std": np.std(widths),
            "min": np.min(widths),
            "max": np.max(widths),
            "median": np.median(widths),
            "violation_count": len(width_analysis["violations"]),
            "violation_rate": (
                len(width_analysis["violations"]) / len(widths) if widths else 0
            ),
        }

        return width_analysis

    def _line_segments_intersect(self, p1, p2, p3, p4):
        """Check if two line segments intersect using cross product method"""

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def visualize_constraint_test(self, track_id, figsize=(12, 8)):
        """
        Visualize constraint testing results with comprehensive analysis
        """
        constraint_results = self.test_all_constraints(track_id)
        track = self.tracks[track_id - 1]

        fig = plt.figure(figsize=figsize)

        # 1. Track layout with constraint violations highlighted (top-left)
        ax1 = plt.subplot(2, 3, 1)
        self._plot_track_with_constraints(ax1, track, constraint_results)

        # 2. Constraint summary (top-center)
        ax2 = plt.subplot(2, 3, 2)
        self._plot_constraint_summary(ax2, constraint_results)

        # 3. Width analysis (top-right)
        ax3 = plt.subplot(2, 3, 3)
        width_analysis = self.calculate_widths_for_track(track_id)
        self._plot_width_distribution(ax3, width_analysis)

        # 4. Segment angle analysis (bottom-left)
        ax4 = plt.subplot(2, 3, 4)
        self._plot_segment_angles(ax4, track, constraint_results)

        # 5. Polygon analysis (bottom-center)
        ax5 = plt.subplot(2, 3, 5)
        self._plot_polygon_analysis(ax5, track, constraint_results)

        # 6. Overall results (bottom-right)
        ax6 = plt.subplot(2, 3, 6)
        self._plot_overall_results(ax6, constraint_results)

        plt.suptitle(
            f"Track {track_id} - Complete Constraint Analysis",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

        return constraint_results

    def _plot_track_with_constraints(self, ax, track, constraint_results):
        """Plot track with constraint violations highlighted"""
        left_points = np.array(track["left_points"])
        right_points = np.array(track["right_points"])

        # Plot boundaries
        ax.plot(
            left_points[:, 0],
            left_points[:, 1],
            "b-",
            linewidth=3,
            label="Left Boundary",
            alpha=0.7,
        )
        ax.plot(
            right_points[:, 0],
            right_points[:, 1],
            "r-",
            linewidth=3,
            label="Right Boundary",
            alpha=0.7,
        )

        # Highlight segment angle violations
        seg_violations = constraint_results["seg_angle"]["violations"]
        for side, idx, angle in seg_violations:
            if side == "left" and idx < len(left_points) - 1:
                ax.scatter(
                    left_points[idx + 1, 0],
                    left_points[idx + 1, 1],
                    color="red",
                    s=100,
                    marker="X",
                    zorder=5,
                    label="Sharp Turn" if idx == 0 else "",
                )
            elif side == "right" and idx < len(right_points) - 1:
                ax.scatter(
                    right_points[idx + 1, 0],
                    right_points[idx + 1, 1],
                    color="red",
                    s=100,
                    marker="X",
                    zorder=5,
                    label="Sharp Turn" if idx == 0 else "",
                )

        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title("Track with Constraint Violations")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    def _plot_constraint_summary(self, ax, constraint_results):
        """Plot constraint summary pie chart"""
        constraints = ["Segment Angle", "Width", "Polygon"]
        passed = [
            constraint_results["seg_angle"]["passed"],
            constraint_results["width"]["passed"],
            constraint_results["polygon"]["passed"],
        ]

        colors = ["green" if p else "red" for p in passed]
        wedges, texts, autotexts = ax.pie(
            [1, 1, 1], labels=constraints, colors=colors, autopct="%s", startangle=90
        )

        ax.set_title("Constraint Status")

    def _plot_width_distribution(self, ax, width_analysis):
        """Plot width distribution histogram with constraints"""
        widths = width_analysis["widths"]
        stats = width_analysis["stats"]

        # Create histogram
        n, bins, patches = ax.hist(
            widths, bins=20, alpha=0.7, color="skyblue", edgecolor="black"
        )

        # Color bars based on constraints
        wmin, wmax = 2.5, 6.5
        for i, (patch, bin_left, bin_right) in enumerate(
            zip(patches, bins[:-1], bins[1:])
        ):
            if bin_right < wmin or bin_left > wmax:
                patch.set_facecolor("red")
                patch.set_alpha(0.8)
            elif bin_left < wmin or bin_right > wmax:
                patch.set_facecolor("orange")
                patch.set_alpha(0.8)
            else:
                patch.set_facecolor("green")
                patch.set_alpha(0.8)

        # Add constraint lines
        ax.axvline(
            wmin, color="red", linestyle="--", linewidth=2, label=f"Min Width ({wmin}m)"
        )
        ax.axvline(
            wmax, color="red", linestyle="--", linewidth=2, label=f"Max Width ({wmax}m)"
        )
        ax.axvline(
            stats["mean"],
            color="blue",
            linestyle="-",
            linewidth=2,
            label=f'Mean ({stats["mean"]:.2f}m)',
        )

        ax.set_xlabel("Width (meters)")
        ax.set_ylabel("Frequency")
        ax.set_title("Width Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_segment_angles(self, ax, track, constraint_results):
        """Plot segment angle analysis"""
        left_coords = np.array(track["left_points"])
        right_coords = np.array(track["right_points"])

        # Calculate angles for both sides
        left_angles = []
        right_angles = []

        for i in range(len(left_coords) - 2):
            p1, p2, p3 = left_coords[i], left_coords[i + 1], left_coords[i + 2]
            angle = calculate_segment_angle(p1, p2, p3)
            left_angles.append(angle)

        for i in range(len(right_coords) - 2):
            p1, p2, p3 = right_coords[i], right_coords[i + 1], right_coords[i + 2]
            angle = calculate_segment_angle(p1, p2, p3)
            right_angles.append(angle)

        ax.plot(left_angles, "b-", label="Left Side", alpha=0.7)
        ax.plot(right_angles, "r-", label="Right Side", alpha=0.7)
        ax.axhline(np.pi / 2, color="red", linestyle="--", label="90° Threshold")

        ax.set_xlabel("Segment Index")
        ax.set_ylabel("Angle (radians)")
        ax.set_title("Segment Angles")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_polygon_analysis(self, ax, track, constraint_results):
        """Plot polygon analysis"""
        left_coords = np.array(track["left_points"])
        right_coords = np.array(track["right_points"])

        # Create polygon
        polygon_points = []
        polygon_points.extend(left_coords)
        polygon_points.extend(right_coords[::-1])
        polygon_points = np.array(polygon_points)

        # Plot polygon
        polygon = plt.Polygon(
            polygon_points, alpha=0.3, facecolor="lightblue", edgecolor="blue"
        )
        ax.add_patch(polygon)

        # Highlight intersection points
        poly_violations = constraint_results["polygon"]["violations"]
        for i, j in poly_violations:
            ax.scatter(
                polygon_points[i, 0],
                polygon_points[i, 1],
                color="red",
                s=50,
                marker="o",
            )
            ax.scatter(
                polygon_points[j, 0],
                polygon_points[j, 1],
                color="red",
                s=50,
                marker="o",
            )

        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title(f"Polygon Analysis\n{len(poly_violations)} intersections")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    def _plot_overall_results(self, ax, constraint_results):
        """Plot overall constraint results"""
        overall_passed = constraint_results["overall_passed"]

        # Create a simple status display
        ax.text(
            0.5,
            0.7,
            "OVERALL RESULT",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )

        status = "PASSED" if overall_passed else "FAILED"
        color = "green" if overall_passed else "red"
        ax.text(
            0.5,
            0.5,
            status,
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
            color=color,
        )

        # Add details
        details = f"""Segment Angle: {'✓' if constraint_results['seg_angle']['passed'] else '✗'}
Width: {'✓' if constraint_results['width']['passed'] else '✗'}
Polygon: {'✓' if constraint_results['polygon']['passed'] else '✗'}"""

        ax.text(0.5, 0.3, details, ha="center", va="center", fontsize=12)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Overall Status")

    def test_all_tracks_constraints(self):
        """Test constraints for all tracks and provide summary"""
        print("Comprehensive Constraint Testing")
        print("=" * 40)

        all_results = []
        for track in self.tracks:
            results = self.test_all_constraints(track["id"])
            all_results.append(results)

            status = "PASS" if results["overall_passed"] else "FAIL"
            print(f"Track {track['id']}: {status}")

        # Summary statistics
        total_tracks = len(all_results)
        passed_tracks = sum(1 for r in all_results if r["overall_passed"])

        print(
            f"\nSummary: {passed_tracks}/{total_tracks} tracks passed all constraints"
        )

        return all_results


def main():
    """Main function to demonstrate constraint testing"""
    print("Constraint Tester")
    print("-" * 20)

    tester = ConstraintTester()

    if not tester.tracks:
        print("ERROR: No tracks loaded!")
        return

    print(f"Loaded {len(tester.tracks)} tracks")

    # Test all tracks
    all_results = tester.test_all_tracks_constraints()

    # Show detailed analysis for first track
    print("\nShowing detailed analysis for Track 1...")
    tester.visualize_constraint_test(1)

    print("Done!")


if __name__ == "__main__":
    main()
