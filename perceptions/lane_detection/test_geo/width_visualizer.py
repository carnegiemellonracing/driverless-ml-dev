import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geo import bt_decider, constraint_decider, find_matching_points


class TrackWidthVisualizer:
    """
    A comprehensive tool for visualizing and analyzing track width constraints.
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

    def calculate_widths_for_track(self, track_id, wmin=2.5, wmax=6.5):
        """
        Calculate widths for all sections of a track using the matching algorithm.
        Returns detailed width analysis including violations.
        """
        track = self.tracks[track_id - 1]

        # Convert points to the format expected by find_matching_points
        all_points = track["left_points"] + track["right_points"]
        left_indices = list(range(len(track["left_points"])))
        right_indices = list(
            range(
                len(track["left_points"]),
                len(track["left_points"]) + len(track["right_points"]),
            )
        )

        # Calculate matching lines and widths
        matching_lines = find_matching_points(left_indices, right_indices, all_points)

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

    def visualize_track_widths(self, track_id, figsize=(12, 8)):
        """
        Create comprehensive visualization of track widths including:
        1. Track layout with width lines
        2. Width distribution histogram
        3. Width vs position plot
        4. Violation analysis
        """
        width_analysis = self.calculate_widths_for_track(track_id)
        track = self.tracks[track_id - 1]

        fig = plt.figure(figsize=figsize)

        # 1. Track layout with width visualization (top-left)
        ax1 = plt.subplot(2, 2, 1)
        self._plot_track_layout(ax1, track, width_analysis)

        # 2. Width distribution histogram (top-right)
        ax2 = plt.subplot(2, 2, 2)
        self._plot_width_distribution(ax2, width_analysis)

        # 3. Width vs position plot (bottom-left)
        ax3 = plt.subplot(2, 2, 3)
        self._plot_width_vs_position(ax3, width_analysis)

        # 4. Violation analysis (bottom-right)
        ax4 = plt.subplot(2, 2, 4)
        self._plot_violation_analysis(ax4, width_analysis)

        plt.suptitle(
            f"Track {track_id} - Comprehensive Width Analysis",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

        return width_analysis

    def _plot_track_layout(self, ax, track, width_analysis):
        """Plot track layout with color-coded width lines"""
        # Plot left and right boundaries
        left_points = np.array(track["left_points"])
        right_points = np.array(track["right_points"])

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

        # Plot width lines with color coding
        widths = width_analysis["widths"]
        wmin, wmax = 2.5, 6.5

        for i, match in enumerate(width_analysis["matching_lines"]):
            left_pt = match["left_point"]
            right_pt = match["right_point"]
            width = match["width"]

            # Color coding: green (good), yellow (borderline), red (violation)
            if 2.5 <= width <= 6.5:
                color = "green"
                alpha = 0.8
            elif 2.0 <= width <= 7.0:
                color = "orange"
                alpha = 0.6
            else:
                color = "red"
                alpha = 0.9

            ax.plot(
                [left_pt[0], right_pt[0]],
                [left_pt[1], right_pt[1]],
                color=color,
                alpha=alpha,
                linewidth=2,
            )

            # Add width text for extreme cases
            if width < 2.0 or width > 7.0:
                mid_x = (left_pt[0] + right_pt[0]) / 2
                mid_y = (left_pt[1] + right_pt[1]) / 2
                ax.text(
                    mid_x,
                    mid_y,
                    f"{width:.1f}m",
                    fontsize=8,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )

        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_title("Track Layout with Width Lines")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

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

    def _plot_width_vs_position(self, ax, width_analysis):
        """Plot width vs position along track"""
        widths = width_analysis["widths"]
        positions = range(len(widths))

        ax.plot(positions, widths, "b-", linewidth=2, alpha=0.7, label="Track Width")

        # Add constraint bands
        wmin, wmax = 2.5, 6.5
        ax.fill_between(
            positions, wmin, wmax, alpha=0.2, color="green", label="Valid Range"
        )
        ax.axhline(wmin, color="red", linestyle="--", alpha=0.7)
        ax.axhline(wmax, color="red", linestyle="--", alpha=0.7)

        # Highlight violations
        violations = width_analysis["violations"]
        for violation in violations:
            ax.scatter(
                violation["index"],
                violation["width"],
                color="red",
                s=100,
                zorder=5,
                alpha=0.8,
            )

        ax.set_xlabel("Position Along Track")
        ax.set_ylabel("Width (meters)")
        ax.set_title("Width vs Position")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_violation_analysis(self, ax, width_analysis):
        """Plot violation analysis pie chart and stats"""
        violations = width_analysis["violations"]
        stats = width_analysis["stats"]

        # Count violation types
        too_narrow = len([v for v in violations if v["type"] == "too_narrow"])
        too_wide = len([v for v in violations if v["type"] == "too_wide"])
        valid = len(width_analysis["widths"]) - len(violations)

        # Create pie chart
        sizes = [valid, too_narrow, too_wide]
        labels = [
            f"Valid ({valid})",
            f"Too Narrow ({too_narrow})",
            f"Too Wide ({too_wide})",
        ]
        colors = ["green", "orange", "red"]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
        )

        # Add text box with statistics
        stats_text = f"""Statistics:
Mean: {stats['mean']:.2f}m
Std: {stats['std']:.2f}m
Min: {stats['min']:.2f}m
Max: {stats['max']:.2f}m
Violation Rate: {stats['violation_rate']:.1%}"""

        ax.text(
            1.3,
            0.5,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )

        ax.set_title("Violation Analysis")

    def visualize_all_tracks(self, figsize=(12, 8)):
        """Visualize all tracks in one frame for comparison"""
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.flatten()

        for i, track in enumerate(self.tracks):
            if i >= 9:  # Only show first 9 tracks
                break

            ax = axes[i]
            width_analysis = self.calculate_widths_for_track(track["id"])

            # Simple track layout with width lines
            self._plot_track_layout(ax, track, width_analysis)
            ax.set_title(
                f'Track {track["id"]} - {width_analysis["stats"]["violation_rate"]:.1%} violations'
            )

        # Hide unused subplots
        for i in range(len(self.tracks), 9):
            axes[i].set_visible(False)

        plt.suptitle(
            "All Tracks - Width Analysis Overview", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()

    def generate_width_report(self, track_id):
        """Generate detailed text report for a track"""
        width_analysis = self.calculate_widths_for_track(track_id)
        stats = width_analysis["stats"]
        violations = width_analysis["violations"]

        print(f"\n{'='*60}")
        print(f"WIDTH ANALYSIS REPORT - TRACK {track_id}")
        print(f"{'='*60}")

        print(f"\nBASIC STATISTICS:")
        print(f"  Mean Width: {stats['mean']:.3f} meters")
        print(f"  Standard Deviation: {stats['std']:.3f} meters")
        print(f"  Min Width: {stats['min']:.3f} meters")
        print(f"  Max Width: {stats['max']:.3f} meters")
        print(f"  Median Width: {stats['median']:.3f} meters")

        print(f"\nCONSTRAINT VIOLATIONS:")
        print(f"  Total Sections: {len(width_analysis['widths'])}")
        print(f"  Violations: {stats['violation_count']}")
        print(f"  Violation Rate: {stats['violation_rate']:.1%}")

        if violations:
            print(f"\nVIOLATION DETAILS:")
            for violation in violations:
                print(
                    f"  Section {violation['index']}: {violation['type'].replace('_', ' ').title()}"
                )
                print(
                    f"    Width: {violation['width']:.3f}m (threshold: {violation['threshold']:.1f}m)"
                )
                print(f"    Severity: {violation['severity']:.1%}")

        print(f"\n{'='*60}")

        return width_analysis


def main():
    """Main function to demonstrate the width visualizer"""
    print("Track Width Visualizer")
    print("-" * 30)

    visualizer = TrackWidthVisualizer()

    if not visualizer.tracks:
        print("ERROR: No tracks loaded!")
        return

    print(f"Loaded {len(visualizer.tracks)} tracks")

    # Show all tracks overview
    print("Showing all tracks overview...")
    visualizer.visualize_all_tracks()

    # Show detailed analysis of track 1
    print("Showing detailed analysis of Track 1...")
    visualizer.visualize_track_widths(1)

    print("Done!")


if __name__ == "__main__":
    main()
