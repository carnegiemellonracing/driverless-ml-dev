"""
Lane Detection Evaluation Script

Evaluates the trained model on dataset maps using proper perceptual field contexts,
computes metrics, and generates visualizations of predicted lanes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from perceptions.lane_detection.data_loader import (
    cone_maps,
    left_boundaries,
    right_boundaries,
    generate_perceptual_field_data,
)
from perceptions.lane_detection.inference import Classifier
from perceptions.lane_detection.models import PerceptualFieldContext
from perceptions.lane_detection.ranker import IoU


def visualize_prediction(
    cone_map,
    left_gt,
    right_gt,
    ctx: PerceptualFieldContext,
    prediction,
    title="Predicted Lane",
    save_path=None,
):
    """
    Visualize full map with ground truth and predicted lane.
    """
    plt.figure(figsize=(12, 10))

    # Plot ALL cones (full map)
    plt.scatter(
        cone_map[:, 0],
        cone_map[:, 1],
        c="lightgray",
        s=20,
        alpha=0.5,
        label="All Cones",
    )

    # Highlight visible cones
    visible_pts = cone_map[list(ctx.visible_indices)]
    plt.scatter(
        visible_pts[:, 0],
        visible_pts[:, 1],
        c="gray",
        s=40,
        alpha=0.8,
        label="Visible Cones",
    )

    # Plot ground truth boundaries (full)
    if len(left_gt) > 1:
        left_gt_pts = cone_map[left_gt]
        plt.plot(
            left_gt_pts[:, 0],
            left_gt_pts[:, 1],
            "b--",
            linewidth=1,
            alpha=0.5,
            label="GT Left",
        )

    if len(right_gt) > 1:
        right_gt_pts = cone_map[right_gt]
        plt.plot(
            right_gt_pts[:, 0],
            right_gt_pts[:, 1],
            "r--",
            linewidth=1,
            alpha=0.5,
            label="GT Right",
        )

    # Plot car position
    plt.scatter(
        ctx.car_pos[0],
        ctx.car_pos[1],
        c="green",
        s=150,
        marker="*",
        label="Car",
        zorder=10,
    )

    # Draw car heading arrow
    heading_vec = np.array([np.cos(ctx.car_heading), np.sin(ctx.car_heading)])
    plt.arrow(
        ctx.car_pos[0],
        ctx.car_pos[1],
        heading_vec[0] * 3,
        heading_vec[1] * 3,
        head_width=0.5,
        color="green",
        zorder=10,
    )

    # Plot predicted boundaries
    if prediction:
        left_len = len(prediction.left_path)
        right_len = len(prediction.right_path)

        if left_len > 1:
            left_pts = cone_map[prediction.left_path]
            plt.plot(
                left_pts[:, 0],
                left_pts[:, 1],
                "b-",
                linewidth=3,
                label=f"Pred Left ({left_len} pts)",
            )
            plt.scatter(
                left_pts[:, 0], left_pts[:, 1], c="blue", s=80, marker="o", zorder=5
            )

        if right_len > 1:
            right_pts = cone_map[prediction.right_path]
            plt.plot(
                right_pts[:, 0],
                right_pts[:, 1],
                "r-",
                linewidth=3,
                label=f"Pred Right ({right_len} pts)",
            )
            plt.scatter(
                right_pts[:, 0], right_pts[:, 1], c="red", s=80, marker="o", zorder=5
            )

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def evaluate_contexts(
    model_path="perceptions/lane_detection/best_model.pth",
    output_dir="evaluation_results",
    max_contexts_per_map=5,
    perceptual_range=30.0,
    noise: bool = False,
):
    """
    Run evaluation using properly generated perceptual field contexts.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load classifier
    print(f"Loading model from {model_path}...")
    try:
        classifier = Classifier(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    results = []

    # Set samples per point based on noise flag
    samples_per_point = 5 if noise else 1
    print(
        f"\nEvaluating on {len(cone_maps)} maps (Noise: {noise}, Samples/Pt: {samples_per_point})...\n"
    )

    for map_i, cone_map in enumerate(cone_maps):
        left_gt = left_boundaries[map_i]
        right_gt = right_boundaries[map_i]

        # Generate perceptual field contexts for this map
        contexts = generate_perceptual_field_data(
            left_boundary=left_gt,
            right_boundary=right_gt,
            cone_map=cone_map,
            perceptual_range=perceptual_range,
            dmax=5.0,
            samples_per_point=samples_per_point,
        )

        print(f"Map {map_i+1}: {len(cone_map)} cones, {len(contexts)} contexts")

        # Evaluate on a subset of contexts
        for i, ctx in enumerate(contexts[:max_contexts_per_map]):
            prediction = classifier.eval_from_context(ctx)

            if prediction is None:
                print(f"  Context {i+1}: No detection")
                results.append(
                    {
                        "map_idx": map_i,
                        "ctx_idx": i,
                        "detected": False,
                    }
                )
                continue

            # Debug: Print path lengths
            left_len = len(prediction.left_path)
            right_len = len(prediction.right_path)

            # Compute IoU
            iou_score = IoU(ctx, prediction)

            print(
                f"  Context {i+1}: Left={left_len} pts, Right={right_len} pts, IoU={iou_score:.2f}"
            )

            results.append(
                {
                    "map_idx": map_i,
                    "ctx_idx": i,
                    "detected": True,
                    "iou": iou_score,
                    "left_len": left_len,
                    "right_len": right_len,
                }
            )

            # Visualize prediction with full map
            visualize_prediction(
                cone_map,
                left_gt,
                right_gt,
                ctx,
                prediction,
                title=f"Map {map_i+1} Context {i+1} (L:{left_len}, R:{right_len}, IoU:{iou_score:.2f})",
                save_path=f"{output_dir}/map{map_i+1}_ctx{i+1}.png",
            )

    # Summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)

    detected = [r for r in results if r["detected"]]
    if detected:
        avg_iou = np.mean([r.get("iou", 0) for r in detected])
        avg_left = np.mean([r.get("left_len", 0) for r in detected])
        avg_right = np.mean([r.get("right_len", 0) for r in detected])

        print(f"Contexts Evaluated: {len(results)}")
        print(
            f"Lanes Detected: {len(detected)} ({100*len(detected)/len(results):.1f}%)"
        )
        print(f"Average Left Path Length: {avg_left:.1f}")
        print(f"Average Right Path Length: {avg_right:.1f}")
        print(f"Average IoU: {avg_iou:.3f}")
    else:
        print("No lanes detected!")

    print(f"\nVisualizations saved to '{output_dir}/'")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate lane detection model")
    parser.add_argument(
        "--model",
        default="perceptions/lane_detection/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--contexts", type=int, default=3, help="Max contexts to evaluate per map"
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="Enable noise in context generation (samples_per_point=5)",
    )

    args = parser.parse_args()

    evaluate_contexts(
        model_path=args.model,
        output_dir=args.output,
        max_contexts_per_map=args.contexts,
        noise=args.noise,
    )
