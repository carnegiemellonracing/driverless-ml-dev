import argparse
import csv
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from PIL import Image

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def xywh_to_xyxy(cx, cy, w, h):
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_gt(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = [float(x) for x in parts[1:5]]
            boxes.append((cls, xywh_to_xyxy(cx*img_w, cy*img_h, w*img_w, h*img_h)))
    return boxes


def compute_ap(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]))


def ap_for_class(detections, n_gt, iou_thresh):
    detections.sort(key=lambda x: x[0], reverse=True)
    total_gt = sum(n_gt.values())
    if total_gt == 0:
        return 0.0

    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    ignore = np.zeros(len(detections))

    matched_valid = defaultdict(set)

    for i, (conf, img_id, pred_box, gt_list) in enumerate(detections):
        best_valid_iou, best_valid_idx = 0.0, -1
        best_ignore_iou = 0.0

        for gt_idx, gt_box, gt_ignored in gt_list:
            ov = iou(pred_box, gt_box)

            if gt_ignored:
                if ov > best_ignore_iou:
                    best_ignore_iou = ov
            else:
                if gt_idx in matched_valid[img_id]:
                    continue
                if ov > best_valid_iou:
                    best_valid_iou, best_valid_idx = ov, gt_idx

        if best_valid_iou >= iou_thresh and best_valid_idx >= 0:
            matched_valid[img_id].add(best_valid_idx)
            tp[i] = 1
        elif best_ignore_iou >= iou_thresh:
            ignore[i] = 1
        else:
            fp[i] = 1

    keep = ignore == 0
    tp, fp = tp[keep], fp[keep]

    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
    recalls = cum_tp / total_gt
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    return compute_ap(recalls, precisions)


def collect_inference_data(model, source, labels, conf_thresh):
    image_files = sorted(f for f in os.listdir(source)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
    print(f"Images: {len(image_files)}")

    image_data = []
    results = model.predict(source=source, stream=True, verbose=False, conf=conf_thresh)

    for i, r in enumerate(results):
        if (i+1) % max(1, len(image_files)//10) == 0:
            print(f"  {100*(i+1)//len(image_files)}%", flush=True)

        img_id = Path(r.path).stem
        img_w, img_h = Image.open(r.path).size
        gt_all = load_gt(os.path.join(labels, img_id + ".txt"), img_w, img_h)

        preds = []
        if r.boxes is not None and len(r.boxes) > 0:
            pred_boxes = r.boxes.xyxy.cpu().numpy()
            pred_confs = r.boxes.conf.cpu().numpy()
            pred_cls = r.boxes.cls.cpu().numpy().astype(int)
            for j in range(len(pred_boxes)):
                preds.append((int(pred_cls[j]), float(pred_confs[j]), pred_boxes[j]))

        image_data.append({"img_id": img_id, "gt_all": gt_all, "preds": preds})

    return image_data


def evaluate_at_cutoff(cutoff, image_data, class_names, exclude_set):
    detections_by_class = defaultdict(list)
    gt_count_by_class = defaultdict(lambda: defaultdict(int))
    total, filtered = 0, 0

    for img in image_data:
        img_id = img["img_id"]

        gt_by_class = defaultdict(list)
        for gt_idx, (cls, box) in enumerate(img["gt_all"]):
            if cls in exclude_set:
                continue
            gt_area = (box[2]-box[0]) * (box[3]-box[1])
            ignored = gt_area < cutoff
            gt_by_class[cls].append((gt_idx, box, ignored))
            if not ignored:
                gt_count_by_class[cls][img_id] += 1

        for cls, conf, box in img["preds"]:
            if cls in exclude_set:
                continue
            total += 1
            area = (box[2]-box[0]) * (box[3]-box[1])
            if area < cutoff:
                filtered += 1
                continue
            detections_by_class[cls].append(
                (conf, img_id, box, gt_by_class.get(cls, []))
            )

    iou_thresholds = np.linspace(0.50, 0.95, 10)
    ap50_cls, ap50_95_cls = {}, {}

    for cls in range(len(class_names)):
        dets = detections_by_class.get(cls, [])
        n_gt = gt_count_by_class.get(cls, {})
        ap50_cls[cls] = ap_for_class(list(dets), n_gt, 0.50)
        ap50_95_cls[cls] = float(np.mean(
            [ap_for_class(list(dets), n_gt, t) for t in iou_thresholds]
        ))

    classes_with_gt = [c for c in ap50_cls if sum(gt_count_by_class.get(c, {}).values()) > 0]
    if classes_with_gt:
        map50 = float(np.mean([ap50_cls[c] for c in classes_with_gt]))
        map50_95 = float(np.mean([ap50_95_cls[c] for c in classes_with_gt]))
    else:
        map50, map50_95 = 0.0, 0.0

    return {
        "cutoff": cutoff, "map50": map50, "map50_95": map50_95,
        "total": total, "filtered": filtered, "kept": total - filtered,
        "classes_used": len(classes_with_gt),
    }


def print_summary_table(rows):
    header = f"{'cutoff':>8}  {'mAP@50':>8}  {'mAP@50-95':>10}  {'total':>6}  {'filtered':>8}  {'kept':>6}  {'classes':>7}"
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in rows:
        print(f"{r['cutoff']:>8.0f}  {r['map50']:>8.4f}  {r['map50_95']:>10.4f}"
              f"  {r['total']:>6}  {r['filtered']:>8}  {r['kept']:>6}  {r['classes_used']:>7}")
    print(sep)


def save_csv(rows, path):
    fieldnames = ["cutoff", "map50", "map50_95", "total", "filtered", "kept", "classes_used"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved → {path}")


def save_plots(rows, prefix):
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plots.")
        return

    cutoffs = [r["cutoff"] for r in rows]
    plots = [
        ("map50",    [r["map50"] for r in rows],    "mAP@50"),
        ("map50_95", [r["map50_95"] for r in rows],  "mAP@50-95"),
    ]

    for suffix, yvals, ylabel in plots:
        fig, ax = plt.subplots()
        ax.plot(cutoffs, yvals, marker="o")
        ax.set_xlabel("Area Cutoff (px²)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Area Cutoff")
        ax.grid(True, alpha=0.3)
        path = f"{prefix}_{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="Recalculate mAP with area cutoff sweep")
    parser.add_argument("--model",           type=str,   default="ml_data/26s_tuned_best.pt")
    parser.add_argument("--source",          type=str,   default="ml_data/fsoco_yolo/images/val")
    parser.add_argument("--labels",          type=str,   default="ml_data/fsoco_yolo/labels/val")
    parser.add_argument("--conf-thresh",     type=float, default=0.001)
    parser.add_argument("--exclude-classes", type=int,   nargs="*", default=[])
    # edit the default list below to change which cutoffs are swept
    parser.add_argument("--area-cutoffs",    type=float, nargs="+", default=[0])
    parser.add_argument("--plot-prefix",     type=str,   default="area_cutoff_sweep")
    args = parser.parse_args()

    exclude_set = set(args.exclude_classes)
    CLASS_NAMES = {0: "unknown_cone", 1: "yellow_cone", 2: "blue_cone",
                   3: "orange_cone", 4: "large_orange_cone"}

    model = YOLO(args.model)
    image_data = collect_inference_data(model, args.source, args.labels, args.conf_thresh)

    cutoffs = sorted(args.area_cutoffs)
    print(f"\nSweeping {len(cutoffs)} cutoff(s): {cutoffs} | Excluded classes: {args.exclude_classes}")

    results = [evaluate_at_cutoff(c, image_data, CLASS_NAMES, exclude_set) for c in cutoffs]

    print_summary_table(results)
    save_csv(results, f"{args.plot_prefix}.csv")
    save_plots(results, args.plot_prefix)


if __name__ == "__main__":
    main()