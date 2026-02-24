import argparse
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from PIL import Image


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
    mpre = np.concatenate(([1.0], precisions, [0.0]))
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
    matched = defaultdict(set)

    for i, (conf, img_id, pred_box, gt_list) in enumerate(detections):
        best_iou, best_idx, best_ignore = 0.0, -1, False
        for gt_idx, gt_box, gt_ignored in gt_list:
            if gt_idx in matched[img_id]:
                continue
            ov = iou(pred_box, gt_box)
            if ov > best_iou:
                best_iou, best_idx, best_ignore = ov, gt_idx, gt_ignored

        if best_iou >= iou_thresh and best_idx >= 0:
            matched[img_id].add(best_idx)
            if best_ignore:
                ignore[i] = 1
            else:
                tp[i] = 1
        else:
            fp[i] = 1

    # remove ignored detections
    keep = ignore == 0
    tp, fp = tp[keep], fp[keep]

    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
    return compute_ap(cum_tp / total_gt, cum_tp / (cum_tp + cum_fp))


def main():
    parser = argparse.ArgumentParser(description="Recalculate mAP with area cutoff filter")
    parser.add_argument("--model", type=str, default="ml_data/26s_tuned_best.pt")
    parser.add_argument("--source", type=str, default="ml_data/fsoco_yolo/images/")
    parser.add_argument("--labels", type=str, default="ml_data/fsoco_yolo/labels/")
    parser.add_argument("--area-cutoff", type=float, default=0, help="Min bbox area (px²)")
    parser.add_argument("--conf-thresh", type=float, default=0.001)
    args = parser.parse_args()

    CLASS_NAMES = {0: "unknown_cone", 1: "yellow_cone", 2: "blue_cone",
                   3: "orange_cone", 4: "large_orange_cone"}

    model = YOLO(args.model)

    image_files = sorted(f for f in os.listdir(args.source)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
    print(f"Images: {len(image_files)} | Area cutoff: {args.area_cutoff} px²")

    detections_by_class = defaultdict(list)
    gt_count_by_class = defaultdict(lambda: defaultdict(int))
    total, filtered = 0, 0

    results = model.predict(source=args.source, stream=True, verbose=False, conf=args.conf_thresh)

    for i, r in enumerate(results):
        if (i+1) % max(1, len(image_files)//10) == 0:
            print(f"  {100*(i+1)//len(image_files)}%", flush=True)

        img_id = Path(r.path).stem
        img_w, img_h = Image.open(r.path).size
        gt_all = load_gt(os.path.join(args.labels, img_id + ".txt"), img_w, img_h)

        gt_by_class = defaultdict(list)
        for gt_idx, (cls, box) in enumerate(gt_all):
            gt_area = (box[2]-box[0]) * (box[3]-box[1])
            ignored = gt_area < args.area_cutoff
            gt_by_class[cls].append((gt_idx, box, ignored))
            if not ignored:
                gt_count_by_class[cls][img_id] += 1

        if r.boxes is None or len(r.boxes) == 0:
            continue

        pred_boxes = r.boxes.xyxy.cpu().numpy()
        pred_confs = r.boxes.conf.cpu().numpy()
        pred_cls = r.boxes.cls.cpu().numpy().astype(int)

        for j in range(len(pred_boxes)):
            total += 1
            box = pred_boxes[j]
            area = (box[2]-box[0]) * (box[3]-box[1])
            if area < args.area_cutoff:
                filtered += 1
                continue
            detections_by_class[pred_cls[j]].append(
                (float(pred_confs[j]), img_id, box, gt_by_class.get(pred_cls[j], []))
            )

    print(f"Preds: {total} total, {filtered} filtered, {total-filtered} kept\n")

    iou_thresholds = np.linspace(0.50, 0.95, 10)
    ap50_cls, ap50_95_cls = {}, {}

    for cls in range(len(CLASS_NAMES)):
        dets = detections_by_class.get(cls, [])
        n_gt = gt_count_by_class.get(cls, {})
        ap50_cls[cls] = ap_for_class(list(dets), n_gt, 0.50)
        ap50_95_cls[cls] = float(np.mean([ap_for_class(list(dets), n_gt, t) for t in iou_thresholds]))

    # include classes with gt
    classes_with_gt = [cls for cls in ap50_cls if sum(gt_count_by_class.get(cls, {}).values()) > 0]
    if classes_with_gt:
        map50 = float(np.mean([ap50_cls[cls] for cls in classes_with_gt]))
        map50_95 = float(np.mean([ap50_95_cls[cls] for cls in classes_with_gt]))
    else:
        map50, map50_95 = 0.0, 0.0

    print(f"\n  Classes in mAP avg: {len(classes_with_gt)}/{len(CLASS_NAMES)}")
    print(f"  mAP@50:    {map50:.4f}")
    print(f"  mAP@50-95: {map50_95:.4f}")


if __name__ == "__main__":
    main()