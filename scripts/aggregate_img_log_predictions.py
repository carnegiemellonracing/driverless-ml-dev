#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


NAME_RE = re.compile(r"(\d+)_([lr])\.bmp$", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate prediction boxes from BMP log frames.")
    p.add_argument("--img-dir", default="ml_data/img_logs", help="Folder containing BMP files or subfolders with BMP files.")
    p.add_argument("--skip-frames", type=int, default=10, help="Ignore first N frames per camera side.")
    p.add_argument("--min-box-area", type=int, default=250, help="Minimum box area in pixels to keep.")
    p.add_argument("--out-dir", default=None, help="Output folder (default: <img-dir>/aggregate).")
    p.add_argument("--show", action="store_true", help="Show output windows.")
    return p.parse_args()


def resolve_img_dir(path: Path) -> Path:
    if list(path.glob("*.bmp")):
        return path
    for sub in sorted(path.iterdir()):
        if sub.is_dir() and list(sub.glob("*.bmp")):
            return sub
    raise FileNotFoundError(f"No BMP files found in: {path}")


def split_side_files(img_dir: Path):
    side_files = {"l": [], "r": []}
    for f in img_dir.glob("*.bmp"):
        m = NAME_RE.match(f.name)
        if not m:
            continue
        ts = int(m.group(1))
        side = m.group(2).lower()
        side_files[side].append((ts, f))
    for side in side_files:
        side_files[side].sort(key=lambda x: x[0])
        side_files[side] = [f for _, f in side_files[side]]
    return side_files


def prediction_mask(img: np.ndarray) -> np.ndarray:
    # Decode BMP pixels and keep vivid overlay graphics (no fixed class-color palette).
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 100)).astype(np.uint8) * 255
    # Drop static text/hud area at top.
    mask[:130, :] = 0
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


def extract_boxes(mask: np.ndarray, min_box_area: int):
    h, w = mask.shape[:2]
    max_area = 0.06 * h * w
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < 40:
            continue
        peri = cv2.arcLength(c, True)
        poly = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(poly) != 4 or not cv2.isContourConvex(poly):
            continue
        pts = poly.reshape(-1, 2).astype(np.int32)
        x, y, bw, bh = cv2.boundingRect(pts)
        area = bw * bh
        if bw < 6 or bh < 6 or area < min_box_area or area > max_area:
            continue
        ar = bh / (bw + 1e-6)
        if ar < 0.9 or ar > 6.0:
            continue
        fill = cv2.contourArea(poly) / (area + 1e-6)
        if fill < 0.02 or fill > 0.95:
            continue
        boxes.append((pts, (x, y, bw, bh)))
    return boxes


def make_vis(heat: np.ndarray, mean_box, base_img: np.ndarray):
    if heat.max() > 0:
        norm = np.clip((heat / heat.max()) * 255.0, 0, 255).astype(np.uint8)
    else:
        norm = np.zeros_like(heat, dtype=np.uint8)
    heat_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    vis = cv2.addWeighted(base_img, 0.65, heat_color, 0.35, 0)
    if mean_box is not None:
        x, y, w, h = [int(round(v)) for v in mean_box]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(vis, "mean box", (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def analyze_side(files, min_box_area: int):
    if not files:
        return None, None

    first = cv2.imread(str(files[0]))
    if first is None:
        return None, None
    h, w = first.shape[:2]
    heat = np.zeros((h, w), np.float32)
    img_sum = np.zeros((h, w, 3), np.float32)

    per_frame_counts = []
    all_boxes = []
    readable = 0
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        readable += 1
        img_sum += img.astype(np.float32)
        boxes = extract_boxes(prediction_mask(img), min_box_area)
        per_frame_counts.append(len(boxes))
        for pts, (x, y, bw, bh) in boxes:
            poly_mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(poly_mask, [pts], 1)
            heat += poly_mask.astype(np.float32)
            all_boxes.append((x, y, bw, bh))

    if readable == 0:
        return None, None

    mean_box = None
    if all_boxes:
        mean_box = np.array(all_boxes, np.float32).mean(axis=0).tolist()
    avg_img = np.clip(img_sum / readable, 0, 255).astype(np.uint8)
    vis = make_vis(heat / max(readable, 1), mean_box, avg_img)

    areas = np.array([bw * bh for _, _, bw, bh in all_boxes], dtype=np.float32) if all_boxes else np.array([], dtype=np.float32)
    stats = {
        "images": int(readable),
        "frames_with_predictions": int(sum(1 for c in per_frame_counts if c > 0)),
        "total_boxes": int(len(all_boxes)),
        "boxes_per_frame_mean": float(np.mean(per_frame_counts)) if per_frame_counts else 0.0,
        "boxes_per_frame_std": float(np.std(per_frame_counts)) if per_frame_counts else 0.0,
        "box_area_mean": float(np.mean(areas)) if len(areas) else 0.0,
        "box_area_std": float(np.std(areas)) if len(areas) else 0.0,
        "mean_box_xywh": [float(v) for v in mean_box] if mean_box is not None else None,
    }
    return stats, vis


def main():
    args = parse_args()
    img_dir = resolve_img_dir(Path(args.img_dir))
    out_dir = Path(args.out_dir) if args.out_dir else img_dir / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    side_files = split_side_files(img_dir)
    used = {side: files[args.skip_frames :] for side, files in side_files.items()}

    results = {
        "source_dir": str(img_dir),
        "skip_frames_per_side": int(args.skip_frames),
        "sides": {},
    }
    for side in ("l", "r"):
        stats, vis = analyze_side(used[side], args.min_box_area)
        results["sides"][side] = stats if stats is not None else {"images": 0, "total_boxes": 0}
        if vis is not None:
            cv2.imwrite(str(out_dir / f"avg_boxes_{side}.png"), vis)
            if args.show:
                cv2.imshow(f"avg_boxes_{side}", vis)

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nWrote outputs to: {out_dir}")
    if args.show:
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
