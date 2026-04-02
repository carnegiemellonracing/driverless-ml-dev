import json
import shutil
import random
import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

SPLIT_NAMES = ("train", "val", "test")


def convert_supervisely_to_yolo(fsoco_raw, fsoco_yolo, split=(0.8, 0.1, 0.1)):
    for s in SPLIT_NAMES:
        (fsoco_yolo / 'images' / s).mkdir(parents=True, exist_ok=True)
        (fsoco_yolo / 'labels' / s).mkdir(parents=True, exist_ok=True)
    
    # hard code class names
    classes = ['blue_cone', 'unknown_cone', 'orange_cone', 'large_orange_cone', 'yellow_cone']
    class_map = {name: i for i, name in enumerate(classes)}
    print(f"Classes (bounding boxes only): {classes}\n")
    print(f"Class mapping: {class_map}\n")

    ann_dir = fsoco_raw / 'ann'
    img_dir = fsoco_raw / 'img'
    
    ann_files = list(ann_dir.glob('*.json'))
    print(f"Total annotations: {len(ann_files)}")
    
    random.shuffle(ann_files)
    n1 = int(len(ann_files) * split[0])
    n2 = int(len(ann_files) * (split[0] + split[1]))
    splits = {
        'train': ann_files[:n1],
        'val': ann_files[n1:n2],
        'test': ann_files[n2:]
    }
    
    for split_name, anns in splits.items():
        print(f"\nConverting {split_name}: {len(anns)} images")
        
        for ann_path in tqdm(anns):
            img_name = ann_path.stem
            img_path = img_dir / img_name
            
            if not img_path.exists():
                alt_name = img_name.replace('.jpg', '.png') if '.jpg' in img_name else img_name.replace('.png', '.jpg')
                img_path = img_dir / alt_name
                if not img_path.exists():
                    continue

            with Image.open(img_path) as img:
                w, h = img.size

            with open(ann_path, "r", encoding="utf-8") as f:
                ann = json.load(f)

            yolo_labels = []
            for obj in ann.get('objects', []):
                cls = obj['classTitle']
                if cls not in class_map or obj['geometryType'] != 'rectangle':
                    continue
                
                points = obj['points']['exterior']
                x1, y1 = points[0]
                x2, y2 = points[1]

                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = abs(x2 - x1) / w
                height = abs(y2 - y1) / h

                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                yolo_labels.append(f"{class_map[cls]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            if yolo_labels:
                shutil.copy(img_path, fsoco_yolo / 'images' / split_name / img_path.name)
                with open(fsoco_yolo / 'labels' / split_name / f"{img_path.stem}.txt", 'w') as f:
                    f.write('\n'.join(yolo_labels))
    
    print("\nConversion complete")
    print(f"\nDataset splits:")
    for split_name in SPLIT_NAMES:
        n_imgs = len(list((fsoco_yolo / 'images' / split_name).glob('*.[jp][pn]g')))
        print(f"  {split_name}: {n_imgs} images")
    
    return classes

def load_classes_from_meta(fsoco_mod: Path):
    meta_path = fsoco_mod / "meta.json"
    if not meta_path.exists():
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        classes = [c["title"] for c in meta["classes"] if c.get("shape") == "rectangle"]
        print(f"Loaded {len(classes)} classes from meta.json: {classes}")
        return classes
    except Exception as e:
        print(f"Warning: Failed to load classes from meta.json: {e}")
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert Supervisely-style FSOCO data to YOLO format.")
    parser.add_argument("source", type=Path, help="Path to working dataset directory (must contain ann/ and img/).")
    parser.add_argument("dest", type=Path, help="Path to output YOLO dataset directory.")
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.8, 0.1, 0.1),
        help="Train/val/test split ratios (default: 0.8 0.1 0.1).",
    )
    args = parser.parse_args(argv)

    fsoco_mod = Path(args.source)
    fsoco_yolo = Path(args.dest)

    if not (fsoco_mod.exists() and fsoco_mod.is_dir() and any(fsoco_mod.iterdir())):
        raise FileNotFoundError(f"Working dataset not found or empty: {fsoco_mod}")

    if abs(sum(args.split) - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {args.split} (sum={sum(args.split):.6f})")

    classes = load_classes_from_meta(fsoco_mod)
    yolo_exists = fsoco_yolo.exists() and any(fsoco_yolo.iterdir())

    if yolo_exists:
        print(f"Skipping conversion: YOLO dataset already populated {fsoco_yolo}")
    else:
        print(f"Using working dataset: {fsoco_mod}")
        classes = convert_supervisely_to_yolo(fsoco_mod, fsoco_yolo, split=tuple(args.split))

    if classes is None:
        raise RuntimeError(
            "Failed to load class definitions.\n"
            f"Ensure meta.json exists in {fsoco_mod} or run the conversion."
        )

    print(f"\nFinal classes list ({len(classes)} classes): {classes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
