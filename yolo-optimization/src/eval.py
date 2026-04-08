import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

import dataset

def run_eval(weights_path, params_path):
  """
  Evaluation using unseen test set on trained model
  """
  print(f"Load config file from {params_path}")
  with open(params_path, 'r') as f:
    params = yaml.safe_load(f)
    
  print(f"Loading model weights from {weights_path}")
  try:
    model = YOLO(weights_path)
  except Exception as e:
    print(f"Failed to load model: {e}")
    return None

  print("Preparing Dataset")
  data_yaml = dataset.prepare()
  
  print("Constructing output path")
  project = params.get("project")
  name = f"{params.get('name')}/eval"
  
  print("Metric Evaluation on Test Split")
  print(f"Saving results to {project}/{name}")
  
  metrics = model.val(
    data=data_yaml,
    split='test',
    conf=0.001,
    verbose=True,
    plots=True,
    project=project,
    name=name
  )
  
  print(f"Test mAP50:    {metrics.box.map50:.4f}")
  print(f"Test mAP50-95: {metrics.box.map:.4f}")
  print(f"Results saved to: {metrics.save_dir}")

  print("Running visual inference check")
  with open(data_yaml, "r") as f:
    data_cfg = yaml.safe_load(f)

  dataset_root = Path(data_cfg.get("path", Path(data_yaml).parent))
  split_path = data_cfg.get("test") or data_cfg.get("val") or data_cfg.get("train")
  if not split_path:
    print("Visual inference check skipped: no dataset split found in yaml")
    return metrics

  split_path = Path(split_path)
  if not split_path.is_absolute():
    split_path = dataset_root / split_path

  max_images = int(params.get("visual_check_images", 4))
  exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
  sample_images = []

  if split_path.is_file():
    with open(split_path, "r") as f:
      for line in f:
        candidate = Path(line.strip())
        if not candidate.is_absolute():
          candidate = dataset_root / candidate
        if candidate.suffix.lower() in exts and candidate.exists():
          sample_images.append(candidate)
        if len(sample_images) >= max_images:
          break
  elif split_path.is_dir():
    for candidate in split_path.iterdir():
      if candidate.is_file() and candidate.suffix.lower() in exts:
        sample_images.append(candidate)
      if len(sample_images) >= max_images:
        break

    if not sample_images:
      for candidate in split_path.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in exts:
          sample_images.append(candidate)
        if len(sample_images) >= max_images:
          break

  if not sample_images:
    print(f"Visual inference check skipped: no images found in {split_path}")
    return metrics

  results = model.predict(
    source=[str(p) for p in sample_images],
    conf=0.25,
    save=True,
    verbose=False,
    project=project,
    name=f"{params.get('name')}/eval/visual_check",
    exist_ok=True,
  )

  save_dir = results[0].save_dir if results else "unknown"
  print(f"Visual inference images saved to: {save_dir}")

  return metrics
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="YOLO Evaluation Pipeline")
  
  parser.add_argument("--weights", type=str, required=True, help="(e.g., experiments/run/weights/best.pt)")
  parser.add_argument("--params", type=str, default="yolo-optimization/configs/hyperparams.yaml")

  args = parser.parse_args()
  
  run_eval(args.weights, args.params)
