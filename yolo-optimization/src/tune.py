import argparse
import yaml
from ray import tune
from ultralytics import YOLO
from ultralytics import settings
import mlflow
import os
from pathlib import Path

import dataset

BEST_METRIC_KEYS = (
  "metrics/mAP50-95(B)",
  "metrics/mAP50-95(M)",
  "metrics/mAP50-95(P)",
)


def build_search_space(space_config):
  search_space = {}
  for param, spec in space_config.items():
    dist_type = spec[0]
    if dist_type == "uniform":
      search_space[param] = tune.uniform(spec[1], spec[2])
    elif dist_type == "loguniform":
      search_space[param] = tune.loguniform(spec[1], spec[2])
    elif dist_type == "choice":
      search_space[param] = tune.choice(spec[1])
  return search_space


def log_best_trial_summary(result):
  if result is None:
    return

  best_result = None
  best_metric_key = None
  for metric_key in BEST_METRIC_KEYS:
    try:
      best_result = result.get_best_result(metric=metric_key, mode="max")
      best_metric_key = metric_key
      break
    except Exception:
      continue

  if best_result is None:
    print("Could not determine best Ray Tune trial from known mAP metrics.")
    return

  metrics = getattr(best_result, "metrics", {}) or {}
  trial_path = getattr(best_result, "path", None) or getattr(best_result, "log_dir", "")
  trial_dir = Path(trial_path)
  best_weights = trial_dir / "weights" / "best.pt"

  print("\n=== Best Trial Summary ===")
  print(f"{best_metric_key}: {metrics.get(best_metric_key)}")
  print(f"Trial directory: {trial_dir}")
  print(f"Best weights: {best_weights}")


def run_tuning(model_name, config_path, resume=False):
  print(f"Loading config from {config_path}")
  with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
  
  tuning_cfg = config["tuning"]
  train_args = config.get("train_args", {})
  search_space = build_search_space(config["search_space"])
  
  print("Preparing dataset")
  data_yaml = dataset.prepare()
  print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
  import torch
  print("Torch CUDA available:", torch.cuda.is_available())
  print("Torch CUDA device count:", torch.cuda.device_count())
  
  os.environ["MLFLOW_TRACKING_URI"] = "file:///root/driverless-ml-dev/ml_data/experiments/mlflow_tracking"
  os.environ["MLFLOW_EXPERIMENT_NAME"] = f"{train_args.get('name', 'ray_tune')}_{model_name}"
  settings.update({
    "tensorboard": False,
    "mlflow": True
  })
  
  print(f"Loading model {model_name}")
  model = YOLO(model_name)
  
  tune_kwargs = {
    "data": str(data_yaml),
    "space": search_space,
    "epochs": tuning_cfg["epochs"],
    "iterations": tuning_cfg["iterations"],
    "gpu_per_trial": tuning_cfg.get("gpu_per_trial", 1),
    "grace_period": tuning_cfg["grace_period"],
    "use_ray": tuning_cfg.get("use_ray", True),
    "resume": resume,
    **train_args
  }
  
  print(f"Starting tuning: {tuning_cfg['iterations']} trials, {tuning_cfg['epochs']} epochs each")
  result = model.tune(**tune_kwargs)
  if tune_kwargs.get("use_ray", True):
    log_best_trial_summary(result)
  
  return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for YOLO models.")

    parser.add_argument("--model", type=str, default="yolo26s.pt")
    parser.add_argument("--config", type=str, default="yolo-optimization/configs/tune_config.yaml")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    run_tuning(args.model, args.config, args.resume)
