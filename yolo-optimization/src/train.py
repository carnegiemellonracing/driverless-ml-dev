# load data, model, train, validate, benchmark
import os
import argparse
import yaml
from pathlib import Path
import mlflow
from ultralytics import YOLO
from ultralytics import settings

import dataset

def run_training(model_name, params_path):
  """
  Data Prep + Experiment Setup + Training
  """
  
  # Prepare dataset by pointing to fsoco_yolo/fsoco.yaml config path
  print("Preparing Dataset")
  data_yaml_path = dataset.prepare()
  
  # Load params
  print(f"Loading params from {params_path}")
  with open(params_path, 'r') as f:
    params = yaml.safe_load(f)
    
  # Set up experiment tracking to mounted directory, set tensorboard to false explicitly
  os.environ["MLFLOW_TRACKING_URI"] = "file:///root/driverless-ml-dev/ml_data/experiments/mlflow_tracking"
  os.environ["MLFLOW_EXPERIMENT_NAME"] = params.get("name")
  settings.update({
    "tensorboard": False,
    "mlflow": True
  })
  
  # Load model
  model = YOLO(model_name)
  
  print(f"Starting training for {model_name}")
  
  # Train
  results = model.train(
    data=str(data_yaml_path),
    **params
  )

  return results

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="YOLO Training Pipeline")
  
  parser.add_argument("--model", type=str, default="yolov8n.pt")
  parser.add_argument("--params", type=str, default="yolo-optimization/configs/hyperparams.yaml")
  
  args = parser.parse_args()
  
  run_training(args.model, args.params)