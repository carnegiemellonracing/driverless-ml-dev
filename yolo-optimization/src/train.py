# load data, model, train, validate, benchmark
import argparse
import yaml
from pathlib import Path
import mlflow
from ultralytics import YOLO

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
    
  # Set up experiment tracking to mounted directory
  mlflow.set_tracking_uri("file:///root/driverless-ml-dev/ml_data/experiments/mlruns")
  exp_name = params.get("name")
  mlflow.set_experiment(exp_name)
  print(f"MLflow Experiment set to {exp_name}")
  
  # Load model
  model = YOLO(model_name)
  
  print(f"Starting training for {model_name}")
  
  # Train
  results = model.train(
    data=str(data_yaml_path),
    **params # May warn about extra arguments, may recognize unwanted arguments
  )
  
  # Final metrics from validation
  metrics = model.val()
  
  print(f"Final metrics: {metrics}")
  return results

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="YOLO Training Pipeline")
  
  parser.add_argument("--model", type=str, default="yolov8n.pt")
  parser.add_argument("--params", type=str, default="yolo-optimization/configs/hyperparams.yaml")
  
  args = parser.parse_args()
  
  run_training(args.model, args.params)