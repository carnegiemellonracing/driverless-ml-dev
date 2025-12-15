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
  
  # TODO: Implement visual inference check
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="YOLO Evaluation Pipeline")
  
  parser.add_argument("--weights", type=str, required=True, help="(e.g., experiments/run/weights/best.pt)")
  parser.add_argument("--params", type=str, default="yolo-optimization/configs/hyperparams.yaml")

  args = parser.parse_args()
  
  run_eval(args.weights, args.params)
