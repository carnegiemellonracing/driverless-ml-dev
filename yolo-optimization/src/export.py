import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "ml_data/models"

def run_export(weights_path, format_type, dynamic=False):
  """
  exports yolo model in given format
  """

  # Load model weights
  print(f"Loading model weights from {weights_path}")
  try:
    model = YOLO(weights_path)
  except Exception as e:
    print(f"Failed to load model: {e}")
    return None
  
  print(f"Exporting to format {format_type}")
  export_kwargs = {
    'format': format_type,
    'device': 0,
    'dynamic': dynamic
  }
  
  exported_path = model.export(**export_kwargs)
  
  print(f"Model exported to {exported_path}")
  
  # OPTIONAL: copy to ml_data
  
  return exported_path


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="YOLO Export Tool")
  
  parser.add_argument("--weights", type=str, required=True, help="(e.g., experiments/run/weights/best.pt)")
  parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "engine", "torchscript"])
  
  args = parser.parse_args()
  
  run_export(args.weights, args.format)