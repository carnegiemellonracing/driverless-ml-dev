# TODO: implement preprocessing steps
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FSOCO_YAML = ROOT / 'ml_data/fsoco_yolo/fsoco.yaml'

def prepare():
  """
  Verify config exists and return path to trainer
  """
  if not FSOCO_YAML.exists():
    raise FileNotFoundError(f"Config not found at {FSOCO_YAML}.")
  
  print(f"Using dataset config at {FSOCO_YAML}")
  return FSOCO_YAML