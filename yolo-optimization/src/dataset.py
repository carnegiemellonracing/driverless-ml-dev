from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from shutil import copy2, rmtree
import sys

ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_DIR = ROOT / "yolo-optimization/src/preprocessing"
if str(PREPROCESSING_DIR) not in sys.path:
  sys.path.insert(0, str(PREPROCESSING_DIR))

FSOCO_RAW = ROOT / "ml_data/fsoco_raw"
FSOCO_MOD = ROOT / "ml_data/fsoco_mod"
FSOCO_EDIT = ROOT / "ml_data/fsoco_edit"
FSOCO_YOLO = ROOT / "ml_data/fsoco_yolo"
FSOCO_YAML = FSOCO_YOLO / "fsoco.yaml"


def _load_module(module_name: str, file_path: Path):
  spec = spec_from_file_location(module_name, file_path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load module from {file_path}")
  module = module_from_spec(spec)
  previous_module = sys.modules.get(module_name)
  sys.modules[module_name] = module
  try:
    spec.loader.exec_module(module)
  except Exception:
    if previous_module is None:
      sys.modules.pop(module_name, None)
    else:
      sys.modules[module_name] = previous_module
    raise
  return module


def _is_populated_dir(path: Path) -> bool:
  return path.exists() and path.is_dir() and any(path.iterdir())


def _cleanup_intermediate_copies() -> None:
  for path in (FSOCO_MOD, FSOCO_EDIT):
    if not path.exists():
      continue
    if path.is_dir():
      rmtree(path)
      print(f"Removed intermediate dataset copy: {path}")
    else:
      path.unlink()
      print(f"Removed intermediate dataset file: {path}")


def prepare(*, cleanup_intermediate_copies: bool = True):
  """
  Prepare dataset by running preprocessing in sequence:
  fsoco-to-yolo -> editing -> convert -> yaml

  Args:
    cleanup_intermediate_copies: When True, remove intermediate folders
      (fsoco_mod and fsoco_edit) after preprocessing finishes.
  """
  try:
    if _is_populated_dir(FSOCO_YOLO) and FSOCO_YAML.exists():
      yaml_step = _load_module("preprocess_yaml", PREPROCESSING_DIR / "yaml.py")
      yaml_step.normalize_existing_yaml(FSOCO_YOLO)
      print(f"Skipping preprocessing: final dataset already exists at {FSOCO_YOLO}")
      return FSOCO_YAML

    if not FSOCO_RAW.exists():
      raise FileNotFoundError(f"Raw dataset not found at {FSOCO_RAW}")

    fsoco_to_yolo = _load_module("fsoco_to_yolo", PREPROCESSING_DIR / "fsoco-to-yolo.py")
    editing = _load_module("editing", PREPROCESSING_DIR / "editing.py")
    convert = _load_module("preprocess_convert", PREPROCESSING_DIR / "convert.py")
    yaml_step = _load_module("preprocess_yaml", PREPROCESSING_DIR / "yaml.py")

    print("Step 1/4: fsoco to yolo (flatten raw dataset)")
    fsoco_to_yolo.copy_and_flatten_dataset(FSOCO_RAW, FSOCO_MOD, overwrite=False)

    print("Step 2/4: editing")
    (FSOCO_EDIT / "img").mkdir(parents=True, exist_ok=True)
    (FSOCO_EDIT / "ann").mkdir(parents=True, exist_ok=True)
    editing.split_dataset_on_x_axis(
      aspect_ratio=1.6, # aspect ratio of seecams https://www.e-consystems.com/industrial-cameras/ar0234-usb3-global-shutter-camera.asp#
      images_folder_path=FSOCO_MOD / "img",
      labels_folder_path=FSOCO_MOD / "ann",
      output_images_folder_path=FSOCO_EDIT / "img",
      output_labels_folder_path=FSOCO_EDIT / "ann",
      delete_images_without_labels=False,
      workers=4
    )
    meta_src = FSOCO_MOD / "meta.json"
    if meta_src.exists():
      copy2(meta_src, FSOCO_EDIT / "meta.json")

    print("Step 3/4: convert")
    if _is_populated_dir(FSOCO_YOLO):
      print(f"Skipping conversion: YOLO dataset already populated {FSOCO_YOLO}")
      classes = convert.load_classes_from_meta(FSOCO_EDIT)
    else:
      classes = convert.convert_supervisely_to_yolo(FSOCO_EDIT, FSOCO_YOLO)

    print("Step 4/4: yaml")
    data_yaml = yaml_step.main(
      yolo_dir=FSOCO_YOLO,
      meta=FSOCO_EDIT / "meta.json",
      classes=classes,
      overwrite=False,
    )

    print(f"Using dataset config at {data_yaml}")
    return data_yaml
  finally:
    if cleanup_intermediate_copies:
      _cleanup_intermediate_copies()


def main() -> Path:
  return prepare()


if __name__ == "__main__":
  main()
