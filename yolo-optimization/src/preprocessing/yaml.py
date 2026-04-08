import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLO_DIR = ROOT / "ml_data/fsoco_yolo"
DEFAULT_META_PATH = ROOT / "ml_data/fsoco_mod/meta.json"


def _load_yaml_module():
    script_dir = Path(__file__).resolve().parent
    original_sys_path = sys.path[:]
    try:
        sys.path = [p for p in sys.path if Path(p).resolve() != script_dir]
        module = importlib.import_module("yaml")
    finally:
        sys.path = original_sys_path

    if not hasattr(module, "safe_dump") or not hasattr(module, "safe_load"):
        raise ImportError("PyYAML is required but could not be imported correctly.")
    return module


yaml_lib = _load_yaml_module()


def load_classes_from_meta(meta_path: Path) -> list[str] | None:
    if not meta_path.exists():
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    classes = [c["title"] for c in meta.get("classes", []) if c.get("shape") == "rectangle"]
    return classes or None


def create_or_show_yaml(
    yolo_dir: Path,
    classes: list[str],
    *,
    overwrite: bool = False,
) -> Path:
    config_path = yolo_dir / "fsoco.yaml"

    if config_path.exists() and not overwrite:
        print(f"Skipping YAML creation: config already exists: {config_path}")
        return config_path

    config = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": classes,
    }
    yolo_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml_lib.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Config saved: {config_path}")
    print(yaml_lib.dump(config, default_flow_style=False, sort_keys=False))
    return config_path


def normalize_existing_yaml(yolo_dir: Path) -> Path:
    """Normalize fsoco.yaml for cross-platform portability."""
    config_path = yolo_dir / "fsoco.yaml"
    if not config_path.exists():
        return config_path

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml_lib.safe_load(f)

    if not isinstance(config, dict):
        raise RuntimeError(f"Unexpected YAML structure in {config_path}")

    changed = False
    if config.get("path") != ".":
        config["path"] = "."
        changed = True

    defaults = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
    }
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
            changed = True

    if changed:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml_lib.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Normalized dataset YAML for portability: {config_path}")

    return config_path


def main(
    *,
    yolo_dir: Path = DEFAULT_YOLO_DIR,
    meta: Path = DEFAULT_META_PATH,
    classes: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    if classes is None:
        classes = load_classes_from_meta(meta)
    if classes is None:
        raise RuntimeError(
            "Classes not defined. Provide classes=... or ensure meta.json contains rectangle classes."
        )

    return create_or_show_yaml(yolo_dir, classes, overwrite=overwrite)


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create or display fsoco.yaml for YOLO training.")
    parser.add_argument(
        "--yolo-dir",
        type=Path,
        default=DEFAULT_YOLO_DIR,
        help=f"Path to YOLO dataset directory (default: {DEFAULT_YOLO_DIR}).",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=DEFAULT_META_PATH,
        help=f"Path to meta.json for class names (default: {DEFAULT_META_PATH}).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        help="Class names. If omitted, classes are loaded from --meta.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite fsoco.yaml if it already exists.",
    )
    args = parser.parse_args(argv)

    main(
        yolo_dir=args.yolo_dir,
        meta=args.meta,
        classes=args.classes,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
