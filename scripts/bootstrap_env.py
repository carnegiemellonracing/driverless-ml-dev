#!/usr/bin/env python3
"""Bootstrap this repository on a fresh machine."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = PROJECT_ROOT / "yolo-optimization" / "requirements.txt"
DEFAULT_VENV_DIR = PROJECT_ROOT / ".venv"

REQUIRED_DIRS = [
    PROJECT_ROOT / "ml_data",
    PROJECT_ROOT / "ml_data" / "models",
    PROJECT_ROOT / "ml_data" / "experiments",
]


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def ensure_python_version(python_cmd: str) -> None:
    check_cmd = [
        python_cmd,
        "-c",
        "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)",
    ]
    try:
        subprocess.run(check_cmd, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"Python executable not found: {python_cmd}") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit("Python 3.10+ is required for this project.") from exc


def resolve_venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def create_folders() -> None:
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
    print("Created/verified project folders in ml_data/.")


def install_dependencies(python_exe: Path, requirements_path: Path) -> None:
    run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([str(python_exe), "-m", "pip", "install", "-r", str(requirements_path)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap local project environment.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to create the virtual environment (default: current Python).",
    )
    parser.add_argument(
        "--requirements",
        default=str(DEFAULT_REQUIREMENTS),
        help="Path to requirements.txt file.",
    )
    parser.add_argument(
        "--venv",
        default=str(DEFAULT_VENV_DIR),
        help="Path to virtual environment directory.",
    )
    parser.add_argument(
        "--skip-venv",
        action="store_true",
        help="Install dependencies into --python directly without creating a virtual environment.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Only create folders/venv; do not install dependencies.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requirements = Path(args.requirements).resolve()
    venv_dir = Path(args.venv).resolve()

    if not requirements.exists():
        raise SystemExit(f"requirements file not found: {requirements}")

    ensure_python_version(args.python)
    create_folders()

    install_python = Path(args.python)
    if not args.skip_venv:
        if not venv_dir.exists():
            run([args.python, "-m", "venv", str(venv_dir)])
            print(f"Created virtual environment at {venv_dir}")
        else:
            print(f"Using existing virtual environment at {venv_dir}")
        install_python = resolve_venv_python(venv_dir)

    if not args.skip_install:
        install_dependencies(install_python, requirements)
        print("Dependencies installed.")
    else:
        print("Skipped dependency install.")

    print("\nSetup complete.")
    if not args.skip_venv:
        if os.name == "nt":
            print(r"Activate with: .\.venv\Scripts\Activate.ps1")
        else:
            print("Activate with: source .venv/bin/activate")
    print("Place raw dataset files in: ml_data/fsoco_raw/")
    print("Then run scripts from repository root.")


if __name__ == "__main__":
    main()
