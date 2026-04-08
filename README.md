# driverless-ml-dev
Machine Learning-related scripts and architecture for Carnegie Mellon Racing

## Setup on a new machine
Run from the repo root:

```bash
scripts/setup.sh
```

```bat
scripts\setup.bat
```

What it does:
- Creates required project folders in `ml_data/`
- Creates a local virtual environment in `.venv/`
- Installs Python dependencies from `yolo-optimization/requirements.txt`
- Supports `--skip-venv` and `--skip-install`

After setup:
1. Activate the virtual environment (`.venv\Scripts\Activate.ps1` on PowerShell, `source .venv/bin/activate` on macOS/Linux).
2. Put raw dataset files in `ml_data/fsoco_raw/`.
3. Run training/eval scripts from the repo root.
