#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
REQ_FILE="${REPO_ROOT}/yolo-optimization/requirements.txt"

SKIP_VENV=0
SKIP_INSTALL=0
for arg in "$@"; do
  case "${arg}" in
    --skip-venv) SKIP_VENV=1 ;;
    --skip-install) SKIP_INSTALL=1 ;;
  esac
done

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "ERROR: requirements file not found at ${REQ_FILE}" >&2
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "ERROR: Python was not found. Install Python 3.10+ and try again." >&2
  exit 1
fi

"${PYTHON_CMD}" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1 || {
  echo "ERROR: Python 3.10+ is required." >&2
  exit 1
}

mkdir -p \
  "${REPO_ROOT}/ml_data/fsoco_raw" \
  "${REPO_ROOT}/ml_data/fsoco_mod" \
  "${REPO_ROOT}/ml_data/fsoco_edit/img" \
  "${REPO_ROOT}/ml_data/fsoco_edit/ann" \
  "${REPO_ROOT}/ml_data/fsoco_yolo" \
  "${REPO_ROOT}/ml_data/models" \
  "${REPO_ROOT}/ml_data/experiments/mlflow_tracking"
echo "Created/verified project folders in ml_data/."

INSTALL_PY="${PYTHON_CMD}"
if [[ "${SKIP_VENV}" -eq 0 ]]; then
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "+ ${PYTHON_CMD} -m venv ${VENV_DIR}"
    "${PYTHON_CMD}" -m venv "${VENV_DIR}"
    echo "Created virtual environment at ${VENV_DIR}"
  else
    echo "Using existing virtual environment at ${VENV_DIR}"
  fi
  INSTALL_PY="${VENV_DIR}/bin/python"
fi

if [[ "${SKIP_INSTALL}" -eq 0 ]]; then
  echo "+ ${INSTALL_PY} -m pip install --upgrade pip setuptools wheel"
  "${INSTALL_PY}" -m pip install --upgrade pip setuptools wheel
  echo "+ ${INSTALL_PY} -m pip install -r ${REQ_FILE}"
  "${INSTALL_PY}" -m pip install -r "${REQ_FILE}"
  echo "Dependencies installed."
else
  echo "Skipped dependency install."
fi

echo
echo "Setup complete."
if [[ "${SKIP_VENV}" -eq 0 ]]; then
  echo "Activate with: source .venv/bin/activate"
fi
echo "Place raw dataset files in: ml_data/fsoco_raw/"
echo "Then run scripts from repository root."
