#!/bin/bash
python -m pip install papermill >/dev/null 2>&1 || pip install papermill >/dev/null 2>&1

SUFFIXES=("8n" "8s" "10n" "10s" "11n" "11s")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer a Python 3.12.3 kernel if available; create one if the interpreter exists
TARGET_PY_VER="3.12.3"
PY_BIN="${PYTHON_BIN:-python3}"
CUR_VER="$($PY_BIN -c 'import platform; print(platform.python_version())' 2>/dev/null || echo "")"
if [ "$CUR_VER" != "$TARGET_PY_VER" ]; then
    for CAND in python3.12 python3.12.3 /usr/local/bin/python3.12 /usr/bin/python3.12; do
        if command -v "$CAND" >/dev/null 2>&1; then
            PY_BIN="$CAND"
            break
        fi
    done
fi

CUR_VER="$($PY_BIN -c 'import platform; print(platform.python_version())' 2>/dev/null || echo "")"
if [ "$CUR_VER" = "$TARGET_PY_VER" ]; then
    if ! jupyter kernelspec list 2>/dev/null | grep -q "py3123"; then
        "$PY_BIN" -m pip install -q ipykernel >/dev/null 2>&1 || true
        "$PY_BIN" -m ipykernel install --user --name py3123 --display-name "Python 3.12.3" >/dev/null 2>&1 || true
    fi
    : "${KERNEL_NAME:=py3123}"
fi

# Select an available kernel, prefer requested name (possibly set above), else fallback to python3
if [ -z "${KERNEL_NAME:-}" ]; then
    KERNEL_NAME="driverless-ml"
fi

if jupyter kernelspec list 2>/dev/null | grep -q "${KERNEL_NAME}"; then
    KERNEL="${KERNEL_NAME}"
elif jupyter kernelspec list 2>/dev/null | grep -q "python3"; then
    KERNEL="python3"
else
    python -m pip install -q ipykernel >/dev/null 2>&1 || pip install -q ipykernel >/dev/null 2>&1
    python -m ipykernel install --user --name python3 >/dev/null 2>&1 || true
    KERNEL="python3"
fi

FAILED=()
SUCCEEDED=()

echo "Starting notebook batch execution..."

for SUFFIX in "${SUFFIXES[@]}"; do
    NOTEBOOK="ultralytics_cli_${SUFFIX}.ipynb"
    OUTPUT="ultralytics_cli_${SUFFIX}_output.ipynb"
    
    echo ""
    echo "Running: $NOTEBOOK"
    
    if [ -f "$NOTEBOOK" ]; then
        if papermill "$NOTEBOOK" "$OUTPUT" --kernel "$KERNEL"; then
            echo "✓ Completed: $NOTEBOOK"
            SUCCEEDED+=("$NOTEBOOK")
        else
            echo "✗ Failed: $NOTEBOOK"
            FAILED+=("$NOTEBOOK")
        fi
    else
        echo "✗ Not found: $NOTEBOOK"
        FAILED+=("$NOTEBOOK")
    fi
done

echo ""
echo "========================================"
echo "Execution Summary:"
echo "Total: ${#SUFFIXES[@]}"
echo "Succeeded: ${#SUCCEEDED[@]}"
echo "Failed: ${#FAILED[@]}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed notebooks:"
    for nb in "${FAILED[@]}"; do
        echo "  - $nb"
    done
    exit 1
fi

echo "========================================"
echo "All notebooks processed successfully!"