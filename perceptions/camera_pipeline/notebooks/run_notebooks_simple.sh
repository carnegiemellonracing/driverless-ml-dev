#!/bin/bash

SUFFIXES=("8n" "8s" "10n" "10s" "11n" "11s")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FAILED=()
SUCCEEDED=()

echo "Starting notebook batch execution..."

for SUFFIX in "${SUFFIXES[@]}"; do
    NOTEBOOK="ultralytics_cli_${SUFFIX}.ipynb"
    OUTPUT="ultralytics_cli_${SUFFIX}_output.ipynb"
    
    echo ""
    echo "Running: $NOTEBOOK"
    
    if [ -f "$NOTEBOOK" ]; then
        if papermill "$NOTEBOOK" "$OUTPUT" --kernel driverless-ml; then
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