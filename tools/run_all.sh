#!/usr/bin/env bash
# Reproduce the paper's controlled experiments end to end.
# Usage: ./tools/run_all.sh [--gpu N]
#
# Requires: prepared data under ./data (see tools/prepare_data.py) and the
# CLIP ViT-B/16 weights (downloaded automatically by the transformers backend).
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="gpu"
if [[ "${1:-}" == "--gpu" ]]; then DEVICE="gpu"; fi
if [[ "${1:-}" == "--cpu" ]]; then DEVICE="cpu"; fi

export PYTHONUNBUFFERED=1

run() { # $1 = config path
    echo "=============================================="
    echo ">>> $1"
    echo "=============================================="
    python main.py benchmark --config "$1" --set device=$DEVICE
}

run configs/tcrt/office31.yaml
run configs/tcrt/office_home.yaml
run configs/tcrt/visda.yaml
run configs/tcrt/domainnet.yaml
run configs/tcrt/domainnet_inductive.yaml

echo ""
echo "All main benchmarks finished. Results under outputs/<benchmark>/results.json"
echo "Run mechanism analyses with:"
echo "  python main.py diagnose --config configs/tcrt/diagnostics.yaml --kind ranking,coverage_risk,stage,graph,probe,additivity,transport,properties"
