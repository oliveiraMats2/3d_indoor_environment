#!/usr/bin/env bash
# Full pipeline then the viewer for one experiment: ./run.sh [configs/<experiment>.yaml]
set -euo pipefail
cd "$(dirname "$0")"
config="${1:-configs/default.yaml}"
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi
python main.py --config "$config"
python view.py --config "$config"
