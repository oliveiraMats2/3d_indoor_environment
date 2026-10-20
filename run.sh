#!/bin/sh
# ./run.sh [configs/<experiment>.yaml]: the whole pipeline, driven only by the YAML (activate .venv first)
exec python3 "$(dirname "$0")/main.py" --config "${1:-configs/default.yaml}"
