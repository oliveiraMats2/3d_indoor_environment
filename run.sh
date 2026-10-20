#!/bin/sh
# ./run.sh [<config>.yaml]: the whole pipeline, driven only by the YAML (bare names: configs/, configs/experiments/)
exec python3 "$(dirname "$0")/main.py" "${1:-configs/default.yaml}"
