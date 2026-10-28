#!/bin/sh
# ./run.sh [<config>.yaml ...]: the whole pipeline once per config, in order, driven only by the YAML
# (bare names: configs/, configs/experiments/; none -> configs/default.yaml). Stops at the first failure.
ROOT=$(dirname "$0")
PY="$ROOT/.venv-gpu/bin/python"; [ -x "$PY" ] || PY=python3
[ $# -gt 0 ] || set -- configs/default.yaml
for cfg in "$@"; do
    echo "run.sh: $PY main.py $cfg"
    "$PY" "$ROOT/main.py" "$cfg" || { rc=$?; echo "run.sh: FAILED on $cfg (exit $rc); remaining configs skipped" >&2; exit $rc; }
done
