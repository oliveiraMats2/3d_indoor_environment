#!/bin/sh
# ./run.sh [--view] [<config>.yaml ...]: the whole pipeline once per config, in order, driven only by the YAML
# (bare names: configs/, configs/experiments/; none -> kitchen_my_home_cpu.yaml, the T2 CPU flow: COLMAP SIFT +
# incremental mapper). The GPU flow (GLOMAP) is run directly: python3 main.py <gpu config>.yaml.
# Stops at the first failure. --view: then opens the interactive viewer (view.py) on the last config; main.py
# already writes the navigation videos and figures without it.
ROOT=$(dirname "$0")
PY="$ROOT/.venv-gpu/bin/python"; [ -x "$PY" ] || PY=python3
VIEW=0; [ "$1" = "--view" ] && { VIEW=1; shift; }
[ $# -gt 0 ] || set -- kitchen_my_home_cpu.yaml
for cfg in "$@"; do
    echo "run.sh: $PY main.py $cfg"
    "$PY" "$ROOT/main.py" "$cfg" || { rc=$?; echo "run.sh: FAILED on $cfg (exit $rc); remaining configs skipped" >&2; exit $rc; }
done
[ $VIEW -eq 1 ] && { echo "run.sh: $PY view.py $cfg"; "$PY" "$ROOT/view.py" "$cfg"; }
exit 0
