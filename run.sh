#!/usr/bin/env bash
# Reproduz o T2: pipeline (main.py) e depois a visualizacao (view.py), os dois guiados pelo mesmo YAML.
set -euo pipefail

uso() {
    cat <<USO
Uso: ./run.sh [CONFIG.yaml]

  CONFIG.yaml  caminho ou nome do config (padrao: config_cozinha.yaml).
               Nomes simples sao procurados na pasta atual, em configs/ e em configs/experiments/.

Executa, nesta ordem:
  1. python main.py CONFIG.yaml   pipeline completo (video -> nuvem/malha, avaliacao, videos)
  2. python view.py CONFIG.yaml   visualizador interativo Open3D do resultado

Ambiente: com "flow: gpu" no config usa .venv-gpu/bin/python (ver docs/install.md); senao
.venv/bin/python se existir, ou "uv run python". Opcoes: -h, --help.
USO
}

case "${1:-}" in
    -h|--help) uso; exit 0 ;;
esac
[ $# -le 1 ] || { uso >&2; exit 2; }

cd "$(dirname "$0")"
CONFIG="${1:-config_cozinha.yaml}"

# fluxo do config (flow: cpu | gpu); nomes simples sao procurados como em main.py
CFG_PATH=""
for c in "$CONFIG" "configs/$CONFIG" "configs/experiments/$CONFIG"; do
    [ -f "$c" ] && { CFG_PATH="$c"; break; }
done
FLOW=""
[ -n "$CFG_PATH" ] && FLOW=$(sed -n 's/^flow:[[:space:]]*\([A-Za-z]*\).*/\1/p' "$CFG_PATH" | head -n 1)

if [ "$FLOW" = "gpu" ]; then
    [ -x .venv-gpu/bin/python ] || { echo "run.sh: flow: gpu, mas .venv-gpu/bin/python nao existe (ver docs/install.md)" >&2; exit 1; }
    PY=(.venv-gpu/bin/python)
elif [ -x .venv/bin/python ]; then
    PY=(.venv/bin/python)
else
    PY=(uv run python)
fi

echo "run.sh: ${PY[*]} main.py $CONFIG"
"${PY[@]}" main.py "$CONFIG"
echo "run.sh: ${PY[*]} view.py $CONFIG"
"${PY[@]}" view.py "$CONFIG"
