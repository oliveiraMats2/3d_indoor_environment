#!/usr/bin/env bash
# Ablacao do SfM na cozinha (T8, R28): roda as variantes em sequencia, um log por variante.
# Uso: bash tools/report/roda_ablacao_cozinha.sh [R]   (R = repo principal com cozinha.MOV e .venv-gpu)
# Cada variante usa main.py de R com o config de W/configs/report; main.py faz chdir para R, entao as saidas
# ficam em R/cozinha_ablacao/<variante>/. Um erro numa variante nao interrompe as demais (cache: true retoma).
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R="${1:-/home/moliveira/Documents/3d_indoor_environment}"
VARIANTES="${VARIANTES:-seq_sem_laco seq_com_laco exaustivo seq_com_laco_146}"
mkdir -p "$R/cozinha_ablacao/logs"
cd "$R" || exit 1
for v in $VARIANTES; do
  cfg="$W/configs/report/ablacao_cozinha_$v.yaml"
  [ -f "$cfg" ] || { echo "config ausente: $cfg"; continue; }
  echo "[$(date '+%F %T')] inicio $v"
  t0=$(date +%s)
  "$R/.venv-gpu/bin/python" "$R/main.py" "$cfg" > "$R/cozinha_ablacao/logs/$v.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] fim $v rc=$rc em $(( $(date +%s) - t0 )) s (log: cozinha_ablacao/logs/$v.log)"
done
echo "[$(date '+%F %T')] ablacao concluida"
