#!/usr/bin/env bash
set -euo pipefail

config_path="${1:?resolved training config is required}"
activate_path="${2:?environment activation script is required}"
project_root="${3:?NFs project root is required}"

if [[ ! -f "${config_path}" ]]; then
  echo "Resolved config is missing: ${config_path}" >&2
  exit 2
fi
if [[ ! -f "${activate_path}" ]]; then
  echo "Environment activation script is missing: ${activate_path}" >&2
  exit 2
fi
if [[ ! -d "${project_root}/src/flashsim_nf" ]]; then
  echo "NFs source tree is missing: ${project_root}" >&2
  exit 2
fi

cd "${project_root}"
source "${activate_path}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

echo "=================================================="
echo "NFs Model 4 training"
echo "Host       : $(hostname)"
echo "Started    : $(date --iso-8601=seconds 2>/dev/null || date)"
echo "Python     : $(python3 --version 2>&1)"
echo "Config     : ${config_path}"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unavailable)"
echo "=================================================="

python3 -c "import flashsim_nf, nflows, torch, uproot"
python3 -u scripts/preflight_model4_training.py --config "${config_path}"
python3 -u scripts/train_model4.py --config "${config_path}"
