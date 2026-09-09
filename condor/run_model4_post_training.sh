#!/usr/bin/env bash
set -euo pipefail

config_path="${1:?resolved post-training config is required}"
activate_path="${2:?environment activation script is required}"
project_root="${3:?NFs project root is required}"
output_root="${4:?NFs output root is required}"
log_subdir="${5:?log subdirectory is required}"
run_name="${6:?run name is required}"

live_directory="${output_root}/condor_logs/live/${log_subdir}"
timestamp="$(date +%Y%m%d_%H%M%S)"
host_label="$(hostname | tr -c 'A-Za-z0-9_.-' '_')"
mkdir -p "${live_directory}"
live_log="${live_directory}/model4_post_training_${run_name}_${timestamp}_${host_label}_pid$$.live.log"
started_epoch="$(date +%s)"

finish_live_log() {
  status=$?
  trap - EXIT
  finished_epoch="$(date +%s)"
  echo "Finished    : $(date --iso-8601=seconds 2>/dev/null || date)"
  echo "Elapsed(s)  : $((finished_epoch - started_epoch))"
  echo "Exit status : ${status}"
  exit "${status}"
}

trap finish_live_log EXIT
exec > >(tee -a "${live_log}") 2>&1

cd "${project_root}"
source "${activate_path}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/flashsim-nf-mpl-${run_name}"
mkdir -p "${MPLCONFIGDIR}"

echo "NFs Model 4 post-training pipeline"
echo "Host       : $(hostname)"
echo "Started    : $(date --iso-8601=seconds 2>/dev/null || date)"
echo "Config     : ${config_path}"
echo "Live log   : ${live_log}"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unavailable)"

python3 -c "import flashsim_nf, matplotlib, nflows, sklearn, torch, uproot"
python3 -u scripts/execute_model4_post_training.py --config "${config_path}"
