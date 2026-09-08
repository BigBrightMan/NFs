#!/usr/bin/env bash
set -euo pipefail

config_path="${1:?resolved training config is required}"
activate_path="${2:?environment activation script is required}"
project_root="${3:?NFs project root is required}"
output_root="${4:?NFs output root is required}"
log_subdir="${5:?log subdirectory is required}"
run_name="${6:?run name is required}"

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
if [[ "${output_root}" != /* ]]; then
  echo "NFs output root must be absolute: ${output_root}" >&2
  exit 2
fi
if [[ "${log_subdir}" == /* || "${log_subdir}" == *".."* || ! "${log_subdir}" =~ ^[A-Za-z0-9_./-]+$ ]]; then
  echo "Invalid log subdirectory: ${log_subdir}" >&2
  exit 2
fi
if [[ ! "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "Invalid run name: ${run_name}" >&2
  exit 2
fi

live_directory="${output_root}/condor_logs/live/${log_subdir}"
timestamp="$(date +%Y%m%d_%H%M%S)"
host_label="$(hostname | tr -c 'A-Za-z0-9_.-' '_')"
mkdir -p "${live_directory}"
live_log="${live_directory}/model4_training_${run_name}_${timestamp}_${host_label}_pid$$.live.log"
started_epoch="$(date +%s)"

finish_live_log() {
  status=$?
  trap - EXIT
  finished_epoch="$(date +%s)"
  echo "=================================================="
  echo "Finished    : $(date --iso-8601=seconds 2>/dev/null || date)"
  echo "Elapsed(s)  : $((finished_epoch - started_epoch))"
  echo "Exit status : ${status}"
  echo "=================================================="
  exit "${status}"
}

trap finish_live_log EXIT
exec > >(tee -a "${live_log}") 2>&1

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
echo "Live log   : ${live_log}"
echo "GPU        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unavailable)"
echo "=================================================="

python3 -c "import flashsim_nf, nflows, torch, uproot"
python3 -u scripts/preflight_model4_training.py --config "${config_path}"
python3 -u scripts/train_model4.py --config "${config_path}"
