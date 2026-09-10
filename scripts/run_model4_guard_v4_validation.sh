#!/usr/bin/env bash
set -euo pipefail

mode="${1:-check}"
project="${NFS_PROJECT_ROOT:-/eos/user/t/tanansub/SWAN_projects/NFs}"
output_root="${NFS_OUTPUT_ROOT:-/eos/user/t/tanansub/SWAN_projects/NFs_output}"
guard_root="${NFS_DATA_ROOT:-/eos/user/t/tanansub/SWAN_projects/NFs_data}/guards"
activate="${NFS_ACTIVATE:-/eos/user/t/tanansub/venvBBfs/bin/activate}"
config_root="${output_root}/resolved_configs/model4_generated_validation_guard_v4"

usage() {
  echo "Usage: bash scripts/run_model4_guard_v4_validation.sh [check|prepare|submit]"
  echo "  check   : read-only guard preflight (default)"
  echo "  prepare : create v4 guards/configs, then print 12 submissions"
  echo "  submit  : submit the prepared 12-job validation matrix"
}

if [[ "$mode" != "check" && "$mode" != "prepare" && "$mode" != "submit" ]]; then
  usage
  exit 2
fi

if [[ ! -f "$activate" ]]; then
  echo "Missing Python environment: $activate" >&2
  exit 1
fi
if [[ ! -d "$project" ]]; then
  echo "Missing NFs project: $project" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$activate"
export PYTHONPATH="${project}/src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/flashsim-mpl-tanansub}"
mkdir -p "$MPLCONFIGDIR"
cd "$project"

if [[ "$mode" == "check" ]]; then
  python3 scripts/create_model4_reference_guards.py \
    --environment cern \
    --guard-root "$guard_root" \
    --expected-count 4 \
    --dry-run
  echo
  echo "Read-only preflight passed. Next:"
  echo "bash scripts/run_model4_guard_v4_validation.sh prepare"
  exit 0
fi

if [[ "$mode" == "prepare" ]]; then
  python3 scripts/create_model4_reference_guards.py \
    --environment cern \
    --guard-root "$guard_root" \
    --expected-count 4

  python3 scripts/create_model4_validation_configs.py \
    --output-root "$output_root" \
    --guard-root "$guard_root" \
    --guard-version-directory train_vs_all_clean_v4 \
    --expected-count 12

  python3 scripts/submit_model4_post_training_matrix.py \
    --config-root "$config_root" \
    --expected-count 12 \
    --dry-run

  echo
  echo "Preparation passed. Review the commands above, then submit with:"
  echo "bash scripts/run_model4_guard_v4_validation.sh submit"
  exit 0
fi

if ! type module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  else
    echo "CERN module command is unavailable" >&2
    exit 1
  fi
fi
module load lxbatch/eossubmit

python3 scripts/submit_model4_post_training_matrix.py \
  --config-root "$config_root" \
  --expected-count 12
