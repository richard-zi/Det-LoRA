#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

LOG_DIR="${FINAL_SUITE_LOG_DIR:-experiments/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${FINAL_SUITE_LOG_FILE:-$LOG_DIR/thesis_experiments_$(date +%Y%m%d_%H%M%S).log}"

run_suite() {
  local config_path="$1"
  echo
  echo "============================================================"
  echo "Running suite: $config_path"
  echo "============================================================"
  uv run python scripts/run_final_suite.py --config "$config_path"
}

{
  echo "Thesis L40 experiments"
  echo "Started: $(date -Is)"
  echo "Working dir: $(pwd)"
  echo "Log: $LOG_FILE"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
  uv --version
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "nvidia-smi not found"
  fi
} | tee -a "$LOG_FILE"

if [[ "${SKIP_HEALTHCHECK:-0}" != "1" ]]; then
  ./scripts/healthcheck_cluster.sh 2>&1 | tee -a "$LOG_FILE"
fi

{
  run_suite "configs/baselines/joint.json"
  run_suite "configs/iterations/iteration1_base.json"
} 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}

{
  echo
  echo "Finished: $(date -Is)"
  echo "Exit code: $status"
} | tee -a "$LOG_FILE"

exit "$status"
