#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

LOG_DIR="${THESIS_EVAL_LOG_DIR:-experiments/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${THESIS_EVAL_LOG_FILE:-$LOG_DIR/thesis_evaluation_$(date +%Y%m%d_%H%M%S).log}"

{
  echo "Thesis L40 evaluation"
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
  ./scripts/healthcheck_evaluation.sh 2>&1 | tee -a "$LOG_FILE"
fi

uv run python scripts/run_thesis_evaluation.py "$@" 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}

{
  echo
  echo "Finished: $(date -Is)"
  echo "Exit code: $status"
} | tee -a "$LOG_FILE"

exit "$status"
