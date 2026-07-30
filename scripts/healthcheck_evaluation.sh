#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PATH="$HOME/.local/bin:$PATH"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

failures=0

check() {
  local name="$1"
  shift
  printf '[check] %s ... ' "$name"
  if "$@" >/tmp/det_lora_eval_healthcheck.out 2>&1; then
    echo "ok"
  else
    echo "failed"
    cat /tmp/det_lora_eval_healthcheck.out
    failures=$((failures + 1))
  fi
}

check "uv available" command -v uv
check "write analysis dir" bash -c 'mkdir -p experiments/analysis && test -w experiments/analysis'
check "joint suite summary present" test -f experiments/suites/thesis_l40_joint_baseline/suite_summary.json
check "main suite summary present" test -f experiments/suites/thesis_l40_main/suite_summary.json
check "extension suite summary present" test -f experiments/suites/thesis_l40_main/extend/suite_summary.json
check "raw labels present" bash -c 'test -f "data/raw/Labels/CSV Format/test_labels.csv"'
check "rf-detr weights present" bash -c 'test -f rf-detr-nano.pth && test -f rf-detr-small.pth && test -f rf-detr-base.pth && test -f rf-detr-medium.pth && test -f rf-detr-large-2026.pth'

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[info] nvidia-smi:"
  nvidia-smi
else
  echo "[warn] nvidia-smi not found. Plots work on CPU, but inference rendering is slower without a GPU."
fi

if command -v uv >/dev/null 2>&1; then
  if ! uv run python - <<'PY'
import json
import shutil
import sys
from pathlib import Path

errors = []

def require(condition, message):
    if not condition:
        errors.append(message)

for summary_path in [
    Path("experiments/suites/thesis_l40_joint_baseline/suite_summary.json"),
    Path("experiments/suites/thesis_l40_main/suite_summary.json"),
    Path("experiments/suites/thesis_l40_main/extend/suite_summary.json"),
]:
    try:
        payload = json.loads(summary_path.read_text())
        require(bool(payload.get("groups")), f"{summary_path} has no groups")
        print(f"[info] {summary_path}: groups={len(payload.get('groups', {}))}")
    except Exception as exc:
        errors.append(f"{summary_path} validation failed: {exc}")

try:
    import torch
    print(f"[info] torch={torch.__version__}")
    print(f"[info] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[info] cuda_device={torch.cuda.get_device_name(0)}")
except Exception as exc:
    errors.append(f"torch import failed: {exc}")

usage = shutil.disk_usage(".")
free_gib = usage.free / (1024 ** 3)
print(f"[info] free_disk_gib={free_gib:.1f}")
require(free_gib >= 10, "Less than 10 GiB free disk space. Evaluation outputs need additional room.")

if errors:
    print("[healthcheck] failed:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)

print("[healthcheck] python checks ok")
PY
  then
    failures=$((failures + 1))
  fi
fi

if [[ "$failures" -ne 0 ]]; then
  echo
  echo "Evaluation healthcheck failed with $failures issue(s). Fix them before starting the evaluation run."
  exit 1
fi

echo
echo "Evaluation healthcheck passed. Ready to start:"
echo "  ./scripts/run_l40_thesis_evaluation.sh"
