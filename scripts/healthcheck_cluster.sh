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
  if "$@" >/tmp/det_lora_healthcheck.out 2>&1; then
    echo "ok"
  else
    echo "failed"
    cat /tmp/det_lora_healthcheck.out
    failures=$((failures + 1))
  fi
}

check "uv available" command -v uv
check "write experiments dir" bash -c 'mkdir -p experiments/logs && test -w experiments/logs'
check "configs present" bash -c 'test -f configs/baselines/joint.json && test -f configs/iterations/iteration1_base.json'
check "raw labels present" bash -c 'test -f "data/raw/Labels/CSV Format/train_labels.csv" && test -f "data/raw/Labels/CSV Format/test_labels.csv"'
check "extension labels present" bash -c 'test -f "data/extension/raw/Labels/CSV Format/train_labels.csv"'
check "rf-detr weights present" bash -c 'test -f rf-detr-nano.pth && test -f rf-detr-small.pth && test -f rf-detr-base.pth && test -f rf-detr-medium.pth && test -f rf-detr-large-2026.pth'

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[info] nvidia-smi:"
  nvidia-smi
else
  echo "[warn] nvidia-smi not found. The run requires an NVIDIA GPU on the cluster node."
  failures=$((failures + 1))
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

try:
    import torch
    print(f"[info] torch={torch.__version__}")
    print(f"[info] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[info] cuda_device={torch.cuda.get_device_name(0)}")
    else:
        errors.append("PyTorch does not see CUDA. Run this on a GPU node or check CUDA/Torch install.")
except Exception as exc:
    errors.append(f"torch import failed: {exc}")

try:
    import det_lora.final_runner as final_runner
except Exception as exc:
    errors.append(f"det_lora import failed: {exc}")
    final_runner = None

for config_path in [
    Path("configs/baselines/joint.json"),
    Path("configs/iterations/iteration1_base.json"),
]:
    try:
        config = json.loads(config_path.read_text())
        if final_runner is not None:
            main_runs = len(final_runner._build_run_specs(config))
            extension_runs = len(final_runner._build_extension_run_specs(config))
            print(f"[info] {config_path}: main_runs={main_runs}, extension_runs={extension_runs}")
            require(main_runs > 0, f"{config_path} builds no main runs")
    except Exception as exc:
        errors.append(f"{config_path} validation failed: {exc}")

raw_images = list(Path("data/raw/Images").glob("*"))
extension_images = list(Path("data/extension/raw/Images").glob("*"))
require(len(raw_images) > 0, "data/raw/Images is empty")
require(len(extension_images) > 0, "data/extension/raw/Images is empty")
print(f"[info] raw_images={len(raw_images)} extension_images={len(extension_images)}")

usage = shutil.disk_usage(".")
free_gib = usage.free / (1024 ** 3)
print(f"[info] free_disk_gib={free_gib:.1f}")
require(free_gib >= 50, "Less than 50 GiB free disk space. The final run can create many checkpoints/logs.")

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
  echo "Healthcheck failed with $failures issue(s). Fix them before starting the final run."
  exit 1
fi

echo
echo "Healthcheck passed. Ready to start:"
echo "  ./scripts/run_l40_thesis_experiments.sh"
