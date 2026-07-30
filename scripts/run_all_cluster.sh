#!/usr/bin/env bash
# All-in-one cluster runner: install -> fix env -> check -> test -> train both suites sequentially.
# Usage (from package root):  bash scripts/run_all_cluster.sh
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

LOG_DIR="experiments/logs"
mkdir -p "$LOG_DIR"

step() { echo; echo "==================== $* ===================="; }

# --- 1. uv install ---
step "1/6 uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

# --- 2. dependencies + headless OpenCV fix (container has no libGL) ---
step "2/6 dependencies"
uv sync --frozen
OPENCV_VERSION=$(uv run python -c "import importlib.metadata as m; print(m.version('opencv-python'))" 2>/dev/null || true)
if [ -n "$OPENCV_VERSION" ]; then
  echo "Swapping opencv-python -> opencv-python-headless (no libGL in container)"
  uv pip uninstall opencv-python
  uv pip install "opencv-python-headless==${OPENCV_VERSION}"
fi
uv run --no-sync python -c "import cv2; print('cv2 ok:', cv2.__version__)"

# --- 3. GPU check ---
step "3/6 GPU"
nvidia-smi || { echo "ERROR: no GPU visible"; exit 1; }
uv run --no-sync python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA ok:', torch.cuda.get_device_name(0))"

# --- 4. data check ---
step "4/6 data"
test -f "data/raw/Labels/CSV Format/train_labels.csv" || { echo "ERROR: data/raw missing (Images/ + Labels/CSV Format/)"; exit 1; }
test -f "data/raw/Labels/CSV Format/test_labels.csv"  || { echo "ERROR: test_labels.csv missing"; exit 1; }
if [ ! -f "data/extension/raw/Labels/CSV Format/train_labels.csv" ]; then
  echo "WARN: data/extension/raw missing -> Track B (extension) of the CL-LoRA suite will fail."
  echo "      Upload the extension data or set extension.enabled=false in configs/iterations/iteration5_shared_adapter.json."
fi

# --- 5. tests ---
step "5/6 tests"
uv run --no-sync pytest -q

# --- 6. training: both suites sequentially, then gate re-eval ---
step "6/6 training"
echo "Suite 1: thesis_l40_iter4 (FFN footprint) ..."
uv run --no-sync python scripts/run_final_suite.py --config configs/iterations/iteration4_extended_footprint.json 2>&1 | tee "$LOG_DIR/iter4_suite.log"

echo "Suite 2: thesis_l40_cllora (CL-LoRA + Track B) ..."
uv run --no-sync python scripts/run_final_suite.py --config configs/iterations/iteration5_shared_adapter.json 2>&1 | tee "$LOG_DIR/cllora_suite.log"

echo "Gate post-hoc sweep on all final checkpoints ..."
for suite in thesis_l40_iter4 thesis_l40_cllora; do
  for ckpt in experiments/suites/$suite/model_*/seed_*/det_lora/final; do
    [ -d "$ckpt" ] || continue
    variant=$(echo "$ckpt" | sed -E 's|.*/model_([^/]+)/.*|\1|')
    seed=$(echo "$ckpt" | sed -E 's|.*/seed_([0-9]+)/.*|\1|')
    echo "--- gate sweep: suite=$suite variant=$variant seed=$seed"
    uv run --no-sync python scripts/ablations/gate_posthoc_sweep.py \
      --suite "$suite" --variant "$variant" --seed "$seed" \
      --fit_max_samples 999999 --data_dir data/raw \
      2>&1 | tee "$LOG_DIR/gate_${suite}_${variant}_seed${seed}.log" || echo "WARN: gate sweep failed for $ckpt"
  done
done

step "DONE"
echo "Results: experiments/suites/thesis_l40_iter4, experiments/suites/thesis_l40_cllora, $LOG_DIR/gate_*.log"
echo "To download: zip -r results.zip experiments/suites/thesis_l40_iter4 experiments/suites/thesis_l40_cllora $LOG_DIR"
