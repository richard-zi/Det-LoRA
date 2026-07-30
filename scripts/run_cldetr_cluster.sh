#!/usr/bin/env bash
# CL-DETR baseline runner (detector-specific IOD comparison, Track A + B).
# Run AFTER the iter4/cllora suites finish so the GPU is not overloaded.
# Usage (from package root):  bash scripts/run_cldetr_cluster.sh
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

LOG_DIR="experiments/logs"
mkdir -p "$LOG_DIR"
step() { echo; echo "==================== $* ===================="; }

# --- 1. uv + deps + headless OpenCV fix (container has no libGL) ---
step "1/5 environment"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
uv --version
uv sync --frozen
OPENCV_VERSION=$(uv run python -c "import importlib.metadata as m; print(m.version('opencv-python'))" 2>/dev/null || true)
if [ -n "$OPENCV_VERSION" ]; then
  echo "Swapping opencv-python -> opencv-python-headless"
  uv pip uninstall opencv-python
  uv pip install "opencv-python-headless==${OPENCV_VERSION}"
fi
uv run --no-sync python -c "import cv2; print('cv2 ok:', cv2.__version__)"

# --- 2. GPU ---
step "2/5 GPU"
nvidia-smi || { echo "ERROR: no GPU visible"; exit 1; }
uv run --no-sync python -c "import torch; assert torch.cuda.is_available(); print('CUDA ok:', torch.cuda.get_device_name(0))"

# --- 3. data ---
step "3/5 data"
test -f "data/raw/Labels/CSV Format/train_labels.csv" || { echo "ERROR: data/raw missing"; exit 1; }
test -f "data/extension/raw/Labels/CSV Format/train_labels.csv" || { echo "ERROR: data/extension/raw missing (Track B)"; exit 1; }

# --- 4. tests (CL-DETR DKD logic) ---
step "4/5 tests"
uv run --no-sync python -m pytest tests/test_cl_detr.py -q

# --- 5. training: CL-DETR suite, Track A + Track B ---
step "5/5 training"
uv run --no-sync python scripts/run_final_suite.py \
  --config configs/baselines/cldetr.json 2>&1 | tee "$LOG_DIR/cldetr_suite.log"

step "DONE"
echo "Results: experiments/suites/thesis_l40_cldetr (Track A) + .../extend (Track B)"
echo "Zum Zurueckgeben: zip -r results_cldetr.zip experiments/suites/thesis_l40_cldetr $LOG_DIR/cldetr_suite.log"
