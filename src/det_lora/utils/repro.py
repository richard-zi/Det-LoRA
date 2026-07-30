"""Reproducibility helpers."""

import os
import platform
import random
from typing import Any, Dict

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch RNGs to a fixed seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collect_runtime_metadata() -> Dict[str, Any]:
    """Capture lightweight runtime metadata for experiment artifacts."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
        "mps_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
    }
