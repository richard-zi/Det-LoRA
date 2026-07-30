"""
Checkpoint/Resume utilities for baselines.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, Set


def load_progress(experiment_dir: Path) -> Dict:
    """Load progress.json if it exists."""
    path = experiment_dir / "progress.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed_tasks": [], "current_task": None}


def save_progress(experiment_dir: Path, completed_tasks: Set[str], current_task: str) -> None:
    """Save progress for resume."""
    progress = {
        "completed_tasks": list(completed_tasks),
        "current_task": current_task,
    }
    with open(experiment_dir / "progress.json", "w") as f:
        json.dump(progress, f, indent=2)


def save_model_checkpoint(
    model,
    experiment_dir: Path,
    task_name: str,
    checkpoint_root: str = "checkpoints",
) -> None:
    """Save full model state after a task."""
    import torch

    checkpoint_dir = experiment_dir / checkpoint_root / task_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(checkpoint_dir / "model.pt"))


def load_model_checkpoint(
    model,
    experiment_dir: Path,
    task_name: str,
    checkpoint_root: str = "checkpoints",
) -> bool:
    """Load model checkpoint. Returns True if successful."""
    import torch

    checkpoint_path = experiment_dir / checkpoint_root / task_name / "model.pt"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"))
        return True
    return False


def prepare_detector_for_checkpoint_load(detector, seen_classes: Iterable[str]) -> None:
    """Expand a fresh detector head to match the classes stored in a checkpoint."""
    for class_name in seen_classes:
        detector.expand_classification_head(class_name)


def is_better_validation_checkpoint(
    current_map50: float,
    current_loss: float,
    best_map50: float,
    best_loss: float,
) -> bool:
    """Select by validation mAP@0.5, falling back to lower validation loss."""
    if current_map50 > best_map50:
        return True
    if current_map50 == best_map50 and current_loss < best_loss:
        return True
    if best_map50 == float("-inf") and current_loss < best_loss:
        return True
    return False


def save_state(experiment_dir: Path, task_name: str, state: Dict) -> None:
    """Save method-specific resume state for a task."""
    import torch

    state_dir = experiment_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, state_dir / f"{task_name}.pt")


def load_state(experiment_dir: Path, task_name: str) -> Dict:
    """Load method-specific resume state for a task."""
    import torch

    state_path = experiment_dir / "state" / f"{task_name}.pt"
    if state_path.exists():
        return torch.load(state_path, map_location="cpu")
    return {}
