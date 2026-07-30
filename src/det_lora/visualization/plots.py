"""
Thesis Visualizations
======================

Publication-quality plots for the master thesis.
All plots use consistent styling: serif fonts, 300 DPI, clean layout.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Thesis-quality plot styling
PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
}

# Colorblind-friendly palette
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]


def _apply_style():
    plt.rcParams.update(PLOT_STYLE)


def plot_training_curves(
    results_path: str,
    output_path: str,
    title: str = "Training Loss per Class",
) -> None:
    """
    Plot training loss curves for all classes overlaid.

    Args:
        results_path: Path to continual experiment results.json
        output_path: Output image path
    """
    _apply_style()

    with open(results_path) as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (class_name, task_data) in enumerate(results["tasks"].items()):
        losses = [h["loss"] for h in task_data["history"]]
        epochs = range(1, len(losses) + 1)
        color = COLORS[i % len(COLORS)]
        ax.plot(epochs, losses, label=class_name, color=color, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved training curves to {output_path}")


def plot_forgetting_matrix(
    evaluation_path: str,
    class_names: List[str],
    output_path: str,
) -> None:
    """
    Plot forgetting heatmap: rows = evaluated class, cols = after task N.

    Args:
        evaluation_path: Path to evaluation.json from run_experiment
        class_names: Ordered list of class names
        output_path: Output image path
    """
    _apply_style()

    with open(evaluation_path) as f:
        data = json.load(f)

    # Build matrix: rows = class, cols = task
    n = len(class_names)
    matrix = np.full((n, n), np.nan)

    for task_str, entry in data.get("history", {}).items():
        task_idx = int(task_str)
        ap_per_class = entry["metrics"].get("AP_per_class@0.5", {})
        for cls_key, ap_val in ap_per_class.items():
            # Find class index
            for ci, cn in enumerate(class_names):
                if cn == cls_key or str(ci) == str(cls_key):
                    matrix[ci, task_idx] = ap_val
                    break

    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), max(6, n * 1.2)))

    im = ax.imshow(matrix, cmap="YlOrRd_r", aspect="auto", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"After Task {i+1}" for i in range(n)], rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_names)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)

    ax.set_title("AP@0.5 per Class After Each Task (Forgetting Matrix)")
    ax.set_xlabel("Training Stage")
    ax.set_ylabel("Evaluated Class")
    plt.colorbar(im, ax=ax, label="AP@0.5")

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved forgetting matrix to {output_path}")


def plot_method_comparison(
    methods_results: Dict[str, Dict[str, float]],
    output_path: str,
    metrics: Optional[List[str]] = None,
) -> None:
    """
    Bar plot comparing multiple methods.

    Args:
        methods_results: {"Det-LoRA": {"mAP@0.5": 0.65, "Forgetting": 0.0, ...}, ...}
        output_path: Output image path
        metrics: Which metrics to plot (default: all)
    """
    _apply_style()

    if metrics is None:
        metrics = list(next(iter(methods_results.values())).keys())

    methods = list(methods_results.keys())
    n_methods = len(methods)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 2), 6))

    for i, method in enumerate(methods):
        values = [methods_results[method].get(m, 0) for m in metrics]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method, color=COLORS[i % len(COLORS)])
        # Value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Method Comparison")
    ax.legend()
    ax.set_ylim(0, 1.1)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved method comparison to {output_path}")


def plot_parameter_efficiency(
    methods: Dict[str, Dict[str, float]],
    output_path: str,
) -> None:
    """
    Scatter plot: trainable parameters vs. mAP.

    Args:
        methods: {"Det-LoRA": {"params": 49152, "mAP@0.5": 0.65}, ...}
        output_path: Output image path
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (name, data) in enumerate(methods.items()):
        ax.scatter(
            data["params"] / 1000,
            data["mAP@0.5"],
            s=150,
            color=COLORS[i % len(COLORS)],
            zorder=5,
            label=name,
        )
        ax.annotate(
            name,
            (data["params"] / 1000, data["mAP@0.5"]),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=10,
        )

    ax.set_xlabel("Trainable Parameters (K)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Parameter Efficiency")
    ax.legend()

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved parameter efficiency plot to {output_path}")


def plot_precision_recall(
    precisions: np.ndarray,
    recalls: np.ndarray,
    ap: float,
    class_name: str,
    output_path: str,
) -> None:
    """
    Plot precision-recall curve for a single class.

    Args:
        precisions: Precision values
        recalls: Recall values
        ap: Average Precision value
        class_name: Name of the class
        output_path: Output image path
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recalls, precisions, color=COLORS[0], linewidth=2)
    ax.fill_between(recalls, precisions, alpha=0.2, color=COLORS[0])

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall: {class_name} (AP={ap:.4f})")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved PR curve to {output_path}")


def generate_all_plots(
    experiment_dir: str,
    class_names: List[str],
    output_dir: str,
) -> None:
    """
    Generate all standard thesis plots from experiment results.

    Args:
        experiment_dir: Path to experiment output (with results.json, evaluation.json)
        class_names: Ordered list of class names
        output_dir: Directory to save all plots
    """
    exp = Path(experiment_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results_path = exp / "results.json"
    eval_path = exp / "evaluation.json"

    if results_path.exists():
        plot_training_curves(str(results_path), str(out / "training_curves.pdf"))

    if eval_path.exists():
        plot_forgetting_matrix(str(eval_path), class_names, str(out / "forgetting_matrix.pdf"))

    print(f"All plots saved to {out}")
