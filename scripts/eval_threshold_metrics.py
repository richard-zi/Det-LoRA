#!/usr/bin/env python
"""Reproduce the threshold-dependent auxiliary metrics of thesis table ``tab:erg_aux``.

The mixed Precision/Recall/F1 values reported at IoU 0.5 are evaluated at the
*per-class F1-optimal confidence threshold*. The ``Precision@0.5`` fields stored in
``evaluation.json`` are threshold-free (they count every raw prediction) and are
therefore far smaller. This script recomputes the reported operating point directly
from the serialized precision/recall/score curves that the evaluator persists per
class (``PR_curve_per_class@0.5``), without any GPU inference.

For each run (method x model variant x seed) the final mixed evaluation is used: the
last task of ``mixed_history`` when present (Det-LoRA), otherwise the last task of
``history`` (single-model baselines, which already evaluate on all seen classes).
Per class the confidence threshold maximizing F1 is selected; a class without any
prediction contributes zeros. Precision, Recall and F1 are macro-averaged over the
classes of a run and then averaged over variants and seeds (Track A).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

# method label -> (suite subdirectory, run subdirectory)
METHOD_SOURCES: Dict[str, Tuple[str, str]] = {
    "Fine-Tuning": ("thesis_l40_main", "finetuning"),
    "EWC": ("thesis_l40_main", "ewc"),
    "Det-LoRA": ("thesis_l40_main", "det_lora"),
    "CL-DETR": ("thesis_l40_cldetr", "cl_detr"),
    "Replay": ("thesis_l40_main", "replay"),
    "Joint Fine-Tuning": ("thesis_l40_joint_baseline", "joint_finetuning"),
}
MODEL_VARIANTS: Sequence[str] = ("nano", "small", "base", "medium", "large")
SEEDS: Sequence[int] = (42, 43, 44)

# reported values of tab:erg_aux for the empirical cross-check (precision, recall, f1)
REPORTED: Dict[str, Tuple[float, float, float]] = {
    "Fine-Tuning": (0.408, 0.361, 0.343),
    "EWC": (0.474, 0.563, 0.497),
    "Det-LoRA": (0.634, 0.675, 0.645),
    "CL-DETR": (0.692, 0.690, 0.682),
    "Replay": (0.716, 0.686, 0.691),
    "Joint Fine-Tuning": (0.862, 0.831, 0.844),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suites_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "suites",
        help="Directory holding the thesis_l40_* suite outputs.",
    )
    parser.add_argument(
        "--iou",
        type=str,
        default="0.5",
        help="IoU level of the PR curve to evaluate (matches the stored key suffix).",
    )
    return parser.parse_args()


def f1_optimal_operating_point(pr_curve: Dict[str, object]) -> Tuple[float, float, float]:
    """Return (precision, recall, f1) at the F1-maximizing confidence threshold.

    The curve stores cumulative precision/recall of the predictions sorted by
    descending score, so each index is one candidate confidence threshold. An empty
    curve (a class without predictions) contributes zeros.
    """
    precisions = pr_curve["precision"]
    recalls = pr_curve["recall"]
    best_precision, best_recall, best_f1 = 0.0, 0.0, 0.0
    for precision, recall in zip(precisions, recalls):
        denominator = precision + recall
        f1 = (2.0 * precision * recall / denominator) if denominator > 0 else 0.0
        if f1 > best_f1:
            best_precision, best_recall, best_f1 = precision, recall, f1
    return best_precision, best_recall, best_f1


def final_mixed_metrics(evaluation: Dict[str, object]) -> Dict[str, object]:
    """Metrics block of the last mixed-evaluation task of a run."""
    history = evaluation.get("mixed_history") or evaluation["history"]
    last_task = max(history, key=int)
    return history[last_task]["metrics"]


def run_macro_metrics(evaluation_path: Path, iou: str) -> Tuple[float, float, float]:
    """Macro-averaged (precision, recall, f1) at F1-optimal thresholds for one run."""
    evaluation = json.loads(evaluation_path.read_text())
    metrics = final_mixed_metrics(evaluation)
    pr_curves = metrics[f"PR_curve_per_class@{iou}"]
    operating_points = [f1_optimal_operating_point(curve) for curve in pr_curves.values()]
    precisions, recalls, f1s = zip(*operating_points)
    return mean(precisions), mean(recalls), mean(f1s)


def aggregate_method(
    suites_dir: Path, suite: str, run: str, iou: str
) -> Tuple[float, float, float, int]:
    """Average macro metrics over all available variant/seed runs of a method."""
    per_run: List[Tuple[float, float, float]] = []
    for variant in MODEL_VARIANTS:
        for seed in SEEDS:
            path = (
                suites_dir / suite / f"model_{variant}" / f"seed_{seed}" / run / "evaluation.json"
            )
            if path.exists():
                per_run.append(run_macro_metrics(path, iou))
    if not per_run:
        raise FileNotFoundError(f"No evaluation.json found for {suite}/{run} under {suites_dir}")
    precisions, recalls, f1s = zip(*per_run)
    return mean(precisions), mean(recalls), mean(f1s), len(per_run)


def main() -> None:
    args = parse_args()
    header = f"{'Verfahren':<20} {'n':>3} {'Precision':>21} {'Recall':>18} {'F1':>18}"
    print(header)
    print("-" * len(header))
    for method, (suite, run) in METHOD_SOURCES.items():
        precision, recall, f1, n = aggregate_method(args.suites_dir, suite, run, args.iou)
        reported = REPORTED.get(method)

        def cell(value: float, expected: float | None) -> str:
            if expected is None:
                return f"{value:.3f}"
            flag = "ok" if round(value, 3) == round(expected, 3) else "DIFF"
            return f"{value:.3f} (ber. {expected:.3f} {flag})"

        rp, rr, rf = reported if reported else (None, None, None)
        print(
            f"{method:<20} {n:>3} {cell(precision, rp):>21} {cell(recall, rr):>18} {cell(f1, rf):>18}"
        )


if __name__ == "__main__":
    main()
