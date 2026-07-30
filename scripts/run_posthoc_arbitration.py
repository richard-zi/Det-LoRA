#!/usr/bin/env python3
"""Post-hoc Det-LoRA adapter arbitration for completed thesis checkpoints.

This script does not retrain adapters. It loads final Det-LoRA checkpoints from
an exported L40 run, fits compact arbitration state on validation data, and
evaluates mixed joint inference before/after arbitration on test data.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.arbitration import apply_adapter_arbitration, fit_adapter_arbitration_state
from det_lora.evaluation.evaluator import (
    ContinualEvaluator,
    _limit_prediction,
    collect_det_lora_joint_predictions,
)
from det_lora.evaluation.metrics import compute_map
from det_lora.model.det_lora import DetLoRA
from det_lora.model.detector import RFDETRDetector
from det_lora.train import _det_lora_class_id_mapping, collate_fn

DEFAULT_EXPORT_ROOT = Path.home() / "Documents/Richard/Uni/Masterthesis/cluster_l40_export"
DEFAULT_CLASSES = (
    "military_tank",
    "military_truck",
    "military_aircraft",
    "military_helicopter",
    "civilian_car",
    "civilian_aircraft",
)
DEFAULT_SUITES = ("thesis_l40_joint_baseline", "thesis_l40_main")
DEFAULT_MODELS = ("nano", "small", "base", "medium", "large")
DEFAULT_SEEDS = (42, 43, 44)
CORE_METRICS = (
    "mAP@0.5",
    "mAP@0.75",
    "mAP@0.95",
    "mAP@0.5:0.95",
    "Precision@0.5",
    "Recall@0.5",
    "F1@0.5",
    "MicroPrecision@0.5",
    "MicroRecall@0.5",
    "MicroF1@0.5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate post-hoc adapter arbitration on final Det-LoRA checkpoints."
    )
    parser.add_argument("--export_root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument("--fit_split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--tune_split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--eval_split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--fit_max_samples", type=int, default=120)
    parser.add_argument("--tune_max_samples", type=int, default=120)
    parser.add_argument("--tune_sample_offset", type=int, default=None)
    parser.add_argument("--eval_max_samples", type=int, default=260)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_detections", type=int, default=100)
    parser.add_argument("--arbitration_max_detections", type=int, default=100)
    parser.add_argument("--confidence_threshold", type=float, default=None)
    parser.add_argument("--optimize_metric", default="mAP@0.5:0.95")
    parser.add_argument(
        "--accept_tolerance",
        type=float,
        default=0.0,
        help="Minimum tune-set metric delta required before arbitration is accepted.",
    )
    parser.add_argument(
        "--disable_tune_guard",
        action="store_true",
        help="Always use the fitted arbitration state even if tune-set metrics get worse.",
    )
    parser.add_argument("--include_curves", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit_runs", type=int, default=None)
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=None,
        help="Optional CPU thread cap for local Mac runs.",
    )
    return parser.parse_args()


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(_to_builtin(payload), handle, indent=2, default=str)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "suite",
        "model",
        "seed",
        "checkpoint_dir",
        "status",
        "baseline_mAP@0.5",
        "arbitrated_mAP@0.5",
        "delta_mAP@0.5",
        "baseline_mAP@0.5:0.95",
        "arbitrated_mAP@0.5:0.95",
        "delta_mAP@0.5:0.95",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_dir(export_root: Path, suite: str, model: str, seed: int) -> Path:
    return (
        export_root
        / "experiments"
        / "suites"
        / suite
        / f"model_{model}"
        / f"seed_{seed}"
        / "det_lora"
        / "final"
    )


def iter_specs(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    count = 0
    for suite in args.suites:
        for model in args.models:
            for seed in args.seeds:
                yield {
                    "suite": suite,
                    "model": model,
                    "seed": int(seed),
                    "checkpoint_dir": checkpoint_dir(args.export_root, suite, model, int(seed)),
                }
                count += 1
                if args.limit_runs is not None and count >= args.limit_runs:
                    return


def make_loader(
    *,
    data_dir: Path,
    split: str,
    det_lora: DetLoRA,
    detector: RFDETRDetector,
    classes: Sequence[str],
    batch_size: int,
    num_workers: int,
    seed: int,
    max_samples: Optional[int],
    sample_offset: int = 0,
) -> DataLoader:
    dataset = load_dataset_from_raw(
        raw_dir=str(data_dir),
        class_filter=list(classes),
        split=split,
        class_id_offset=detector.base_num_classes,
        img_size=detector.resolution,
        seed=seed,
        max_samples=max_samples,
        sample_offset=sample_offset,
        class_id_mapping=_det_lora_class_id_mapping(det_lora, list(classes)),
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples loaded from {data_dir} split={split}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


def load_det_lora(checkpoint: Path, model: str) -> tuple[RFDETRDetector, DetLoRA]:
    detector = RFDETRDetector(variant=model)
    det_lora = DetLoRA(detector=detector)
    det_lora.load_all(str(checkpoint))
    return detector, det_lora


def evaluate(
    det_lora: DetLoRA,
    loader: DataLoader,
    classes: Sequence[str],
    args: argparse.Namespace,
    *,
    use_adapter_arbitration: bool,
) -> Dict[str, Any]:
    evaluator = ContinualEvaluator(
        det_lora,
        confidence_threshold=args.confidence_threshold,
        max_detections_per_image=args.max_detections,
        use_adapter_arbitration=use_adapter_arbitration,
        use_shared_encoder_cache=True,
    )
    return evaluator.evaluate_det_lora_joint(
        dataloader=loader,
        class_names=list(classes),
        include_curves=args.include_curves,
    )


def collect_predictions(
    det_lora: DetLoRA,
    loader: DataLoader,
    classes: Sequence[str],
    args: argparse.Namespace,
) -> tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]], List[int]]:
    predictions, ground_truths, target_class_ids = collect_det_lora_joint_predictions(
        det_lora,
        loader,
        list(classes),
        confidence_threshold=args.confidence_threshold,
        use_shared_encoder_cache=True,
    )
    if args.arbitration_max_detections is not None:
        predictions = [
            _limit_prediction(prediction, args.arbitration_max_detections)
            for prediction in predictions
        ]
    return predictions, ground_truths, target_class_ids


def fit_guarded_arbitration(
    det_lora: DetLoRA,
    fit_loader: DataLoader,
    tune_loader: DataLoader,
    classes: Sequence[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    fit_predictions, fit_ground_truths, target_class_ids = collect_predictions(
        det_lora,
        fit_loader,
        classes,
        args,
    )
    fitted_state = fit_adapter_arbitration_state(
        fit_predictions,
        fit_ground_truths,
        target_class_ids,
        optimize_metric=args.optimize_metric,
    )

    tune_predictions, tune_ground_truths, tune_target_class_ids = collect_predictions(
        det_lora,
        tune_loader,
        classes,
        args,
    )
    baseline_tune_metrics = compute_map(
        tune_predictions,
        tune_ground_truths,
        target_class_ids=tune_target_class_ids,
    )
    adjusted_tune_predictions = apply_adapter_arbitration(tune_predictions, fitted_state)
    arbitrated_tune_metrics = compute_map(
        adjusted_tune_predictions,
        tune_ground_truths,
        target_class_ids=tune_target_class_ids,
    )
    baseline_value = float(
        baseline_tune_metrics.get(args.optimize_metric, baseline_tune_metrics.get("mAP@0.5", 0.0))
    )
    arbitrated_value = float(
        arbitrated_tune_metrics.get(
            args.optimize_metric,
            arbitrated_tune_metrics.get("mAP@0.5", 0.0),
        )
    )
    delta = arbitrated_value - baseline_value
    accepted = args.disable_tune_guard or delta > float(args.accept_tolerance)

    fitted_state["tune_guard"] = {
        "enabled": not args.disable_tune_guard,
        "accepted": bool(accepted),
        "optimize_metric": args.optimize_metric,
        "accept_tolerance": float(args.accept_tolerance),
        "baseline_metric": baseline_value,
        "arbitrated_metric": arbitrated_value,
        "delta": delta,
        "baseline_metrics": baseline_tune_metrics,
        "arbitrated_metrics": arbitrated_tune_metrics,
    }
    if accepted:
        det_lora.set_adapter_arbitration_state(fitted_state)
        return fitted_state

    identity_state = {
        "identity": True,
        "reason": "tune_guard_rejected",
        "rejected_state": fitted_state,
        "tune_guard": fitted_state["tune_guard"],
    }
    det_lora.set_adapter_arbitration_state(identity_state)
    return identity_state


def metric_row(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for metric in CORE_METRICS:
        value = metrics.get(metric)
        if isinstance(value, (int, float)):
            row[f"{prefix}_{metric}"] = float(value)
    return row


def run_one(args: argparse.Namespace, spec: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint = Path(spec["checkpoint_dir"])
    run_dir = args.output_dir / spec["suite"] / f"model_{spec['model']}" / f"seed_{spec['seed']}"
    result_path = run_dir / "posthoc_arbitration_result.json"
    state_path = run_dir / "adapter_arbitration_state.json"

    base_row = {
        "suite": spec["suite"],
        "model": spec["model"],
        "seed": spec["seed"],
        "checkpoint_dir": str(checkpoint),
        "result_path": str(result_path),
        "state_path": str(state_path),
    }

    if not (checkpoint / "det_lora_registry.json").exists():
        return {**base_row, "status": "missing_checkpoint"}
    if result_path.exists() and not args.overwrite:
        try:
            existing = json.loads(result_path.read_text())
            return {**base_row, "status": "existing", **existing.get("summary_row", {})}
        except json.JSONDecodeError:
            pass
    if args.dry_run:
        return {**base_row, "status": "dry_run"}

    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"\n[run] {spec['suite']} model={spec['model']} seed={spec['seed']} "
        f"checkpoint={checkpoint}",
        flush=True,
    )
    detector, det_lora = load_det_lora(checkpoint, spec["model"])
    classes = list(args.classes)
    fit_loader = make_loader(
        data_dir=args.data_dir,
        split=args.fit_split,
        det_lora=det_lora,
        detector=detector,
        classes=classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=spec["seed"],
        max_samples=args.fit_max_samples,
    )
    tune_offset = (
        int(args.tune_sample_offset)
        if args.tune_sample_offset is not None
        else int(args.fit_max_samples or 0)
    )
    tune_loader = make_loader(
        data_dir=args.data_dir,
        split=args.tune_split,
        det_lora=det_lora,
        detector=detector,
        classes=classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=spec["seed"],
        max_samples=args.tune_max_samples,
        sample_offset=tune_offset,
    )
    eval_loader = make_loader(
        data_dir=args.data_dir,
        split=args.eval_split,
        det_lora=det_lora,
        detector=detector,
        classes=classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=spec["seed"],
        max_samples=args.eval_max_samples,
    )

    print("[eval] baseline mixed joint inference", flush=True)
    baseline_metrics = evaluate(
        det_lora,
        eval_loader,
        classes,
        args,
        use_adapter_arbitration=False,
    )

    print(
        "[fit] adapter arbitration "
        f"fit_samples={len(fit_loader.dataset)} tune_samples={len(tune_loader.dataset)} "
        f"max_detections={args.arbitration_max_detections}",
        flush=True,
    )
    arbitration_state = fit_guarded_arbitration(
        det_lora,
        fit_loader,
        tune_loader,
        classes,
        args,
    )
    write_json(state_path, arbitration_state)

    print("[eval] arbitrated mixed joint inference", flush=True)
    arbitrated_metrics = evaluate(
        det_lora,
        eval_loader,
        classes,
        args,
        use_adapter_arbitration=True,
    )

    summary_row = {
        **base_row,
        "status": "ok",
        "fit_samples": len(fit_loader.dataset),
        "eval_samples": len(eval_loader.dataset),
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_row.update(metric_row("baseline", baseline_metrics))
    summary_row.update(metric_row("arbitrated", arbitrated_metrics))
    for metric in CORE_METRICS:
        before = summary_row.get(f"baseline_{metric}")
        after = summary_row.get(f"arbitrated_{metric}")
        if before is not None and after is not None:
            summary_row[f"delta_{metric}"] = float(after) - float(before)

    payload = {
        "config": {
            "export_root": str(args.export_root),
            "data_dir": str(args.data_dir),
            "classes": classes,
            "fit_split": args.fit_split,
            "tune_split": args.tune_split,
            "eval_split": args.eval_split,
            "fit_max_samples": args.fit_max_samples,
            "tune_max_samples": args.tune_max_samples,
            "tune_sample_offset": tune_offset,
            "eval_max_samples": args.eval_max_samples,
            "batch_size": args.batch_size,
            "max_detections": args.max_detections,
            "arbitration_max_detections": args.arbitration_max_detections,
        },
        "spec": base_row,
        "baseline_metrics": baseline_metrics,
        "arbitrated_metrics": arbitrated_metrics,
        "arbitration_state_path": str(state_path),
        "summary_row": summary_row,
    }
    write_json(result_path, payload)
    print(
        "[done] "
        f"mAP50 {summary_row.get('baseline_mAP@0.5', 0.0):.4f} -> "
        f"{summary_row.get('arbitrated_mAP@0.5', 0.0):.4f}; "
        f"mAP50:95 {summary_row.get('baseline_mAP@0.5:0.95', 0.0):.4f} -> "
        f"{summary_row.get('arbitrated_mAP@0.5:0.95', 0.0):.4f}",
        flush=True,
    )

    del fit_loader, tune_loader, eval_loader, det_lora, detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    return summary_row


def main() -> None:
    args = parse_args()
    args.export_root = args.export_root.expanduser().resolve()
    if args.data_dir is None:
        args.data_dir = args.export_root / "data" / "raw"
    args.data_dir = args.data_dir.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = args.export_root / "experiments" / "analysis" / "posthoc_arbitration"
    args.output_dir = args.output_dir.expanduser().resolve()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    print("[info] export_root:", args.export_root)
    print("[info] data_dir:   ", args.data_dir)
    print("[info] output_dir: ", args.output_dir)
    print("[info] suites:     ", ", ".join(args.suites))
    print("[info] models:     ", ", ".join(args.models))
    print("[info] seeds:      ", ", ".join(str(seed) for seed in args.seeds))
    print("[info] classes:    ", ", ".join(args.classes))

    rows: List[Dict[str, Any]] = []
    for spec in iter_specs(args):
        try:
            rows.append(run_one(args, spec))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            error_row = {
                "suite": spec["suite"],
                "model": spec["model"],
                "seed": spec["seed"],
                "checkpoint_dir": str(spec["checkpoint_dir"]),
                "status": "error",
                "error": repr(exc),
            }
            rows.append(error_row)
            print(f"[error] {error_row}", flush=True)
            write_json(
                args.output_dir
                / spec["suite"]
                / f"model_{spec['model']}"
                / f"seed_{spec['seed']}"
                / "posthoc_arbitration_error.json",
                error_row,
            )

    write_csv(args.output_dir / "posthoc_arbitration_runs.csv", rows)
    write_json(
        args.output_dir / "posthoc_arbitration_summary.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rows": rows,
        },
    )
    print("[complete]", args.output_dir)


if __name__ == "__main__":
    main()
