"""
Det-LoRA Continual Learning Experiment
=========================================

Runs a full class-incremental continual learning experiment with task-level resume.

Features:
- Trains adapters sequentially and evaluates on all seen classes after each task
- Saves finalized checkpoints after each task (adapter weights + progress.json)
- Resumes from the last completed task

Usage:
    # Start new experiment
    python -m det_lora.run_experiment --classes military_tank military_truck --epochs 30

    # Resume crashed experiment
    python -m det_lora.run_experiment --resume experiments/continual_20260315_123456
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader

from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.evaluator import (
    ContinualEvaluator,
    aggregate_classwise_metrics,
    refresh_adapter_arbitration,
    refresh_adapter_calibration,
    refresh_shared_quality_calibrator,
    summarize_mixed_confusion,
)
from det_lora.model.det_lora import DetLoRA
from det_lora.model.detector import LORA_TARGET_PRESETS, RFDETRDetector
from det_lora.train import (
    SyntheticDetectionDataset,
    _make_adapter_hard_negatives,
    collate_fn,
    evaluate_detection_metrics,
    evaluate_loss,
    train_one_epoch,
)
from det_lora.utils import (
    collect_runtime_metadata,
    expand_model_variants,
    resolve_variant_settings,
    set_global_seed,
)


def _load_progress(experiment_dir: Path) -> Dict:
    """Load progress.json if it exists."""
    progress_path = experiment_dir / "progress.json"
    if progress_path.exists():
        with open(progress_path) as f:
            return json.load(f)
    return {"completed_tasks": [], "current_task": None}


def _save_progress(experiment_dir: Path, progress: Dict) -> None:
    """Save progress.json for resume capability."""
    with open(experiment_dir / "progress.json", "w") as f:
        json.dump(progress, f, indent=2)


def _load_evaluation_histories(experiment_dir: Path) -> tuple[Dict[int, Dict], Dict[int, Dict]]:
    """Load persisted matched + mixed evaluation histories if present."""
    evaluation_path = experiment_dir / "evaluation.json"
    if not evaluation_path.exists():
        return {}, {}

    with open(evaluation_path) as f:
        data = json.load(f)

    def _parse_history(raw_history: Dict) -> Dict[int, Dict]:
        history = {}
        for task_str, entry in raw_history.items():
            history[int(task_str)] = {
                "metrics": entry["metrics"],
                "class_names": entry.get("class_names", []),
            }
        return history

    matched_history = _parse_history(data.get("matched_history", {}))
    mixed_history = _parse_history(data.get("mixed_history", data.get("history", {})))
    return matched_history, mixed_history


def _to_builtin(value):
    """Recursively convert nested scalars into JSON-safe builtins."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def _serialize_history(history: Dict[int, Dict]) -> Dict[str, Dict]:
    serialized = {}
    for task_idx, entry in history.items():
        serialized[str(task_idx)] = {
            "metrics": _to_builtin(entry["metrics"]),
            "class_names": entry.get("class_names", []),
        }
    return serialized


def _evaluate_matched_seen_classes(
    det_lora: DetLoRA,
    seen_classes: List[str],
    data_dir: str,
    batch_size: int,
    resolution: int,
    class_id_offset: int,
    seed: int,
    include_curves: bool = False,
) -> Dict:
    """Evaluate each seen class on its own matched class slice."""
    per_class_metrics = {}
    evaluator = ContinualEvaluator(det_lora)

    for class_name in seen_classes:
        dataset = load_dataset_from_raw(
            raw_dir=data_dir,
            class_filter=class_name,
            split="test",
            class_id_offset=class_id_offset,
            img_size=resolution,
            seed=seed,
            class_id_mapping=_det_lora_class_id_mapping(det_lora, [class_name]),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        per_class_metrics[class_name] = evaluator.evaluate_det_lora_joint(
            dataloader=dataloader,
            class_names=[class_name],
            include_curves=include_curves,
        )

    return aggregate_classwise_metrics(per_class_metrics)


def _make_raw_dataloader(
    *,
    raw_dir: str,
    class_filter,
    split: str,
    class_id_offset: int,
    img_size: int,
    seed: int,
    batch_size: int,
    max_samples: Optional[int] = None,
    class_id_mapping: Optional[Dict[str, int]] = None,
) -> DataLoader:
    """Build a deterministic raw-data dataloader for calibration/evaluation."""
    dataset = load_dataset_from_raw(
        raw_dir=raw_dir,
        class_filter=class_filter,
        split=split,
        class_id_offset=class_id_offset,
        img_size=img_size,
        seed=seed,
        max_samples=max_samples,
        class_id_mapping=class_id_mapping,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )


def _prefix_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _det_lora_class_id_mapping(det_lora: DetLoRA, class_names: List[str]) -> Dict[str, int]:
    return {class_name: det_lora.get_class_id(class_name) for class_name in class_names}


def run_continual_experiment(
    classes: List[str],
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    model_variant: str = "medium",
    data_dir: str = "data/raw",
    save_dir: str = "experiments",
    synthetic: bool = False,
    max_samples: Optional[int] = None,
    resume_dir: Optional[str] = None,
    seed: int = 42,
    experiment_name: Optional[str] = None,
    metrics_eval_every: int = 5,
    enable_shared_quality_calibrator: bool = True,
    use_adapter_arbitration: bool = False,
    use_hard_negatives: bool = True,
    symmetric_hard_negatives: bool = False,
    lora_target_preset: str = "default",
    use_dora: bool = False,
    merge_consistency_weight: float = 0.0,
    use_shared_adapter: bool = False,
    shared_drift_weight: float = 1.0,
    preset_name: Optional[str] = None,
) -> Dict:
    """
    Run a full class-incremental continual learning experiment.

    For each class in sequence:
    1. Add new LoRA adapter (freeze previous)
    2. Train on new class data
    3. Save checkpoint
    4. After ALL training: evaluate on ALL classes

    Supports resume: if resume_dir is set, loads progress and continues.
    """
    # Determine experiment directory
    set_global_seed(seed)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    if resume_dir:
        experiment_dir = Path(resume_dir)
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Resume dir not found: {resume_dir}")
        print(f"[Experiment] Resuming from {experiment_dir}")
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = experiment_name or f"continual_{model_variant}_{timestamp}"
        experiment_dir = Path(save_dir) / run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Load progress
    progress = _load_progress(experiment_dir)
    completed_tasks = set(progress.get("completed_tasks", []))

    print(f"\n{'='*60}")
    print(f"Det-LoRA Continual Learning Experiment")
    print(f"{'='*60}")
    print(f"Classes: {' -> '.join(classes)}")
    print(f"Epochs per class: {epochs}")
    print(f"Model: RF-DETR {model_variant}")
    print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print(f"Output: {experiment_dir}")
    if completed_tasks:
        print(f"Resuming - already completed: {completed_tasks}")
    print(f"{'='*60}\n")

    # Initialize model
    detector = RFDETRDetector(variant=model_variant)
    det_lora = DetLoRA(
        detector=detector,
        default_rank=lora_rank,
        default_alpha=lora_alpha,
        lora_target_preset=lora_target_preset,
        use_dora=use_dora,
        use_shared_adapter=use_shared_adapter,
    )
    resolution = detector.resolution

    # Results tracking
    results_path = experiment_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        all_results.setdefault("config", {})
        all_results.setdefault("tasks", {})
        all_results.setdefault("evaluation_after_task", {})
        all_results.setdefault("matched_evaluation_after_task", {})
        all_results.setdefault("mixed_evaluation_after_task", {})
        all_results["config"].setdefault("output_dir", str(experiment_dir))
        all_results["config"].setdefault("metrics_eval_every", metrics_eval_every)
        all_results["config"]["enable_shared_quality_calibrator"] = enable_shared_quality_calibrator
        all_results["config"]["use_adapter_arbitration"] = use_adapter_arbitration
        all_results["config"]["use_hard_negatives"] = use_hard_negatives
        all_results["config"].setdefault("run_metadata", {})
        all_results["config"]["run_metadata"].setdefault("started_at", started_at)
        all_results["config"]["run_metadata"].setdefault("resumed", bool(resume_dir))
        all_results["config"]["run_metadata"].setdefault("finished_at", None)
    else:
        all_results = {
            "config": {
                "classes": classes,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "model_variant": f"RF-DETR {model_variant}",
                "resolution": resolution,
                "seed": seed,
                "metrics_eval_every": metrics_eval_every,
                "preset": preset_name,
                "enable_shared_quality_calibrator": enable_shared_quality_calibrator,
                "use_adapter_arbitration": use_adapter_arbitration,
                "use_hard_negatives": use_hard_negatives,
                "eval_split": "test",
                "runtime": collect_runtime_metadata(),
                "output_dir": str(experiment_dir),
                "run_metadata": {
                    "started_at": started_at,
                    "finished_at": None,
                    "resumed": bool(resume_dir),
                },
            },
            "tasks": {},
            "evaluation_after_task": {},
            "matched_evaluation_after_task": {},
            "mixed_evaluation_after_task": {},
        }

    matched_history, mixed_history = _load_evaluation_histories(experiment_dir)
    matched_evaluator = ContinualEvaluator(
        det_lora,
        use_shared_quality_calibrator=enable_shared_quality_calibrator,
        use_adapter_arbitration=use_adapter_arbitration,
    )
    matched_evaluator.history = matched_history
    mixed_evaluator = ContinualEvaluator(
        det_lora,
        use_shared_quality_calibrator=enable_shared_quality_calibrator,
        use_adapter_arbitration=use_adapter_arbitration,
    )
    mixed_evaluator.history = mixed_history

    # ===== TRAINING PHASE =====
    print("\n>>> TRAINING PHASE <<<\n")

    for task_idx, class_name in enumerate(classes):
        # Check if already completed
        if class_name in completed_tasks:
            print(
                f"[Task {task_idx+1}/{len(classes)}] '{class_name}' - SKIPPED (already completed)"
            )
            # Load full checkpoint (head + LoRA) from saved state
            checkpoint_dir = experiment_dir / f"checkpoints/task_{task_idx}_{class_name}"
            if checkpoint_dir.exists():
                det_lora.load_all(str(checkpoint_dir))
                print(f"  Restored from {checkpoint_dir}")
            else:
                raise FileNotFoundError(
                    f"Missing finalized checkpoint for completed task '{class_name}': {checkpoint_dir}"
                )
            continue

        print(f"\n{'='*60}")
        print(f"TASK {task_idx + 1}/{len(classes)}: Training '{class_name}'")
        print(f"{'='*60}")

        # 1. Add class
        det_lora.add_class(class_name, rank=lora_rank, alpha=lora_alpha)

        # 2. Create dataset
        if synthetic:
            num_classes = detector.get_num_classes()
            train_dataset = SyntheticDetectionDataset(100, num_classes, resolution)
            val_dataset = SyntheticDetectionDataset(25, num_classes, resolution)
        else:
            train_dataset = load_dataset_from_raw(
                raw_dir=data_dir,
                class_filter=class_name,
                split="train",
                class_id_offset=detector.base_num_classes,
                img_size=resolution,
                seed=seed,
                max_samples=max_samples,
                class_id_mapping=_det_lora_class_id_mapping(det_lora, [class_name]),
            )
            val_dataset = load_dataset_from_raw(
                raw_dir=data_dir,
                class_filter=class_name,
                split="val",
                class_id_offset=detector.base_num_classes,
                img_size=resolution,
                seed=seed,
                max_samples=max_samples,
                class_id_mapping=_det_lora_class_id_mapping(det_lora, [class_name]),
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        hard_negative_counts: Dict[str, int] = {}
        if not synthetic and use_hard_negatives and len(train_dataset) > 0:
            # Symmetric mode: also use later (confusable) classes as negatives, so
            # every adapter learns a discriminative -- not merely one-vs-background
            # -- boundary. Relaxes strict incrementality (other classes' images
            # serve as empty-target negatives); declared as a design choice.
            negative_classes = (
                [c for c in classes if c != class_name] if symmetric_hard_negatives else None
            )
            negative_datasets, hard_negative_counts = _make_adapter_hard_negatives(
                det_lora=det_lora,
                raw_dir=data_dir,
                target_class=class_name,
                detector=detector,
                img_size=resolution,
                seed=seed,
                max_samples_per_class=len(train_dataset),
                negative_classes=negative_classes,
            )
            if negative_datasets:
                train_dataset = ConcatDataset([train_dataset, *negative_datasets])
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=0,
                )

        print(f"Training samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
        if hard_negative_counts:
            print(f"Hard negatives: {hard_negative_counts}")

        # 3. Optimizer: head gets weight_decay=0 to prevent AdamW from
        # drifting gradient-masked COCO neurons (wd applies even at grad=0)
        lora_params = []
        head_params = []
        for name, param in det_lora.model.named_parameters():
            if not param.requires_grad:
                continue
            if "class_embed" in name or "enc_out_class" in name:
                head_params.append(param)
            else:
                lora_params.append(param)
        optimizer = AdamW(
            [
                {"params": lora_params, "weight_decay": weight_decay},
                {"params": head_params, "weight_decay": 0.0},
            ],
            lr=lr,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        trainable_count = sum(p.numel() for p in lora_params + head_params)
        print(f"Trainable params: {trainable_count:,}")

        # 4. Train
        if merge_consistency_weight > 0:
            anchored_modules = det_lora.enable_merge_consistency()
            print(
                f"[MergeConsistency] {anchored_modules} anchored modules "
                f"(weight={merge_consistency_weight})"
            )

        task_history = []
        for epoch in range(1, epochs + 1):
            metrics = train_one_epoch(
                det_lora=det_lora,
                dataloader=train_loader,
                optimizer=optimizer,
                epoch=epoch,
                use_orthogonal_loss=False,  # Not needed - adapters are independent (unload approach)
                merge_consistency_weight=merge_consistency_weight,
                shared_drift_weight=shared_drift_weight if use_shared_adapter else 0.0,
            )
            val_metrics = evaluate_loss(det_lora, val_loader)
            metrics.update(_prefix_metrics(val_metrics, "val_"))
            if metrics_eval_every > 0 and (epoch % metrics_eval_every == 0 or epoch == epochs):
                det_metrics = evaluate_detection_metrics(det_lora, val_loader, [class_name])
                metrics.update(_prefix_metrics(det_metrics, "val_"))
            scheduler.step()
            task_history.append(metrics)

            if epoch % 5 == 0 or epoch == epochs:
                loss_terms = " | ".join(
                    f"{label}: {metrics[key]:.4f}"
                    for key, label in (
                        ("cls_loss", "Cls"),
                        ("bbox_loss", "BBox"),
                        ("giou_loss", "GIoU"),
                    )
                    if key in metrics
                )
                val_terms = " | ".join(
                    f"{label}: {metrics[key]:.4f}"
                    for key, label in (
                        ("val_cls_loss", "ValCls"),
                        ("val_bbox_loss", "ValBBox"),
                        ("val_giou_loss", "ValGIoU"),
                    )
                    if key in metrics
                )
                det_summary = ""
                if "val_mAP@0.5" in metrics:
                    det_summary = (
                        f" | mAP50: {metrics['val_mAP@0.5']:.4f}"
                        f" | mAP75: {metrics.get('val_mAP@0.75', 0.0):.4f}"
                        f" | Prec: {metrics.get('val_Precision@0.5', 0.0):.4f}"
                        f" | Rec: {metrics.get('val_Recall@0.5', 0.0):.4f}"
                        f" | F1: {metrics.get('val_F1@0.5', 0.0):.4f}"
                    )
                print(
                    f"  Epoch {epoch}/{epochs} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Orth: {metrics['orth_loss']:.4f} | "
                    f"Stab: {metrics.get('stability_loss', 0.0):.4f} | "
                    f"Val: {metrics['val_loss']:.4f}"
                    + (f" | {loss_terms}" if loss_terms else "")
                    + (f" | {val_terms}" if val_terms else "")
                    + det_summary
                )

        # 5. Finalize: save LoRA adapter, restore base, freeze head neuron
        det_lora.finalize_task(save_dir=str(experiment_dir / "adapters"))

        # 6. Refresh calibration without replay:
        # - current class gets positive samples from its own validation set
        # - previous classes treat the new class validation images as negatives
        # - the current class also treats all previous-class validation images as
        #   negatives so calibration is symmetric across seen classes
        seen_classes = [cls for cls in classes[: task_idx + 1] if cls in det_lora.trained_classes]
        if not synthetic:
            previous_classes = [
                previous_class
                for previous_class in det_lora.trained_classes
                if previous_class != class_name
            ]
            previous_val_loaders = [
                _make_raw_dataloader(
                    raw_dir=data_dir,
                    class_filter=previous_class,
                    split="val",
                    class_id_offset=detector.base_num_classes,
                    img_size=resolution,
                    seed=seed,
                    batch_size=batch_size,
                    max_samples=max_samples,
                    class_id_mapping=_det_lora_class_id_mapping(det_lora, [previous_class]),
                )
                for previous_class in previous_classes
            ]
            refresh_adapter_calibration(
                det_lora,
                class_name,
                positive_dataloader=val_loader,
                negative_dataloaders=previous_val_loaders,
            )
            for previous_class in previous_classes:
                refresh_adapter_calibration(
                    det_lora,
                    previous_class,
                    negative_dataloaders=[val_loader],
                )

            seen_val_loader = None
            if enable_shared_quality_calibrator or use_adapter_arbitration:
                seen_val_loader = _make_raw_dataloader(
                    raw_dir=data_dir,
                    class_filter=seen_classes,
                    split="val",
                    class_id_offset=detector.base_num_classes,
                    img_size=resolution,
                    seed=seed,
                    batch_size=batch_size,
                    max_samples=max_samples,
                    class_id_mapping=_det_lora_class_id_mapping(det_lora, seen_classes),
                )
            if enable_shared_quality_calibrator and seen_val_loader is not None:
                refresh_shared_quality_calibrator(
                    det_lora,
                    seen_val_loader,
                    seen_classes,
                )
            else:
                det_lora.set_shared_quality_calibrator({})
            is_final_task = task_idx == len(classes) - 1
            # Disabled for full L40 suite stability: refresh_adapter_arbitration can hang after
            # final-class evaluation. Post-hoc arbitration should be evaluated separately.
            det_lora.set_adapter_arbitration_state({})

        # 7. Save checkpoint (post-merge, plain model + calibration state)
        checkpoint_dir = experiment_dir / f"checkpoints/task_{task_idx}_{class_name}"
        det_lora.save_all(str(checkpoint_dir))
        print(f"  Checkpoint saved: {checkpoint_dir}")

        # 8. Update progress
        completed_tasks.add(class_name)
        progress["completed_tasks"] = list(completed_tasks)
        progress["current_task"] = class_name
        _save_progress(experiment_dir, progress)

        # 9. Store results
        all_results["tasks"][class_name] = {
            "task_idx": task_idx,
            "trainable_params": trainable_count,
            "train_samples": len(train_dataset),
            "hard_negative_counts": hard_negative_counts,
            "history": task_history,
            "final_loss": task_history[-1]["loss"],
        }

        # 10. Evaluate on all seen classes after this task using both:
        # - matched class-only slices for retention
        # - the mixed seen-class set for joint continual-learning behavior
        if not synthetic:
            eval_dataset = load_dataset_from_raw(
                raw_dir=data_dir,
                class_filter=seen_classes,
                split="test",
                class_id_offset=detector.base_num_classes,
                img_size=resolution,
                seed=seed,
                class_id_mapping=_det_lora_class_id_mapping(det_lora, seen_classes),
            )
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
            mixed_metrics = mixed_evaluator.evaluate_det_lora_joint(
                dataloader=eval_loader,
                class_names=seen_classes,
                task_idx=task_idx,
                include_curves=True,
            )
            matched_metrics = _evaluate_matched_seen_classes(
                det_lora=det_lora,
                seen_classes=seen_classes,
                data_dir=data_dir,
                batch_size=batch_size,
                resolution=resolution,
                class_id_offset=detector.base_num_classes,
                seed=seed,
                include_curves=True,
            )
            matched_evaluator.history[task_idx] = {
                "metrics": matched_metrics,
                "class_names": seen_classes,
            }
            all_results["evaluation_after_task"][class_name] = mixed_metrics
            all_results["mixed_evaluation_after_task"][class_name] = mixed_metrics
            all_results["matched_evaluation_after_task"][class_name] = matched_metrics
            all_results.setdefault("calibration", {})[class_name] = {
                trained_class: det_lora.get_calibrator(trained_class)
                for trained_class in det_lora.trained_classes
            }
            all_results.setdefault("shared_quality_calibrator", {})[
                class_name
            ] = det_lora.shared_quality_calibrator

        # Save intermediate results
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\nCompleted task '{class_name}' (loss: {task_history[-1]['loss']:.4f})")

    if mixed_evaluator.history:
        latest_task_idx = max(mixed_evaluator.history.keys())
        all_results["mixed_final_evaluation"] = mixed_evaluator.history[latest_task_idx]["metrics"]
        # Keep the historical key for downstream consumers that expect a single final eval.
        all_results["final_evaluation"] = all_results["mixed_final_evaluation"]
    if matched_evaluator.history:
        latest_task_idx = max(matched_evaluator.history.keys())
        all_results["matched_final_evaluation"] = matched_evaluator.history[latest_task_idx][
            "metrics"
        ]
    all_results["matched_forgetting"] = _to_builtin(matched_evaluator.compute_forgetting())
    all_results["matched_forward_transfer"] = _to_builtin(
        matched_evaluator.compute_forward_transfer()
    )
    all_results["mixed_forgetting"] = _to_builtin(mixed_evaluator.compute_forgetting())
    all_results["mixed_forward_transfer"] = _to_builtin(mixed_evaluator.compute_forward_transfer())
    all_results["mixed_confusion_summary"] = summarize_mixed_confusion(
        matched_evaluator.history,
        mixed_evaluator.history,
    )
    all_results["output_dir"] = str(experiment_dir)
    all_results["config"].setdefault("run_metadata", {})
    all_results["config"]["run_metadata"]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save final model + results
    det_lora.save_all(str(experiment_dir / "final"))

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    evaluation_payload = {
        "matched_history": _serialize_history(matched_evaluator.history),
        "matched_forgetting": _to_builtin(matched_evaluator.compute_forgetting()),
        "matched_forward_transfer": _to_builtin(matched_evaluator.compute_forward_transfer()),
        "mixed_history": _serialize_history(mixed_evaluator.history),
        "mixed_forgetting": _to_builtin(mixed_evaluator.compute_forgetting()),
        "mixed_forward_transfer": _to_builtin(mixed_evaluator.compute_forward_transfer()),
        "mixed_confusion_summary": _to_builtin(
            summarize_mixed_confusion(matched_evaluator.history, mixed_evaluator.history)
        ),
        # Backward-compatible aliases for older tooling.
        "history": _serialize_history(mixed_evaluator.history),
        "forgetting": _to_builtin(matched_evaluator.compute_forgetting()),
    }
    with open(experiment_dir / "evaluation.json", "w") as f:
        json.dump(evaluation_payload, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Tasks: {len(classes)}")
    print(f"Adapters: {len(det_lora.adapters)}")
    print(f"Results: {results_path}")
    print(det_lora.summary())

    return all_results


def main():
    base_defaults = {
        "epochs": 30,
        "batch_size": 4,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "lora_rank": 8,
        "lora_alpha": 16,
        "metrics_eval_every": 5,
    }
    parser = argparse.ArgumentParser(description="Det-LoRA Continual Learning Experiment")
    parser.add_argument(
        "--classes",
        nargs="+",
        type=str,
        default=[
            "military_tank",
            "military_truck",
            "military_aircraft",
            "military_helicopter",
            "civilian_car",
            "civilian_aircraft",
        ],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["medium"],
        help="RF-DETR variant(s): nano, small, base, medium, large, or 'all'",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Variant-specific hyperparameter preset, e.g. l40_final",
    )
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--resume", type=str, default=None, help="Resume from experiment dir")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metrics_eval_every",
        type=int,
        default=None,
        help="Evaluate validation detection metrics every N epochs (0 disables)",
    )
    parser.add_argument(
        "--enable_shared_quality_calibrator",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--disable_shared_quality_calibrator",
        action="store_true",
        help="Disable the shared quality/objectness calibrator (enabled by default)",
    )
    parser.add_argument(
        "--use_adapter_arbitration",
        action="store_true",
        help="Fit compact adapter arbitration on validation data for joint Det-LoRA inference",
    )
    parser.add_argument(
        "--disable_hard_negatives",
        action="store_true",
        help="Disable empty-target hard negatives from already learned classes",
    )
    parser.add_argument(
        "--symmetric_hard_negatives",
        action="store_true",
        help="Use ALL other classes (incl. later, confusable ones) as empty-target "
        "hard negatives so each adapter learns a discriminative boundary.",
    )
    parser.add_argument(
        "--lora_targets",
        choices=tuple(LORA_TARGET_PRESETS),
        default="default",
        help="Adapter footprint: default (thesis main suite) or localization presets",
    )
    parser.add_argument(
        "--lora_dora",
        action="store_true",
        help="Use weight-decomposed LoRA (DoRA) instead of vanilla LoRA",
    )
    parser.add_argument(
        "--merge_consistency_weight",
        type=float,
        default=0.0,
        help="DuET-style sign-consistency penalty vs. previous adapters (enables merging)",
    )
    parser.add_argument(
        "--cl_lora",
        action="store_true",
        help="CL-LoRA mode: task-shared adapter (fixed orthogonal down-projection) "
        "in addition to the per-class adapters",
    )
    parser.add_argument(
        "--shared_drift_weight",
        type=float,
        default=1.0,
        help="Importance-weighted L2 anchor protecting the shared adapter (CL-LoRA mode)",
    )
    args = parser.parse_args()

    models = expand_model_variants(args.model)

    for model_idx, model_variant in enumerate(models):
        resolved = resolve_variant_settings(
            variant=model_variant,
            preset_name=args.preset,
            base_defaults=base_defaults,
            overrides={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "metrics_eval_every": args.metrics_eval_every,
            },
        )
        if len(models) > 1:
            print(f"\n{'#'*60}")
            print(f"# MODEL {model_idx + 1}/{len(models)}: RF-DETR {model_variant}")
            print(f"{'#'*60}")
        if args.preset:
            print(
                "[Preset] "
                f"{args.preset} -> batch_size={resolved['batch_size']}, "
                f"lr={resolved['lr']}, lora_rank={resolved['lora_rank']}, "
                f"lora_alpha={resolved['lora_alpha']}, epochs={resolved['epochs']}"
            )

        run_continual_experiment(
            classes=args.classes,
            epochs=int(resolved["epochs"]),
            batch_size=int(resolved["batch_size"]),
            lr=float(resolved["lr"]),
            weight_decay=float(resolved["weight_decay"]),
            lora_rank=int(resolved["lora_rank"]),
            lora_alpha=int(resolved["lora_alpha"]),
            model_variant=model_variant,
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            resume_dir=args.resume,
            synthetic=args.synthetic,
            max_samples=args.max_samples,
            seed=args.seed,
            metrics_eval_every=int(resolved["metrics_eval_every"]),
            enable_shared_quality_calibrator=not args.disable_shared_quality_calibrator,
            use_adapter_arbitration=args.use_adapter_arbitration,
            use_hard_negatives=not args.disable_hard_negatives,
            symmetric_hard_negatives=args.symmetric_hard_negatives,
            lora_target_preset=args.lora_targets,
            use_dora=args.lora_dora,
            merge_consistency_weight=args.merge_consistency_weight,
            use_shared_adapter=args.cl_lora,
            shared_drift_weight=args.shared_drift_weight,
            preset_name=args.preset,
        )


if __name__ == "__main__":
    main()
