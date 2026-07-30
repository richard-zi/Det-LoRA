"""
Det-LoRA Training Script
==========================

Train a single Det-LoRA adapter for a new class or extend an existing one.

Usage:
    # Add new class
    uv run python -m det_lora.train --class_name military_tank --epochs 50

    # Extend existing class with new data
    uv run python -m det_lora.train --class_name military_tank --extend --epochs 20

    # Quick test with synthetic data
    uv run python -m det_lora.train --class_name test --epochs 5 --synthetic
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.evaluator import (
    ContinualEvaluator,
    _select_adapter_versions,
    refresh_adapter_arbitration,
    refresh_adapter_calibration,
)
from det_lora.model.det_lora import DetLoRA
from det_lora.model.detector import LORA_TARGET_PRESETS, RFDETRDetector
from det_lora.utils import collect_runtime_metadata, resolve_variant_settings, set_global_seed


class SyntheticDetectionDataset(Dataset):
    """Synthetic dataset for testing the pipeline without real data."""

    def __init__(self, num_samples: int = 100, num_classes: int = 81, img_size: int = 640):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        pixel_values = torch.randn(3, self.img_size, self.img_size)

        # Random number of objects (1-5)
        num_objects = torch.randint(1, 6, (1,)).item()
        boxes = torch.rand(num_objects, 4) * 0.5 + 0.1  # cxcywh in [0.1, 0.6]
        boxes[:, 2:] = boxes[:, 2:].clamp(0.05, 0.5)  # Ensure valid sizes
        class_labels = torch.randint(0, self.num_classes, (num_objects,))

        return {
            "pixel_values": pixel_values,
            "labels": {
                "class_labels": class_labels,
                "boxes": boxes,
            },
            "sample_id": idx,
        }


class EmptyTargetDataset(Dataset):
    """Use known non-target-class images as hard negatives for one adapter."""

    def __init__(self, wrapped: Dataset):
        self.wrapped = wrapped

    def __len__(self) -> int:
        return len(self.wrapped)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = dict(self.wrapped[idx])
        sample["labels"] = {
            "labels": torch.empty((0,), dtype=torch.long),
            "boxes": torch.empty((0, 4), dtype=torch.float32),
        }
        return sample


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate for variable-length targets."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    sample_ids = torch.tensor(
        [int(item.get("sample_id", -1)) for item in batch],
        dtype=torch.long,
    )
    return {"pixel_values": pixel_values, "labels": labels, "sample_ids": sample_ids}


def _extract_loss_components(loss_dict: Optional[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    """Keep the most useful RF-DETR loss components for logging."""
    if not loss_dict:
        return {}

    component_map = {
        "loss_ce": "cls_loss",
        "loss_bbox": "bbox_loss",
        "loss_giou": "giou_loss",
        "cardinality_error": "cardinality_error",
    }
    components = {}
    for source_key, target_key in component_map.items():
        if source_key in loss_dict:
            components[target_key] = float(loss_dict[source_key].detach().item())
    return components


def _prefix_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    """Prefix metric names when flattening train/val/eval results into history."""
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _compute_teacher_anchor_loss(
    student_logits: torch.Tensor,
    student_boxes: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_boxes: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Distill the current extend step toward the pre-update teacher outputs."""
    cls_loss = F.mse_loss(student_logits.sigmoid(), teacher_logits.sigmoid())
    box_loss = F.l1_loss(student_boxes, teacher_boxes)
    total = cls_loss + box_loss
    return {
        "teacher_anchor_loss": total,
        "teacher_cls_loss": cls_loss,
        "teacher_box_loss": box_loss,
    }


def _get_teacher_class_ids(det_lora: DetLoRA) -> List[int]:
    """Track background + all incrementally learned classes during extend distillation."""
    detector = getattr(det_lora, "detector", None)
    trained_classes = getattr(det_lora, "trained_classes", [])
    if detector is None or not hasattr(detector, "base_num_classes") or not trained_classes:
        return []

    teacher_class_ids = [detector.base_num_classes - 1]
    teacher_class_ids.extend(det_lora.get_class_id(class_name) for class_name in trained_classes)
    return sorted(set(teacher_class_ids))


def _det_lora_class_id_mapping(det_lora: DetLoRA, class_names: List[str]) -> Dict[str, int]:
    return {class_name: det_lora.get_class_id(class_name) for class_name in class_names}


def _make_calibration_loader(
    det_lora: DetLoRA,
    raw_dir: str,
    class_name: str,
    detector: RFDETRDetector,
    batch_size: int,
    seed: int,
    max_samples: Optional[int],
) -> Optional[DataLoader]:
    dataset = load_dataset_from_raw(
        raw_dir=raw_dir,
        class_filter=class_name,
        split="val",
        class_id_offset=detector.base_num_classes,
        img_size=detector.resolution,
        seed=seed,
        max_samples=max_samples,
        class_id_mapping=_det_lora_class_id_mapping(det_lora, [class_name]),
    )
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )


def _make_adapter_hard_negatives(
    det_lora: DetLoRA,
    raw_dir: str,
    target_class: str,
    detector: RFDETRDetector,
    img_size: int,
    seed: int,
    max_samples_per_class: int,
    negative_classes: Optional[List[str]] = None,
) -> tuple[List[Dataset], Dict[str, int]]:
    # Default: only already-learned classes (strict incremental). When
    # `negative_classes` is given, use that explicit pool instead -- this enables
    # a symmetric/retroactive hardening pass in which an adapter also sees
    # later-learned, confusable classes as negatives.
    source_classes = negative_classes if negative_classes is not None else det_lora.trained_classes
    negative_datasets: List[Dataset] = []
    negative_counts: Dict[str, int] = {}
    for negative_class in source_classes:
        if negative_class == target_class:
            continue
        # Negatives become empty-target samples, so the concrete class id is
        # irrelevant. Fall back to a dummy id for not-yet-registered (future)
        # classes, which enables symmetric/retroactive hard negatives.
        try:
            class_id_mapping = _det_lora_class_id_mapping(det_lora, [negative_class])
        except ValueError:
            class_id_mapping = {negative_class: detector.base_num_classes}
        dataset = load_dataset_from_raw(
            raw_dir=raw_dir,
            class_filter=negative_class,
            split="train",
            class_id_offset=detector.base_num_classes,
            img_size=img_size,
            seed=seed,
            max_samples=max_samples_per_class,
            class_id_mapping=class_id_mapping,
        )
        if len(dataset) == 0:
            continue
        negative_datasets.append(EmptyTargetDataset(dataset))
        negative_counts[negative_class] = len(dataset)
    return negative_datasets, negative_counts


def _select_distillation_logits(
    logits: torch.Tensor,
    class_ids: Optional[List[int]],
) -> torch.Tensor:
    """Keep only the generic incremental class subset relevant for teacher anchoring."""
    if not class_ids:
        return logits

    valid_ids = [class_id for class_id in class_ids if 0 <= class_id < logits.shape[-1]]
    if not valid_ids:
        return logits[..., :0]

    index = torch.tensor(valid_ids, dtype=torch.long, device=logits.device)
    return torch.index_select(logits, dim=-1, index=index)


def _extract_distillation_targets(
    outputs: Dict[str, Any],
    class_ids: Optional[List[int]],
) -> Dict[str, Any]:
    """Extract a compact multi-scale teacher target from RF-DETR outputs."""
    targets: Dict[str, Any] = {
        "pred_logits": _select_distillation_logits(outputs["pred_logits"], class_ids),
        "pred_boxes": outputs["pred_boxes"],
    }

    enc_outputs = outputs.get("enc_outputs")
    if enc_outputs is not None:
        targets["enc_outputs"] = {
            "pred_logits": _select_distillation_logits(enc_outputs["pred_logits"], class_ids),
            "pred_boxes": enc_outputs["pred_boxes"],
        }

    aux_outputs = outputs.get("aux_outputs") or []
    if aux_outputs:
        targets["aux_outputs"] = [
            {
                "pred_logits": _select_distillation_logits(aux_output["pred_logits"], class_ids),
                "pred_boxes": aux_output["pred_boxes"],
            }
            for aux_output in aux_outputs
        ]

    return targets


def _compute_multiscale_teacher_anchor_loss(
    student_outputs: Dict[str, Any],
    teacher_outputs: Dict[str, Any],
    encoder_weight: float = 0.5,
    aux_weight: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """Anchor final, encoder, and decoder auxiliary outputs to the pre-update teacher."""
    final_losses = _compute_teacher_anchor_loss(
        student_outputs["pred_logits"],
        student_outputs["pred_boxes"],
        teacher_outputs["pred_logits"],
        teacher_outputs["pred_boxes"],
    )
    total_loss = final_losses["teacher_anchor_loss"]
    encoder_loss = torch.tensor(0.0, device=student_outputs["pred_logits"].device)
    aux_loss = torch.tensor(0.0, device=student_outputs["pred_logits"].device)

    student_enc = student_outputs.get("enc_outputs")
    teacher_enc = teacher_outputs.get("enc_outputs")
    if student_enc is not None and teacher_enc is not None:
        encoder_losses = _compute_teacher_anchor_loss(
            student_enc["pred_logits"],
            student_enc["pred_boxes"],
            teacher_enc["pred_logits"],
            teacher_enc["pred_boxes"],
        )
        encoder_loss = encoder_losses["teacher_anchor_loss"]
        total_loss = total_loss + encoder_weight * encoder_loss

    student_aux = student_outputs.get("aux_outputs") or []
    teacher_aux = teacher_outputs.get("aux_outputs") or []
    aux_levels = min(len(student_aux), len(teacher_aux))
    if aux_levels > 0:
        aux_terms = []
        for idx in range(aux_levels):
            aux_terms.append(
                _compute_teacher_anchor_loss(
                    student_aux[idx]["pred_logits"],
                    student_aux[idx]["pred_boxes"],
                    teacher_aux[idx]["pred_logits"],
                    teacher_aux[idx]["pred_boxes"],
                )["teacher_anchor_loss"]
            )
        aux_loss = torch.stack(aux_terms).mean()
        total_loss = total_loss + aux_weight * aux_loss

    return {
        "teacher_anchor_loss": total_loss,
        "teacher_final_loss": final_losses["teacher_anchor_loss"],
        "teacher_cls_loss": final_losses["teacher_cls_loss"],
        "teacher_box_loss": final_losses["teacher_box_loss"],
        "teacher_encoder_loss": encoder_loss,
        "teacher_aux_loss": aux_loss,
    }


@torch.no_grad()
def build_teacher_cache(
    det_lora: DetLoRA,
    dataloader: DataLoader,
    class_ids: Optional[List[int]] = None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Cache pre-update predictions for each training sample during `extend`."""
    det_lora.model.eval()
    device = det_lora.device
    teacher_cache: Dict[int, Dict[str, torch.Tensor]] = {}

    for batch in dataloader:
        sample_ids = batch.get("sample_ids")
        if sample_ids is None:
            continue

        pixel_values = batch["pixel_values"].to(device)
        outputs = det_lora.forward(pixel_values=pixel_values)
        distilled_outputs = _extract_distillation_targets(outputs, class_ids)

        for batch_idx, sample_id in enumerate(sample_ids.tolist()):
            teacher_entry: Dict[str, Any] = {
                "pred_logits": distilled_outputs["pred_logits"][batch_idx]
                .detach()
                .cpu()
                .to(torch.float32)
                .clone(),
                "pred_boxes": distilled_outputs["pred_boxes"][batch_idx]
                .detach()
                .cpu()
                .to(torch.float32)
                .clone(),
            }
            enc_outputs = distilled_outputs.get("enc_outputs")
            if enc_outputs is not None:
                teacher_entry["enc_outputs"] = {
                    "pred_logits": enc_outputs["pred_logits"][batch_idx]
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .clone(),
                    "pred_boxes": enc_outputs["pred_boxes"][batch_idx]
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .clone(),
                }
            aux_outputs = distilled_outputs.get("aux_outputs") or []
            if aux_outputs:
                teacher_entry["aux_outputs"] = [
                    {
                        "pred_logits": aux_output["pred_logits"][batch_idx]
                        .detach()
                        .cpu()
                        .to(torch.float32)
                        .clone(),
                        "pred_boxes": aux_output["pred_boxes"][batch_idx]
                        .detach()
                        .cpu()
                        .to(torch.float32)
                        .clone(),
                    }
                    for aux_output in aux_outputs
                ]
            teacher_cache[int(sample_id)] = teacher_entry

    return teacher_cache


def train_one_epoch(
    det_lora: DetLoRA,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    use_orthogonal_loss: bool = True,
    stability_loss_weight: float = 0.0,
    merge_consistency_weight: float = 0.0,
    shared_drift_weight: float = 0.0,
    teacher_cache: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    teacher_anchor_weight: float = 0.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    det_lora.model.train()
    device = det_lora.device

    total_loss = 0.0
    total_orth_loss = 0.0
    total_stability_loss = 0.0
    total_teacher_anchor_loss = 0.0
    total_teacher_final_loss = 0.0
    total_teacher_cls_loss = 0.0
    total_teacher_box_loss = 0.0
    total_teacher_encoder_loss = 0.0
    total_teacher_aux_loss = 0.0
    num_batches = 0
    component_totals: Dict[str, float] = {}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]

        # Move targets to device (RF-DETR uses 'labels' key, not 'class_labels')
        device_targets = []
        for label in labels:
            lbl_key = "labels" if "labels" in label else "class_labels"
            device_targets.append(
                {
                    "labels": label[lbl_key].to(device),
                    "boxes": label["boxes"].to(device),
                }
            )

        # Forward pass
        outputs = det_lora.forward(pixel_values=pixel_values, targets=device_targets)
        loss = outputs["loss"]
        loss_components = _extract_loss_components(outputs.get("loss_dict"))

        # Orthogonal regularization
        orth_loss = torch.tensor(0.0, device=device)
        if use_orthogonal_loss:
            orth_loss = det_lora.orthogonal_loss()
            loss = loss + orth_loss

        stability_loss = torch.tensor(0.0, device=device)
        if stability_loss_weight > 0:
            stability_loss = det_lora.stability_loss() * stability_loss_weight
            loss = loss + stability_loss

        if merge_consistency_weight > 0:
            loss = loss + det_lora.merge_consistency_loss() * merge_consistency_weight

        if shared_drift_weight > 0:
            loss = loss + det_lora.shared_drift_loss() * shared_drift_weight

        teacher_anchor_loss = torch.tensor(0.0, device=device)
        teacher_cls_loss = torch.tensor(0.0, device=device)
        teacher_box_loss = torch.tensor(0.0, device=device)
        sample_ids = batch.get("sample_ids")
        if teacher_cache and teacher_anchor_weight > 0 and sample_ids is not None:
            teacher_entries = [
                teacher_cache.get(int(sample_id)) for sample_id in sample_ids.tolist()
            ]
            if all(entry is not None for entry in teacher_entries):
                was_training = det_lora.model.training
                det_lora.model.eval()
                anchor_outputs = _extract_distillation_targets(
                    det_lora.forward(pixel_values=pixel_values),
                    _get_teacher_class_ids(det_lora),
                )
                if was_training:
                    det_lora.model.train()
                teacher_outputs: Dict[str, Any] = {
                    "pred_logits": torch.stack(
                        [entry["pred_logits"] for entry in teacher_entries if entry is not None]
                    ).to(device=device, dtype=anchor_outputs["pred_logits"].dtype),
                    "pred_boxes": torch.stack(
                        [entry["pred_boxes"] for entry in teacher_entries if entry is not None]
                    ).to(device=device, dtype=anchor_outputs["pred_boxes"].dtype),
                }
                if anchor_outputs.get("enc_outputs") is not None and all(
                    "enc_outputs" in entry for entry in teacher_entries if entry is not None
                ):
                    teacher_outputs["enc_outputs"] = {
                        "pred_logits": torch.stack(
                            [
                                entry["enc_outputs"]["pred_logits"]
                                for entry in teacher_entries
                                if entry is not None
                            ]
                        ).to(
                            device=device,
                            dtype=anchor_outputs["enc_outputs"]["pred_logits"].dtype,
                        ),
                        "pred_boxes": torch.stack(
                            [
                                entry["enc_outputs"]["pred_boxes"]
                                for entry in teacher_entries
                                if entry is not None
                            ]
                        ).to(
                            device=device,
                            dtype=anchor_outputs["enc_outputs"]["pred_boxes"].dtype,
                        ),
                    }
                teacher_aux_levels = min(
                    len(anchor_outputs.get("aux_outputs") or []),
                    min(
                        len(entry.get("aux_outputs") or [])
                        for entry in teacher_entries
                        if entry is not None
                    ),
                )
                if teacher_aux_levels > 0:
                    teacher_outputs["aux_outputs"] = []
                    for idx in range(teacher_aux_levels):
                        teacher_outputs["aux_outputs"].append(
                            {
                                "pred_logits": torch.stack(
                                    [
                                        entry["aux_outputs"][idx]["pred_logits"]
                                        for entry in teacher_entries
                                        if entry is not None
                                    ]
                                ).to(
                                    device=device,
                                    dtype=anchor_outputs["aux_outputs"][idx]["pred_logits"].dtype,
                                ),
                                "pred_boxes": torch.stack(
                                    [
                                        entry["aux_outputs"][idx]["pred_boxes"]
                                        for entry in teacher_entries
                                        if entry is not None
                                    ]
                                ).to(
                                    device=device,
                                    dtype=anchor_outputs["aux_outputs"][idx]["pred_boxes"].dtype,
                                ),
                            }
                        )
                teacher_losses = _compute_multiscale_teacher_anchor_loss(
                    anchor_outputs,
                    teacher_outputs,
                )
                teacher_anchor_loss = teacher_losses["teacher_anchor_loss"]
                teacher_cls_loss = teacher_losses["teacher_cls_loss"]
                teacher_box_loss = teacher_losses["teacher_box_loss"]
                teacher_final_loss = teacher_losses["teacher_final_loss"]
                teacher_encoder_loss = teacher_losses["teacher_encoder_loss"]
                teacher_aux_loss = teacher_losses["teacher_aux_loss"]
                loss = loss + teacher_anchor_weight * teacher_anchor_loss
            else:
                teacher_final_loss = torch.tensor(0.0, device=device)
                teacher_encoder_loss = torch.tensor(0.0, device=device)
                teacher_aux_loss = torch.tensor(0.0, device=device)
        else:
            teacher_final_loss = torch.tensor(0.0, device=device)
            teacher_encoder_loss = torch.tensor(0.0, device=device)
            teacher_aux_loss = torch.tensor(0.0, device=device)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(det_lora.get_trainable_params(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_orth_loss += orth_loss.item()
        total_stability_loss += stability_loss.item()
        total_teacher_anchor_loss += teacher_anchor_loss.item()
        total_teacher_final_loss += teacher_final_loss.item()
        total_teacher_cls_loss += teacher_cls_loss.item()
        total_teacher_box_loss += teacher_box_loss.item()
        total_teacher_encoder_loss += teacher_encoder_loss.item()
        total_teacher_aux_loss += teacher_aux_loss.item()
        for key, value in loss_components.items():
            component_totals[key] = component_totals.get(key, 0.0) + value
        num_batches += 1

        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "orth": f"{orth_loss.item():.4f}",
                "stab": f"{stability_loss.item():.4f}",
                "teach": f"{teacher_anchor_loss.item():.4f}",
            }
        )

    avg_loss = total_loss / max(num_batches, 1)
    avg_orth = total_orth_loss / max(num_batches, 1)
    avg_stability = total_stability_loss / max(num_batches, 1)
    metrics = {
        "loss": avg_loss,
        "orth_loss": avg_orth,
        "stability_loss": avg_stability,
        "teacher_anchor_loss": total_teacher_anchor_loss / max(num_batches, 1),
        "teacher_final_loss": total_teacher_final_loss / max(num_batches, 1),
        "teacher_cls_loss": total_teacher_cls_loss / max(num_batches, 1),
        "teacher_box_loss": total_teacher_box_loss / max(num_batches, 1),
        "teacher_encoder_loss": total_teacher_encoder_loss / max(num_batches, 1),
        "teacher_aux_loss": total_teacher_aux_loss / max(num_batches, 1),
    }
    for key, value in component_totals.items():
        metrics[key] = value / max(num_batches, 1)
    return metrics


@torch.no_grad()
def evaluate_loss(
    det_lora: DetLoRA,
    dataloader: DataLoader,
) -> Dict[str, float]:
    """Evaluate validation loss without updating the model."""
    det_lora.model.eval()
    device = det_lora.device

    total_loss = 0.0
    num_batches = 0
    component_totals: Dict[str, float] = {}

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]

        device_targets = []
        for label in labels:
            lbl_key = "labels" if "labels" in label else "class_labels"
            device_targets.append(
                {
                    "labels": label[lbl_key].to(device),
                    "boxes": label["boxes"].to(device),
                }
            )

        outputs = det_lora.forward(pixel_values=pixel_values, targets=device_targets)
        total_loss += outputs["loss"].item()
        loss_components = _extract_loss_components(outputs.get("loss_dict"))
        for key, value in loss_components.items():
            component_totals[key] = component_totals.get(key, 0.0) + value
        num_batches += 1

    metrics = {"loss": total_loss / max(num_batches, 1)}
    for key, value in component_totals.items():
        metrics[key] = value / max(num_batches, 1)
    return metrics


@torch.no_grad()
def evaluate_detection_metrics(
    det_lora: DetLoRA,
    dataloader: DataLoader,
    class_names: List[str],
    include_curves: bool = False,
) -> Dict[str, Any]:
    """Evaluate detection metrics for the currently active Det-LoRA model state."""
    evaluator = ContinualEvaluator(det_lora)
    class_ids = [det_lora.get_class_id(class_name) for class_name in class_names]
    return evaluator.evaluate_standard_detector(
        detector=det_lora.detector,
        dataloader=dataloader,
        class_names=class_names,
        class_ids=class_ids,
        include_curves=include_curves,
    )


def train_adapter(
    class_name: str,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_target_preset: str = "default",
    use_dora: bool = False,
    model_variant: str = "medium",
    data_dir: str = "data/raw",
    test_data_dir: Optional[str] = None,
    arbitration_data_dir: Optional[str] = None,
    save_dir: str = "experiments",
    extend: bool = False,
    synthetic: bool = False,
    max_samples: Optional[int] = None,
    sample_offset: int = 0,
    load_dir: Optional[str] = None,
    seed: int = 42,
    extend_strategy: str = "warm_start",
    version_selection_strategy: str = "anchor_latest",
    metrics_eval_every: int = 1,
    stability_loss_weight: float = 1e-5,
    teacher_anchor_weight: float = 0.05,
    use_hard_negatives: bool = True,
    hard_negative_classes: Optional[List[str]] = None,
    use_adapter_arbitration: bool = False,
    merge_consistency_weight: float = 0.0,
    shared_drift_weight: float = 0.0,
    preset_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Dict:
    """
    Train a Det-LoRA adapter for a single class.

    Args:
        class_name: Class to train
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_target_preset: Named adapter footprint (see LORA_TARGET_PRESETS)
        use_dora: Use weight-decomposed LoRA (DoRA) instead of vanilla LoRA
        model_variant: HuggingFace model name
        data_dir: Path to raw data
        test_data_dir: Separate raw data path for extend evaluation
        arbitration_data_dir: Raw data path for fitting joint-inference arbitration
        save_dir: Experiment output directory
        extend: If True, extend existing class (data-incremental)
        synthetic: Use synthetic data for testing
        max_samples: Limit training samples
        sample_offset: Skip filtered training samples before applying max_samples
        extend_strategy: "warm_start" continues the active adapter; "grow_freeze" adds a fresh version
        version_selection_strategy: Adapter-version selection for grow-freeze evaluation
        metrics_eval_every: Evaluate detection metrics on the validation split every N epochs
        stability_loss_weight: L2 anchor strength for extending an existing class
        teacher_anchor_weight: Distill toward the pre-update model during `extend`
        use_hard_negatives: Mix non-target seen-class images as empty-target negatives
        use_adapter_arbitration: Fit compact region-classifier arbitration for joint inference

    Returns:
        Training results dict
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(save_dir) / (experiment_name or f"train_{class_name}_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    evaluation_data_dir = test_data_dir or data_dir
    arbitration_raw_dir = arbitration_data_dir or data_dir
    if extend_strategy not in {"warm_start", "grow_freeze"}:
        raise ValueError("extend_strategy must be one of: warm_start, grow_freeze")
    if version_selection_strategy not in {"all", "anchor_latest", "latest"}:
        raise ValueError("version_selection_strategy must be one of: all, anchor_latest, latest")

    print(f"\n{'='*60}")
    print(f"Det-LoRA Training: {'Extending' if extend else 'New class'} '{class_name}'")
    print(f"{'='*60}")

    # Initialize detector + Det-LoRA
    detector = RFDETRDetector(variant=model_variant)
    det_lora = DetLoRA(
        detector=detector,
        default_rank=lora_rank,
        default_alpha=lora_alpha,
        lora_target_preset=lora_target_preset,
        use_dora=use_dora,
    )

    pre_extend_metrics = None
    pre_extend_target_metrics = None
    if load_dir:
        det_lora.load_all(load_dir)
        if extend and not synthetic and det_lora.trained_classes:
            pre_extend_classes = list(det_lora.trained_classes)
            resolution = detector.resolution
            pre_extend_dataset = load_dataset_from_raw(
                raw_dir=evaluation_data_dir,
                class_filter=pre_extend_classes,
                split="test",
                class_id_offset=detector.base_num_classes,
                img_size=resolution,
                seed=seed,
                class_id_mapping=_det_lora_class_id_mapping(det_lora, pre_extend_classes),
            )
            pre_extend_loader = DataLoader(
                pre_extend_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
            pre_extend_metrics = ContinualEvaluator(det_lora).evaluate_det_lora_joint(
                dataloader=pre_extend_loader,
                class_names=pre_extend_classes,
                include_curves=True,
                version_selection_by_class=(
                    {class_name: version_selection_strategy}
                    if extend_strategy == "grow_freeze"
                    else None
                ),
            )

            target_pre_extend_dataset = load_dataset_from_raw(
                raw_dir=evaluation_data_dir,
                class_filter=class_name,
                split="test",
                class_id_offset=detector.base_num_classes,
                img_size=resolution,
                seed=seed,
                class_id_mapping=_det_lora_class_id_mapping(det_lora, [class_name]),
            )
            target_pre_extend_loader = DataLoader(
                target_pre_extend_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
            pre_extend_evaluator = ContinualEvaluator(det_lora)
            if extend_strategy == "grow_freeze":
                pre_extend_target_metrics = pre_extend_evaluator.evaluate_det_lora_version_ensemble(
                    dataloader=target_pre_extend_loader,
                    class_name=class_name,
                    include_curves=True,
                    version_selection_strategy=version_selection_strategy,
                )
            else:
                pre_extend_target_metrics = pre_extend_evaluator.evaluate_det_lora_joint(
                    dataloader=target_pre_extend_loader,
                    class_names=[class_name],
                    include_curves=True,
                )

    # Add or extend class
    if extend:
        if not load_dir:
            raise ValueError("--extend requires --load_dir with a finalized Det-LoRA checkpoint")
        if extend_strategy == "grow_freeze":
            adapter_name = det_lora.extend_class_with_fresh_adapter(
                class_name,
                rank=lora_rank,
                alpha=lora_alpha,
            )
        else:
            adapter_name = det_lora.extend_class(class_name, rank=lora_rank, alpha=lora_alpha)
    else:
        adapter_name = det_lora.add_class(class_name, rank=lora_rank, alpha=lora_alpha)

    print(det_lora.summary())

    # Create dataset
    resolution = detector.resolution  # e.g. 576 for medium
    hard_negative_counts: Dict[str, int] = {}
    adapter_arbitration_state = None
    if synthetic:
        num_classes = detector.get_num_classes()
        train_dataset = SyntheticDetectionDataset(
            num_samples=200, num_classes=num_classes, img_size=resolution
        )
        val_dataset = SyntheticDetectionDataset(
            num_samples=50, num_classes=num_classes, img_size=resolution
        )
    else:
        class_id_offset = detector.base_num_classes  # 90 (COCO classes)
        train_dataset = load_dataset_from_raw(
            raw_dir=data_dir,
            class_filter=class_name,
            split="train",
            class_id_offset=class_id_offset,
            img_size=resolution,
            seed=seed,
            max_samples=max_samples,
            sample_offset=sample_offset,
            class_id_mapping=_det_lora_class_id_mapping(det_lora, [class_name]),
        )
        val_dataset = load_dataset_from_raw(
            raw_dir=data_dir,
            class_filter=class_name,
            split="val",
            class_id_offset=class_id_offset,
            img_size=resolution,
            seed=seed,
            class_id_mapping=_det_lora_class_id_mapping(det_lora, [class_name]),
        )
        if use_hard_negatives and len(train_dataset) > 0:
            negative_datasets, hard_negative_counts = _make_adapter_hard_negatives(
                det_lora=det_lora,
                raw_dir=data_dir,
                target_class=class_name,
                detector=detector,
                img_size=resolution,
                seed=seed,
                max_samples_per_class=len(train_dataset),
                negative_classes=hard_negative_classes,
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
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    teacher_cache = None
    teacher_class_ids: List[int] = []
    if extend and extend_strategy == "warm_start" and teacher_anchor_weight > 0:
        teacher_class_ids = _get_teacher_class_ids(det_lora)
        teacher_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        teacher_cache = build_teacher_cache(
            det_lora,
            teacher_loader,
            class_ids=teacher_class_ids,
        )

    print(f"\nDataset: {len(train_dataset)} train, {len(val_dataset)} val")
    if hard_negative_counts:
        print(f"Hard negatives: {hard_negative_counts}")
    if teacher_cache is not None:
        print(
            f"Teacher cache: {len(teacher_cache)} samples | "
            f"Anchor weight: {teacher_anchor_weight}"
        )

    # Optimizer: separate param groups so head gets weight_decay=0.
    # AdamW applies weight decay even when grad=0, which would drift
    # gradient-masked COCO neurons. Disabling wd on head prevents this.
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

    print(f"Trainable params: {sum(p.numel() for p in lora_params + head_params):,}")
    print(f"LR: {lr}, Epochs: {epochs}, Batch: {batch_size}")
    print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")

    # Training loop
    history = []
    best_val_loss = float("inf")
    best_val_map50 = float("-inf")
    best_epoch = 0
    best_dir = experiment_dir / "best"

    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            det_lora=det_lora,
            dataloader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            use_orthogonal_loss=len(det_lora.adapters) > 1,
            stability_loss_weight=(
                stability_loss_weight if extend and extend_strategy == "warm_start" else 0.0
            ),
            merge_consistency_weight=merge_consistency_weight,
            shared_drift_weight=shared_drift_weight,
            teacher_cache=teacher_cache,
            teacher_anchor_weight=(
                teacher_anchor_weight if extend and extend_strategy == "warm_start" else 0.0
            ),
        )

        scheduler.step()
        val_metrics = evaluate_loss(det_lora, val_loader)
        val_detection_metrics = None
        if (
            not synthetic
            and metrics_eval_every > 0
            and (epoch % metrics_eval_every == 0 or epoch == epochs)
        ):
            val_detection_metrics = evaluate_detection_metrics(
                det_lora=det_lora,
                dataloader=val_loader,
                class_names=[class_name],
            )

        # Log
        entry = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            **train_metrics,
            **_prefix_metrics(val_metrics, "val_"),
        }
        if val_detection_metrics is not None:
            entry.update(_prefix_metrics(val_detection_metrics, "val_"))
        history.append(entry)

        train_loss_terms = " | ".join(
            f"{label}: {train_metrics[key]:.4f}"
            for key, label in (
                ("cls_loss", "Cls"),
                ("bbox_loss", "BBox"),
                ("giou_loss", "GIoU"),
            )
            if key in train_metrics
        )
        val_loss_terms = " | ".join(
            f"{label}: {val_metrics[key]:.4f}"
            for key, label in (
                ("cls_loss", "ValCls"),
                ("bbox_loss", "ValBBox"),
                ("giou_loss", "ValGIoU"),
            )
            if key in val_metrics
        )
        detection_summary = ""
        if val_detection_metrics is not None:
            detection_summary = (
                f" | mAP50: {val_detection_metrics['mAP@0.5']:.4f}"
                f" | mAP75: {val_detection_metrics['mAP@0.75']:.4f}"
                f" | Prec: {val_detection_metrics['Precision@0.5']:.4f}"
                f" | Rec: {val_detection_metrics['Recall@0.5']:.4f}"
                f" | F1: {val_detection_metrics['F1@0.5']:.4f}"
            )
        print(
            f"  Epoch {epoch}/{epochs} | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"Orth: {train_metrics['orth_loss']:.4f} | "
            f"Stab: {train_metrics['stability_loss']:.4f} | "
            f"Teach: {train_metrics['teacher_anchor_loss']:.4f} | "
            f"Val: {val_metrics['loss']:.4f}"
            + (f" | {train_loss_terms}" if train_loss_terms else "")
            + (f" | {val_loss_terms}" if val_loss_terms else "")
            + detection_summary
            + f" | LR: {entry['lr']:.6f}"
        )

        current_val_map50 = (
            float(val_detection_metrics["mAP@0.5"])
            if val_detection_metrics is not None
            else float("-inf")
        )
        is_better = False
        if current_val_map50 > best_val_map50:
            is_better = True
        elif current_val_map50 == best_val_map50 and val_metrics["loss"] < best_val_loss:
            is_better = True
        elif best_val_map50 == float("-inf") and val_metrics["loss"] < best_val_loss:
            is_better = True

        if is_better:
            best_val_loss = val_metrics["loss"]
            best_val_map50 = max(best_val_map50, current_val_map50)
            best_epoch = epoch
            det_lora.save_all(str(best_dir))

    # Restore the best validation checkpoint, finalize the adapter, then save
    # a clean reusable checkpoint with the adapter registered in the registry.
    det_lora.load_all(str(best_dir))
    det_lora.finalize_task(save_dir=str(experiment_dir / "adapters"))
    if not synthetic:
        previous_classes = [
            seen_class for seen_class in det_lora.trained_classes if seen_class != class_name
        ]
        previous_val_loaders = []
        for previous_class in previous_classes:
            loader = _make_calibration_loader(
                det_lora=det_lora,
                raw_dir=data_dir,
                class_name=previous_class,
                detector=detector,
                batch_size=batch_size,
                seed=seed,
                max_samples=max_samples,
            )
            if loader is not None:
                previous_val_loaders.append(loader)
        if extend and extend_strategy == "grow_freeze":
            original_version = det_lora._active_versions.get(class_name)
            for version_id in _select_adapter_versions(
                det_lora,
                class_name,
                version_selection_strategy,
            ):
                det_lora.activate_adapter_version(class_name, version_id)
                refresh_adapter_calibration(
                    det_lora,
                    class_name,
                    positive_dataloader=val_loader,
                    negative_dataloaders=previous_val_loaders,
                )
            if original_version is not None:
                det_lora.activate_adapter_version(class_name, original_version)
        else:
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
        if use_adapter_arbitration and len(det_lora.trained_classes) > 1:
            arbitration_classes = list(det_lora.trained_classes)
            arbitration_dataset = load_dataset_from_raw(
                raw_dir=arbitration_raw_dir,
                class_filter=arbitration_classes,
                split="val",
                class_id_offset=detector.base_num_classes,
                img_size=resolution,
                seed=seed,
                max_samples=max_samples,
                class_id_mapping=_det_lora_class_id_mapping(det_lora, arbitration_classes),
            )
            arbitration_loader = DataLoader(
                arbitration_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
            adapter_arbitration_state = refresh_adapter_arbitration(
                det_lora,
                arbitration_loader,
                arbitration_classes,
                version_selection_by_class=(
                    {class_name: version_selection_strategy}
                    if extend and extend_strategy == "grow_freeze"
                    else None
                ),
            )
    det_lora.save_all(str(experiment_dir / "final"))

    test_metrics = None
    test_target_metrics = None
    target_extension_delta = None
    mixed_extension_delta = None
    if not synthetic:
        seen_classes = list(det_lora.trained_classes)
        test_dataset = load_dataset_from_raw(
            raw_dir=evaluation_data_dir,
            class_filter=seen_classes,
            split="test",
            class_id_offset=detector.base_num_classes,
            img_size=resolution,
            seed=seed,
            class_id_mapping=_det_lora_class_id_mapping(det_lora, seen_classes),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        evaluator = ContinualEvaluator(
            det_lora,
            use_adapter_arbitration=use_adapter_arbitration,
        )
        test_metrics = evaluator.evaluate_det_lora_joint(
            dataloader=test_loader,
            class_names=seen_classes,
            include_curves=True,
            version_selection_by_class=(
                {class_name: version_selection_strategy}
                if extend and extend_strategy == "grow_freeze"
                else None
            ),
        )
        target_test_dataset = load_dataset_from_raw(
            raw_dir=evaluation_data_dir,
            class_filter=class_name,
            split="test",
            class_id_offset=detector.base_num_classes,
            img_size=resolution,
            seed=seed,
            class_id_mapping=_det_lora_class_id_mapping(det_lora, [class_name]),
        )
        target_test_loader = DataLoader(
            target_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        test_target_metrics = evaluator.evaluate_det_lora_joint(
            dataloader=target_test_loader,
            class_names=[class_name],
            include_curves=True,
        )
        if extend and extend_strategy == "grow_freeze":
            test_target_metrics = evaluator.evaluate_det_lora_version_ensemble(
                dataloader=target_test_loader,
                class_name=class_name,
                include_curves=True,
                version_selection_strategy=version_selection_strategy,
            )
        if pre_extend_metrics is not None:
            mixed_extension_delta = {
                key: float(test_metrics.get(key, 0.0) - pre_extend_metrics.get(key, 0.0))
                for key in (
                    "mAP@0.5",
                    "mAP@0.75",
                    "mAP@0.95",
                    "mAP@0.5:0.95",
                    "Precision@0.5",
                    "Precision@0.95",
                    "Recall@0.5",
                    "Recall@0.95",
                    "F1@0.5",
                    "F1@0.95",
                )
            }
        if pre_extend_target_metrics is not None and test_target_metrics is not None:
            target_extension_delta = {
                key: float(
                    test_target_metrics.get(key, 0.0) - pre_extend_target_metrics.get(key, 0.0)
                )
                for key in (
                    "mAP@0.5",
                    "mAP@0.75",
                    "mAP@0.95",
                    "mAP@0.5:0.95",
                    "Precision@0.5",
                    "Precision@0.95",
                    "Recall@0.5",
                    "Recall@0.95",
                    "F1@0.5",
                    "F1@0.95",
                )
            }

    # Save training history
    results = {
        "class_name": class_name,
        "adapter_name": adapter_name,
        "extend": extend,
        "mode": "extend" if extend else "train",
        "extend_strategy": extend_strategy if extend else None,
        "version_selection_strategy": (
            version_selection_strategy if extend and extend_strategy == "grow_freeze" else None
        ),
        "output_dir": str(experiment_dir),
        "source_experiment_dir": load_dir,
        "source_checkpoint_dir": load_dir,
        "train_data_dir": data_dir,
        "test_data_dir": evaluation_data_dir,
        "arbitration_data_dir": arbitration_raw_dir,
        "target_class": class_name,
        "final_checkpoint_task": class_name,
        "final_checkpoint_dir": str(experiment_dir / "final"),
        "model_variant": model_variant,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_target_preset": lora_target_preset,
        "use_dora": use_dora,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "max_samples": max_samples,
        "sample_offset": sample_offset,
        "hard_negative_counts": hard_negative_counts,
        "use_hard_negatives": use_hard_negatives,
        "adapter_arbitration_state": adapter_arbitration_state,
        "metrics_eval_every": metrics_eval_every,
        "stability_loss_weight": (
            stability_loss_weight if extend and extend_strategy == "warm_start" else 0.0
        ),
        "teacher_anchor_weight": (
            teacher_anchor_weight if extend and extend_strategy == "warm_start" else 0.0
        ),
        "use_adapter_arbitration": use_adapter_arbitration,
        "preset": preset_name,
        "teacher_cache_size": 0 if teacher_cache is None else len(teacher_cache),
        "teacher_class_ids": teacher_class_ids,
        "runtime": collect_runtime_metadata(),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_map50": None if best_val_map50 == float("-inf") else best_val_map50,
        "test_metrics_scope": list(det_lora.trained_classes) if not synthetic else [],
        "history": history,
        "test_metrics": test_metrics,
        "test_target_metrics": test_target_metrics,
        "pre_extend_metrics": pre_extend_metrics,
        "pre_extend_target_metrics": pre_extend_target_metrics,
        "mixed_extension_delta": mixed_extension_delta,
        "target_extension_delta": target_extension_delta,
        "mixed_final_evaluation": test_metrics,
        "matched_final_evaluation": test_target_metrics,
    }

    with open(experiment_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    evaluation_payload = {
        "class_name": class_name,
        "extend": extend,
        "mode": "extend" if extend else "train",
        "extend_strategy": extend_strategy if extend else None,
        "version_selection_strategy": (
            version_selection_strategy if extend and extend_strategy == "grow_freeze" else None
        ),
        "max_samples": max_samples,
        "sample_offset": sample_offset,
        "hard_negative_counts": hard_negative_counts,
        "use_hard_negatives": use_hard_negatives,
        "adapter_arbitration_state": adapter_arbitration_state,
        "pre_extend_metrics": pre_extend_metrics,
        "pre_extend_target_metrics": pre_extend_target_metrics,
        "test_metrics": test_metrics,
        "test_target_metrics": test_target_metrics,
        "mixed_extension_delta": mixed_extension_delta,
        "target_extension_delta": target_extension_delta,
        "mixed_final_evaluation": test_metrics,
        "matched_final_evaluation": test_target_metrics,
        "train_data_dir": data_dir,
        "test_data_dir": evaluation_data_dir,
        "arbitration_data_dir": arbitration_raw_dir,
        "history": history,
    }
    with open(experiment_dir / "evaluation.json", "w") as f:
        json.dump(evaluation_payload, f, indent=2, default=str)

    print(f"\nTraining complete! Results saved to {experiment_dir}")
    return results


def main():
    base_defaults = {
        "epochs": 50,
        "batch_size": 4,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "lora_rank": 8,
        "lora_alpha": 16,
        "metrics_eval_every": 1,
    }
    parser = argparse.ArgumentParser(description="Train Det-LoRA adapter")
    parser.add_argument("--class_name", type=str, required=True, help="Class name to train")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--lora_rank", type=int, default=None, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha")
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
        "--model",
        type=str,
        default="medium",
        help="RF-DETR variant: nano, small, base, medium, large",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Variant-specific hyperparameter preset, e.g. l40_final",
    )
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Raw data directory")
    parser.add_argument(
        "--arbitration_data_dir",
        type=str,
        default=None,
        help="Raw data directory used to fit joint-inference arbitration",
    )
    parser.add_argument("--save_dir", type=str, default="experiments", help="Save directory")
    parser.add_argument("--extend", action="store_true", help="Extend existing class")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples")
    parser.add_argument(
        "--sample_offset",
        type=int,
        default=0,
        help="Skip filtered training samples before applying --max_samples",
    )
    parser.add_argument(
        "--load_dir", type=str, default=None, help="Load existing Det-LoRA checkpoint"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--extend_strategy",
        choices=("warm_start", "grow_freeze"),
        default="warm_start",
        help="Extension strategy for existing classes",
    )
    parser.add_argument(
        "--version_selection_strategy",
        choices=("all", "anchor_latest", "latest"),
        default="anchor_latest",
        help="Adapter-version selection for grow-freeze evaluation",
    )
    parser.add_argument(
        "--metrics_eval_every",
        type=int,
        default=None,
        help="Evaluate validation detection metrics every N epochs (0 disables)",
    )
    parser.add_argument(
        "--stability_loss_weight",
        type=float,
        default=1e-5,
        help="L2 anchor regularization when extending an existing class",
    )
    parser.add_argument(
        "--teacher_anchor_weight",
        type=float,
        default=0.05,
        help="Teacher distillation strength when extending an existing class",
    )
    parser.add_argument(
        "--disable_hard_negatives",
        action="store_true",
        help="Disable empty-target hard negatives from already learned classes",
    )
    parser.add_argument(
        "--use_adapter_arbitration",
        action="store_true",
        help="Fit compact region-classifier arbitration for joint Det-LoRA inference",
    )

    args = parser.parse_args()

    resolved = resolve_variant_settings(
        variant=args.model,
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
    if args.preset:
        print(
            "[Preset] "
            f"{args.preset} -> batch_size={resolved['batch_size']}, "
            f"lr={resolved['lr']}, lora_rank={resolved['lora_rank']}, "
            f"lora_alpha={resolved['lora_alpha']}, epochs={resolved['epochs']}"
        )

    train_adapter(
        class_name=args.class_name,
        epochs=int(resolved["epochs"]),
        batch_size=int(resolved["batch_size"]),
        lr=float(resolved["lr"]),
        weight_decay=float(resolved["weight_decay"]),
        lora_rank=int(resolved["lora_rank"]),
        lora_alpha=int(resolved["lora_alpha"]),
        lora_target_preset=args.lora_targets,
        use_dora=args.lora_dora,
        model_variant=args.model,
        data_dir=args.data_dir,
        arbitration_data_dir=args.arbitration_data_dir,
        save_dir=args.save_dir,
        extend=args.extend,
        synthetic=args.synthetic,
        max_samples=args.max_samples,
        sample_offset=args.sample_offset,
        load_dir=args.load_dir,
        seed=args.seed,
        extend_strategy=args.extend_strategy,
        version_selection_strategy=args.version_selection_strategy,
        metrics_eval_every=int(resolved["metrics_eval_every"]),
        stability_loss_weight=args.stability_loss_weight,
        teacher_anchor_weight=args.teacher_anchor_weight,
        use_hard_negatives=not args.disable_hard_negatives,
        use_adapter_arbitration=args.use_adapter_arbitration,
        preset_name=args.preset,
    )


if __name__ == "__main__":
    main()
