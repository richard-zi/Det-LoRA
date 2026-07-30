"""
Continual Learning Evaluator
==============================

Evaluates continual-learning runs and tracks per-task histories for
forgetting and transfer metrics.
"""

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from det_lora.evaluation.arbitration import (
    apply_adapter_arbitration,
    fit_adapter_arbitration_state,
)
from det_lora.evaluation.conflict_gate import apply_conflict_gate, fit_pair_gate
from det_lora.evaluation.metrics import compute_iou_matrix, compute_map
from det_lora.model.det_lora import DetLoRA
from det_lora.model.detector import RFDETRDetector


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _empty_prediction() -> Dict[str, np.ndarray]:
    return {
        "boxes": np.empty((0, 4), dtype=np.float32),
        "scores": np.empty((0,), dtype=np.float32),
        "labels": np.empty((0,), dtype=np.int64),
        "query_ids": np.empty((0,), dtype=np.int64),
    }


def _filter_ground_truth(
    gt: Dict[str, torch.Tensor],
    target_class_ids: Sequence[int],
) -> Dict[str, np.ndarray]:
    labels = gt.get("labels", gt.get("class_labels"))
    boxes = gt["boxes"]
    if labels.numel() == 0:
        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "labels": np.empty((0,), dtype=np.int64),
        }

    target_set = set(int(cls_id) for cls_id in target_class_ids)
    keep = torch.tensor(
        [int(label.item()) in target_set for label in labels],
        dtype=torch.bool,
        device=labels.device,
    )

    return {
        "boxes": cxcywh_to_xyxy(boxes[keep]).cpu().numpy(),
        "labels": labels[keep].cpu().numpy(),
    }


def _append_prediction(
    prediction: Dict[str, np.ndarray],
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_id: int,
    query_ids: Optional[torch.Tensor] = None,
    metadata: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, np.ndarray]:
    if boxes.numel() == 0:
        return prediction

    pred_boxes = cxcywh_to_xyxy(boxes).cpu().numpy()
    pred_scores = scores.cpu().numpy()
    pred_labels = np.full(pred_scores.shape[0], class_id, dtype=np.int64)
    pred_query_ids = (
        query_ids.detach().cpu().numpy().astype(np.int64)
        if query_ids is not None
        else np.full(pred_scores.shape[0], -1, dtype=np.int64)
    )
    pred_metadata = {}
    for key, value in (metadata or {}).items():
        if isinstance(value, torch.Tensor):
            pred_metadata[key] = value.detach().cpu().numpy()
        else:
            pred_metadata[key] = np.asarray(value)

    if prediction["boxes"].size == 0:
        prediction["boxes"] = pred_boxes
        prediction["scores"] = pred_scores
        prediction["labels"] = pred_labels
        prediction["query_ids"] = pred_query_ids
        for key, value in pred_metadata.items():
            prediction[key] = value
        return prediction

    prediction["boxes"] = np.concatenate([prediction["boxes"], pred_boxes], axis=0)
    prediction["scores"] = np.concatenate([prediction["scores"], pred_scores], axis=0)
    prediction["labels"] = np.concatenate([prediction["labels"], pred_labels], axis=0)
    prediction["query_ids"] = np.concatenate([prediction["query_ids"], pred_query_ids], axis=0)
    for key, value in pred_metadata.items():
        if key not in prediction:
            prediction[key] = value
        else:
            prediction[key] = np.concatenate([prediction[key], value], axis=0)
    return prediction


def _prediction_keep_mask(
    scores: torch.Tensor,
    confidence_threshold: Optional[float],
) -> torch.Tensor:
    """Keep all scores for ranking-based AP unless an explicit cutoff is requested."""
    if confidence_threshold is None:
        return torch.ones_like(scores, dtype=torch.bool)
    return scores >= confidence_threshold


def _limit_prediction(
    prediction: Dict[str, np.ndarray],
    max_detections: Optional[int],
) -> Dict[str, np.ndarray]:
    """
    Keep the top-scoring detections per image.

    COCO-style AP is ranking-based and should not depend on a hard score cutoff.
    To keep evaluation stable and bounded for DETR-style dense outputs, we cap
    the number of detections per image after sorting by score.
    """
    if max_detections is None or prediction["scores"].shape[0] <= max_detections:
        return prediction

    keep = np.argsort(prediction["scores"])[::-1][:max_detections]
    limited = {}
    for key, value in prediction.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == prediction["scores"].shape[:1]:
            limited[key] = value[keep]
        else:
            limited[key] = value
    return limited


def _nms_prediction(
    prediction: Dict[str, np.ndarray],
    iou_threshold: float = 0.6,
) -> Dict[str, np.ndarray]:
    """Suppress duplicate detections when multiple adapter versions vote for one class."""
    if prediction["scores"].shape[0] <= 1:
        return prediction

    keep: List[int] = []
    labels = prediction["labels"]
    scores = prediction["scores"]
    boxes = prediction["boxes"]

    for class_id in sorted(set(int(label) for label in labels.tolist())):
        class_indices = np.flatnonzero(labels == class_id)
        ordered = class_indices[np.argsort(scores[class_indices])[::-1]]
        while ordered.size > 0:
            current = int(ordered[0])
            keep.append(current)
            if ordered.size == 1:
                break
            ious = compute_iou_matrix(
                boxes[[current]].astype(np.float32),
                boxes[ordered[1:]].astype(np.float32),
            )[0]
            ordered = ordered[1:][ious < iou_threshold]

    keep_array = np.array(keep, dtype=np.int64)
    keep_array = keep_array[np.argsort(scores[keep_array])[::-1]]
    return {
        key: (
            value[keep_array]
            if isinstance(value, np.ndarray) and value.shape[:1] == scores.shape[:1]
            else value
        )
        for key, value in prediction.items()
    }


def _select_adapter_versions(
    det_lora: DetLoRA,
    class_name: str,
    strategy: str,
) -> List[str]:
    """Select class-internal adapter versions for replay-free extension inference."""
    version_entries = getattr(det_lora, "_adapter_versions", {}).get(class_name, [])
    version_ids = [str(entry["version_id"]) for entry in version_entries]
    if not version_ids:
        raise ValueError(f"No adapter versions available for '{class_name}'")

    if strategy == "all":
        return version_ids
    if strategy == "anchor_latest":
        if len(version_ids) <= 2:
            return version_ids
        return [version_ids[0], version_ids[-1]]
    if strategy == "latest":
        return [version_ids[-1]]

    raise ValueError("version_selection_strategy must be one of: all, anchor_latest, latest")


def aggregate_classwise_metrics(
    class_metrics_by_name: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate single-class evaluations into a matched-retention summary."""
    if not class_metrics_by_name:
        return {}

    mean_keys = (
        "mAP@0.5",
        "mAP@0.75",
        "mAP@0.95",
        "mAP@0.5:0.95",
        "Precision@0.5",
        "Precision@0.75",
        "Precision@0.95",
        "Recall@0.5",
        "Recall@0.75",
        "Recall@0.95",
        "F1@0.5",
        "F1@0.75",
        "F1@0.95",
        "MicroPrecision@0.5",
        "MicroPrecision@0.75",
        "MicroPrecision@0.95",
        "MicroRecall@0.5",
        "MicroRecall@0.75",
        "MicroRecall@0.95",
        "MicroF1@0.5",
        "MicroF1@0.75",
        "MicroF1@0.95",
    )
    aggregate: Dict[str, Any] = {
        key: float(
            np.mean([float(metrics.get(key, 0.0)) for metrics in class_metrics_by_name.values()])
        )
        for key in mean_keys
    }

    aggregate["TP@0.5"] = int(
        sum(int(metrics.get("TP@0.5", 0)) for metrics in class_metrics_by_name.values())
    )
    aggregate["TP@0.75"] = int(
        sum(int(metrics.get("TP@0.75", 0)) for metrics in class_metrics_by_name.values())
    )
    aggregate["TP@0.95"] = int(
        sum(int(metrics.get("TP@0.95", 0)) for metrics in class_metrics_by_name.values())
    )
    aggregate["FP@0.5"] = int(
        sum(int(metrics.get("FP@0.5", 0)) for metrics in class_metrics_by_name.values())
    )
    aggregate["FP@0.75"] = int(
        sum(int(metrics.get("FP@0.75", 0)) for metrics in class_metrics_by_name.values())
    )
    aggregate["FP@0.95"] = int(
        sum(int(metrics.get("FP@0.95", 0)) for metrics in class_metrics_by_name.values())
    )
    aggregate["FN@0.5"] = int(
        sum(int(metrics.get("FN@0.5", 0)) for metrics in class_metrics_by_name.values())
    )
    aggregate["FN@0.75"] = int(
        sum(int(metrics.get("FN@0.75", 0)) for metrics in class_metrics_by_name.values())
    )
    aggregate["FN@0.95"] = int(
        sum(int(metrics.get("FN@0.95", 0)) for metrics in class_metrics_by_name.values())
    )
    aggregate["AP_per_class@0.5"] = {
        class_name: float(
            metrics.get("AP_per_class@0.5", {}).get(class_name, metrics.get("mAP@0.5", 0.0))
        )
        for class_name, metrics in class_metrics_by_name.items()
    }
    aggregate["AP_per_class@0.75"] = {
        class_name: float(
            metrics.get("AP_per_class@0.75", {}).get(class_name, metrics.get("mAP@0.75", 0.0))
        )
        for class_name, metrics in class_metrics_by_name.items()
    }
    aggregate["AP_per_class@0.95"] = {
        class_name: float(
            metrics.get("AP_per_class@0.95", {}).get(class_name, metrics.get("mAP@0.95", 0.0))
        )
        for class_name, metrics in class_metrics_by_name.items()
    }
    aggregate["Precision_per_class@0.5"] = {
        class_name: float(
            metrics.get("Precision_per_class@0.5", {}).get(
                class_name,
                metrics.get("Precision@0.5", 0.0),
            )
        )
        for class_name, metrics in class_metrics_by_name.items()
    }
    aggregate["Precision_per_class@0.75"] = {
        class_name: float(
            metrics.get("Precision_per_class@0.75", {}).get(
                class_name,
                metrics.get("Precision@0.75", 0.0),
            )
        )
        for class_name, metrics in class_metrics_by_name.items()
    }
    aggregate["Precision_per_class@0.95"] = {
        class_name: float(
            metrics.get("Precision_per_class@0.95", {}).get(
                class_name,
                metrics.get("Precision@0.95", 0.0),
            )
        )
        for class_name, metrics in class_metrics_by_name.items()
    }
    aggregate["Recall_per_class@0.5"] = {
        class_name: float(
            metrics.get("Recall_per_class@0.5", {}).get(
                class_name,
                metrics.get("Recall@0.5", 0.0),
            )
        )
        for class_name, metrics in class_metrics_by_name.items()
    }
    aggregate["Recall_per_class@0.75"] = {
        class_name: float(
            metrics.get("Recall_per_class@0.75", {}).get(
                class_name,
                metrics.get("Recall@0.75", 0.0),
            )
        )
        for class_name, metrics in class_metrics_by_name.items()
    }
    aggregate["Recall_per_class@0.95"] = {
        class_name: float(
            metrics.get("Recall_per_class@0.95", {}).get(
                class_name,
                metrics.get("Recall@0.95", 0.0),
            )
        )
        for class_name, metrics in class_metrics_by_name.items()
    }
    aggregate["F1_per_class@0.5"] = {
        class_name: float(
            metrics.get("F1_per_class@0.5", {}).get(class_name, metrics.get("F1@0.5", 0.0))
        )
        for class_name, metrics in class_metrics_by_name.items()
    }
    aggregate["F1_per_class@0.75"] = {
        class_name: float(
            metrics.get("F1_per_class@0.75", {}).get(class_name, metrics.get("F1@0.75", 0.0))
        )
        for class_name, metrics in class_metrics_by_name.items()
    }
    aggregate["F1_per_class@0.95"] = {
        class_name: float(
            metrics.get("F1_per_class@0.95", {}).get(class_name, metrics.get("F1@0.95", 0.0))
        )
        for class_name, metrics in class_metrics_by_name.items()
    }
    for curve_key in ("PR_curve_per_class@0.5", "PR_curve_per_class@0.95"):
        curve_data = {}
        for class_name, metrics in class_metrics_by_name.items():
            class_curve = metrics.get(curve_key, {}).get(class_name)
            if class_curve is not None:
                curve_data[class_name] = class_curve
        if curve_data:
            aggregate[curve_key] = curve_data
    aggregate["per_class_evaluations"] = class_metrics_by_name
    return aggregate


def summarize_mixed_confusion(
    matched_history: Dict[int, Dict[str, Any]],
    mixed_history: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare matched-retention and mixed-set AP to expose confusion effects."""
    summary: Dict[str, Any] = {}
    for task_idx, matched_entry in matched_history.items():
        mixed_entry = mixed_history.get(task_idx)
        if mixed_entry is None:
            continue

        matched_metrics = matched_entry["metrics"]
        mixed_metrics = mixed_entry["metrics"]
        matched_ap = matched_metrics.get("AP_per_class@0.5", {})
        mixed_ap = mixed_metrics.get("AP_per_class@0.5", {})
        summary[str(task_idx)] = {
            "mAP@0.5_gap": float(
                matched_metrics.get("mAP@0.5", 0.0) - mixed_metrics.get("mAP@0.5", 0.0)
            ),
            "AP_per_class@0.5_gap": {
                class_name: float(matched_ap.get(class_name, 0.0) - mixed_ap.get(class_name, 0.0))
                for class_name in sorted(set(matched_ap) | set(mixed_ap))
            },
        }
    return summary


@torch.no_grad()
def collect_score_bank_for_class(
    det_lora: DetLoRA,
    dataloader: DataLoader,
    class_name: str,
    known_absent: bool = False,
    iou_threshold: float = 0.5,
    max_predictions_per_image: int = 25,
    max_negative_scores_per_image: int = 10,
    max_bank_size: int = 2048,
) -> Dict[str, List[float]]:
    """
    Collect compact positive/negative score samples for a single adapter.

    When `known_absent=True`, every sampled score is treated as a negative. This
    is used to calibrate old adapters on newly seen class images without replay.
    """
    det_lora.set_eval_mode()
    class_id = det_lora.get_class_id(class_name)
    det_lora.load_adapter_for_eval(class_name)
    device = det_lora.device

    positives: List[float] = []
    negatives: List[float] = []

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        outputs = det_lora.forward(pixel_values=pixel_values)
        class_scores = outputs["pred_logits"][:, :, class_id].sigmoid()
        pred_boxes = outputs["pred_boxes"]

        for sample_idx in range(class_scores.shape[0]):
            scores = class_scores[sample_idx]
            boxes = cxcywh_to_xyxy(pred_boxes[sample_idx]).cpu().numpy()
            order = torch.argsort(scores, descending=True)[:max_predictions_per_image]
            selected_scores = scores[order].detach().cpu().tolist()
            selected_boxes = boxes[order.cpu().numpy()]

            if known_absent:
                negatives.extend(
                    float(score) for score in selected_scores[:max_negative_scores_per_image]
                )
                continue

            gt = _filter_ground_truth(batch["labels"][sample_idx], [class_id])
            gt_boxes = gt["boxes"]
            if gt_boxes.size == 0:
                negatives.extend(
                    float(score) for score in selected_scores[:max_negative_scores_per_image]
                )
                continue

            matched_gt: set[int] = set()
            matched_pred: set[int] = set()
            ious = compute_iou_matrix(
                np.asarray(selected_boxes, dtype=np.float32),
                np.asarray(gt_boxes, dtype=np.float32),
            )

            for pred_idx, score in enumerate(selected_scores):
                best_iou = 0.0
                best_gt = -1
                for gt_idx in range(ious.shape[1]):
                    if gt_idx in matched_gt:
                        continue
                    iou = float(ious[pred_idx, gt_idx])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt_idx
                if best_iou >= iou_threshold and best_gt >= 0:
                    positives.append(float(score))
                    matched_gt.add(best_gt)
                    matched_pred.add(pred_idx)

            negatives_added = 0
            for pred_idx, score in enumerate(selected_scores):
                if pred_idx in matched_pred:
                    continue
                negatives.append(float(score))
                negatives_added += 1
                if negatives_added >= max_negative_scores_per_image:
                    break

    det_lora.unload_adapter()
    return {
        "positive_scores": det_lora._compress_scores(positives, limit=max_bank_size),
        "negative_scores": det_lora._compress_scores(negatives, limit=max_bank_size),
    }


def refresh_adapter_calibration(
    det_lora: DetLoRA,
    class_name: str,
    positive_dataloader: Optional[DataLoader] = None,
    negative_dataloaders: Optional[Sequence[DataLoader]] = None,
    max_bank_size: int = 2048,
) -> Dict[str, float]:
    """Update score banks and refit a class calibrator."""
    if positive_dataloader is not None:
        bank = collect_score_bank_for_class(
            det_lora,
            positive_dataloader,
            class_name,
            known_absent=False,
            max_bank_size=max_bank_size,
        )
        det_lora.record_score_bank(
            class_name,
            positive_scores=bank["positive_scores"],
            negative_scores=bank["negative_scores"],
            max_bank_size=max_bank_size,
        )

    for dataloader in negative_dataloaders or []:
        bank = collect_score_bank_for_class(
            det_lora,
            dataloader,
            class_name,
            known_absent=True,
            max_bank_size=max_bank_size,
        )
        det_lora.record_score_bank(
            class_name,
            negative_scores=bank["negative_scores"],
            max_bank_size=max_bank_size,
        )

    return det_lora.fit_calibrator(class_name)


def extract_shared_quality_features(prediction: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Build class-agnostic quality features from decoder embeddings plus geometry.

    This intentionally ignores the predicted class identity so the learned
    quality model stays generic as new classes are added.
    """
    scores = prediction["scores"].astype(np.float32)
    if scores.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float32)

    embedding_features = prediction.get("quality_features")
    if embedding_features is not None:
        embedding_features = np.asarray(embedding_features, dtype=np.float32)
        if embedding_features.shape[:1] != scores.shape[:1]:
            embedding_features = None

    boxes = prediction["boxes"].astype(np.float32)
    widths = np.clip(boxes[:, 2] - boxes[:, 0], 1e-6, None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], 1e-6, None)
    areas = np.clip(widths * heights, 1e-6, None)
    logits = _logit_np(scores)
    scalar_features = np.stack(
        [
            scores,
            logits,
            np.log(areas),
            np.log(widths / heights),
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    if embedding_features is None:
        return scalar_features
    return np.concatenate([embedding_features, scalar_features], axis=1)


def _logit_np(scores: np.ndarray) -> np.ndarray:
    """Stable numpy logit for probability-like scores."""
    clipped = np.clip(scores.astype(np.float32), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
    """Stable numpy sigmoid."""
    logits = logits.astype(np.float32, copy=False)
    positive = logits >= 0.0
    out = np.empty_like(logits, dtype=np.float32)
    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    out[~positive] = exp_logits / (1.0 + exp_logits)
    return out


def _fit_logistic_reranker(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    steps: int,
    lr: float,
    base_logits: Optional[np.ndarray] = None,
    ranking_loss_weight: float = 0.0,
    ranking_margin: float = 0.5,
    ranking_topk: int = 512,
) -> Dict[str, Any]:
    """Fit a lightweight linear logistic model, optionally as a residual."""
    positive_idx = np.flatnonzero(targets > 0.5)
    negative_idx = np.flatnonzero(targets <= 0.5)
    feature_dim = int(features.shape[1]) if features.ndim == 2 else 0

    if features.shape[0] == 0:
        return {
            "identity": True,
            "reason": "no_predictions",
            "feature_dim": feature_dim,
            "positive_count": 0,
            "negative_count": 0,
            "train_negative_count": 0,
            "mode": "direct" if base_logits is None else "residual",
        }

    if positive_idx.size == 0:
        return {
            "identity": True,
            "reason": "no_positive_matches",
            "feature_dim": feature_dim,
            "positive_count": 0,
            "negative_count": int(negative_idx.size),
            "train_negative_count": int(negative_idx.size),
            "mode": "direct" if base_logits is None else "residual",
        }

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1e-5)
    x_norm = (x - mean) / std
    base = (
        torch.tensor(base_logits, dtype=torch.float32)
        if base_logits is not None
        else torch.zeros_like(y)
    )

    weight = torch.zeros((x_norm.shape[1],), dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
    positives = float(positive_idx.size)
    negatives = float(negative_idx.size)
    optimizer = torch.optim.Adam([weight, bias], lr=lr)
    pos_weight = torch.tensor(negatives / max(positives, 1.0), dtype=torch.float32)
    baseline_loss = float(F.binary_cross_entropy_with_logits(base, y, pos_weight=pos_weight).item())

    for _ in range(steps):
        optimizer.zero_grad()
        residual = x_norm @ weight + bias
        logits = base + residual
        bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)

        if ranking_loss_weight > 0.0:
            positive_logits = logits[y > 0.5]
            negative_logits = logits[y <= 0.5]
            if positive_logits.numel() > 0 and negative_logits.numel() > 0:
                hard_negatives = negative_logits.topk(
                    min(int(ranking_topk), int(negative_logits.numel()))
                ).values
                ranking_loss = F.softplus(
                    ranking_margin - (positive_logits[:, None] - hard_negatives[None, :])
                ).mean()
            else:
                ranking_loss = torch.tensor(0.0, dtype=torch.float32)
        else:
            ranking_loss = torch.tensor(0.0, dtype=torch.float32)

        loss = bce + ranking_loss_weight * ranking_loss
        loss = loss + 1e-3 * (weight.pow(2).mean() + bias.pow(2))
        loss.backward()
        optimizer.step()

    trained_logits = base + (x_norm @ weight.detach() + bias.detach())
    trained_loss = float(
        F.binary_cross_entropy_with_logits(
            trained_logits,
            y,
            pos_weight=pos_weight,
        ).item()
    )

    return {
        "weight": weight.detach().cpu().tolist(),
        "bias": float(bias.detach().cpu().item()),
        "mean": mean.detach().cpu().tolist(),
        "std": std.detach().cpu().tolist(),
        "feature_dim": int(features.shape[1]),
        "positive_count": int(positive_idx.size),
        "negative_count": int(negative_idx.size),
        "train_negative_count": int(negative_idx.size),
        "baseline_loss": baseline_loss,
        "trained_loss": trained_loss,
        "mode": "direct" if base_logits is None else "residual",
    }


def _apply_linear_reranker(features: np.ndarray, reranker: Dict[str, Any]) -> np.ndarray:
    """Apply a fitted linear reranker and return its raw linear output."""
    weight = np.asarray(reranker["weight"], dtype=np.float32)
    mean = np.asarray(reranker["mean"], dtype=np.float32)
    std = np.asarray(reranker["std"], dtype=np.float32)
    bias = float(reranker["bias"])
    feature_dim = min(
        features.shape[1],
        weight.shape[0],
        mean.shape[0],
        std.shape[0],
    )
    normalized = (features[:, :feature_dim] - mean[:feature_dim]) / std[:feature_dim]
    return normalized @ weight[:feature_dim] + bias


def label_tp_for_class(
    prediction: Dict[str, np.ndarray],
    gt: Dict[str, np.ndarray],
    class_id: int,
    iou_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Label detections of one class as TP/FP using greedy AP-style matching."""
    pred_mask = prediction["labels"] == class_id
    gt_mask = gt["labels"] == class_id
    pred_boxes = prediction["boxes"][pred_mask].astype(np.float32)
    pred_scores = prediction["scores"][pred_mask].astype(np.float32)
    gt_boxes = gt["boxes"][gt_mask].astype(np.float32)
    labels = np.zeros(pred_boxes.shape[0], dtype=np.float32)

    if pred_boxes.shape[0] == 0:
        return labels, pred_mask

    order = np.argsort(pred_scores)[::-1]
    matched_gt: set[int] = set()

    if gt_boxes.shape[0] > 0:
        ious = compute_iou_matrix(pred_boxes[order], gt_boxes)
        for rank_idx, pred_idx in enumerate(order):
            best_gt = -1
            best_iou = 0.0
            for gt_idx in range(gt_boxes.shape[0]):
                if gt_idx in matched_gt:
                    continue
                iou = float(ious[rank_idx, gt_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_idx
            if best_iou >= iou_threshold and best_gt >= 0:
                labels[pred_idx] = 1.0
                matched_gt.add(best_gt)

    return labels, pred_mask


def fit_shared_quality_calibrator(
    predictions: Sequence[Dict[str, np.ndarray]],
    ground_truths: Sequence[Dict[str, np.ndarray]],
    target_class_ids: Sequence[int],
    steps: int = 300,
    lr: float = 0.03,
    ranking_loss_weight: float = 0.1,
    ranking_margin: float = 0.5,
    ranking_topk: int = 512,
) -> Dict[str, Any]:
    """
    Fit one class-agnostic quality model over all seen classes.

    The model predicts whether a detection is a true positive independent of
    which expert/class produced it. This is a lightweight proxy for a shared
    objectness/quality head without modifying the training loop yet.
    """
    feature_rows: List[np.ndarray] = []
    target_rows: List[np.ndarray] = []

    for prediction, gt in zip(predictions, ground_truths):
        features = extract_shared_quality_features(prediction)
        if features.shape[0] == 0:
            continue
        per_prediction_targets = np.zeros(features.shape[0], dtype=np.float32)
        has_target = np.zeros(features.shape[0], dtype=bool)

        for class_id in target_class_ids:
            tp_labels, pred_mask = label_tp_for_class(prediction, gt, class_id)
            if pred_mask.sum() == 0:
                continue
            per_prediction_targets[pred_mask] = tp_labels
            has_target[pred_mask] = True

        if np.any(has_target):
            feature_rows.append(features[has_target])
            target_rows.append(per_prediction_targets[has_target])

    if not feature_rows:
        return {
            "identity": True,
            "reason": "no_predictions",
            "feature_dim": 0,
            "positive_count": 0,
            "negative_count": 0,
            "train_negative_count": 0,
        }

    all_features = np.concatenate(feature_rows, axis=0)
    all_targets = np.concatenate(target_rows, axis=0)
    return _fit_logistic_reranker(
        all_features,
        all_targets,
        steps=steps,
        lr=lr,
        ranking_loss_weight=ranking_loss_weight,
        ranking_margin=ranking_margin,
        ranking_topk=ranking_topk,
    )


def apply_shared_quality_calibrator(
    predictions: Sequence[Dict[str, np.ndarray]],
    calibrator: Optional[Dict[str, Any]],
) -> List[Dict[str, np.ndarray]]:
    """Replace raw scores with shared quality probabilities when available."""
    if not calibrator or calibrator.get("identity"):
        return list(predictions)

    adjusted_predictions: List[Dict[str, np.ndarray]] = []
    for prediction in predictions:
        features = extract_shared_quality_features(prediction)
        if features.shape[0] == 0:
            adjusted_predictions.append(prediction)
            continue
        adjusted = dict(prediction)
        adjusted["scores"] = _sigmoid_np(_apply_linear_reranker(features, calibrator))
        adjusted_predictions.append(adjusted)
    return adjusted_predictions


@torch.no_grad()
def collect_det_lora_joint_predictions(
    det_lora: DetLoRA,
    dataloader: DataLoader,
    class_names: Sequence[str],
    confidence_threshold: Optional[float] = None,
    use_shared_encoder_cache: bool = True,
    version_selection_by_class: Optional[Dict[str, str]] = None,
) -> tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]], List[int]]:
    """Collect raw merged predictions for the actual Det-LoRA joint inference path."""
    det_lora.set_eval_mode()
    device = det_lora.device
    target_class_ids = [det_lora.get_class_id(class_name) for class_name in class_names]
    all_predictions: List[Dict[str, np.ndarray]] = []
    all_ground_truths: List[Dict[str, np.ndarray]] = []

    for batch in dataloader:
        for sample_idx in range(len(batch["labels"])):
            all_predictions.append(_empty_prediction())
            all_ground_truths.append(
                _filter_ground_truth(batch["labels"][sample_idx], target_class_ids)
            )

    has_version_selection = bool(version_selection_by_class)
    has_shared_encoder_context = use_shared_encoder_cache and all(
        hasattr(det_lora.detector, attr)
        for attr in ("extract_shared_encoder_context", "forward_from_shared_encoder_context")
    )
    supports_shared_encoder = not has_version_selection and has_shared_encoder_context

    if supports_shared_encoder:
        det_lora.prepare_eval_adapter_cache(list(class_names))
        try:
            image_offset = 0
            for batch in tqdm(dataloader, desc="Evaluating joint", leave=False):
                pixel_values = batch["pixel_values"].to(device)
                shared_context = det_lora.detector.extract_shared_encoder_context(pixel_values)

                for class_name, class_id in zip(class_names, target_class_ids):
                    det_lora.activate_cached_eval_adapter(class_name)
                    outputs = det_lora.detector.forward_from_shared_encoder_context(shared_context)
                    class_scores = outputs["pred_logits"][:, :, class_id].sigmoid()
                    class_scores = det_lora.calibrate_scores(class_name, class_scores)
                    pred_boxes = outputs["pred_boxes"]
                    decoder_embeddings = outputs.get("decoder_embeddings")
                    proposal_embeddings = outputs.get("proposal_embeddings")
                    query_ids = torch.arange(class_scores.shape[1], device=class_scores.device)

                    for sample_idx in range(class_scores.shape[0]):
                        keep = _prediction_keep_mask(class_scores[sample_idx], confidence_threshold)
                        final_boxes = pred_boxes[sample_idx][keep]
                        metadata = None
                        if decoder_embeddings is not None or proposal_embeddings is not None:
                            metadata = {}
                            if decoder_embeddings is not None:
                                metadata["quality_features"] = decoder_embeddings[sample_idx][keep]
                            if proposal_embeddings is not None:
                                metadata["arbitration_features"] = proposal_embeddings[sample_idx][
                                    keep
                                ]

                        all_predictions[image_offset + sample_idx] = _append_prediction(
                            all_predictions[image_offset + sample_idx],
                            final_boxes,
                            class_scores[sample_idx][keep],
                            class_id,
                            query_ids=query_ids[keep],
                            metadata=metadata,
                        )

                image_offset += pixel_values.shape[0]
        finally:
            det_lora.clear_eval_adapter_cache()
    else:
        for class_name, class_id in zip(class_names, target_class_ids):
            original_version = getattr(det_lora, "_active_versions", {}).get(class_name)
            if version_selection_by_class and class_name in version_selection_by_class:
                version_ids = _select_adapter_versions(
                    det_lora,
                    class_name,
                    version_selection_by_class[class_name],
                )
            else:
                version_ids = [original_version] if original_version is not None else [None]

            try:
                for version_id in version_ids:
                    if version_id is not None:
                        det_lora.activate_adapter_version(class_name, str(version_id))
                    det_lora.load_adapter_for_eval(class_name)
                    image_offset = 0

                    for batch in tqdm(
                        dataloader,
                        desc=(
                            f"Evaluating {class_name}:{version_id}"
                            if version_id is not None
                            else f"Evaluating {class_name}"
                        ),
                        leave=False,
                    ):
                        pixel_values = batch["pixel_values"].to(device)
                        if has_shared_encoder_context:
                            shared_context = det_lora.detector.extract_shared_encoder_context(
                                pixel_values
                            )
                            outputs = det_lora.detector.forward_from_shared_encoder_context(
                                shared_context
                            )
                        else:
                            outputs = det_lora.forward(pixel_values=pixel_values)
                        class_scores = outputs["pred_logits"][:, :, class_id].sigmoid()
                        class_scores = det_lora.calibrate_scores(class_name, class_scores)
                        pred_boxes = outputs["pred_boxes"]
                        decoder_embeddings = outputs.get("decoder_embeddings")
                        proposal_embeddings = outputs.get("proposal_embeddings")
                        query_ids = torch.arange(class_scores.shape[1], device=class_scores.device)

                        for sample_idx in range(class_scores.shape[0]):
                            keep = _prediction_keep_mask(
                                class_scores[sample_idx],
                                confidence_threshold,
                            )
                            final_boxes = pred_boxes[sample_idx][keep]
                            metadata = None
                            if decoder_embeddings is not None or proposal_embeddings is not None:
                                metadata = {}
                                if decoder_embeddings is not None:
                                    metadata["quality_features"] = decoder_embeddings[sample_idx][
                                        keep
                                    ]
                                if proposal_embeddings is not None:
                                    metadata["arbitration_features"] = proposal_embeddings[
                                        sample_idx
                                    ][keep]

                            all_predictions[image_offset + sample_idx] = _append_prediction(
                                all_predictions[image_offset + sample_idx],
                                final_boxes,
                                class_scores[sample_idx][keep],
                                class_id,
                                query_ids=query_ids[keep],
                                metadata=metadata,
                            )

                        image_offset += class_scores.shape[0]

                    det_lora.unload_adapter()
            finally:
                det_lora.unload_adapter()
                if original_version is not None:
                    det_lora.activate_adapter_version(class_name, str(original_version))

    if has_version_selection:
        all_predictions = [_nms_prediction(prediction) for prediction in all_predictions]

    return all_predictions, all_ground_truths, target_class_ids


@torch.no_grad()
def collect_det_lora_version_ensemble_predictions(
    det_lora: DetLoRA,
    dataloader: DataLoader,
    class_name: str,
    confidence_threshold: Optional[float] = None,
    version_selection_strategy: str = "anchor_latest",
) -> tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]], List[int]]:
    """Collect predictions by merging selected frozen adapter versions for one class."""
    det_lora.set_eval_mode()
    device = det_lora.device
    class_id = det_lora.get_class_id(class_name)
    target_class_ids = [class_id]
    all_predictions: List[Dict[str, np.ndarray]] = []
    all_ground_truths: List[Dict[str, np.ndarray]] = []

    for batch in dataloader:
        for sample_idx in range(len(batch["labels"])):
            all_predictions.append(_empty_prediction())
            all_ground_truths.append(
                _filter_ground_truth(batch["labels"][sample_idx], target_class_ids)
            )

    version_ids = _select_adapter_versions(
        det_lora,
        class_name,
        version_selection_strategy,
    )

    original_version = getattr(det_lora, "_active_versions", {}).get(class_name)
    try:
        for version_id in version_ids:
            det_lora.activate_adapter_version(class_name, version_id)
            det_lora.load_adapter_for_eval(class_name)
            image_offset = 0

            for batch in tqdm(
                dataloader,
                desc=f"Evaluating {class_name}:{version_id}",
                leave=False,
            ):
                pixel_values = batch["pixel_values"].to(device)
                outputs = det_lora.forward(pixel_values=pixel_values)
                class_scores = outputs["pred_logits"][:, :, class_id].sigmoid()
                class_scores = det_lora.calibrate_scores(class_name, class_scores)
                pred_boxes = outputs["pred_boxes"]
                query_ids = torch.arange(class_scores.shape[1], device=class_scores.device)

                for sample_idx in range(class_scores.shape[0]):
                    keep = _prediction_keep_mask(
                        class_scores[sample_idx],
                        confidence_threshold,
                    )
                    all_predictions[image_offset + sample_idx] = _append_prediction(
                        all_predictions[image_offset + sample_idx],
                        pred_boxes[sample_idx][keep],
                        class_scores[sample_idx][keep],
                        class_id,
                        query_ids=query_ids[keep],
                    )

                image_offset += class_scores.shape[0]

            det_lora.unload_adapter()
    finally:
        det_lora.unload_adapter()
        if original_version is not None:
            det_lora.activate_adapter_version(class_name, str(original_version))

    return (
        [_nms_prediction(prediction) for prediction in all_predictions],
        all_ground_truths,
        target_class_ids,
    )


def refresh_shared_quality_calibrator(
    det_lora: DetLoRA,
    dataloader: DataLoader,
    class_names: Sequence[str],
    steps: int = 300,
    lr: float = 0.03,
    use_shared_encoder_cache: bool = True,
) -> Dict[str, Any]:
    """Fit and store one shared quality/objectness calibrator on mixed validation data."""
    predictions, ground_truths, target_class_ids = collect_det_lora_joint_predictions(
        det_lora,
        dataloader,
        class_names,
        use_shared_encoder_cache=use_shared_encoder_cache,
    )
    calibrator = fit_shared_quality_calibrator(
        predictions,
        ground_truths,
        target_class_ids,
        steps=steps,
        lr=lr,
    )
    det_lora.set_shared_quality_calibrator(calibrator)
    return calibrator


def refresh_conflict_gate(
    det_lora: DetLoRA,
    dataloader: DataLoader,
    class_names: Sequence[str],
    use_shared_encoder_cache: bool = True,
) -> Dict[str, Any]:
    """Fit and store the post-hoc cross-adapter conflict gate on calibration data.

    Builds one anisotropic Mahalanobis classifier per genuinely confusable class
    pair from the cross-adapter embeddings of conflicting objects. Replay-free:
    only compact per-pair Gaussian statistics are kept, adapters stay frozen.
    """
    predictions, ground_truths, target_class_ids = collect_det_lora_joint_predictions(
        det_lora,
        dataloader,
        class_names,
        use_shared_encoder_cache=use_shared_encoder_cache,
    )
    state = fit_pair_gate(predictions, ground_truths, target_class_ids)
    det_lora.set_conflict_gate(state)
    return state


def refresh_adapter_arbitration(
    det_lora: DetLoRA,
    dataloader: DataLoader,
    class_names: Sequence[str],
    version_selection_by_class: Optional[Dict[str, str]] = None,
    max_detections_per_image: Optional[int] = 150,
) -> Dict[str, Any]:
    """Fit and store compact class-prototype arbitration for joint adapter inference."""
    predictions, ground_truths, target_class_ids = collect_det_lora_joint_predictions(
        det_lora,
        dataloader,
        class_names,
        use_shared_encoder_cache=True,
        version_selection_by_class=version_selection_by_class,
    )
    if max_detections_per_image is not None:
        predictions = [
            _limit_prediction(prediction, max_detections_per_image) for prediction in predictions
        ]
    state = fit_adapter_arbitration_state(
        predictions,
        ground_truths,
        target_class_ids,
    )
    det_lora.set_adapter_arbitration_state(state)
    return state


class ContinualEvaluator:
    """
    Evaluator for continual learning experiments.

    Supports:
    - Det-LoRA joint evaluation by loading each learned adapter once
    - Standard detector evaluation for baseline methods
    - History tracking for forgetting / transfer summaries
    """

    def __init__(
        self,
        det_lora: Optional[DetLoRA] = None,
        confidence_threshold: Optional[float] = None,
        max_detections_per_image: Optional[int] = 100,
        use_shared_quality_calibrator: bool = False,
        use_adapter_arbitration: bool = False,
        use_conflict_gate: bool = False,
        use_shared_encoder_cache: bool = True,
    ):
        self.det_lora = det_lora
        self.confidence_threshold = confidence_threshold
        self.max_detections_per_image = max_detections_per_image
        self.use_shared_quality_calibrator = use_shared_quality_calibrator
        self.use_adapter_arbitration = use_adapter_arbitration
        self.use_conflict_gate = use_conflict_gate
        self.use_shared_encoder_cache = use_shared_encoder_cache
        self.history: Dict[int, Dict[str, Any]] = {}

    def _rename_ap_keys(
        self,
        metrics: Dict[str, Any],
        class_name_by_id: Dict[int, str],
    ) -> Dict[str, Any]:
        renamed = dict(metrics)
        for key in (
            "AP_per_class@0.5",
            "AP_per_class@0.75",
            "AP_per_class@0.95",
            "Precision_per_class@0.5",
            "Precision_per_class@0.75",
            "Precision_per_class@0.95",
            "Recall_per_class@0.5",
            "Recall_per_class@0.75",
            "Recall_per_class@0.95",
            "F1_per_class@0.5",
            "F1_per_class@0.75",
            "F1_per_class@0.95",
            "PR_curve_per_class@0.5",
            "PR_curve_per_class@0.95",
        ):
            if key not in metrics:
                continue
            if key.startswith("PR_curve_per_class@"):
                renamed[key] = {
                    class_name_by_id.get(int(cls_id), str(cls_id)): value
                    for cls_id, value in metrics.get(key, {}).items()
                }
            else:
                renamed[key] = {
                    class_name_by_id.get(int(cls_id), str(cls_id)): float(value)
                    for cls_id, value in metrics.get(key, {}).items()
                }
        return renamed

    def _record_history(
        self,
        task_idx: Optional[int],
        class_names: Sequence[str],
        metrics: Dict[str, Any],
    ) -> None:
        if task_idx is None:
            return
        self.history[task_idx] = {
            "metrics": metrics,
            "class_names": list(class_names),
        }

    def _postprocess_predictions(
        self,
        predictions: Sequence[Dict[str, np.ndarray]],
    ) -> List[Dict[str, np.ndarray]]:
        processed = list(predictions)
        if (
            self.use_shared_quality_calibrator
            and self.det_lora is not None
            and getattr(self.det_lora, "shared_quality_calibrator", {})
        ):
            processed = apply_shared_quality_calibrator(
                processed,
                self.det_lora.shared_quality_calibrator,
            )
        if (
            self.use_adapter_arbitration
            and self.det_lora is not None
            and getattr(self.det_lora, "adapter_arbitration_state", {})
        ):
            processed = apply_adapter_arbitration(
                processed,
                self.det_lora.adapter_arbitration_state,
            )
        if (
            self.use_conflict_gate
            and self.det_lora is not None
            and getattr(self.det_lora, "conflict_gate", {})
        ):
            processed = apply_conflict_gate(processed, self.det_lora.conflict_gate)
        return [
            _limit_prediction(prediction, self.max_detections_per_image) for prediction in processed
        ]

    @torch.no_grad()
    def evaluate_standard_detector(
        self,
        detector: RFDETRDetector,
        dataloader: DataLoader,
        class_names: Sequence[str],
        class_ids: Sequence[int],
        task_idx: Optional[int] = None,
        include_curves: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate a standard detector on a fixed subset of classes."""
        detector.model.eval()
        device = detector.device
        target_class_ids = [int(class_id) for class_id in class_ids]
        class_name_by_id = dict(zip(target_class_ids, class_names))

        all_predictions: List[Dict[str, np.ndarray]] = []
        all_ground_truths: List[Dict[str, np.ndarray]] = []

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            outputs = detector.forward(pixel_values=pixel_values)
            logits = outputs["pred_logits"]
            pred_boxes = outputs["pred_boxes"]

            for sample_idx in range(logits.shape[0]):
                probs = logits[sample_idx].sigmoid()
                prediction = _empty_prediction()

                for class_name, class_id in zip(class_names, target_class_ids):
                    class_scores = probs[:, class_id]
                    if self.det_lora is not None:
                        class_scores = self.det_lora.calibrate_scores(class_name, class_scores)
                    keep = _prediction_keep_mask(
                        class_scores,
                        self.confidence_threshold,
                    )
                    prediction = _append_prediction(
                        prediction,
                        pred_boxes[sample_idx][keep],
                        class_scores[keep],
                        class_id,
                    )

                all_predictions.append(prediction)
                all_ground_truths.append(
                    _filter_ground_truth(batch["labels"][sample_idx], target_class_ids)
                )

        all_predictions = self._postprocess_predictions(all_predictions)

        metrics = compute_map(
            all_predictions,
            all_ground_truths,
            target_class_ids=target_class_ids,
            include_curves=include_curves,
        )
        named_metrics = self._rename_ap_keys(metrics, class_name_by_id)
        self._record_history(task_idx, class_names, named_metrics)
        return named_metrics

    @torch.no_grad()
    def evaluate_det_lora_joint(
        self,
        dataloader: DataLoader,
        class_names: Sequence[str],
        task_idx: Optional[int] = None,
        include_curves: bool = False,
        version_selection_by_class: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate Det-LoRA in the actual continual-learning inference setting:
        each learned class uses its own adapter and predictions are merged.
        """
        if self.det_lora is None:
            raise ValueError("Det-LoRA evaluator requires a DetLoRA instance")

        all_predictions, all_ground_truths, target_class_ids = collect_det_lora_joint_predictions(
            self.det_lora,
            dataloader,
            class_names,
            confidence_threshold=self.confidence_threshold,
            use_shared_encoder_cache=self.use_shared_encoder_cache,
            version_selection_by_class=version_selection_by_class,
        )
        class_name_by_id = dict(zip(target_class_ids, class_names))

        all_predictions = self._postprocess_predictions(all_predictions)

        metrics = compute_map(
            all_predictions,
            all_ground_truths,
            target_class_ids=target_class_ids,
            include_curves=include_curves,
        )
        named_metrics = self._rename_ap_keys(metrics, class_name_by_id)
        self._record_history(task_idx, class_names, named_metrics)
        return named_metrics

    @torch.no_grad()
    def evaluate_det_lora_version_ensemble(
        self,
        dataloader: DataLoader,
        class_name: str,
        task_idx: Optional[int] = None,
        include_curves: bool = False,
        version_selection_strategy: str = "anchor_latest",
    ) -> Dict[str, Any]:
        """Evaluate one class by merging predictions from all frozen adapter versions."""
        if self.det_lora is None:
            raise ValueError("Det-LoRA evaluator requires a DetLoRA instance")

        all_predictions, all_ground_truths, target_class_ids = (
            collect_det_lora_version_ensemble_predictions(
                self.det_lora,
                dataloader,
                class_name,
                confidence_threshold=self.confidence_threshold,
                version_selection_strategy=version_selection_strategy,
            )
        )
        all_predictions = self._postprocess_predictions(all_predictions)

        metrics = compute_map(
            all_predictions,
            all_ground_truths,
            target_class_ids=target_class_ids,
            include_curves=include_curves,
        )
        named_metrics = self._rename_ap_keys(metrics, {target_class_ids[0]: class_name})
        self._record_history(task_idx, [class_name], named_metrics)
        return named_metrics

    def compute_forgetting(self) -> Dict[str, float]:
        """Forgetting = best historical AP - latest AP for each class."""
        if len(self.history) < 2:
            return {}

        latest_task = max(self.history.keys())
        latest_ap = self.history[latest_task]["metrics"].get("AP_per_class@0.5", {})
        best_ap_per_class: Dict[str, float] = {}

        for task_idx in sorted(self.history.keys()):
            for cls_key, ap in (
                self.history[task_idx]["metrics"].get("AP_per_class@0.5", {}).items()
            ):
                if cls_key not in best_ap_per_class or ap > best_ap_per_class[cls_key]:
                    best_ap_per_class[cls_key] = ap

        return {
            cls_key: max(0.0, best_ap - latest_ap.get(cls_key, 0.0))
            for cls_key, best_ap in best_ap_per_class.items()
        }

    def compute_forward_transfer(self) -> Dict[str, float]:
        """Track the overall mAP improvement from first to latest task."""
        if len(self.history) < 2:
            return {}

        first_task = min(self.history.keys())
        latest_task = max(self.history.keys())
        return {
            "mAP@0.5_improvement": (
                self.history[latest_task]["metrics"]["mAP@0.5"]
                - self.history[first_task]["metrics"]["mAP@0.5"]
            ),
        }

    def get_summary(self) -> str:
        """Get a formatted summary of all evaluations."""
        lines = [
            "=" * 60,
            "Continual Learning Evaluation Summary",
            "=" * 60,
        ]

        for task_idx in sorted(self.history.keys()):
            entry = self.history[task_idx]
            metrics = entry["metrics"]
            classes = entry.get("class_names", [])

            lines.append(f"\nAfter Task {task_idx + 1} ({', '.join(classes)}):")
            lines.append(f"  mAP@0.5:      {metrics['mAP@0.5']:.4f}")
            lines.append(f"  mAP@0.75:     {metrics.get('mAP@0.75', 0.0):.4f}")
            lines.append(f"  mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
            lines.append(f"  Precision@0.5:{metrics.get('Precision@0.5', 0.0):.4f}")
            lines.append(f"  Recall@0.5:   {metrics.get('Recall@0.5', 0.0):.4f}")
            lines.append(f"  F1@0.5:       {metrics.get('F1@0.5', 0.0):.4f}")

            for cls_key, ap in sorted(metrics.get("AP_per_class@0.5", {}).items()):
                lines.append(f"  {cls_key}: AP@0.5={ap:.4f}")

        forgetting = self.compute_forgetting()
        if forgetting:
            lines.append("\nForgetting (best AP - current AP):")
            for cls_key, drop in sorted(forgetting.items()):
                status = "ZERO" if drop < 0.001 else f"{drop:.4f}"
                lines.append(f"  {cls_key}: {status}")
            lines.append(f"  Average Forgetting: {float(np.mean(list(forgetting.values()))):.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save_results(self, path: str) -> None:
        """Save evaluation history and derived continual-learning metrics."""
        results = {
            "history": {},
            "forgetting": {k: float(v) for k, v in self.compute_forgetting().items()},
            "forward_transfer": {k: float(v) for k, v in self.compute_forward_transfer().items()},
        }

        for task_idx, entry in self.history.items():
            results["history"][str(task_idx)] = {
                "metrics": _to_builtin(entry["metrics"]),
                "class_names": entry.get("class_names", []),
            }

        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Evaluator] Results saved to {path}")


def _to_builtin(value: Any) -> Any:
    """Recursively convert numpy / torch-like scalars into JSON-safe builtins."""
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value
