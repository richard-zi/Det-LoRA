"""
Detection Evaluation Metrics
==============================

COCO-style mAP computation plus precision / recall style summaries for
object-detection evaluation.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def compute_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between predicted and ground truth boxes.

    Args:
        pred_boxes: [N, 4] in xyxy format
        gt_boxes: [M, 4] in xyxy format

    Returns:
        IoU matrix [N, M]
    """
    x1 = np.maximum(pred_boxes[:, 0:1], gt_boxes[:, 0:1].T)
    y1 = np.maximum(pred_boxes[:, 1:2], gt_boxes[:, 1:2].T)
    x2 = np.minimum(pred_boxes[:, 2:3], gt_boxes[:, 2:3].T)
    y2 = np.minimum(pred_boxes[:, 3:4], gt_boxes[:, 3:4].T)

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    union = area_pred[:, None] + area_gt[None, :] - intersection
    return intersection / np.maximum(union, 1e-8)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using 101-point interpolation (COCO style)."""
    recall_thresholds = np.linspace(0, 1, 101)
    interpolated_precisions = np.zeros_like(recall_thresholds)

    for i, t in enumerate(recall_thresholds):
        mask = recalls >= t
        if mask.any():
            interpolated_precisions[i] = precisions[mask].max()

    return float(interpolated_precisions.mean())


def compute_ap_per_class(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    target_class_ids: Optional[Sequence[int]] = None,
) -> Dict[int, float]:
    """
    Compute AP per class at a given IoU threshold.

    Args:
        predictions: List of {"boxes": [N,4] xyxy, "scores": [N], "labels": [N]}
        ground_truths: List of {"boxes": [M,4] xyxy, "labels": [M]}
        iou_threshold: IoU threshold for matching

    Returns:
        Dict mapping class_id -> AP
    """
    target_set = (
        {int(cls_id) for cls_id in target_class_ids} if target_class_ids is not None else None
    )

    # Collect predictions and ground truths per class. When target_class_ids is
    # given, restrict evaluation to exactly that subset instead of averaging in
    # unrelated false-positive classes.
    all_classes = set(target_set or [])
    gt_per_class: Dict[int, List] = {}
    pred_per_class: Dict[int, List] = {}

    for img_idx, gt in enumerate(ground_truths):
        for j, label in enumerate(gt["labels"]):
            label_int = int(label)
            if target_set is not None and label_int not in target_set:
                continue
            all_classes.add(label_int)
            gt_per_class.setdefault(label_int, []).append(
                {"img_idx": img_idx, "box": gt["boxes"][j], "matched": False}
            )

    for img_idx, pred in enumerate(predictions):
        for j, label in enumerate(pred["labels"]):
            label_int = int(label)
            if target_set is not None and label_int not in target_set:
                continue
            all_classes.add(label_int)
            pred_per_class.setdefault(label_int, []).append(
                {
                    "img_idx": img_idx,
                    "box": pred["boxes"][j],
                    "score": float(pred["scores"][j]),
                }
            )

    # Compute AP per class
    ap_per_class = {}
    for cls_id in sorted(all_classes):
        class_metrics = _compute_class_metrics(
            pred_per_class.get(cls_id, []),
            gt_per_class.get(cls_id, []),
            iou_threshold=iou_threshold,
        )
        if class_metrics is None:
            continue
        ap_per_class[cls_id] = class_metrics["ap"]

    return ap_per_class


def _compute_class_metrics(
    preds: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    iou_threshold: float,
) -> Optional[Dict[str, Any]]:
    """
    Compute per-class detection statistics at a single IoU threshold.

    Returns None if the class has no ground-truth boxes in the evaluation set.
    """
    if not gts:
        return None

    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    gts = [dict(gt, matched=False) for gt in gts]

    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    n_gt = len(gts)

    for i, pred in enumerate(preds):
        best_iou = 0.0
        best_gt_idx = -1

        pred_box = np.array(pred["box"]).reshape(1, -1)
        for gt_idx, gt in enumerate(gts):
            if gt["img_idx"] != pred["img_idx"]:
                continue
            if gt["matched"]:
                continue

            gt_box = np.array(gt["box"]).reshape(1, -1)
            iou = compute_iou_matrix(pred_box, gt_box)[0, 0]

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[i] = 1
            gts[best_gt_idx]["matched"] = True
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / max(n_gt, 1)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-8)
    score_thresholds = np.asarray([pred["score"] for pred in preds], dtype=float)

    ap = compute_ap(recalls, precisions)
    final_tp = int(tp.sum())
    final_fp = int(fp.sum())
    final_fn = max(n_gt - final_tp, 0)

    precision = final_tp / max(final_tp + final_fp, 1)
    recall = final_tp / max(n_gt, 1)
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "ap": float(ap),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(final_tp),
        "fp": float(final_fp),
        "fn": float(final_fn),
        "num_gt": float(n_gt),
        "num_pred": float(len(preds)),
        "precision_curve": precisions.astype(float).tolist(),
        "recall_curve": recalls.astype(float).tolist(),
        "score_thresholds": score_thresholds.astype(float).tolist(),
    }


def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresholds: Optional[List[float]] = None,
    target_class_ids: Optional[Sequence[int]] = None,
    include_curves: bool = False,
) -> Dict[str, Any]:
    """
    Compute COCO-style mAP metrics.

    Args:
        predictions: List of {"boxes": [N,4] xyxy, "scores": [N], "labels": [N]}
        ground_truths: List of {"boxes": [M,4] xyxy, "labels": [M]}
        iou_thresholds: IoU thresholds (default: [0.5] and [0.5:0.95:0.05])

    Returns:
        Dict with "mAP@0.5", "mAP@0.5:0.95", per-class APs, and optionally
        per-class precision-recall curve data when include_curves is True.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5 to 0.95

    stats_at_50 = _compute_stats_per_class(
        predictions,
        ground_truths,
        0.5,
        target_class_ids=target_class_ids,
    )
    stats_at_75 = _compute_stats_per_class(
        predictions,
        ground_truths,
        0.75,
        target_class_ids=target_class_ids,
    )
    stats_at_95 = _compute_stats_per_class(
        predictions,
        ground_truths,
        0.95,
        target_class_ids=target_class_ids,
    )
    ap_at_50 = {cls_id: stats["ap"] for cls_id, stats in stats_at_50.items()}
    ap_at_75 = {cls_id: stats["ap"] for cls_id, stats in stats_at_75.items()}
    ap_at_95 = {cls_id: stats["ap"] for cls_id, stats in stats_at_95.items()}
    precision_at_50 = {cls_id: stats["precision"] for cls_id, stats in stats_at_50.items()}
    recall_at_50 = {cls_id: stats["recall"] for cls_id, stats in stats_at_50.items()}
    f1_at_50 = {cls_id: stats["f1"] for cls_id, stats in stats_at_50.items()}
    precision_at_75 = {cls_id: stats["precision"] for cls_id, stats in stats_at_75.items()}
    recall_at_75 = {cls_id: stats["recall"] for cls_id, stats in stats_at_75.items()}
    f1_at_75 = {cls_id: stats["f1"] for cls_id, stats in stats_at_75.items()}
    precision_at_95 = {cls_id: stats["precision"] for cls_id, stats in stats_at_95.items()}
    recall_at_95 = {cls_id: stats["recall"] for cls_id, stats in stats_at_95.items()}
    f1_at_95 = {cls_id: stats["f1"] for cls_id, stats in stats_at_95.items()}

    mean_ap_50 = float(np.mean(list(ap_at_50.values()))) if ap_at_50 else 0.0
    mean_ap_75 = float(np.mean(list(ap_at_75.values()))) if ap_at_75 else 0.0
    mean_ap_95 = float(np.mean(list(ap_at_95.values()))) if ap_at_95 else 0.0
    mean_precision_50 = float(np.mean(list(precision_at_50.values()))) if precision_at_50 else 0.0
    mean_precision_75 = float(np.mean(list(precision_at_75.values()))) if precision_at_75 else 0.0
    mean_precision_95 = float(np.mean(list(precision_at_95.values()))) if precision_at_95 else 0.0
    mean_recall_50 = float(np.mean(list(recall_at_50.values()))) if recall_at_50 else 0.0
    mean_recall_75 = float(np.mean(list(recall_at_75.values()))) if recall_at_75 else 0.0
    mean_recall_95 = float(np.mean(list(recall_at_95.values()))) if recall_at_95 else 0.0
    mean_f1_50 = float(np.mean(list(f1_at_50.values()))) if f1_at_50 else 0.0
    mean_f1_75 = float(np.mean(list(f1_at_75.values()))) if f1_at_75 else 0.0
    mean_f1_95 = float(np.mean(list(f1_at_95.values()))) if f1_at_95 else 0.0

    # mAP@0.5:0.95
    all_aps = []
    for iou_t in iou_thresholds:
        ap = compute_ap_per_class(
            predictions,
            ground_truths,
            iou_t,
            target_class_ids=target_class_ids,
        )
        all_aps.append(ap)

    # Average over IoU thresholds
    all_classes = set()
    for ap in all_aps:
        all_classes.update(ap.keys())

    mean_ap_5095 = 0.0
    if all_classes:
        class_aps = []
        for cls_id in all_classes:
            cls_mean = np.mean([ap.get(cls_id, 0.0) for ap in all_aps])
            class_aps.append(cls_mean)
        mean_ap_5095 = float(np.mean(class_aps))

    total_tp_50 = int(sum(stats["tp"] for stats in stats_at_50.values()))
    total_fp_50 = int(sum(stats["fp"] for stats in stats_at_50.values()))
    total_fn_50 = int(sum(stats["fn"] for stats in stats_at_50.values()))
    total_tp_75 = int(sum(stats["tp"] for stats in stats_at_75.values()))
    total_fp_75 = int(sum(stats["fp"] for stats in stats_at_75.values()))
    total_fn_75 = int(sum(stats["fn"] for stats in stats_at_75.values()))
    total_tp_95 = int(sum(stats["tp"] for stats in stats_at_95.values()))
    total_fp_95 = int(sum(stats["fp"] for stats in stats_at_95.values()))
    total_fn_95 = int(sum(stats["fn"] for stats in stats_at_95.values()))
    micro_precision_50 = total_tp_50 / max(total_tp_50 + total_fp_50, 1)
    micro_recall_50 = total_tp_50 / max(total_tp_50 + total_fn_50, 1)
    micro_f1_50 = 0.0
    if micro_precision_50 + micro_recall_50 > 0:
        micro_f1_50 = (
            2 * micro_precision_50 * micro_recall_50 / (micro_precision_50 + micro_recall_50)
        )
    micro_precision_75 = total_tp_75 / max(total_tp_75 + total_fp_75, 1)
    micro_recall_75 = total_tp_75 / max(total_tp_75 + total_fn_75, 1)
    micro_f1_75 = 0.0
    if micro_precision_75 + micro_recall_75 > 0:
        micro_f1_75 = (
            2 * micro_precision_75 * micro_recall_75 / (micro_precision_75 + micro_recall_75)
        )
    micro_precision_95 = total_tp_95 / max(total_tp_95 + total_fp_95, 1)
    micro_recall_95 = total_tp_95 / max(total_tp_95 + total_fn_95, 1)
    micro_f1_95 = 0.0
    if micro_precision_95 + micro_recall_95 > 0:
        micro_f1_95 = (
            2 * micro_precision_95 * micro_recall_95 / (micro_precision_95 + micro_recall_95)
        )

    result: Dict[str, Any] = {
        "mAP@0.5": mean_ap_50,
        "mAP@0.75": mean_ap_75,
        "mAP@0.95": mean_ap_95,
        "mAP@0.5:0.95": mean_ap_5095,
        "Precision@0.5": mean_precision_50,
        "Precision@0.75": mean_precision_75,
        "Precision@0.95": mean_precision_95,
        "Recall@0.5": mean_recall_50,
        "Recall@0.75": mean_recall_75,
        "Recall@0.95": mean_recall_95,
        "F1@0.5": mean_f1_50,
        "F1@0.75": mean_f1_75,
        "F1@0.95": mean_f1_95,
        "MicroPrecision@0.5": float(micro_precision_50),
        "MicroPrecision@0.75": float(micro_precision_75),
        "MicroPrecision@0.95": float(micro_precision_95),
        "MicroRecall@0.5": float(micro_recall_50),
        "MicroRecall@0.75": float(micro_recall_75),
        "MicroRecall@0.95": float(micro_recall_95),
        "MicroF1@0.5": float(micro_f1_50),
        "MicroF1@0.75": float(micro_f1_75),
        "MicroF1@0.95": float(micro_f1_95),
        "TP@0.5": total_tp_50,
        "TP@0.75": total_tp_75,
        "TP@0.95": total_tp_95,
        "FP@0.5": total_fp_50,
        "FP@0.75": total_fp_75,
        "FP@0.95": total_fp_95,
        "FN@0.5": total_fn_50,
        "FN@0.75": total_fn_75,
        "FN@0.95": total_fn_95,
        "AP_per_class@0.5": ap_at_50,
        "Precision_per_class@0.5": precision_at_50,
        "Recall_per_class@0.5": recall_at_50,
        "F1_per_class@0.5": f1_at_50,
        "AP_per_class@0.75": ap_at_75,
        "Precision_per_class@0.75": precision_at_75,
        "Recall_per_class@0.75": recall_at_75,
        "F1_per_class@0.75": f1_at_75,
        "AP_per_class@0.95": ap_at_95,
        "Precision_per_class@0.95": precision_at_95,
        "Recall_per_class@0.95": recall_at_95,
        "F1_per_class@0.95": f1_at_95,
    }

    if include_curves:

        def _curve_payload(stats_by_class: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
            return {
                cls_id: {
                    "precision": list(stats.get("precision_curve", [])),
                    "recall": list(stats.get("recall_curve", [])),
                    "score_thresholds": list(stats.get("score_thresholds", [])),
                    "ap": float(stats.get("ap", 0.0)),
                    "tp": float(stats.get("tp", 0.0)),
                    "fp": float(stats.get("fp", 0.0)),
                    "fn": float(stats.get("fn", 0.0)),
                    "num_gt": float(stats.get("num_gt", 0.0)),
                    "num_pred": float(stats.get("num_pred", 0.0)),
                }
                for cls_id, stats in stats_by_class.items()
            }

        result["PR_curve_per_class@0.5"] = _curve_payload(stats_at_50)
        result["PR_curve_per_class@0.95"] = _curve_payload(stats_at_95)

    return result


def _compute_stats_per_class(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    target_class_ids: Optional[Sequence[int]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Compute detailed per-class detection statistics at a single IoU threshold."""
    target_set = (
        {int(cls_id) for cls_id in target_class_ids} if target_class_ids is not None else None
    )

    all_classes = set(target_set or [])
    gt_per_class: Dict[int, List] = {}
    pred_per_class: Dict[int, List] = {}

    for img_idx, gt in enumerate(ground_truths):
        for j, label in enumerate(gt["labels"]):
            label_int = int(label)
            if target_set is not None and label_int not in target_set:
                continue
            all_classes.add(label_int)
            gt_per_class.setdefault(label_int, []).append(
                {"img_idx": img_idx, "box": gt["boxes"][j]}
            )

    for img_idx, pred in enumerate(predictions):
        for j, label in enumerate(pred["labels"]):
            label_int = int(label)
            if target_set is not None and label_int not in target_set:
                continue
            all_classes.add(label_int)
            pred_per_class.setdefault(label_int, []).append(
                {
                    "img_idx": img_idx,
                    "box": pred["boxes"][j],
                    "score": float(pred["scores"][j]),
                }
            )

    stats_per_class = {}
    for cls_id in sorted(all_classes):
        class_metrics = _compute_class_metrics(
            pred_per_class.get(cls_id, []),
            gt_per_class.get(cls_id, []),
            iou_threshold=iou_threshold,
        )
        if class_metrics is not None:
            stats_per_class[cls_id] = class_metrics
    return stats_per_class
