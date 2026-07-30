"""
Adapter arbitration for Det-LoRA joint inference.

The state is intentionally compact: class prototypes and scalar weights only.
It does not retain old images or raw validation detections.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from det_lora.evaluation.metrics import compute_iou_matrix, compute_map


def _logit_np(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores.astype(np.float32), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32, copy=False)
    positive = logits >= 0.0
    out = np.empty_like(logits, dtype=np.float32)
    out[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    out[~positive] = exp_logits / (1.0 + exp_logits)
    return out


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-8, None)


def _prediction_features(prediction: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    features = prediction.get("arbitration_features", prediction.get("quality_features"))
    if features is None:
        return None
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != prediction["scores"].shape[0]:
        return None
    return _normalize_rows(features)


def _matched_feature_rows(
    prediction: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    class_id: int,
    iou_threshold: float,
) -> List[np.ndarray]:
    features = _prediction_features(prediction)
    if features is None:
        return []

    pred_mask = prediction["labels"] == class_id
    gt_mask = ground_truth["labels"] == class_id
    pred_indices = np.flatnonzero(pred_mask)
    gt_boxes = ground_truth["boxes"][gt_mask].astype(np.float32)
    if pred_indices.size == 0 or gt_boxes.shape[0] == 0:
        return []

    pred_boxes = prediction["boxes"][pred_indices].astype(np.float32)
    pred_scores = prediction["scores"][pred_indices].astype(np.float32)
    order = np.argsort(pred_scores)[::-1]
    ious = compute_iou_matrix(pred_boxes[order], gt_boxes)
    matched_gt: set[int] = set()
    rows: List[np.ndarray] = []

    for rank_idx, local_pred_idx in enumerate(order):
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
            rows.append(features[pred_indices[local_pred_idx]])
            matched_gt.add(best_gt)

    return rows


def _build_prototypes(
    predictions: Sequence[Dict[str, np.ndarray]],
    ground_truths: Sequence[Dict[str, np.ndarray]],
    target_class_ids: Sequence[int],
    iou_threshold: float,
    max_classifier_rows_per_image: int = 24,
) -> tuple[Dict[str, List[float]], Dict[str, int], np.ndarray, np.ndarray]:
    prototypes: Dict[str, List[float]] = {}
    counts: Dict[str, int] = {}
    feature_rows: List[np.ndarray] = []
    label_rows: List[int] = []
    classifier_feature_rows: List[np.ndarray] = []
    classifier_label_rows: List[int] = []
    target_id_set = {int(class_id) for class_id in target_class_ids}

    for class_id in target_class_ids:
        rows: List[np.ndarray] = []
        for prediction, ground_truth in zip(predictions, ground_truths):
            rows.extend(
                _matched_feature_rows(
                    prediction,
                    ground_truth,
                    int(class_id),
                    iou_threshold,
                )
            )
        if not rows:
            continue
        feature_rows.extend(rows)
        label_rows.extend([int(class_id)] * len(rows))
        prototype = np.stack(rows, axis=0).mean(axis=0, dtype=np.float32)
        prototype = prototype / np.clip(np.linalg.norm(prototype), 1e-8, None)
        prototypes[str(int(class_id))] = prototype.astype(np.float32).tolist()
        counts[str(int(class_id))] = len(rows)

    for prediction, ground_truth in zip(predictions, ground_truths):
        features = _prediction_features(prediction)
        if features is None or prediction["scores"].shape[0] == 0:
            continue

        gt_labels = ground_truth["labels"].astype(np.int64)
        gt_keep = np.flatnonzero(np.isin(gt_labels, list(target_id_set)))
        pred_keep = np.flatnonzero(
            np.isin(prediction["labels"].astype(np.int64), list(target_id_set))
        )
        if gt_keep.size == 0 or pred_keep.size == 0:
            continue

        pred_scores = prediction["scores"][pred_keep].astype(np.float32)
        pred_keep = pred_keep[np.argsort(pred_scores)[::-1][:max_classifier_rows_per_image]]
        ious = compute_iou_matrix(
            prediction["boxes"][pred_keep].astype(np.float32),
            ground_truth["boxes"][gt_keep].astype(np.float32),
        )
        for row_idx, pred_idx in enumerate(pred_keep):
            best_gt = int(np.argmax(ious[row_idx]))
            if float(ious[row_idx, best_gt]) < iou_threshold:
                continue
            classifier_feature_rows.append(features[pred_idx])
            classifier_label_rows.append(int(gt_labels[gt_keep[best_gt]]))

    if feature_rows:
        classifier_features = (
            np.stack(classifier_feature_rows, axis=0).astype(np.float32)
            if classifier_feature_rows
            else np.stack(feature_rows, axis=0).astype(np.float32)
        )
        classifier_labels = (
            np.asarray(classifier_label_rows, dtype=np.int64)
            if classifier_label_rows
            else np.asarray(label_rows, dtype=np.int64)
        )
        return (
            prototypes,
            counts,
            classifier_features,
            classifier_labels,
        )
    return prototypes, counts, np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)


def _fit_region_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    target_class_ids: Sequence[int],
    *,
    steps: int = 400,
    lr: float = 0.05,
) -> Dict[str, Any]:
    if features.shape[0] == 0:
        return {"identity": True, "reason": "no_features"}

    class_ids = [int(class_id) for class_id in target_class_ids]
    class_to_index = {class_id: idx for idx, class_id in enumerate(class_ids)}
    y_indices = np.asarray([class_to_index[int(label)] for label in labels], dtype=np.int64)
    if len(set(y_indices.tolist())) < 2:
        return {"identity": True, "reason": "single_class"}

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(y_indices, dtype=torch.long)
    weight = torch.zeros((len(class_ids), x.shape[1]), dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((len(class_ids),), dtype=torch.float32, requires_grad=True)
    counts = torch.bincount(y, minlength=len(class_ids)).float()
    class_weights = counts.sum() / counts.clamp_min(1.0)
    class_weights = class_weights / class_weights.mean().clamp_min(1e-8)
    optimizer = torch.optim.Adam([weight, bias], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        logits = x @ weight.t() + bias
        loss = F.cross_entropy(logits, y, weight=class_weights)
        loss = loss + 1e-3 * weight.pow(2).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = x @ weight.t() + bias
        accuracy = float((logits.argmax(dim=1) == y).float().mean().item())

    return {
        "identity": False,
        "class_ids": class_ids,
        "weight": weight.detach().cpu().tolist(),
        "bias": bias.detach().cpu().tolist(),
        "train_count": int(features.shape[0]),
        "train_accuracy": accuracy,
    }


def _cluster_prediction_indices(
    boxes: np.ndarray,
    overlap_iou_threshold: float,
) -> List[np.ndarray]:
    if boxes.shape[0] == 0:
        return []

    remaining = set(range(boxes.shape[0]))
    clusters: List[np.ndarray] = []
    iou_matrix = compute_iou_matrix(boxes.astype(np.float32), boxes.astype(np.float32))

    while remaining:
        seed = min(remaining)
        cluster = {seed}
        frontier = {seed}
        while frontier:
            current = frontier.pop()
            neighbours = {
                idx
                for idx in remaining
                if idx not in cluster and float(iou_matrix[current, idx]) >= overlap_iou_threshold
            }
            cluster.update(neighbours)
            frontier.update(neighbours)
        remaining.difference_update(cluster)
        clusters.append(np.array(sorted(cluster), dtype=np.int64))

    return clusters


def apply_adapter_arbitration(
    predictions: Sequence[Dict[str, np.ndarray]],
    state: Optional[Dict[str, Any]],
) -> List[Dict[str, np.ndarray]]:
    """Re-score overlapping multi-adapter detections with compact class prototypes."""
    if not state or state.get("identity"):
        return list(predictions)

    prototypes = {
        int(class_id): np.asarray(vector, dtype=np.float32)
        for class_id, vector in state.get("class_prototypes", {}).items()
    }
    if not prototypes:
        return list(predictions)

    prototype_weight = float(state.get("prototype_weight", 0.0))
    classifier_weight = float(state.get("classifier_weight", 0.0))
    loser_penalty = float(state.get("loser_penalty", 0.0))
    overlap_iou_threshold = float(state.get("overlap_iou_threshold", 0.6))
    class_biases = {
        int(class_id): float(bias) for class_id, bias in state.get("class_biases", {}).items()
    }
    adjusted_predictions: List[Dict[str, np.ndarray]] = []

    for prediction in predictions:
        features = _prediction_features(prediction)
        if features is None or prediction["scores"].shape[0] == 0:
            adjusted_predictions.append(prediction)
            continue

        scores = prediction["scores"].astype(np.float32)
        labels = prediction["labels"].astype(np.int64)
        logits = _logit_np(scores)
        similarities = np.zeros_like(scores, dtype=np.float32)
        classifier_scores = np.zeros_like(scores, dtype=np.float32)

        for row_idx, class_id in enumerate(labels.tolist()):
            prototype = prototypes.get(int(class_id))
            if prototype is None or prototype.shape[0] != features.shape[1]:
                continue
            similarities[row_idx] = float(np.dot(features[row_idx], prototype))

        classifier = state.get("region_classifier", {})
        if classifier and not classifier.get("identity"):
            classifier_class_ids = [int(class_id) for class_id in classifier.get("class_ids", [])]
            classifier_index = {class_id: idx for idx, class_id in enumerate(classifier_class_ids)}
            weight = np.asarray(classifier["weight"], dtype=np.float32)
            bias = np.asarray(classifier["bias"], dtype=np.float32)
            if weight.ndim == 2 and weight.shape[1] == features.shape[1]:
                all_classifier_scores = features @ weight.T + bias
                for row_idx, class_id in enumerate(labels.tolist()):
                    class_idx = classifier_index.get(int(class_id))
                    if class_idx is not None:
                        classifier_scores[row_idx] = float(
                            all_classifier_scores[row_idx, class_idx]
                        )

        biases = np.array(
            [class_biases.get(int(class_id), 0.0) for class_id in labels.tolist()],
            dtype=np.float32,
        )
        adjusted_logits = (
            logits
            + biases
            + prototype_weight * similarities
            + classifier_weight * classifier_scores
        )
        clusters = _cluster_prediction_indices(
            prediction["boxes"].astype(np.float32),
            overlap_iou_threshold,
        )

        for cluster in clusters:
            if cluster.size <= 1 or np.unique(labels[cluster]).size <= 1:
                continue
            winner = int(cluster[np.argmax(adjusted_logits[cluster])])
            losing = cluster[labels[cluster] != labels[winner]]
            adjusted_logits[losing] -= loser_penalty

        adjusted = dict(prediction)
        adjusted["scores"] = _sigmoid_np(adjusted_logits)
        adjusted_predictions.append(adjusted)

    return adjusted_predictions


def simplify_joint_predictions_for_display(
    predictions: Sequence[Dict[str, np.ndarray]],
    *,
    score_threshold: float = 0.5,
    relative_score_margin: Optional[float] = None,
    iou_threshold: float = 0.6,
    max_detections_per_image: int = 5,
    class_agnostic: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """
    Reduce dense ranking predictions to a small visual/demo set.

    This is intentionally separate from AP evaluation. AP needs many ranked
    detections; visual inference needs one readable object-level output.
    """
    simplified: List[Dict[str, np.ndarray]] = []
    for prediction in predictions:
        scores = prediction["scores"].astype(np.float32)
        effective_threshold = float(score_threshold)
        if relative_score_margin is not None and scores.shape[0] > 0:
            top_score = float(np.max(scores))
            effective_threshold = max(effective_threshold, top_score - float(relative_score_margin))
        keep = np.flatnonzero(scores >= effective_threshold)
        if keep.size == 0:
            simplified.append({key: value[:0] for key, value in prediction.items()})
            continue

        ordered = keep[np.argsort(scores[keep])[::-1]]
        selected: List[int] = []
        boxes = prediction["boxes"].astype(np.float32)
        labels = prediction["labels"].astype(np.int64)

        for candidate in ordered:
            if len(selected) >= max_detections_per_image:
                break
            if selected:
                selected_boxes = boxes[np.asarray(selected, dtype=np.int64)]
                ious = compute_iou_matrix(boxes[[candidate]], selected_boxes)[0]
                if class_agnostic:
                    has_overlap = np.any(ious >= iou_threshold)
                else:
                    selected_labels = labels[np.asarray(selected, dtype=np.int64)]
                    has_overlap = np.any(
                        (ious >= iou_threshold) & (selected_labels == labels[candidate])
                    )
                if has_overlap:
                    continue
            selected.append(int(candidate))

        selected_array = np.asarray(selected, dtype=np.int64)
        simplified.append(
            {
                key: (
                    value[selected_array]
                    if isinstance(value, np.ndarray) and value.shape[:1] == scores.shape[:1]
                    else value
                )
                for key, value in prediction.items()
            }
        )

    return simplified


def fit_adapter_arbitration_state(
    predictions: Sequence[Dict[str, np.ndarray]],
    ground_truths: Sequence[Dict[str, np.ndarray]],
    target_class_ids: Sequence[int],
    *,
    prototype_iou_threshold: float = 0.5,
    overlap_iou_threshold: float = 0.6,
    max_classifier_rows_per_image: int = 24,
    prototype_weight_grid: Sequence[float] = (0.0, 1.5),
    classifier_weight_grid: Sequence[float] = (0.0, 0.5),
    loser_penalty_grid: Sequence[float] = (0.0, 1.5),
    class_bias_grid: Sequence[float] = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0),
    optimize_metric: str = "mAP@0.5:0.95",
) -> Dict[str, Any]:
    """Fit compact arbitration state and tune its scalar weights on validation predictions."""
    prototypes, counts, matched_features, matched_labels = _build_prototypes(
        predictions,
        ground_truths,
        target_class_ids,
        prototype_iou_threshold,
        max_classifier_rows_per_image=max_classifier_rows_per_image,
    )
    if not prototypes:
        return {
            "identity": True,
            "reason": "no_matched_feature_prototypes",
            "prototype_iou_threshold": float(prototype_iou_threshold),
            "overlap_iou_threshold": float(overlap_iou_threshold),
            "class_prototypes": {},
            "prototype_counts": {},
        }

    best_state: Dict[str, Any] = {
        "identity": False,
        "prototype_iou_threshold": float(prototype_iou_threshold),
        "overlap_iou_threshold": float(overlap_iou_threshold),
        "max_classifier_rows_per_image": int(max_classifier_rows_per_image),
        "class_prototypes": prototypes,
        "prototype_counts": counts,
        "prototype_weight": 0.0,
        "classifier_weight": 0.0,
        "loser_penalty": 0.0,
        "class_biases": {str(int(class_id)): 0.0 for class_id in target_class_ids},
        "region_classifier": _fit_region_classifier(
            matched_features,
            matched_labels,
            target_class_ids,
        ),
        "optimize_metric": optimize_metric,
        "validation_metric": float("-inf"),
    }

    for prototype_weight in prototype_weight_grid:
        for classifier_weight in classifier_weight_grid:
            for loser_penalty in loser_penalty_grid:
                candidate = {
                    **best_state,
                    "prototype_weight": float(prototype_weight),
                    "classifier_weight": float(classifier_weight),
                    "loser_penalty": float(loser_penalty),
                }
                adjusted = apply_adapter_arbitration(predictions, candidate)
                metrics = compute_map(
                    adjusted,
                    ground_truths,
                    target_class_ids=target_class_ids,
                )
                metric_value = float(metrics.get(optimize_metric, metrics.get("mAP@0.5", 0.0)))
                if metric_value > float(best_state["validation_metric"]):
                    best_state["prototype_weight"] = float(prototype_weight)
                    best_state["classifier_weight"] = float(classifier_weight)
                    best_state["loser_penalty"] = float(loser_penalty)
                    best_state["validation_metric"] = metric_value
                    best_state["validation_mAP@0.5"] = float(metrics.get("mAP@0.5", 0.0))
                    best_state["validation_mAP@0.95"] = float(metrics.get("mAP@0.95", 0.0))
                    best_state["validation_mAP@0.5:0.95"] = float(metrics.get("mAP@0.5:0.95", 0.0))

    for _ in range(2):
        for class_id in target_class_ids:
            class_key = str(int(class_id))
            current_biases = dict(best_state.get("class_biases", {}))
            best_bias = float(current_biases.get(class_key, 0.0))
            best_metric = float(best_state["validation_metric"])
            for class_bias in class_bias_grid:
                candidate_biases = dict(current_biases)
                candidate_biases[class_key] = float(class_bias)
                candidate = {
                    **best_state,
                    "class_biases": candidate_biases,
                }
                adjusted = apply_adapter_arbitration(predictions, candidate)
                metrics = compute_map(
                    adjusted,
                    ground_truths,
                    target_class_ids=target_class_ids,
                )
                metric_value = float(metrics.get(optimize_metric, metrics.get("mAP@0.5", 0.0)))
                if metric_value > best_metric:
                    best_bias = float(class_bias)
                    best_metric = metric_value
                    best_state["validation_metric"] = metric_value
                    best_state["validation_mAP@0.5"] = float(metrics.get("mAP@0.5", 0.0))
                    best_state["validation_mAP@0.95"] = float(metrics.get("mAP@0.95", 0.0))
                    best_state["validation_mAP@0.5:0.95"] = float(metrics.get("mAP@0.5:0.95", 0.0))
            best_state["class_biases"][class_key] = best_bias

    return best_state
