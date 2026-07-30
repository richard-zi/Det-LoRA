"""Post-hoc cross-adapter conflict gate (replay-free, adapters frozen).

Resolves residual cross-class confusion at the *decision layer* of Det-LoRA's joint
inference, without retraining or modifying any adapter. The motivation and empirical
validation are documented in the thesis (third design iteration): an already-trained
per-class adapter cannot be re-hardened against a confusable class post-hoc (the
classes are entangled in its committed representation), but the confusion *can* be
resolved at inference by comparing the independent experts in a shared embedding
space -- in line with exemplar-free prototype methods such as FeCAM, EASE and RanPAC.

Mechanism (class-agnostic, generic over all pairs):
  fit:   for every class pair (i, j) that genuinely co-fires on the same object in
         the calibration data, fit a 2-class anisotropic Mahalanobis classifier on the
         cross-adapter embedding [emb|adapter_i, emb|adapter_j] (mean + covariance per
         class). Only confusable pairs accumulate enough samples; well-separated pairs
         are skipped automatically.
  apply: whenever >= 2 class adapters fire confidently on the same object, the relevant
         pair classifier picks the winner and the losing detections' scores are damped.
         Non-conflicting detections are left untouched, so well-separated classes are
         not affected.

Only compact per-pair Gaussian statistics ("values") are stored -- no raw data, so the
replay-free property is preserved.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

DEFAULT_FLOOR = 0.1  # cluster membership floor
DEFAULT_RESOLVE_TAU = 0.3  # both detections must exceed this for the gate to act
DEFAULT_CLUSTER_IOU = 0.5
DEFAULT_GT_IOU = 0.5
DEFAULT_MIN_SAMPLES = 60  # min objects per class to build a pair classifier
DEFAULT_SHRINKAGE = 1e-1
DEFAULT_PENALTY = 0.5  # score multiplier applied to losing detections


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def _clusters(prediction: Dict[str, np.ndarray], floor: float, cluster_iou: float):
    """Greedy class-agnostic spatial clustering of high-score detections.

    Yields dicts {members:[idx], by_class:{class_id: best_member_idx}}.
    """
    scores = prediction["scores"].astype(np.float32)
    boxes = prediction["boxes"].astype(np.float32)
    labels = prediction["labels"]
    idx = np.where(scores >= floor)[0]
    if idx.size == 0:
        return []
    order = idx[np.argsort(scores[idx])[::-1]]
    ious = _iou_matrix(boxes, boxes)
    used = np.zeros(len(scores), dtype=bool)
    out = []
    for seed in order:
        if used[seed]:
            continue
        members = [int(m) for m in order if not used[m] and ious[seed, m] >= cluster_iou]
        for m in members:
            used[m] = True
        by_class: Dict[int, int] = {}
        for m in members:
            c = int(labels[m])
            if c not in by_class or scores[m] > scores[by_class[c]]:
                by_class[c] = m
        out.append({"members": members, "by_class": by_class})
    return out


def fit_pair_gate(
    predictions: Sequence[Dict[str, np.ndarray]],
    ground_truths: Sequence[Dict[str, np.ndarray]],
    class_ids: Sequence[int],
    floor: float = DEFAULT_FLOOR,
    cluster_iou: float = DEFAULT_CLUSTER_IOU,
    gt_iou: float = DEFAULT_GT_IOU,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    shrinkage: float = DEFAULT_SHRINKAGE,
) -> Dict[str, Any]:
    """Fit one 2-class Mahalanobis classifier per confusable class pair.

    Returns a serializable state dict consumable by ``apply_conflict_gate``.
    """
    class_ids = list(class_ids)
    bank: Dict[tuple, Dict[int, List[np.ndarray]]] = {
        pair: {pair[0]: [], pair[1]: []} for pair in combinations(class_ids, 2)
    }
    for prediction, gt in zip(predictions, ground_truths):
        embeddings = prediction.get("quality_features")
        if embeddings is None:
            continue
        boxes = prediction["boxes"].astype(np.float32)
        gt_boxes = np.asarray(gt["boxes"], dtype=np.float32)
        gt_labels = np.asarray(gt["labels"])
        if gt_boxes.shape[0] == 0:
            continue
        for cluster in _clusters(prediction, floor, cluster_iou):
            present = [c for c in cluster["by_class"] if c in class_ids]
            if len(present) < 2:
                continue
            seed_box = boxes[cluster["by_class"][present[0]]][None, :]
            ious = _iou_matrix(seed_box, gt_boxes)[0]
            best = int(ious.argmax())
            if ious[best] < gt_iou:
                continue
            true_class = int(gt_labels[best])
            if true_class not in class_ids:
                continue
            for i, j in combinations(sorted(present), 2):
                if true_class not in (i, j):
                    continue
                feature = np.concatenate(
                    [embeddings[cluster["by_class"][i]], embeddings[cluster["by_class"][j]]]
                )
                bank[(i, j)][true_class].append(feature)

    pairs: Dict[str, Any] = {}
    for pair, by_class in bank.items():
        if len(by_class[pair[0]]) < min_samples or len(by_class[pair[1]]) < min_samples:
            continue
        means, precisions = [], []
        for c in pair:
            x = np.stack(by_class[c]).astype(np.float64)
            means.append(x.mean(0))
            cov = np.cov(x.T) + np.eye(x.shape[1]) * shrinkage
            precisions.append(np.linalg.pinv(cov))
        pairs[f"{pair[0]},{pair[1]}"] = {
            "classes": [int(pair[0]), int(pair[1])],
            "means": [m.astype(np.float32) for m in means],
            "precisions": [p.astype(np.float32) for p in precisions],
            "counts": [len(by_class[pair[0]]), len(by_class[pair[1]])],
        }
    return {
        "pairs": pairs,
        "floor": floor,
        "cluster_iou": cluster_iou,
        "resolve_tau": DEFAULT_RESOLVE_TAU,
        "penalty": DEFAULT_PENALTY,
    }


def _mahalanobis(feature: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> float:
    diff = feature.astype(np.float64) - mean.astype(np.float64)
    return float(diff @ precision.astype(np.float64) @ diff)


def apply_conflict_gate(
    predictions: Sequence[Dict[str, np.ndarray]],
    state: Dict[str, Any],
    penalty: Optional[float] = None,
    resolve_tau: Optional[float] = None,
) -> List[Dict[str, np.ndarray]]:
    """Damp losing detections of genuine cross-class object conflicts."""
    pairs = state.get("pairs", {})
    if not pairs:
        return list(predictions)
    floor = float(state.get("floor", DEFAULT_FLOOR))
    cluster_iou = float(state.get("cluster_iou", DEFAULT_CLUSTER_IOU))
    penalty = float(state.get("penalty", DEFAULT_PENALTY)) if penalty is None else penalty
    resolve_tau = (
        float(state.get("resolve_tau", DEFAULT_RESOLVE_TAU)) if resolve_tau is None else resolve_tau
    )
    lookup = {tuple(entry["classes"]): entry for entry in pairs.values()}
    gated: List[Dict[str, np.ndarray]] = []
    for prediction in predictions:
        embeddings = prediction.get("quality_features")
        if embeddings is None:
            gated.append(prediction)
            continue
        scores = prediction["scores"].astype(np.float32).copy()
        raw = prediction["scores"].astype(np.float32)
        labels = prediction["labels"]
        for cluster in _clusters(prediction, floor, cluster_iou):
            present = [c for c in cluster["by_class"] if raw[cluster["by_class"][c]] >= resolve_tau]
            if len(present) < 2:
                continue
            losers: set = set()
            for i, j in combinations(sorted(present), 2):
                entry = lookup.get((i, j))
                if entry is None:
                    continue
                feature = np.concatenate(
                    [embeddings[cluster["by_class"][i]], embeddings[cluster["by_class"][j]]]
                )
                d_i = _mahalanobis(feature, entry["means"][0], entry["precisions"][0])
                d_j = _mahalanobis(feature, entry["means"][1], entry["precisions"][1])
                losers.add(j if d_i <= d_j else i)
            if not losers:
                continue
            for m in cluster["members"]:
                if int(labels[m]) in losers:
                    scores[m] = scores[m] * penalty
        new_prediction = dict(prediction)
        new_prediction["scores"] = scores
        gated.append(new_prediction)
    return gated
