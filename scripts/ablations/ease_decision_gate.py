"""Generic post-hoc decision-layer gate (EASE/RanPAC/FeCAM-style, replay-free).

CLASS-AGNOSTIC and architecture-level: no pair is hardwired. For every class PAIR
that actually co-fires in training (i.e. is genuinely confusable; this emerges
automatically from the data), a small discriminative Mahalanobis classifier is fit
on the cross-adapter embedding [emb|adapter_i , emb|adapter_j] of conflict objects.
At joint inference, whenever two class adapters both fire confidently on the same
object, the relevant pair classifier decides the winner and the loser's detections
are suppressed. Non-confusable pairs almost never co-fire confidently, so the gate
leaves easy classes untouched.

Only compact per-pair Gaussian statistics are stored (means+precision), replay-free
(no raw data), adapters stay frozen, fully incremental. Reports overall mixed mAP AND
per-class AP (to confirm easy classes are not harmed) plus a naive score-only control.

Usage (from the repo root):
  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --no-sync python scripts/ablations/ease_decision_gate.py --seed 42
"""

import argparse
from itertools import combinations

import numpy as np
import torch
from torch.utils.data import DataLoader

from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.evaluator import (
    apply_shared_quality_calibrator,
    collect_det_lora_joint_predictions,
    compute_map,
)
from det_lora.model.det_lora import DetLoRA
from det_lora.model.detector import RFDETRDetector
from det_lora.train import _det_lora_class_id_mapping, collate_fn

CLASSES = [
    "military_tank",
    "military_truck",
    "military_aircraft",
    "military_helicopter",
    "civilian_car",
    "civilian_aircraft",
]
RAW = "data/raw"
FLOOR = 0.1  # cluster membership floor
RESOLVE_TAU = 0.3  # both detections must exceed this for the gate to act (genuine conflict)
CLUSTER_IOU = 0.5
GT_IOU = 0.5
MIN_PAIR_SAMPLES = 60  # min objects per class to build a pair classifier
SHRINKAGE = 1e-1

_args = argparse.ArgumentParser()
_args.add_argument("--seed", type=int, default=42)
_args.add_argument("--variant", default="nano")
_args.add_argument("--ckpt", default=None)
ARGS = _args.parse_args()
SEED = ARGS.seed
CKPT = (
    ARGS.ckpt
    or f"experiments/suites/thesis_l40_main/model_{ARGS.variant}/seed_{SEED}/det_lora/final"
)

det = RFDETRDetector(variant=ARGS.variant)
dl = DetLoRA(detector=det)
dl.load_all(CKPT)
ids = [dl.get_class_id(c) for c in CLASSES]
NAME = {dl.get_class_id(c): c for c in CLASSES}


def iou_mat(a, b):
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(aa[:, None] + ab[None, :] - inter, 1e-9, None)


def loader(split, n, seed):
    mapping = _det_lora_class_id_mapping(dl, CLASSES)
    ds = load_dataset_from_raw(
        raw_dir=RAW,
        class_filter=CLASSES,
        split=split,
        img_size=det.resolution,
        seed=seed,
        max_samples=n,
        class_id_mapping=mapping,
    )
    return DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)


def clusters(pred):
    scores = pred["scores"].astype(np.float32)
    boxes = pred["boxes"].astype(np.float32)
    labels = pred["labels"]
    idx = np.where(scores >= FLOOR)[0]
    if idx.size == 0:
        return []
    order = idx[np.argsort(scores[idx])[::-1]]
    ious = iou_mat(boxes, boxes)
    used = np.zeros(len(scores), bool)
    out = []
    for seed in order:
        if used[seed]:
            continue
        members = [int(m) for m in order if not used[m] and ious[seed, m] >= CLUSTER_IOU]
        for m in members:
            used[m] = True
        by_class = {}
        for m in members:
            c = int(labels[m])
            if c not in by_class or scores[m] > scores[by_class[c]]:
                by_class[c] = m
        out.append({"members": members, "by_class": by_class})
    return out


def fit_pair_classifiers(preds, gts):
    """For every confusable class pair (i<j) that co-fires on the same object in
    training, collect [emb_i, emb_j] with the GT label and fit a 2-class Mahalanobis."""
    bank = {pair: {pair[0]: [], pair[1]: []} for pair in combinations(ids, 2)}
    for pred, gt in zip(preds, gts):
        emb = pred.get("quality_features")
        if emb is None:
            continue
        scores = pred["scores"].astype(np.float32)
        gt_boxes = gt["boxes"].astype(np.float32)
        gt_labels = gt["labels"]
        if gt_boxes.shape[0] == 0:
            continue
        for cl in clusters(pred):
            present = [c for c in cl["by_class"] if c in ids]
            if len(present) < 2:
                continue
            box = pred["boxes"][cl["by_class"][present[0]]][None, :].astype(np.float32)
            ious = iou_mat(box, gt_boxes)[0]
            best = int(ious.argmax())
            if ious[best] < GT_IOU:
                continue
            true_c = int(gt_labels[best])
            if true_c not in ids:
                continue
            for i, j in combinations(sorted(present), 2):
                if true_c not in (i, j):
                    continue
                feat = np.concatenate([emb[cl["by_class"][i]], emb[cl["by_class"][j]]])
                bank[(i, j)][true_c].append(feat)
    classifiers = {}
    for pair, by in bank.items():
        if len(by[pair[0]]) < MIN_PAIR_SAMPLES or len(by[pair[1]]) < MIN_PAIR_SAMPLES:
            continue
        means, precisions = [], []
        for c in pair:
            x = np.stack(by[c]).astype(np.float64)
            means.append(x.mean(0))
            cov = np.cov(x.T) + np.eye(x.shape[1]) * SHRINKAGE
            precisions.append(np.linalg.pinv(cov))
        classifiers[pair] = (means, precisions, len(by[pair[0]]), len(by[pair[1]]))
    return classifiers


def maha(feat, mean, prec):
    d = feat.astype(np.float64) - mean
    return float(d @ prec @ d)


def apply_gate(preds, classifiers, penalty, mode="maha"):
    gated = []
    n_act = 0
    for pred in preds:
        scores = pred["scores"].astype(np.float32).copy()
        raw = pred["scores"].astype(np.float32)
        emb = pred.get("quality_features")
        labels = pred["labels"]
        for cl in clusters(pred):
            present = [
                c for c in cl["by_class"] if c in ids and raw[cl["by_class"][c]] >= RESOLVE_TAU
            ]
            if len(present) < 2:
                continue
            losers = set()
            for i, j in combinations(sorted(present), 2):
                if mode == "maha":
                    if (i, j) not in classifiers:
                        continue
                    means, precisions, _, _ = classifiers[(i, j)]
                    feat = np.concatenate([emb[cl["by_class"][i]], emb[cl["by_class"][j]]])
                    win = (i, j)[
                        int(
                            np.argmin(
                                [
                                    maha(feat, means[0], precisions[0]),
                                    maha(feat, means[1], precisions[1]),
                                ]
                            )
                        )
                    ]
                else:  # naive: higher score wins
                    win = i if raw[cl["by_class"][i]] >= raw[cl["by_class"][j]] else j
                losers.add(j if win == i else i)
            if not losers:
                continue
            n_act += 1
            for m in cl["members"]:
                if int(labels[m]) in losers:
                    scores[m] *= penalty
        new = dict(pred)
        new["scores"] = scores
        gated.append(new)
    return gated, n_act


def main():
    print("Collecting TRAIN predictions (all classes, for pair classifiers)...")
    tr_p, tr_g, _ = collect_det_lora_joint_predictions(dl, loader("train", None, SEED), CLASSES)
    classifiers = fit_pair_classifiers(tr_p, tr_g)
    print("Confusable pairs with classifier:")
    for (i, j), (_, _, ni, nj) in classifiers.items():
        print(f"  {NAME[i][:9]:9} vs {NAME[j][:9]:9}  (n={ni}/{nj})")

    print("Collecting TEST predictions...")
    te_p, te_g, _ = collect_det_lora_joint_predictions(dl, loader("test", None, SEED), CLASSES)
    te_c = apply_shared_quality_calibrator(te_p, dl.shared_quality_calibrator)
    base = compute_map(te_c, te_g, target_class_ids=ids, include_curves=False)

    def line(tag, m):
        a = m["AP_per_class@0.5"]
        per = " ".join(f"{NAME[c].split('_')[1][:4]}={a[c]:.2f}" for c in ids)
        print(f"{tag:24} mAP@.5={m['mAP@0.5']:.4f} mAP={m['mAP@0.5:0.95']:.4f} | {per}")

    print("\n=========== GENERIC pairwise decision gate ===========")
    line("dichte Baseline B", base)
    naive, n = apply_gate(te_c, classifiers, 0.0, mode="naive")
    line(
        f"naiv (Score, n={n})", compute_map(naive, te_g, target_class_ids=ids, include_curves=False)
    )
    for penalty in [0.0, 0.5, 0.75]:
        gated, n = apply_gate(te_c, classifiers, penalty, mode="maha")
        line(
            f"gate (penalty={penalty}, n={n})",
            compute_map(gated, te_g, target_class_ids=ids, include_curves=False),
        )


if __name__ == "__main__":
    main()
