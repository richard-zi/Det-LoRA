#!/usr/bin/env python
"""FeCAM-style re-ranking (Goswami et al., NeurIPS 2023): anisotropic Mahalanobis
distance with per-class covariance + correlation normalization, fit ONLY on train
embeddings (replay-free, adapters untouched). Tests whether this beats the naive
cosine-NCM (M2) and improves over config C. nano/seed 42, exploratory.
"""
from pathlib import Path

import numpy as np
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
CKPT = "experiments/suites/thesis_l40_main/model_nano/seed_42/det_lora/final"
TRAIN_SAMPLES = 1500


def iou_xyxy(a, b):
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


det = RFDETRDetector(variant="nano")
dl = DetLoRA(detector=det)
dl.load_all(CKPT)
ids = [dl.get_class_id(c) for c in CLASSES]
mapping = _det_lora_class_id_mapping(dl, CLASSES)


def loader(split, n, seed):
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


# collect TP embeddings per class on train
tr_p, tr_g, _ = collect_det_lora_joint_predictions(dl, loader("train", TRAIN_SAMPLES, 42), CLASSES)
feats = {c: [] for c in ids}
for p, gt in zip(tr_p, tr_g):
    emb = p.get("quality_features")
    if emb is None or p["scores"].size == 0 or gt["boxes"].size == 0:
        continue
    emb = np.asarray(emb, np.float32)
    ious = iou_xyxy(p["boxes"].astype(np.float32), gt["boxes"].astype(np.float32))
    for i in range(p["boxes"].shape[0]):
        lab = int(p["labels"][i])
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= 0.5 and int(gt["labels"][j]) == lab:
            feats[lab].append(emb[i])
for c in ids:
    print(f"  class {c}: {len(feats[c])} TP embeddings")

D = next(len(v[0]) for v in feats.values() if v)


def build_gaussians(shrink, corr_norm):
    mus, precis = {}, {}
    for c in ids:
        X = np.stack(feats[c]).astype(np.float64)
        mu = X.mean(0)
        Xc = X - mu
        cov = (Xc.T @ Xc) / max(len(X) - 1, 1)
        if corr_norm:  # FeCAM correlation normalization
            d = np.sqrt(np.clip(np.diag(cov), 1e-8, None))
            cov = cov / np.outer(d, d)
        cov = (1 - shrink) * cov + shrink * np.eye(D) * np.trace(cov) / D
        mus[c] = mu
        precis[c] = np.linalg.inv(cov + 1e-6 * np.eye(D))
    return mus, precis


def mahalanobis_post(emb, mus, precis, corr_norm, T):
    en = emb.astype(np.float64)
    dists = np.zeros((en.shape[0], len(ids)))
    for k, c in enumerate(ids):
        diff = en - mus[c]
        if corr_norm:
            d = np.sqrt(
                np.clip(np.diag(np.linalg.inv(precis[c])), 1e-8, None)
            )  # approx; keep simple
        dists[:, k] = np.einsum("ni,ij,nj->n", diff, precis[c], diff)
    post = np.exp(-dists / (2 * T))
    post /= np.clip(post.sum(1, keepdims=True), 1e-12, None)
    return post


# test predictions = config C
te_p, te_g, _ = collect_det_lora_joint_predictions(dl, loader("test", None, 42), CLASSES)
te_c = apply_shared_quality_calibrator(te_p, dl.shared_quality_calibrator)
bm = compute_map(te_c, te_g, target_class_ids=ids, include_curves=False)
b5, b9 = bm["mAP@0.5"], bm["mAP@0.5:0.95"]
print(f"\nBaseline C: mAP@0.5={b5:.3f}  mAP@.5:.95={b9:.3f}\n")
print(
    f"{'corrNorm':>9}{'shrink':>8}{'T':>6}{'alpha':>7}{'mAP@0.5':>10}{'d':>9}{'mAP@.5:.95':>12}{'d':>9}"
)

cid_index = {c: k for k, c in enumerate(ids)}
FLOOR = 0.1  # only rerank high-score detections (that is where the confusion sits)
for corr_norm in [True, False]:
    for shrink in [0.5]:
        mus, precis = build_gaussians(shrink, corr_norm)
        for T in [10.0, 100.0]:
            for alpha in [0.5, 1.0]:
                rer = []
                for p in te_c:
                    q = dict(p)
                    emb = p.get("quality_features")
                    s = p["scores"].astype(np.float32).copy()
                    if emb is None or s.size == 0:
                        rer.append(q)
                        continue
                    idx = np.where(s >= FLOOR)[0]
                    if idx.size:
                        post = mahalanobis_post(
                            np.asarray(emb, np.float32)[idx], mus, precis, corr_norm, T
                        )
                        w = np.array(
                            [post[k, cid_index[int(p["labels"][idx[k]])]] for k in range(idx.size)]
                        )
                        s[idx] = (s[idx] ** (1 - alpha)) * (w.astype(np.float32) ** alpha)
                    q["scores"] = s
                    rer.append(q)
                m = compute_map(rer, te_g, target_class_ids=ids, include_curves=False)
                m5, m9 = m["mAP@0.5"], m["mAP@0.5:0.95"]
                print(
                    f"{str(corr_norm):>9}{shrink:>8.1f}{T:>6.0f}{alpha:>7.1f}{m5:>10.3f}{m5-b5:>+9.3f}{m9:>12.3f}{m9-b9:>+9.3f}",
                    flush=True,
                )
