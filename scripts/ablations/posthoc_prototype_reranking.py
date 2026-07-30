#!/usr/bin/env python
"""M2: post-hoc prototype (NCM) re-ranking on top of full calibration (config C).
Prototypes are fit ONLY on training data (replay-free, no adapter retraining).
Tests whether combining the calibrated score with an embedding-NCM posterior
improves mixed mAP. Exploratory sweep over (alpha, T) on nano/seed 42.
"""
import json

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
CKPT = "experiments/suites/thesis_l40_main/model_nano/seed_42/det_lora/final"
VARIANT, SEED = "nano", 42
TRAIN_SAMPLES = 800


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


def get_loader(det, split, samples, seed):
    mapping = _det_lora_class_id_mapping(det_lora_global, CLASSES)
    ds = load_dataset_from_raw(
        raw_dir=RAW,
        class_filter=CLASSES,
        split=split,
        img_size=det.resolution,
        seed=seed,
        max_samples=samples,
        class_id_mapping=mapping,
    )
    return DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)


def collect(det, loader):
    preds, gts, ids = collect_det_lora_joint_predictions(det, loader, CLASSES)
    return preds, gts, ids


def build_prototypes(preds, gts, ids):
    """Per-class mean L2-normalized embedding of TP detections (matched to GT)."""
    id_list = list(ids)
    acc = {cid: [] for cid in id_list}
    for p, gt in zip(preds, gts):
        emb = p.get("quality_features")
        if emb is None or p["scores"].size == 0:
            continue
        emb = np.asarray(emb, np.float32)
        if gt["boxes"].size == 0:
            continue
        ious = iou_xyxy(p["boxes"].astype(np.float32), gt["boxes"].astype(np.float32))
        for i in range(p["boxes"].shape[0]):
            lab = int(p["labels"][i])
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= 0.5 and int(gt["labels"][j]) == lab:
                acc[lab].append(emb[i])
    protos = {}
    for cid, lst in acc.items():
        if lst:
            v = np.mean(np.stack(lst), 0)
            protos[cid] = v / (np.linalg.norm(v) + 1e-9)
    return protos


def rerank(preds, protos, ids, alpha, T):
    cid_order = list(ids)
    P = np.stack([protos[c] for c in cid_order])  # (C,D)
    out = []
    for p in preds:
        q = dict(p)
        emb = p.get("quality_features")
        if emb is None or p["scores"].size == 0:
            out.append(q)
            continue
        emb = np.asarray(emb, np.float32)
        en = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        sims = en @ P.T  # (N,C) cosine
        post = np.exp(sims / T)
        post /= post.sum(1, keepdims=True)
        w = np.empty(p["scores"].shape[0], np.float32)
        for i in range(w.shape[0]):
            ci = cid_order.index(int(p["labels"][i]))
            w[i] = post[i, ci]
        q["scores"] = (p["scores"] ** (1 - alpha)) * (w**alpha)
        out.append(q)
    return out


def mAP(preds, gts, ids):
    m = compute_map(preds, gts, target_class_ids=list(ids), include_curves=False)
    return m["mAP@0.5"], m["mAP@0.5:0.95"]


print("loading model + data ...")
det = RFDETRDetector(variant=VARIANT)
det_lora_global = DetLoRA(detector=det)
det_lora_global.load_all(CKPT)

tr_loader = get_loader(det, "train", TRAIN_SAMPLES, SEED)
te_loader = get_loader(det, "test", None, SEED)

tr_p, tr_g, ids = collect(det_lora_global, tr_loader)
protos = build_prototypes(tr_p, tr_g, ids)
print("built prototypes for classes:", [c for c in ids if c in protos], "/", list(ids))

te_p, te_g, ids = collect(det_lora_global, te_loader)
# apply stage 2 (quality calibrator) -> matches config C
te_c = apply_shared_quality_calibrator(te_p, det_lora_global.shared_quality_calibrator)
base5, base9 = mAP(te_c, te_g, ids)
print(f"\nBaseline C (Nano/42): mAP@0.5={base5:.3f}  mAP@.5:.95={base9:.3f}\n")

print(f"{'alpha':>6}{'T':>6}{'mAP@0.5':>10}{'d@0.5':>9}{'mAP@.5:.95':>12}{'d':>9}")
for T in [0.05, 0.1, 0.2]:
    for alpha in [0.2, 0.4, 0.6, 0.8, 1.0]:
        rp = rerank(te_c, protos, ids, alpha, T)
        m5, m9 = mAP(rp, te_g, ids)
        print(f"{alpha:>6.1f}{T:>6.2f}{m5:>10.3f}{m5-base5:>+9.3f}{m9:>12.3f}{m9-base9:>+9.3f}")
