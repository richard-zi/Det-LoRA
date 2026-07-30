#!/usr/bin/env python
"""M1: post-hoc cross-class competitive suppression (soft-NMS across classes).
For overlapping detections of DIFFERENT classes (same object), demote the
lower-scoring class hypotheses. Pure inference-time post-processing, no params
fit on data. Tested on top of config C (full calibration), nano/seed 42.
"""
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


def cross_class_softnms(preds, iou_thr, mode, score_floor=0.05):
    """Only consider detections above score_floor for competition (where confusion
    lives); leaves the low-score tail untouched. Vectorized IoU on the subset."""
    out = []
    for p in preds:
        q = dict(p)
        n = p["scores"].shape[0]
        s = p["scores"].astype(np.float32).copy()
        if n == 0:
            out.append(q)
            continue
        idx = np.where(s >= score_floor)[0]
        if idx.size > 1:
            boxes = p["boxes"].astype(np.float32)[idx]
            labs = p["labels"][idx]
            ss = s[idx]
            ious = iou_xyxy(boxes, boxes)
            diff_class = labs[:, None] != labs[None, :]
            stronger = ss[None, :] > ss[:, None]
            overlap = ious >= iou_thr
            # i is demoted if some j is stronger, different class, overlapping
            conflict = diff_class & stronger & overlap
            for ii in range(idx.size):
                js = np.where(conflict[ii])[0]
                if js.size:
                    if mode == "soft":
                        ss[ii] *= 1.0 - float(ious[ii, js].max())
                    else:
                        ss[ii] = 0.0
            s[idx] = ss
        q["scores"] = s
        out.append(q)
    return out


print("loading ...")
det = RFDETRDetector(variant="nano")
dl = DetLoRA(detector=det)
dl.load_all(CKPT)
mapping = _det_lora_class_id_mapping(dl, CLASSES)
ds = load_dataset_from_raw(
    raw_dir=RAW,
    class_filter=CLASSES,
    split="test",
    img_size=det.resolution,
    seed=42,
    max_samples=None,
    class_id_mapping=mapping,
)
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)
preds, gts, ids = collect_det_lora_joint_predictions(dl, loader, CLASSES)
te_c = apply_shared_quality_calibrator(preds, dl.shared_quality_calibrator)


def mAP(p):
    m = compute_map(p, gts, target_class_ids=list(ids), include_curves=False)
    return m["mAP@0.5"], m["mAP@0.5:0.95"]


b5, b9 = mAP(te_c)
print(f"\nBaseline C: mAP@0.5={b5:.3f}  mAP@.5:.95={b9:.3f}\n")
print(f"{'mode':>6}{'iou':>6}{'mAP@0.5':>10}{'d':>9}{'mAP@.5:.95':>12}{'d':>9}")
for mode in ["soft", "hard"]:
    for thr in [0.5, 0.7, 0.9]:
        rp = cross_class_softnms(te_c, thr, mode)
        m5, m9 = mAP(rp)
        print(f"{mode:>6}{thr:>6.1f}{m5:>10.3f}{m5-b5:>+9.3f}{m9:>12.3f}{m9-b9:>+9.3f}")
