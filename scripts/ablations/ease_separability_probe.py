"""Decision-layer separability probe (EASE/RanPAC feasibility check).

Question: is military_aircraft vs civilian_aircraft separable in a SHARED,
leakage-free feature space well enough to beat the ~0.79/0.70 the cross-adapter
score-arbiter already reached? Only if substantially higher would an EASE/RanPAC-
style decision-layer method have headroom to beat the calibrated dense baseline.

Leakage control: every aircraft object (military AND civilian) is embedded under
the SAME fixed adapter, so the classifier cannot exploit adapter identity (the
failure mode of the per-detection learned arbiter). We report:
  - single fixed adapter (military), (civilian),
  - EASE-style concatenation of the object's embedding under BOTH aircraft adapters.
Classifiers: NCM (cosine), Mahalanobis (FeCAM-style, shrinkage), logistic regression.
Evaluated on a held-out split; chance = class prior.

This is an exploratory, removable diagnostic (no production code touched).

Usage (from the repo root):
  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --no-sync python scripts/ablations/ease_separability_probe.py
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from det_lora.data.dataset import load_dataset_from_raw
from det_lora.model.det_lora import DetLoRA
from det_lora.model.detector import RFDETRDetector
from det_lora.train import _det_lora_class_id_mapping, collate_fn

CKPT = "experiments/suites/thesis_l40_main/model_nano/seed_42/det_lora/final"
RAW = "data/raw"
PAIR = ["military_aircraft", "civilian_aircraft"]
IOU_MATCH = 0.5


def cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    x, y, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = cxcywh_to_xyxy(a)
    b = cxcywh_to_xyxy(b)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


@torch.no_grad()
def collect_object_features(det_lora, loader, adapters):
    """For each GT aircraft object, gather its embedding under each adapter (the
    max-IoU query). Returns dict adapter->[N,D], plus labels [N] (0/1) and a mask
    of objects matched under ALL adapters (for the concatenation variant)."""
    device = det_lora.device
    det_lora.set_eval_mode()
    det_lora.prepare_eval_adapter_cache(adapters)
    pair_ids = {det_lora.get_class_id(c): i for i, c in enumerate(PAIR)}

    per_adapter = {a: [] for a in adapters}
    labels = []
    matched_all = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        context = det_lora.detector.extract_shared_encoder_context(pixel_values)
        adapter_boxes, adapter_emb = {}, {}
        for adapter in adapters:
            det_lora.activate_cached_eval_adapter(adapter)
            outputs = det_lora.detector.forward_from_shared_encoder_context(context)
            adapter_boxes[adapter] = outputs["pred_boxes"].float().cpu().numpy()
            adapter_emb[adapter] = outputs["decoder_embeddings"].float().cpu().numpy()

        for sample_idx in range(pixel_values.shape[0]):
            gt = batch["labels"][sample_idx]
            gt_labels = gt["labels"].cpu().numpy()
            gt_boxes = gt["boxes"].cpu().numpy()
            keep = np.array([lbl in pair_ids for lbl in gt_labels])
            if not keep.any():
                continue
            gt_boxes = gt_boxes[keep]
            gt_cls = np.array([pair_ids[int(lbl)] for lbl in gt_labels[keep]])
            for obj_idx in range(gt_boxes.shape[0]):
                gt_box = gt_boxes[obj_idx : obj_idx + 1]
                emb_per_adapter = {}
                ok_all = True
                for adapter in adapters:
                    ious = iou_matrix(adapter_boxes[adapter][sample_idx], gt_box)[:, 0]
                    best = int(ious.argmax())
                    if ious[best] < IOU_MATCH:
                        ok_all = False
                    emb_per_adapter[adapter] = adapter_emb[adapter][sample_idx][best]
                for adapter in adapters:
                    per_adapter[adapter].append(emb_per_adapter[adapter])
                labels.append(int(gt_cls[obj_idx]))
                matched_all.append(ok_all)
    out = {a: np.stack(v) for a, v in per_adapter.items()}
    return out, np.array(labels), np.array(matched_all)


def ncm_accuracy(xtr, ytr, xte, yte):
    xtr = xtr / np.clip(np.linalg.norm(xtr, axis=1, keepdims=True), 1e-8, None)
    xte = xte / np.clip(np.linalg.norm(xte, axis=1, keepdims=True), 1e-8, None)
    protos = np.stack([xtr[ytr == c].mean(0) for c in (0, 1)])
    protos = protos / np.clip(np.linalg.norm(protos, axis=1, keepdims=True), 1e-8, None)
    pred = (xte @ protos.T).argmax(1)
    return (pred == yte).mean()


def mahalanobis_accuracy(xtr, ytr, xte, yte):
    preds = []
    means, precisions = [], []
    for c in (0, 1):
        xc = xtr[ytr == c]
        mean = xc.mean(0)
        cov = np.cov(xc.T) + np.eye(xc.shape[1]) * 1e-2  # shrinkage
        means.append(mean)
        precisions.append(np.linalg.pinv(cov))
    for x in xte:
        dists = [float((x - means[c]) @ precisions[c] @ (x - means[c])) for c in (0, 1)]
        preds.append(int(np.argmin(dists)))
    return (np.array(preds) == yte).mean()


def logreg_accuracy(xtr, ytr, xte, yte):
    """Standardized logistic regression (torch, L2-regularized) -- the linear view."""
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-8
    xtr_n = torch.tensor((xtr - mu) / sd, dtype=torch.float32)
    xte_n = torch.tensor((xte - mu) / sd, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    weight = torch.zeros(xtr_n.shape[1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=0.05, weight_decay=1e-3)
    for _ in range(400):
        optimizer.zero_grad()
        logits = xtr_n @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ytr_t)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = ((xte_n @ weight + bias) > 0).long().numpy()
    return (pred == yte).mean()


def report(name, xtr, ytr, xte, yte):
    print(f"\n--- {name}  (train n={len(ytr)}, test n={len(yte)}, dim={xtr.shape[1]}) ---")
    prior = max(yte.mean(), 1 - yte.mean())
    print(f"  Chance (Mehrheitsklasse): {prior:.3f}")
    print(f"  NCM (cosine):    {ncm_accuracy(xtr, ytr, xte, yte):.3f}")
    print(f"  Mahalanobis:     {mahalanobis_accuracy(xtr, ytr, xte, yte):.3f}")
    print(f"  LogReg:          {logreg_accuracy(xtr, ytr, xte, yte):.3f}")


def make_loader(split, seed, det_lora):
    ds = load_dataset_from_raw(
        raw_dir=RAW,
        class_filter=PAIR,
        split=split,
        class_id_offset=det_lora.detector.base_num_classes,
        img_size=det_lora.detector.resolution,
        seed=seed,
        class_id_mapping=_det_lora_class_id_mapping(det_lora, PAIR),
    )
    return DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)


def main():
    detector = RFDETRDetector(variant="nano")
    det_lora = DetLoRA(detector=detector, default_rank=8, default_alpha=16)
    det_lora.load_all(CKPT)

    tr = collect_object_features(det_lora, make_loader("train", 42, det_lora), PAIR)
    te = collect_object_features(det_lora, make_loader("test", 42, det_lora), PAIR)
    (xtr, ytr, mtr), (xte, yte, mte) = tr, te

    print("\n=========== EASE/RanPAC Feasibility: aircraft-pair separability ===========")
    print("reference: the cross-adapter score arbiter reached 0.79 overall / 0.70 on the pair.")

    # single fixed adapter (leakage-free: same adapter for both classes)
    for adapter in PAIR:
        report(f"single adapter = {adapter}", xtr[adapter], ytr, xte[adapter], yte)

    # EASE-style: concatenation of embeddings under BOTH aircraft adapters
    cat_tr = np.concatenate([xtr[PAIR[0]], xtr[PAIR[1]]], axis=1)
    cat_te = np.concatenate([xte[PAIR[0]], xte[PAIR[1]]], axis=1)
    report(
        "EASE concat (mil-adapter | civ-adapter embedding)",
        cat_tr[mtr],
        ytr[mtr],
        cat_te[mte],
        yte[mte],
    )


if __name__ == "__main__":
    main()
