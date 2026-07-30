#!/usr/bin/env python
"""Learned arbiter: a DISCRIMINATIVE classifier on the frozen decoder embeddings,
trained replay-free on stored train statistics (adapters untouched). Unlike the
generative NCM/FeCAM re-rankers, a softmax classifier learns the decision boundary
directly and can separate close class means if any linear/non-linear signal exists.

Diagnostics: reports the arbiter's held-out classification accuracy (does the
embedding even carry the distinction?) AND the effect of blending its posterior
into the calibrated scores on mixed mAP (config C).
nano/seed 42, exploratory.
"""
import numpy as np
import torch
import torch.nn as nn
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
DEV = "cpu"  # small classifier; CPU is fine and deterministic
torch.manual_seed(0)


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
cidx = {c: k for k, c in enumerate(ids)}
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


def gather_tp(preds, gts):
    X, y = [], []
    for p, gt in zip(preds, gts):
        emb = p.get("quality_features")
        if emb is None or p["scores"].size == 0 or gt["boxes"].size == 0:
            continue
        emb = np.asarray(emb, np.float32)
        ious = iou_xyxy(p["boxes"].astype(np.float32), gt["boxes"].astype(np.float32))
        for i in range(p["boxes"].shape[0]):
            lab = int(p["labels"][i])
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= 0.5 and int(gt["labels"][j]) == lab:
                X.append(emb[i])
                y.append(cidx[lab])
    return np.stack(X), np.array(y)


print("Sammle Trainings-Embeddings ...")
tr_p, tr_g, _ = collect_det_lora_joint_predictions(dl, loader("train", TRAIN_SAMPLES, 42), CLASSES)
X, y = gather_tp(tr_p, tr_g)
print(f"  {X.shape[0]} TP-Embeddings, Dim={X.shape[1]}")

mu, sd = X.mean(0), X.std(0) + 1e-6
Xn = (X - mu) / sd
# 80/20 train/val split (diagnose separability)
rng = np.random.RandomState(0)
perm = rng.permutation(len(Xn))
ntr = int(0.8 * len(Xn))
tr_i, va_i = perm[:ntr], perm[ntr:]
Xtr = torch.tensor(Xn[tr_i], dtype=torch.float32)
ytr = torch.tensor(y[tr_i])
Xva = torch.tensor(Xn[va_i], dtype=torch.float32)
yva = torch.tensor(y[va_i])
D = X.shape[1]
C = len(ids)


def train_clf(kind):
    if kind == "linear":
        net = nn.Linear(D, C)
    else:
        net = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, C))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    best_acc, best_state = 0, None
    for ep in range(300):
        net.train()
        opt.zero_grad()
        out = net(Xtr)
        loss = lossf(out, ytr)
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == 299:
            net.eval()
            with torch.no_grad():
                acc = (net(Xva).argmax(1) == yva).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    return net, best_acc


# test predictions (config C)
te_p, te_g, _ = collect_det_lora_joint_predictions(dl, loader("test", None, 42), CLASSES)
te_c = apply_shared_quality_calibrator(te_p, dl.shared_quality_calibrator)
bm = compute_map(te_c, te_g, target_class_ids=ids, include_curves=False)
b5, b9 = bm["mAP@0.5"], bm["mAP@0.5:0.95"]
print(f"\nBaseline C: mAP@0.5={b5:.3f}  mAP@.5:.95={b9:.3f}")

FLOOR = 0.1
for kind in ["linear", "mlp"]:
    net, vacc = train_clf(kind)
    print(f"\n[{kind}] Arbiter Val-Accuracy (6-Klassen, Embedding-Trennbarkeit): {vacc:.3f}")
    print(f"{'alpha':>6}{'mAP@0.5':>10}{'d':>9}{'mAP@.5:.95':>12}{'d':>9}")
    for alpha in [0.3, 0.5, 0.7, 1.0]:
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
                en = (np.asarray(emb, np.float32)[idx] - mu) / sd
                with torch.no_grad():
                    post = torch.softmax(net(torch.tensor(en, dtype=torch.float32)), 1).numpy()
                w = np.array([post[k, cidx[int(p["labels"][idx[k]])]] for k in range(idx.size)])
                s[idx] = (s[idx] ** (1 - alpha)) * (w.astype(np.float32) ** alpha)
            q["scores"] = s
            rer.append(q)
        m = compute_map(rer, te_g, target_class_ids=ids, include_curves=False)
        m5, m9 = m["mAP@0.5"], m["mAP@0.5:0.95"]
        print(f"{alpha:>6.1f}{m5:>10.3f}{m5-b5:>+9.3f}{m9:>12.3f}{m9-b9:>+9.3f}", flush=True)

    # DECISIVE TEST: re-assign each detection's class label to the arbiter argmax.
    # If the embedding truly separates classes -> huge gain. If leakage (embedding
    # encodes the active adapter) -> argmax == original label -> no change.
    rer = []
    n_changed = 0
    n_tot = 0
    id_arr = np.array(ids)
    for p in te_c:
        q = dict(p)
        emb = p.get("quality_features")
        s = p["scores"].astype(np.float32)
        if emb is None or s.size == 0:
            rer.append(q)
            continue
        labs = p["labels"].copy()
        idx = np.where(s >= FLOOR)[0]
        if idx.size:
            en = (np.asarray(emb, np.float32)[idx] - mu) / sd
            with torch.no_grad():
                pred = net(torch.tensor(en, dtype=torch.float32)).argmax(1).numpy()
            new = id_arr[pred]
            n_changed += int((new != labs[idx]).sum())
            n_tot += idx.size
            labs[idx] = new
        q["labels"] = labs
        rer.append(q)
    m = compute_map(rer, te_g, target_class_ids=ids, include_curves=False)
    print(
        f"[{kind}] RE-ASSIGN labels: mAP@0.5={m['mAP@0.5']:.3f} ({m['mAP@0.5']-b5:+.3f}), "
        f"mAP@.5:.95={m['mAP@0.5:0.95']:.3f} ({m['mAP@0.5:0.95']-b9:+.3f}); "
        f"Labels geaendert: {n_changed}/{n_tot} ({100*n_changed/max(n_tot,1):.1f}%)",
        flush=True,
    )
