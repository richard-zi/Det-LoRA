#!/usr/bin/env python
"""Cross-Adapter Arbiter (post-training, replay-free, adapters frozen).

Per physical object we build a SYMMETRIC cross-adapter score profile: the 6-vector
[max score of adapter_k on this object]. Unlike the per-detection embedding (which
leaks the active adapter's signature), this profile lives in a shared space.
A small discriminative arbiter is trained on TRAIN object profiles -> true class,
then used at test to re-weight each detection's score by the arbiter posterior of
its own class. nano/seed 42, exploratory.
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
FLOOR = 0.1
CLUSTER_IOU = 0.5
torch.manual_seed(0)

det = RFDETRDetector(variant="nano")
dl = DetLoRA(detector=det)
dl.load_all(CKPT)
ids = [dl.get_class_id(c) for c in CLASSES]
C = len(ids)
cidx = {c: k for k, c in enumerate(ids)}
mapping = _det_lora_class_id_mapping(dl, CLASSES)


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


def clusters_of(pred):
    """Greedy class-agnostic clustering of high-score detections into objects.
    Returns list of (seed_idx, member_idxs, profile[C], box)."""
    s = pred["scores"].astype(np.float32)
    boxes = pred["boxes"].astype(np.float32)
    labs = pred["labels"]
    idx = np.where(s >= FLOOR)[0]
    if idx.size == 0:
        return []
    order = idx[np.argsort(s[idx])[::-1]]
    used = np.zeros(len(s), bool)
    out = []
    ious = iou_mat(boxes, boxes)
    for seed in order:
        if used[seed]:
            continue
        members = [int(m) for m in order if not used[m] and ious[seed, m] >= CLUSTER_IOU]
        for m in members:
            used[m] = True
        profile = np.zeros(C, np.float32)
        for m in members:
            k = cidx[int(labs[m])]
            profile[k] = max(profile[k], float(s[m]))
        out.append((int(seed), members, profile, boxes[seed]))
    return out


def geom(box):
    w = max(box[2] - box[0], 1e-6)
    h = max(box[3] - box[1], 1e-6)
    return np.array([np.log(w * h), np.log(w / h)], np.float32)


# ---- collect train, build object-level dataset ----
print("collecting training profiles ...")
tr_p, tr_g, _ = collect_det_lora_joint_predictions(dl, loader("train", TRAIN_SAMPLES, 42), CLASSES)
Xtr, ytr = [], []
for p, gt in zip(tr_p, tr_g):
    if gt["boxes"].size == 0:
        continue
    cl = clusters_of(p)
    if not cl:
        continue
    seed_boxes = np.stack([c[3] for c in cl])
    g = iou_mat(seed_boxes, gt["boxes"].astype(np.float32))
    for ci, (_, _, profile, box) in enumerate(cl):
        j = int(np.argmax(g[ci]))
        if g[ci, j] >= 0.5:
            Xtr.append(np.concatenate([profile, geom(box)]))
            ytr.append(cidx[int(gt["labels"][j])])
Xtr = np.stack(Xtr)
ytr = np.array(ytr)
print(f"  {len(Xtr)} object profiles, feature dim={Xtr.shape[1]}")
print("  class distribution:", {CLASSES[k]: int((ytr == k).sum()) for k in range(C)})

mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
Xn = (Xtr - mu) / sd
rng = np.random.RandomState(0)
perm = rng.permutation(len(Xn))
ntr = int(0.8 * len(Xn))
tri, vai = perm[:ntr], perm[ntr:]
Xt = torch.tensor(Xn[tri], dtype=torch.float32)
yt = torch.tensor(ytr[tri])
Xv = torch.tensor(Xn[vai], dtype=torch.float32)
yv = torch.tensor(ytr[vai])
D = Xtr.shape[1]

net = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, C))
opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
lossf = nn.CrossEntropyLoss()
best, bstate = 0, None
for ep in range(400):
    net.train()
    opt.zero_grad()
    loss = lossf(net(Xt), yt)
    loss.backward()
    opt.step()
    if ep % 20 == 0 or ep == 399:
        net.eval()
        with torch.no_grad():
            acc = (net(Xv).argmax(1) == yv).float().mean().item()
        if acc > best:
            best = acc
            bstate = {k: v.clone() for k, v in net.state_dict().items()}
net.load_state_dict(bstate)
net.eval()
print(f"\narbiter val accuracy (object profiles, 6 classes): {best:.3f}")
# crucial sub-question: mil_aircraft vs civ_aircraft separability
ma, ca = cidx[dl.get_class_id("military_aircraft")], cidx[dl.get_class_id("civilian_aircraft")]
with torch.no_grad():
    pv = net(Xv).argmax(1).numpy()
mask = np.isin(yv.numpy(), [ma, ca])
if mask.sum():
    print(
        f"  of which mil/civ aircraft val accuracy: {(pv[mask]==yv.numpy()[mask]).mean():.3f}  (n={int(mask.sum())})"
    )

# ---- test: re-weight by arbiter posterior of own class ----
te_p, te_g, _ = collect_det_lora_joint_predictions(dl, loader("test", None, 42), CLASSES)
te_c = apply_shared_quality_calibrator(te_p, dl.shared_quality_calibrator)
bm = compute_map(te_c, te_g, target_class_ids=ids, include_curves=False)
b5, b9 = bm["mAP@0.5"], bm["mAP@0.5:0.95"]
print(f"\n(context) dense baseline C: mAP@0.5={b5:.3f}  mAP@.5:.95={b9:.3f}")


# Fair comparison under the SAME object clustering: one detection per object,
# score = highest adapter score; only the class ASSIGNMENT differs:
#   naive  = class of the strongest adapter (winner-take-all, current behavior)
#   arbiter= class chosen by the cross-adapter arbiter
def build(mode):
    out = []
    for p in te_c:
        cl = clusters_of(p)
        boxes, scores, labels = [], [], []
        for _, members, profile, box in cl:
            sc = float(profile.max())
            if mode == "naive":
                k = int(np.argmax(profile))
            else:
                feat = (np.concatenate([profile, geom(box)]) - mu) / sd
                with torch.no_grad():
                    k = int(net(torch.tensor(feat[None], dtype=torch.float32)).argmax(1).item())
            boxes.append(box)
            scores.append(sc)
            labels.append(ids[k])
        out.append(
            {
                "boxes": np.array(boxes, np.float32).reshape(-1, 4),
                "scores": np.array(scores, np.float32),
                "labels": np.array(labels, np.int64),
            }
        )
    return out


for mode in ["naive", "arbiter"]:
    preds = build(mode)
    m = compute_map(preds, te_g, target_class_ids=ids, include_curves=False)
    print(
        f"[{mode:7}] (1 det/object) mAP@0.5={m['mAP@0.5']:.3f}  mAP@.5:.95={m['mAP@0.5:0.95']:.3f}",
        flush=True,
    )

# SELECTIVE application on the DENSE predictions: only on a real conflict
# (>=2 adapters with score>CONF on the same object) dampen the scores of the
# class detections the arbiter did NOT pick. Everything else stays untouched.
print("\nselective arbitration on dense predictions (only on conflict):")
print(
    f"{'CONF':>6}{'penalty':>8}{'mAP@0.5':>10}{'d':>9}{'mAP@.5:.95':>12}{'d':>9}{'#conflicts':>11}"
)
for CONF in [0.5]:
    for penalty in [0.5, 0.2, 0.0]:
        rer = []
        nconf = 0
        for p in te_c:
            q = dict(p)
            s = p["scores"].astype(np.float32).copy()
            for _, members, profile, box in clusters_of(p):
                if int((profile > CONF).sum()) < 2:
                    continue  # no conflict -> keep unchanged
                nconf += 1
                feat = (np.concatenate([profile, geom(box)]) - mu) / sd
                with torch.no_grad():
                    khat = int(net(torch.tensor(feat[None], dtype=torch.float32)).argmax(1).item())
                for m_ in members:
                    if cidx[int(p["labels"][m_])] != khat:
                        s[m_] *= penalty
            q["scores"] = s
            rer.append(q)
        mm = compute_map(rer, te_g, target_class_ids=ids, include_curves=False)
        m5, m9 = mm["mAP@0.5"], mm["mAP@0.5:0.95"]
        print(
            f"{CONF:>6.1f}{penalty:>8.1f}{m5:>10.3f}{m5-b5:>+9.3f}{m9:>12.3f}{m9-b9:>+9.3f}{nconf:>11}",
            flush=True,
        )
