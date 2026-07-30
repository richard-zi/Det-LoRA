#!/usr/bin/env python
"""Build a positive gallery: one clean, correct, high-confidence detection per class."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.arbitration import (
    apply_adapter_arbitration,
    simplify_joint_predictions_for_display,
)
from det_lora.evaluation.evaluator import collect_det_lora_joint_predictions
from det_lora.model.det_lora import DetLoRA
from det_lora.model.detector import RFDETRDetector
from det_lora.train import _det_lora_class_id_mapping, collate_fn

CKPT = "experiments/suites/thesis_l40_main/model_nano/seed_42/det_lora/final"
CLASSES = [
    "military_tank",
    "military_truck",
    "military_aircraft",
    "military_helicopter",
    "civilian_car",
    "civilian_aircraft",
]
RAW = "data/raw"
OUT = Path("outputs/positives")
OUT.mkdir(parents=True, exist_ok=True)
GREEN = (60, 180, 75)


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
id2name = {dl.get_class_id(c): c for c in CLASSES}
mapping = _det_lora_class_id_mapping(dl, CLASSES)
ds = load_dataset_from_raw(
    raw_dir=RAW,
    class_filter=CLASSES,
    split="test",
    img_size=det.resolution,
    seed=7,
    max_samples=500,
    class_id_mapping=mapping,
)
files = [f for f, _ in ds.samples]
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)
preds, gts, _ = collect_det_lora_joint_predictions(dl, loader, CLASSES)
preds = apply_adapter_arbitration(preds, dl.adapter_arbitration_state)
# allow multiple objects: no relative_score_margin, higher detection limit,
# class-agnostic NMS against duplicates.
disp = simplify_joint_predictions_for_display(
    preds,
    score_threshold=0.5,
    relative_score_margin=None,
    iou_threshold=0.5,
    max_detections_per_image=12,
    class_agnostic=True,
)

# best correct, single-object, high-score, HIGH-RESOLUTION example per class
from PIL import Image as _Im

_area_cache = {}


def img_area(fn):
    if fn not in _area_cache:
        try:
            w, h = _Im.open(Path(RAW) / "Images" / fn).size
            _area_cache[fn] = w * h
        except Exception:
            _area_cache[fn] = 0
    return _area_cache[fn]


EXCLUDE = {
    "0f0c81a7303883289bd9d0e9b468b834.jpg",  # dark tank image
    "1st_Fighter_Wing_hosts_coalition_aerial_exercise_28329.jpg",  # already used in the failure-case figure
}
best = {}  # class -> (area, score, idx, det_index)
for i, (d, gt) in enumerate(zip(disp, gts)):
    if d["scores"].size == 0 or gt["boxes"].size == 0:
        continue
    if files[i] in EXCLUDE:
        continue
    gt_names = {id2name.get(int(l), "?") for l in gt["labels"]}
    if len(gt_names) != 1:  # prefer clean single-class images
        continue
    ious = iou_xyxy(d["boxes"].astype(np.float32), gt["boxes"].astype(np.float32))
    for di in range(d["boxes"].shape[0]):
        name = id2name.get(int(d["labels"][di]), "?")
        j = int(np.argmax(ious[di]))
        correct = name in gt_names and ious[di, j] >= 0.5
        sc = float(d["scores"][di])
        if not correct or sc < 0.9:  # only confident, correct detections
            continue
        area = img_area(files[i])
        if area < 250 * 250:  # skip tiny/low-res thumbnails
            continue
        if name not in best or area > best[name][0]:  # prefer largest image
            best[name] = (area, sc, i, di)
best = {k: (v[1], v[2], v[3]) for k, v in best.items()}  # -> (score, idx, di)

font = None
for p in ["/System/Library/Fonts/Helvetica.ttc"]:
    try:
        font = ImageFont.truetype(p, 22)
        break
    except OSError:
        pass
font = font or ImageFont.load_default()

RED = (230, 25, 75)
panels = []
for c in CLASSES:
    if c not in best:
        print("no positive example for", c)
        continue
    sc, i, di = best[c]
    img = Image.open(Path(RAW) / "Images" / files[i]).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    d = disp[i]
    gt = gts[i]
    gt_names = {id2name.get(int(l), "?") for l in gt["labels"]}
    ious_all = (
        iou_xyxy(d["boxes"].astype(np.float32), gt["boxes"].astype(np.float32))
        if gt["boxes"].size
        else None
    )
    order = np.argsort(d["scores"])[::-1]
    n_drawn = 0
    for k in order:  # draw all detections
        name = id2name.get(int(d["labels"][k]), "?")
        correct = name in gt_names and ious_all is not None and ious_all[k].max() >= 0.5
        color = GREEN if correct else RED
        x1, y1, x2, y2 = d["boxes"][k]
        box = [x1 * W, y1 * H, x2 * W, y2 * H]
        draw.rectangle(box, outline=color, width=max(3, W // 220))
        label = f"{name.replace('military_','mil_').replace('civilian_','civ_')} {float(d['scores'][k]):.2f}"
        ty = max(0.0, box[1] - 24)
        tb = draw.textbbox((box[0], ty), label, font=font)
        draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=color)
        draw.text((box[0], ty), label, fill=(255, 255, 255), font=font)
        n_drawn += 1
    panels.append((c, img, sc))
    print(f"{c:<22} best={sc:.2f}  boxes={n_drawn}  {files[i]}")

# compose 2 rows x 3 cols
cell_w, cell_h = 520, 360
cols, rows = 3, 2
canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
for k, (c, img, sc) in enumerate(panels[:6]):
    r, cc = divmod(k, cols)
    im = img.copy()
    im.thumbnail((cell_w - 12, cell_h - 12), Image.LANCZOS)
    ox = cc * cell_w + (cell_w - im.width) // 2
    oy = r * cell_h + (cell_h - im.height) // 2
    canvas.paste(im, (ox, oy))
canvas.save("outputs/positives/gallery.png", quality=94)
print("gallery saved:", canvas.size, "| Panels:", len(panels))
