#!/usr/bin/env python
"""Calibration ablation (inference-only, no retraining) on stored Det-LoRA adapters.
Three configs, arbitration held OFF to isolate the calibration effect:
  A: no calibration   (Platt identity, quality OFF)
  B: Platt only        (Stage 1 on, quality OFF)
  C: both              (Stage 1 + shared quality calibrator)
Metric: mixed mAP@0.5 and mAP@0.5:0.95 over all 6 classes on the test split.
"""
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.evaluator import ContinualEvaluator
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
SUITE = "experiments/suites/thesis_l40_main"
CKPTS = [("nano", 42), ("nano", 43), ("nano", 44), ("base", 42)]
OUT = Path(__file__).parent / "data/calibration_ablation_results.json"

CORE = ["mAP@0.5", "mAP@0.5:0.95"]
AIRCRAFT = ["military_aircraft", "civilian_aircraft"]


def evaluate(det_lora, loader, quality):
    ev = ContinualEvaluator(
        det_lora=det_lora,
        confidence_threshold=None,
        use_shared_quality_calibrator=quality,
        use_adapter_arbitration=False,
        use_shared_encoder_cache=True,
    )
    m = ev.evaluate_det_lora_joint(loader, CLASSES, task_idx=0, include_curves=False)
    out = {k: float(m[k]) for k in CORE if k in m}
    apc = m.get("AP_per_class@0.5", {})
    for c in AIRCRAFT:
        if c in apc:
            out[f"AP@0.5[{c}]"] = float(apc[c])
    return out


def main():
    results = []
    for variant, seed in CKPTS:
        ckpt = f"{SUITE}/model_{variant}/seed_{seed}/det_lora/final"
        if not Path(ckpt).exists():
            print(f"SKIP {ckpt} (missing)")
            continue
        t0 = time.time()
        det = RFDETRDetector(variant=variant)
        dl = DetLoRA(detector=det)
        dl.load_all(ckpt)
        orig_cal = dict(dl._adapter_calibrators)

        mapping = _det_lora_class_id_mapping(dl, CLASSES)
        ds = load_dataset_from_raw(
            raw_dir=RAW,
            class_filter=CLASSES,
            split="test",
            img_size=det.resolution,
            seed=seed,
            max_samples=None,
            class_id_mapping=mapping,
        )
        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)
        n_imgs = len(ds)

        row = {"variant": variant, "seed": seed, "n_test": n_imgs}
        # A: no calibration
        dl._adapter_calibrators = {}
        row["A_nocal"] = evaluate(dl, loader, quality=False)
        # B: Platt only
        dl._adapter_calibrators = dict(orig_cal)
        row["B_platt"] = evaluate(dl, loader, quality=False)
        # C: both
        dl._adapter_calibrators = dict(orig_cal)
        row["C_both"] = evaluate(dl, loader, quality=True)
        row["seconds"] = round(time.time() - t0, 1)
        results.append(row)
        OUT.write_text(json.dumps(results, indent=2))
        print(f"DONE {variant}/{seed} (n={n_imgs}, {row['seconds']}s)")
        for cfg in ["A_nocal", "B_platt", "C_both"]:
            r = row[cfg]
            print(
                f"  {cfg:9} mAP@0.5={r.get('mAP@0.5',0):.3f}  mAP@.5:.95={r.get('mAP@0.5:0.95',0):.3f}"
                f"  mil_air={r.get('AP@0.5[military_aircraft]',0):.3f}  civ_air={r.get('AP@0.5[civilian_aircraft]',0):.3f}"
            )

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nALLE FERTIG -> {OUT}")


if __name__ == "__main__":
    main()
