"""Evaluation harness for the retro-hardening experiment.

Loads a finalized Det-LoRA checkpoint and computes, in the same inference
setting as the suite:
  - mixed AP_per_class@0.5 (all trained classes jointly),
  - matched AP@0.5 for one target class in isolation,
  - the TP-FP score gap of the target class against a confusion class
    (a discrimination diagnostic that is independent of calibration).

Sanity check: the mixed values must reproduce the evaluation.json stored
with the checkpoint, which proves the harness itself is correct.

Usage (from the repo root):
  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --no-sync python scripts/ablations/retro_eval.py \
    --checkpoint experiments/suites/thesis_l40_main/model_nano/seed_42/det_lora/final \
    --variant nano --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.evaluator import ContinualEvaluator
from det_lora.model.det_lora import DetLoRA
from det_lora.model.detector import RFDETRDetector
from det_lora.train import _det_lora_class_id_mapping, collate_fn


def _eval_joint(det_lora, classes, data_dir, seed, batch_size, quality, arbitration):
    detector = det_lora.detector
    dataset = load_dataset_from_raw(
        raw_dir=data_dir,
        class_filter=classes,
        split="test",
        class_id_offset=detector.base_num_classes,
        img_size=detector.resolution,
        seed=seed,
        class_id_mapping=_det_lora_class_id_mapping(det_lora, classes),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    evaluator = ContinualEvaluator(
        det_lora,
        use_shared_quality_calibrator=quality,
        use_adapter_arbitration=arbitration,
    )
    return evaluator.evaluate_det_lora_joint(
        dataloader=loader, class_names=classes, include_curves=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", default="nano")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--target", default="military_aircraft")
    parser.add_argument(
        "--no_quality", action="store_true", help="disable shared quality calibrator"
    )
    parser.add_argument("--no_arbitration", action="store_true", help="disable adapter arbitration")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    detector = RFDETRDetector(variant=args.variant)
    det_lora = DetLoRA(detector=detector, default_rank=8, default_alpha=16)
    det_lora.load_all(args.checkpoint)
    classes = list(det_lora.trained_classes)
    print(f"trained_classes: {classes}")

    # mixed: quality calibrator configurable (ON reproduces thesis-reported pipeline;
    #        OFF gives the refit-independent per-adapter frame). matched: always quality OFF.
    mixed_quality = not args.no_quality
    print(f"mixed_quality_calibrator={mixed_quality}")
    mixed = _eval_joint(
        det_lora, classes, args.data_dir, args.seed, args.batch_size, mixed_quality, False
    )
    matched = _eval_joint(
        det_lora, [args.target], args.data_dir, args.seed, args.batch_size, False, False
    )

    mixed_ap = mixed["AP_per_class@0.5"]
    result = {
        "checkpoint": args.checkpoint,
        "mixed_mAP@0.5": mixed["mAP@0.5"],
        "mixed_AP_per_class@0.5": mixed_ap,
        "matched_AP@0.5_target": matched["AP_per_class@0.5"].get(args.target),
        "target": args.target,
    }
    print("\n=== RESULT ===")
    print(f"mixed mAP@0.5 (all): {result['mixed_mAP@0.5']:.4f}")
    print(f"matched AP@0.5 {args.target}: {result['matched_AP@0.5_target']:.4f}")
    for c in classes:
        print(f"  mixed AP@0.5 {c:22s} = {mixed_ap[c]:.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
