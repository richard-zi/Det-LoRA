"""Zero-shot cross-dataset evaluation of the Track-A Det-LoRA checkpoints on
the extension pool (Shatnawi tank subset + AOD-4 aircraft/helicopter), which
no Track-A adapter has ever seen during training. Mirrors the suite protocol:
matched = per-class slice with only the own adapter (ContinualEvaluator
defaults, Platt calibration only), mixed = joint inference over all six
adapters with the shared quality calibrator. The evaluation set is a fixed,
seeded 20 % slice of the pool (split="val", data seed 42) so every run scores
the identical images.

Usage (from the repo root):
  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/crossdataset_eval.py \
      --suite thesis_l40_iter4 --variant nano --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.evaluator import ContinualEvaluator
from det_lora.model.det_lora import DetLoRA
from det_lora.model.detector import RFDETRDetector
from det_lora.train import _det_lora_class_id_mapping, collate_fn

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/analysis/crossdataset"
OUT.mkdir(parents=True, exist_ok=True)

ALL_CLASSES = [
    "military_tank",
    "military_truck",
    "military_aircraft",
    "military_helicopter",
    "civilian_car",
    "civilian_aircraft",
]
EXT_CLASSES = ["military_tank", "military_aircraft", "military_helicopter"]
DATA_SEED = 42  # fixed evaluation slice, independent of the model seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot cross-dataset evaluation")
    parser.add_argument("--suite", default="thesis_l40_iter4")
    parser.add_argument("--variant", default="nano")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--data_dir", default="data/extension/raw")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    return args


def make_loader(det_lora, detector, args, class_filter, class_names):
    dataset = load_dataset_from_raw(
        raw_dir=args.data_dir,
        class_filter=class_filter,
        split=args.split,
        img_size=detector.resolution,
        seed=DATA_SEED,
        max_samples=args.max_samples,
        class_id_mapping=_det_lora_class_id_mapping(det_lora, class_names),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    return dataset, loader


def evaluate_run(args, seed: int) -> dict:
    checkpoint = (
        REPO / f"experiments/suites/{args.suite}/model_{args.variant}/seed_{seed}/det_lora/final"
    )
    detector = RFDETRDetector(variant=args.variant)
    det_lora = DetLoRA(detector=detector)
    det_lora.load_all(str(checkpoint))

    result: dict = {"suite": args.suite, "variant": args.variant, "seed": seed}

    # Matched: each extension class on its own slice, only its adapter active.
    matched_evaluator = ContinualEvaluator(det_lora)
    matched: dict = {}
    for class_name in EXT_CLASSES:
        dataset, loader = make_loader(det_lora, detector, args, class_name, [class_name])
        metrics = matched_evaluator.evaluate_det_lora_joint(
            dataloader=loader, class_names=[class_name]
        )
        matched[class_name] = {
            "n_images": len(dataset),
            "AP@0.5": metrics["AP_per_class@0.5"][class_name],
            "AP@0.5:0.95": metrics["mAP@0.5:0.95"],
        }
        print(f"[{args.suite}/{args.variant}/seed{seed}] matched {class_name}: {matched[class_name]}")
    result["matched"] = matched

    # Mixed: all six adapters on the union of the three extension slices.
    mixed_evaluator = ContinualEvaluator(det_lora, use_shared_quality_calibrator=True)
    dataset, loader = make_loader(det_lora, detector, args, EXT_CLASSES, ALL_CLASSES)
    metrics = mixed_evaluator.evaluate_det_lora_joint(dataloader=loader, class_names=ALL_CLASSES)
    ap_per_class = metrics["AP_per_class@0.5"]
    present = [c for c in EXT_CLASSES if c in ap_per_class]
    result["mixed"] = {
        "n_images": len(dataset),
        "AP_per_class@0.5": {c: ap_per_class[c] for c in present},
        "mAP@0.5_present": sum(ap_per_class[c] for c in present) / len(present),
        "AP_per_class_full": ap_per_class,
    }
    print(f"[{args.suite}/{args.variant}/seed{seed}] mixed: {result['mixed']['mAP@0.5_present']:.3f}")
    return result


def main() -> None:
    args = parse_args()
    for seed in args.seeds:
        out_path = OUT / f"{args.suite}_{args.variant}_seed{seed}.json"
        if out_path.exists():
            print(f"[Skip] {out_path} exists")
            continue
        result = evaluate_run(args, seed)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
