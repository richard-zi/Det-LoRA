"""Adapter-merging probe: fuse all per-class LoRA adapters into ONE adapter.

Motivation (peer-reviewed): TIES merging (Yadav et al., NeurIPS 2023) and DARE
(Yu et al., ICML 2024) fuse task-specific deltas into a single parameter set;
DuET (Monga et al., ICCV 2025) validates exemplar-free task arithmetic for
incremental object detection. Applied to Det-LoRA, a merged adapter replaces
the N-pass joint inference (one decoder pass per class) with a SINGLE pass:

- cross-adapter co-firing conflicts disappear at the root (one score
  distribution per query instead of N independent ones),
- inference cost drops from N decoder passes to one,
- the per-class adapters remain on disk -> the merge is a reversible
  inference-time optimization; the zero-forgetting design is untouched.

Protocol: merge-method/density grid selected on VAL, single-shot TEST report
against the per-adapter joint baseline. Per-class score calibration is skipped
(monotone per-class transforms cannot change AP).

Usage (from code/):
  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/ablations/adapter_merge_probe.py \
      --variant nano --seed 42 --suite thesis_l40_symhn
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import DataLoader

from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.evaluator import _filter_ground_truth, cxcywh_to_xyxy
from det_lora.evaluation.metrics import compute_map
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
MERGE_VARIANTS = (
    {"name": "linear_mean", "combination_type": "linear", "weight": 1.0 / 6.0, "density": None},
    {"name": "linear_sum", "combination_type": "linear", "weight": 1.0, "density": None},
    {"name": "ties_d0.5", "combination_type": "ties", "weight": 1.0, "density": 0.5},
    {"name": "ties_d0.7", "combination_type": "ties", "weight": 1.0, "density": 0.7},
    {"name": "dare_ties_d0.5", "combination_type": "dare_ties", "weight": 1.0, "density": 0.5},
)
SELECTION_METRIC = "mAP@0.5:0.95"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-class adapters into one")
    parser.add_argument("--variant", default="nano")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suite", default="thesis_l40_symhn")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--val_max_samples", type=int, default=400)
    parser.add_argument("--output_dir", default="experiments/probes/adapter_merge")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Class subset of the checkpoint (default: all six thesis classes)",
    )
    args = parser.parse_args()
    if args.classes:
        global CLASSES
        CLASSES = list(args.classes)
    return args


def make_loader(det_lora, detector, data_dir, split, max_samples, seed):
    mapping = _det_lora_class_id_mapping(det_lora, CLASSES)
    dataset = load_dataset_from_raw(
        raw_dir=data_dir,
        class_filter=CLASSES,
        split=split,
        img_size=detector.resolution,
        seed=seed,
        max_samples=max_samples,
        class_id_mapping=mapping,
    )
    return DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)


@torch.no_grad()
def collect_single_pass(det_lora, dataloader, class_ids):
    """One decoder pass per image; slice scores for every class from it."""
    det_lora.set_eval_mode()
    device = det_lora.device
    predictions, ground_truths = [], []
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        outputs = det_lora.forward(pixel_values=pixel_values)
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        for sample_idx in range(logits.shape[0]):
            xyxy = cxcywh_to_xyxy(boxes[sample_idx]).cpu().numpy()
            sample_boxes, sample_scores, sample_labels = [], [], []
            for class_id in class_ids:
                scores = logits[sample_idx, :, class_id].sigmoid()
                sample_boxes.append(xyxy)
                sample_scores.append(scores.cpu().numpy())
                sample_labels.append(np.full(scores.shape[0], class_id, dtype=np.int64))
            predictions.append(
                {
                    "boxes": np.concatenate(sample_boxes),
                    "scores": np.concatenate(sample_scores),
                    "labels": np.concatenate(sample_labels),
                }
            )
            ground_truths.append(_filter_ground_truth(batch["labels"][sample_idx], class_ids))
    return predictions, ground_truths


def evaluate(predictions, ground_truths, class_ids):
    metrics = compute_map(predictions, ground_truths, target_class_ids=class_ids)
    return {key: metrics[key] for key in ("mAP@0.5", "mAP@0.75", "mAP@0.5:0.95")}


def build_merged_model(detector, adapter_paths, variant):
    """Attach all per-class adapters as named adapters and merge them."""
    first_class = CLASSES[0]
    peft_model = PeftModel.from_pretrained(
        detector.model, adapter_paths[first_class], adapter_name=first_class, is_trainable=False
    )
    for class_name in CLASSES[1:]:
        peft_model.load_adapter(adapter_paths[class_name], adapter_name=class_name)
    kwargs = {}
    if variant["density"] is not None:
        kwargs["density"] = variant["density"]
    peft_model.add_weighted_adapter(
        adapters=list(CLASSES),
        weights=[variant["weight"]] * len(CLASSES),
        adapter_name="merged",
        combination_type=variant["combination_type"],
        **kwargs,
    )
    peft_model.set_adapter("merged")
    return peft_model


def main() -> None:
    args = parse_args()
    checkpoint = args.ckpt or (
        f"experiments/suites/{args.suite}/model_{args.variant}/seed_{args.seed}/det_lora/final"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_key = f"{args.suite}_{args.variant}_seed{args.seed}"

    def fresh_merged_setup(variant):
        """PEFT leaves residual wrapper state across unload/re-wrap cycles, so
        every merge variant gets a pristine detector + checkpoint load."""
        detector = RFDETRDetector(variant=args.variant)
        det_lora = DetLoRA(detector=detector)
        det_lora.load_all(checkpoint)
        adapter_paths = dict(det_lora.adapters)
        for class_name, path in adapter_paths.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Adapter for {class_name} missing: {path}")
        detector.model = build_merged_model(detector, adapter_paths, variant)
        merged_module_count = sum(
            1
            for _, m in detector.model.named_modules()
            if hasattr(m, "lora_A") and "merged" in m.lora_A
        )
        if merged_module_count == 0:
            raise RuntimeError(f"Merge '{variant['name']}' attached to zero modules")
        class_ids = [det_lora.get_class_id(c) for c in CLASSES]
        return detector, det_lora, class_ids

    val_rows = []
    for variant in MERGE_VARIANTS:
        detector, det_lora, class_ids = fresh_merged_setup(variant)
        val_loader = make_loader(
            det_lora, detector, args.data_dir, "val", args.val_max_samples, args.seed
        )
        preds, gts = collect_single_pass(det_lora, val_loader, class_ids)
        row = {"variant": variant["name"], **evaluate(preds, gts, class_ids)}
        val_rows.append(row)
        print(
            f"[Val] {row['variant']:<16} mAP@0.5:0.95={row[SELECTION_METRIC]:.4f} "
            f"mAP@0.75={row['mAP@0.75']:.4f} mAP@0.5={row['mAP@0.5']:.4f}"
        )
        del detector, det_lora

    best = max(val_rows, key=lambda r: r[SELECTION_METRIC])
    best_variant = next(v for v in MERGE_VARIANTS if v["name"] == best["variant"])
    print(f"\n[Selection] best merge on val: {best['variant']} ({best[SELECTION_METRIC]:.4f})")

    detector, det_lora, class_ids = fresh_merged_setup(best_variant)
    test_loader = make_loader(det_lora, detector, args.data_dir, "test", None, args.seed)
    test_preds, test_gts = collect_single_pass(det_lora, test_loader, class_ids)
    test_row = {"variant": best["variant"], **evaluate(test_preds, test_gts, class_ids)}
    print(f"\n=== TEST (one-shot, single-pass merged adapter) ===")
    print(
        f"{test_row['variant']:<16} mAP@0.5:0.95={test_row['mAP@0.5:0.95']:.4f} "
        f"mAP@0.75={test_row['mAP@0.75']:.4f} mAP@0.5={test_row['mAP@0.5']:.4f}"
    )

    result = {
        "checkpoint": checkpoint,
        "val": val_rows,
        "selected": best["variant"],
        "test": test_row,
    }
    (output_dir / f"{run_key}_merge.json").write_text(json.dumps(result, indent=2, default=float))
    print(f"[Merge] Results written to {output_dir / f'{run_key}_merge.json'}")


if __name__ == "__main__":
    main()
