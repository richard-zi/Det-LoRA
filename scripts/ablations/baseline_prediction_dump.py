"""
Per-image test-split detections for the reference baselines (Replay, CL-DETR, Joint).
======================================================================================

The thesis suites stored only aggregated metrics for the baselines. For the paired
bootstrap against Det-LoRA (bootstrap_ci.py) the per-image detections are needed, so
this script reloads the final baseline checkpoint per (method, variant, seed) --
exactly the state that produced the stored mixed test metrics -- runs inference over
the official test split (496 images, shuffle=False, same ordering as the Det-LoRA
caches) and dumps predictions/ground truths in the cache format bootstrap_ci.py
consumes (cache key: baseline_{method}_{variant}_seed{seed}_test_nall.pt).

Validation: the mAP recomputed from each dump is compared against the mixed metrics
stored in the suite's evaluation.json. A per-run deviation above --run_tolerance or a
per-method aggregate deviation above --aggregate_tolerance aborts (indicates a
checkpoint/state or MPS-vs-CUDA mismatch) instead of silently producing detections
that do not correspond to the reported numbers.

Run with PYTORCH_ENABLE_MPS_FALLBACK=1 on Apple Silicon.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from det_lora.baselines.checkpoint import (
    load_model_checkpoint,
    prepare_detector_for_checkpoint_load,
)
from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.evaluator import (
    _append_prediction,
    _empty_prediction,
    _filter_ground_truth,
    _limit_prediction,
    _prediction_keep_mask,
)
from det_lora.evaluation.metrics import compute_map
from det_lora.model.detector import RFDETRDetector
from det_lora.train import collate_fn

CLASSES = [
    "military_tank",
    "military_truck",
    "military_aircraft",
    "military_helicopter",
    "civilian_car",
    "civilian_aircraft",
]
# suite dir, method subdir, final checkpoint task, task index of the final mixed eval
BASELINES: Dict[str, Tuple[str, str, str, int]] = {
    "replay": ("experiments/suites/thesis_l40_main", "replay", "civilian_aircraft", 5),
    "cl_detr": ("experiments/suites/thesis_l40_cldetr", "cl_detr", "civilian_aircraft", 5),
    "joint": ("experiments/suites/thesis_l40_joint_baseline", "joint_finetuning", "joint", 0),
}
# mirrors ContinualEvaluator defaults used for the suite evaluations
MAX_DETECTIONS_PER_IMAGE = 100


@torch.no_grad()
def collect_predictions(
    detector: RFDETRDetector, loader: DataLoader, class_ids: Sequence[int]
) -> Tuple[List[Dict], List[Dict]]:
    """Mirror ContinualEvaluator.evaluate_standard_detector's prediction collection."""
    detector.model.eval()
    device = detector.device
    predictions: List[Dict] = []
    ground_truths: List[Dict] = []
    for batch in tqdm(loader, desc="Inference", leave=False):
        outputs = detector.forward(pixel_values=batch["pixel_values"].to(device))
        logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        for sample_idx in range(logits.shape[0]):
            probs = logits[sample_idx].sigmoid()
            prediction = _empty_prediction()
            for class_id in class_ids:
                class_scores = probs[:, class_id]
                keep = _prediction_keep_mask(class_scores, None)
                prediction = _append_prediction(
                    prediction, pred_boxes[sample_idx][keep], class_scores[keep], class_id
                )
            predictions.append(_limit_prediction(prediction, MAX_DETECTIONS_PER_IMAGE))
            ground_truths.append(_filter_ground_truth(batch["labels"][sample_idx], class_ids))
    return predictions, ground_truths


def stored_mixed_metrics(run_dir: Path, task_idx: int) -> Dict[str, float]:
    with open(run_dir / "evaluation.json") as f:
        history = json.load(f)["history"]
    return history[str(task_idx)]["metrics"]


def dump_run(
    method: str, variant: str, seed: int, data_dir: str, cache_dir: Path, run_tolerance: float
) -> Dict[str, float]:
    suite_dir, subdir, final_task, task_idx = BASELINES[method]
    run_dir = Path(suite_dir) / f"model_{variant}" / f"seed_{seed}" / subdir
    cache_path = cache_dir / f"baseline_{method}_{variant}_seed{seed}_test_nall.pt"

    detector = RFDETRDetector(variant=variant)
    prepare_detector_for_checkpoint_load(detector, CLASSES)
    if not load_model_checkpoint(detector.model, run_dir, final_task):
        raise FileNotFoundError(f"missing checkpoint {run_dir}/checkpoints/{final_task}/model.pt")
    class_ids = [detector.get_class_id(c) for c in CLASSES]

    if cache_path.exists():
        payload = torch.load(str(cache_path), weights_only=False)
        predictions, ground_truths = payload["predictions"], payload["ground_truths"]
        print(f"[Dump] {method} {variant} seed{seed}: cache exists, validating only")
    else:
        dataset = load_dataset_from_raw(
            raw_dir=data_dir,
            class_filter=CLASSES,
            split="test",
            class_id_offset=detector.base_num_classes,
            img_size=detector.resolution,
            seed=seed,
        )
        loader = DataLoader(
            dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0
        )
        predictions, ground_truths = collect_predictions(detector, loader, class_ids)

    metrics = compute_map(predictions, ground_truths, target_class_ids=class_ids)
    stored = stored_mixed_metrics(run_dir, task_idx)
    deltas = {m: metrics[m] - stored[m] for m in ("mAP@0.5", "mAP@0.5:0.95")}
    print(
        f"[Dump] {method} {variant} seed{seed}: "
        f"mAP@0.5:0.95={metrics['mAP@0.5:0.95']:.4f} (stored {stored['mAP@0.5:0.95']:.4f}, "
        f"delta {deltas['mAP@0.5:0.95']:+.4f}) | "
        f"mAP@0.5={metrics['mAP@0.5']:.4f} (stored {stored['mAP@0.5']:.4f}, "
        f"delta {deltas['mAP@0.5']:+.4f})"
    )
    if any(abs(d) > run_tolerance for d in deltas.values()):
        raise RuntimeError(
            f"{method} {variant} seed{seed}: recomputed mAP deviates from the stored suite "
            f"metrics by {deltas} (> {run_tolerance}); aborting instead of dumping mismatched "
            f"detections (check checkpoint state / MPS-vs-CUDA numerics)."
        )

    if not cache_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"predictions": predictions, "ground_truths": ground_truths, "class_ids": class_ids},
            str(cache_path),
        )
        print(f"[Dump] saved {cache_path}")
    return deltas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--methods", nargs="+", default=["replay", "cl_detr", "joint"], choices=list(BASELINES)
    )
    p.add_argument("--variants", nargs="+", default=["nano", "small", "base", "medium", "large"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--cache_dir", default="experiments/probes/gate_posthoc_cache")
    p.add_argument(
        "--aggregate_tolerance",
        type=float,
        default=0.005,
        help="max |mean recomputed - mean stored| mAP per method before aborting",
    )
    p.add_argument(
        "--run_tolerance",
        type=float,
        default=0.01,
        help="max |recomputed - stored| mAP for a single run before aborting",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    all_deltas: Dict[str, Dict[str, float]] = {}
    for method in args.methods:
        method_deltas = []
        for variant in args.variants:
            for seed in args.seeds:
                deltas = dump_run(
                    method, variant, seed, args.data_dir, cache_dir, args.run_tolerance
                )
                all_deltas[f"{method}_{variant}_seed{seed}"] = deltas
                method_deltas.append(deltas)
        for metric in ("mAP@0.5", "mAP@0.5:0.95"):
            aggregate = sum(d[metric] for d in method_deltas) / len(method_deltas)
            print(f"[Dump] {method}: aggregate {metric} deviation {aggregate:+.4f}")
            if abs(aggregate) > args.aggregate_tolerance:
                raise RuntimeError(
                    f"{method}: aggregate {metric} deviation {aggregate:+.4f} exceeds "
                    f"{args.aggregate_tolerance}; MPS reproduction does not match the "
                    f"stored L40 suite metrics."
                )
    summary_path = cache_dir / "baseline_dump_validation.json"
    with open(summary_path, "w") as f:
        json.dump(all_deltas, f, indent=2)
    print(f"[Dump] validation deltas saved to {summary_path}")


if __name__ == "__main__":
    main()
