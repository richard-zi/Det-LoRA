"""Post-hoc conflict-gate refinement sweep with leakage-safe protocol.

Hypotheses (all apply-side, adapters frozen, replay-free):
  H1  The fixed loser penalty (0.5) is not optimal; a harsher/softer damping
      changes mixed mAP.
  H2  A lower/higher conflict threshold (resolve_tau) changes how many genuine
      conflicts the gate resolves.
  H3  Score-weighted box fusion across the co-firing cluster improves the
      winner's localization (targets the diagnosed mAP@0.75 deficit).

Protocol (avoids the tune/test overfitting that sank the earlier arbitration):
  fit        pair classifiers on TRAIN predictions
  selection  full penalty x tau x fusion grid evaluated on VAL only
  report     baseline, thesis default (penalty=0.5, tau=0.3) and the
             val-selected winner evaluated ONCE on TEST

Predictions per split are collected once and cached to disk, so re-running the
sweep with new variants is cheap.

Usage (from code/):
  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/ablations/gate_posthoc_sweep.py \
      --variant nano --seed 42 --suite thesis_l40_symhn
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.conflict_gate import _clusters, _mahalanobis, fit_pair_gate
from det_lora.evaluation.evaluator import (
    apply_shared_quality_calibrator,
    collect_det_lora_joint_predictions,
)
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
PENALTY_GRID = (0.0, 0.25, 0.5, 0.75)
TAU_GRID = (0.2, 0.3)
DEFAULT_PENALTY = 0.5
DEFAULT_TAU = 0.3
SELECTION_MARGIN = 0.002  # val winner must beat default by this to be adopted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe post-hoc gate sweep")
    parser.add_argument("--variant", default="nano")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suite", default="thesis_l40_symhn")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--fit_max_samples", type=int, default=3000)
    parser.add_argument("--val_max_samples", type=int, default=None)
    parser.add_argument("--test_max_samples", type=int, default=None)
    parser.add_argument("--cache_dir", default="experiments/probes/gate_posthoc_cache")
    parser.add_argument("--output_dir", default="experiments/probes/gate_posthoc_sweep")
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


def collect_cached(cache_path: Path, det_lora, dataloader):
    if cache_path.exists():
        payload = torch.load(str(cache_path), weights_only=False)
        return payload["predictions"], payload["ground_truths"]
    predictions, ground_truths, _ = collect_det_lora_joint_predictions(
        det_lora, dataloader, CLASSES
    )
    predictions = apply_shared_quality_calibrator(predictions, det_lora.shared_quality_calibrator)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"predictions": predictions, "ground_truths": ground_truths}, str(cache_path))
    return predictions, ground_truths


def apply_gate_variant(predictions, state, penalty, resolve_tau, fuse_boxes):
    """Gate application with configurable penalty/tau and optional box fusion."""
    lookup = {tuple(entry["classes"]): entry for entry in state["pairs"].values()}
    floor = float(state["floor"])
    cluster_iou = float(state["cluster_iou"])
    gated = []
    for prediction in predictions:
        embeddings = prediction.get("quality_features")
        if embeddings is None or not lookup:
            gated.append(prediction)
            continue
        scores = prediction["scores"].astype(np.float32).copy()
        raw = prediction["scores"].astype(np.float32)
        boxes = prediction["boxes"].astype(np.float32).copy()
        labels = prediction["labels"]
        for cluster in _clusters(prediction, floor, cluster_iou):
            present = [c for c in cluster["by_class"] if raw[cluster["by_class"][c]] >= resolve_tau]
            if len(present) < 2:
                continue
            losers: set = set()
            for i, j in combinations(sorted(present), 2):
                entry = lookup.get((i, j))
                if entry is None:
                    continue
                feature = np.concatenate(
                    [embeddings[cluster["by_class"][i]], embeddings[cluster["by_class"][j]]]
                )
                d_i = _mahalanobis(feature, entry["means"][0], entry["precisions"][0])
                d_j = _mahalanobis(feature, entry["means"][1], entry["precisions"][1])
                losers.add(j if d_i <= d_j else i)
            if not losers:
                continue
            for m in cluster["members"]:
                if int(labels[m]) in losers:
                    scores[m] = scores[m] * penalty
            if fuse_boxes:
                members = np.asarray(cluster["members"], dtype=int)
                weights = raw[members]
                fused = (boxes[members] * weights[:, None]).sum(0) / max(weights.sum(), 1e-9)
                for c, m in cluster["by_class"].items():
                    if c not in losers:
                        boxes[m] = fused
        new_prediction = dict(prediction)
        new_prediction["scores"] = scores
        new_prediction["boxes"] = boxes
        gated.append(new_prediction)
    return gated


def evaluate(predictions, ground_truths, class_ids):
    metrics = compute_map(predictions, ground_truths, target_class_ids=class_ids)
    return {
        "mAP@0.5": metrics["mAP@0.5"],
        "mAP@0.75": metrics["mAP@0.75"],
        "mAP@0.5:0.95": metrics["mAP@0.5:0.95"],
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.ckpt or (
        f"experiments/suites/{args.suite}/model_{args.variant}/seed_{args.seed}/det_lora/final"
    )
    run_key = f"{args.suite}_{args.variant}_seed{args.seed}"
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = RFDETRDetector(variant=args.variant)
    det_lora = DetLoRA(detector=detector)
    det_lora.load_all(checkpoint)
    class_ids = [det_lora.get_class_id(c) for c in CLASSES]

    gate_state_path = cache_dir / f"{run_key}_gate_state_fit{args.fit_max_samples}.pt"
    if gate_state_path.exists():
        state = torch.load(str(gate_state_path), weights_only=False)
        print(f"[Sweep] Loaded cached gate state ({len(state['pairs'])} pairs)")
    else:
        print(f"[Sweep] Collecting TRAIN predictions (fit, max {args.fit_max_samples})...")
        fit_loader = make_loader(
            det_lora, detector, args.data_dir, "train", args.fit_max_samples, args.seed
        )
        fit_preds, fit_gts, _ = collect_det_lora_joint_predictions(det_lora, fit_loader, CLASSES)
        fit_preds = apply_shared_quality_calibrator(fit_preds, det_lora.shared_quality_calibrator)
        state = fit_pair_gate(fit_preds, fit_gts, class_ids)
        gate_state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(gate_state_path))
        print(f"[Sweep] Fitted gate: {len(state['pairs'])} confusable pairs")

    print("[Sweep] Collecting VAL predictions (selection split)...")
    val_loader = make_loader(
        det_lora, detector, args.data_dir, "val", args.val_max_samples, args.seed
    )
    val_preds, val_gts = collect_cached(
        cache_dir / f"{run_key}_val_n{args.val_max_samples or 'all'}.pt", det_lora, val_loader
    )
    print("[Sweep] Collecting TEST predictions (final report split)...")
    test_loader = make_loader(
        det_lora, detector, args.data_dir, "test", args.test_max_samples, args.seed
    )
    test_preds, test_gts = collect_cached(
        cache_dir / f"{run_key}_test_n{args.test_max_samples or 'all'}.pt", det_lora, test_loader
    )

    # --- Selection on VAL only ---
    val_rows = []
    val_rows.append({"variant": "baseline", **evaluate(val_preds, val_gts, class_ids)})
    for penalty in PENALTY_GRID:
        for tau in TAU_GRID:
            for fuse in (False, True):
                gated = apply_gate_variant(val_preds, state, penalty, tau, fuse)
                row = {
                    "variant": f"penalty={penalty},tau={tau},fuse={fuse}",
                    "penalty": penalty,
                    "tau": tau,
                    "fuse": fuse,
                    **evaluate(gated, val_gts, class_ids),
                }
                val_rows.append(row)
                print(
                    f"[Val] {row['variant']:<34} mAP@0.5:0.95={row['mAP@0.5:0.95']:.4f} "
                    f"mAP@0.75={row['mAP@0.75']:.4f} mAP@0.5={row['mAP@0.5']:.4f}"
                )

    grid_rows = [r for r in val_rows if "penalty" in r]
    baseline_row = val_rows[0]
    default_row = next(
        r
        for r in grid_rows
        if r["penalty"] == DEFAULT_PENALTY and r["tau"] == DEFAULT_TAU and not r["fuse"]
    )
    best_row = max(grid_rows, key=lambda r: r["mAP@0.5:0.95"])
    adopted = best_row["mAP@0.5:0.95"] >= default_row["mAP@0.5:0.95"] + SELECTION_MARGIN
    selected = best_row if adopted else default_row
    # Guard against a gate that does not even beat the ungated baseline on val
    # (happens when the pair classifiers are fit on too little data).
    gate_beats_baseline = selected["mAP@0.5:0.95"] > baseline_row["mAP@0.5:0.95"]
    print(
        f"\n[Selection] baseline={baseline_row['mAP@0.5:0.95']:.4f} "
        f"default={default_row['mAP@0.5:0.95']:.4f} "
        f"best={best_row['variant']} ({best_row['mAP@0.5:0.95']:.4f}) "
        f"-> {'ADOPT best' if adopted else 'KEEP default'}"
        f"{'' if gate_beats_baseline else ' [WARN: gate does not beat baseline on val -> recommend NO gate / refit on full train]'}"
    )

    # --- One-shot report on TEST ---
    test_rows = [{"variant": "baseline", **evaluate(test_preds, test_gts, class_ids)}]
    for row in {id(default_row): default_row, id(selected): selected}.values():
        gated = apply_gate_variant(test_preds, state, row["penalty"], row["tau"], row["fuse"])
        test_rows.append({"variant": row["variant"], **evaluate(gated, test_gts, class_ids)})

    print("\n=== TEST (one-shot report) ===")
    for row in test_rows:
        print(
            f"{row['variant']:<34} mAP@0.5:0.95={row['mAP@0.5:0.95']:.4f} "
            f"mAP@0.75={row['mAP@0.75']:.4f} mAP@0.5={row['mAP@0.5']:.4f}"
        )

    result = {
        "checkpoint": checkpoint,
        "fit_max_samples": args.fit_max_samples,
        "gate_pairs": len(state["pairs"]),
        "val": val_rows,
        "selection": {
            "default": default_row,
            "best": best_row,
            "adopted": adopted,
            "gate_beats_baseline": gate_beats_baseline,
        },
        "test": test_rows,
    }
    out_path = output_dir / f"{run_key}_sweep.json"
    out_path.write_text(json.dumps(result, indent=2, default=float))
    print(f"\n[Sweep] Results written to {out_path}")


if __name__ == "__main__":
    main()
