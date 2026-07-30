"""Soft conflict gate evaluated on cached predictions (no GPU needed).

Instead of damping the losing detection by a fixed factor, scale both
conflicting detections by their pairwise Mahalanobis posterior

    w_i = exp(-d_i / (2T)) / (exp(-d_i / (2T)) + exp(-d_j / (2T)))

so that confident gate decisions damp hard and ambiguous ones damp softly.
The temperature T is selected on VAL; TEST is evaluated once.

Requires the prediction caches produced by gate_posthoc_sweep.py.

Usage (from code/):
  uv run python scripts/ablations/gate_soft_eval.py --variant nano --seed 42
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from det_lora.evaluation.conflict_gate import _clusters, _mahalanobis
from det_lora.evaluation.metrics import compute_map

TEMPERATURE_GRID = (16.0, 64.0, 256.0, 1024.0)
HARD_PENALTIES = (0.5, 0.75)
RESOLVE_TAU = 0.3
SELECTION_MARGIN = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Soft gate on cached predictions")
    parser.add_argument("--variant", default="nano")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suite", default="thesis_l40_symhn")
    parser.add_argument("--fit_max_samples", type=int, default=3000)
    parser.add_argument("--val_tag", default="n400")
    parser.add_argument("--cache_dir", default="experiments/probes/gate_posthoc_cache")
    parser.add_argument("--output_dir", default="experiments/probes/gate_posthoc_sweep")
    return parser.parse_args()


def apply_soft_gate(predictions, state, temperature=None, hard_penalty=None):
    """temperature=None with hard_penalty set -> classic hard gate."""
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
        labels = prediction["labels"]
        for cluster in _clusters(prediction, floor, cluster_iou):
            present = [c for c in cluster["by_class"] if raw[cluster["by_class"][c]] >= RESOLVE_TAU]
            if len(present) < 2:
                continue
            class_weight = {}
            losers = set()
            for i, j in combinations(sorted(present), 2):
                entry = lookup.get((i, j))
                if entry is None:
                    continue
                feature = np.concatenate(
                    [embeddings[cluster["by_class"][i]], embeddings[cluster["by_class"][j]]]
                )
                d_i = _mahalanobis(feature, entry["means"][0], entry["precisions"][0])
                d_j = _mahalanobis(feature, entry["means"][1], entry["precisions"][1])
                if temperature is None:
                    losers.add(j if d_i <= d_j else i)
                else:
                    logits = np.array([-d_i, -d_j]) / (2.0 * temperature)
                    logits -= logits.max()
                    w = np.exp(logits)
                    w /= w.sum()
                    class_weight[i] = min(class_weight.get(i, 1.0), float(w[0]))
                    class_weight[j] = min(class_weight.get(j, 1.0), float(w[1]))
            if temperature is None:
                for m in cluster["members"]:
                    if int(labels[m]) in losers:
                        scores[m] = scores[m] * hard_penalty
            else:
                for m in cluster["members"]:
                    weight = class_weight.get(int(labels[m]))
                    if weight is not None:
                        scores[m] = scores[m] * weight
        new_prediction = dict(prediction)
        new_prediction["scores"] = scores
        gated.append(new_prediction)
    return gated


def evaluate(predictions, ground_truths, class_ids):
    metrics = compute_map(predictions, ground_truths, target_class_ids=class_ids)
    return {key: metrics[key] for key in ("mAP@0.5", "mAP@0.75", "mAP@0.5:0.95")}


def main() -> None:
    args = parse_args()
    run_key = f"{args.suite}_{args.variant}_seed{args.seed}"
    cache_dir = Path(args.cache_dir)
    state = torch.load(
        str(cache_dir / f"{run_key}_gate_state_fit{args.fit_max_samples}.pt"),
        weights_only=False,
    )
    val = torch.load(str(cache_dir / f"{run_key}_val_{args.val_tag}.pt"), weights_only=False)
    test = torch.load(str(cache_dir / f"{run_key}_test_nall.pt"), weights_only=False)
    class_ids = sorted({c for entry in state["pairs"].values() for c in entry["classes"]})

    val_rows = []
    for penalty in HARD_PENALTIES:
        gated = apply_soft_gate(val["predictions"], state, hard_penalty=penalty)
        row = {
            "variant": f"hard penalty={penalty}",
            "temperature": None,
            "penalty": penalty,
            **evaluate(gated, val["ground_truths"], class_ids),
        }
        val_rows.append(row)
        print(f"[Val] {row['variant']:<24} mAP@0.5:0.95={row['mAP@0.5:0.95']:.4f}")
    for temperature in TEMPERATURE_GRID:
        gated = apply_soft_gate(val["predictions"], state, temperature=temperature)
        row = {
            "variant": f"soft T={temperature}",
            "temperature": temperature,
            "penalty": None,
            **evaluate(gated, val["ground_truths"], class_ids),
        }
        val_rows.append(row)
        print(f"[Val] {row['variant']:<24} mAP@0.5:0.95={row['mAP@0.5:0.95']:.4f}")

    hard_default = val_rows[0]
    best = max(val_rows, key=lambda r: r["mAP@0.5:0.95"])
    adopted = best["mAP@0.5:0.95"] >= hard_default["mAP@0.5:0.95"] + SELECTION_MARGIN
    selected = best if adopted else hard_default
    print(
        f"\n[Selection] best on val: {best['variant']} "
        f"({best['mAP@0.5:0.95']:.4f} vs default {hard_default['mAP@0.5:0.95']:.4f}) "
        f"-> {'ADOPT' if adopted else 'KEEP default'}"
    )

    test_rows = [
        {"variant": "baseline", **evaluate(test["predictions"], test["ground_truths"], class_ids)}
    ]
    for row in {id(hard_default): hard_default, id(selected): selected}.values():
        gated = apply_soft_gate(
            test["predictions"],
            state,
            temperature=row["temperature"],
            hard_penalty=row["penalty"],
        )
        test_rows.append(
            {"variant": row["variant"], **evaluate(gated, test["ground_truths"], class_ids)}
        )
    print("\n=== TEST (one-shot) ===")
    for row in test_rows:
        print(
            f"{row['variant']:<24} mAP@0.5:0.95={row['mAP@0.5:0.95']:.4f} "
            f"mAP@0.75={row['mAP@0.75']:.4f} mAP@0.5={row['mAP@0.5']:.4f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{run_key}_soft_gate.json").write_text(
        json.dumps(
            {"val": val_rows, "test": test_rows, "adopted": adopted}, indent=2, default=float
        )
    )


if __name__ == "__main__":
    main()
