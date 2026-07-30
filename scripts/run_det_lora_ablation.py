#!/usr/bin/env python3
"""Run small Det-LoRA ablations before committing to expensive cluster retraining."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import det_lora.model.detector as detector_module
from det_lora.run_experiment import run_continual_experiment

CORE_METRICS = ("mAP@0.5", "mAP@0.5:0.95", "Precision@0.5", "Recall@0.5", "F1@0.5")


@dataclass(frozen=True)
class Ablation:
    name: str
    lr: float
    lora_rank: int
    lora_alpha: int
    weight_decay: float
    use_hard_negatives: bool
    enable_shared_quality_calibrator: bool
    target_modules: tuple[str, ...] | None = None


DEFAULT_ABLATIONS = (
    Ablation(
        name="baseline_rank8_alpha16",
        lr=1.5e-4,
        lora_rank=8,
        lora_alpha=16,
        weight_decay=1e-4,
        use_hard_negatives=True,
        enable_shared_quality_calibrator=True,
    ),
    Ablation(
        name="rank16_alpha32",
        lr=1.5e-4,
        lora_rank=16,
        lora_alpha=32,
        weight_decay=1e-4,
        use_hard_negatives=True,
        enable_shared_quality_calibrator=True,
    ),
    Ablation(
        name="rank16_alpha16",
        lr=1.5e-4,
        lora_rank=16,
        lora_alpha=16,
        weight_decay=1e-4,
        use_hard_negatives=True,
        enable_shared_quality_calibrator=True,
    ),
    Ablation(
        name="lower_lr_rank16",
        lr=8e-5,
        lora_rank=16,
        lora_alpha=32,
        weight_decay=1e-4,
        use_hard_negatives=True,
        enable_shared_quality_calibrator=True,
    ),
    Ablation(
        name="no_hard_negatives",
        lr=1.5e-4,
        lora_rank=16,
        lora_alpha=32,
        weight_decay=1e-4,
        use_hard_negatives=False,
        enable_shared_quality_calibrator=True,
    ),
    Ablation(
        name="decoder_ffn_rank8",
        lr=1.5e-4,
        lora_rank=8,
        lora_alpha=16,
        weight_decay=1e-4,
        use_hard_negatives=True,
        enable_shared_quality_calibrator=True,
        target_modules=(
            "cross_attn.value_proj",
            "cross_attn.output_proj",
            "self_attn.out_proj",
            "linear1",
            "linear2",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Det-LoRA ablations")
    parser.add_argument(
        "--output_dir", type=Path, default=Path("experiments/ablations/det_lora_local")
    )
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--model", default="nano")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=12)
    parser.add_argument("--metrics_eval_every", type=int, default=0)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["military_tank", "military_truck", "military_aircraft"],
    )
    parser.add_argument(
        "--ablation_names",
        nargs="+",
        default=[ablation.name for ablation in DEFAULT_ABLATIONS],
    )
    return parser.parse_args()


def flatten_result(
    result: dict[str, Any], ablation: Ablation, args: argparse.Namespace
) -> dict[str, Any]:
    metrics = result.get("mixed_final_evaluation") or result.get("final_evaluation") or {}
    row: dict[str, Any] = {
        "name": ablation.name,
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "max_samples": args.max_samples,
        "lr": ablation.lr,
        "lora_rank": ablation.lora_rank,
        "lora_alpha": ablation.lora_alpha,
        "weight_decay": ablation.weight_decay,
        "use_hard_negatives": ablation.use_hard_negatives,
        "enable_shared_quality_calibrator": ablation.enable_shared_quality_calibrator,
        "target_modules": ",".join(ablation.target_modules or detector_module.LORA_TARGET_MODULES),
        "output_dir": result.get("output_dir") or result.get("config", {}).get("output_dir"),
    }
    for metric in CORE_METRICS:
        row[metric] = metrics.get(metric)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = {name for name in args.ablation_names}
    ablations = [ablation for ablation in DEFAULT_ABLATIONS if ablation.name in selected]
    if not ablations:
        raise ValueError("No ablations selected")

    rows: list[dict[str, Any]] = []
    default_target_modules = tuple(detector_module.LORA_TARGET_MODULES)
    for ablation in ablations:
        run_dir = args.output_dir / ablation.name
        target_modules = tuple(ablation.target_modules or default_target_modules)
        # In-place mutation also updates LORA_TARGET_PRESETS["default"],
        # which aliases this list and is what DetLoRA reads.
        detector_module.LORA_TARGET_MODULES[:] = list(target_modules)
        result = run_continual_experiment(
            classes=args.classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=ablation.lr,
            weight_decay=ablation.weight_decay,
            lora_rank=ablation.lora_rank,
            lora_alpha=ablation.lora_alpha,
            model_variant=args.model,
            data_dir=args.data_dir,
            save_dir=str(run_dir.parent),
            experiment_name=run_dir.name,
            synthetic=False,
            max_samples=args.max_samples,
            seed=args.seed,
            metrics_eval_every=args.metrics_eval_every,
            enable_shared_quality_calibrator=ablation.enable_shared_quality_calibrator,
            use_hard_negatives=ablation.use_hard_negatives,
            preset_name=None,
        )
        rows.append(flatten_result(result, ablation, args))
        write_csv(args.output_dir / "summary.csv", rows)

    detector_module.LORA_TARGET_MODULES[:] = list(default_target_modules)
    summary = {"output_dir": str(args.output_dir), "rows": rows}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
