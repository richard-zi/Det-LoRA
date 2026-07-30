"""Probe for DSR iteration 4 (localization-path adapters).

Diagnosis: the matched ceiling of Det-LoRA trails the joint reference and the
gap widens with IoU strictness (matched mAP@0.75 ~0.64 vs. joint ~0.755),
because the localization path is fully frozen: deformable-attention
sampling_offsets/attention_weights, the decoder FFN and the bbox_embed
refinement MLP receive no adaptation.

Hypothesis: extending the per-class adapter footprint to these modules raises
matched mAP@0.75 / mAP@0.5:0.95 without touching the zero-forgetting design
(adapters stay per-class and removable, the base model stays frozen).

This probe trains ONE class under each footprint preset with identical data,
seed and hyperparameters, and compares matched test metrics.

Usage:
  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/ablations/lora_target_probe.py \
      --class_name military_aircraft --model nano --epochs 10 --seed 42 --preset l40_final
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from det_lora.model.detector import LORA_TARGET_PRESETS
from det_lora.train import train_adapter
from det_lora.utils import resolve_variant_settings

METRIC_KEYS = ("mAP@0.5", "mAP@0.75", "mAP@0.5:0.95")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LoRA target footprints on one class")
    parser.add_argument("--class_name", type=str, default="military_aircraft")
    parser.add_argument("--model", type=str, default="nano")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preset", type=str, default=None, help="Hyperparameter preset, e.g. l40_final"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=tuple(LORA_TARGET_PRESETS),
        default=list(LORA_TARGET_PRESETS),
        help="Footprint presets to compare",
    )
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=None, help="Override preset LoRA rank")
    parser.add_argument("--dora", action="store_true", help="Use DoRA instead of vanilla LoRA")
    parser.add_argument("--tag", type=str, default="", help="Suffix for run names in the summary")
    parser.add_argument("--output_dir", type=str, default="experiments/probes/lora_target_probe")
    return parser.parse_args()


def extract_matched_metrics(results: dict) -> dict:
    matched = results.get("test_target_metrics") or results.get("test_metrics") or {}
    return {key: matched.get(key) for key in METRIC_KEYS}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved = resolve_variant_settings(
        variant=args.model,
        preset_name=args.preset,
        base_defaults={
            "epochs": 50,
            "batch_size": 4,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lora_rank": 8,
            "lora_alpha": 16,
            "metrics_eval_every": 1,
        },
        overrides={"epochs": args.epochs, "lora_rank": args.lora_rank},
    )
    if args.lora_rank is not None:
        resolved["lora_alpha"] = 2 * args.lora_rank

    rows = []
    for footprint in args.targets:
        run_name = f"{args.class_name}_{args.model}_seed{args.seed}_{footprint}{args.tag}"
        print(f"\n{'#' * 60}\n# Footprint: {footprint}\n{'#' * 60}")
        results = train_adapter(
            class_name=args.class_name,
            epochs=int(resolved["epochs"]),
            batch_size=int(resolved["batch_size"]),
            lr=float(resolved["lr"]),
            weight_decay=float(resolved["weight_decay"]),
            lora_rank=int(resolved["lora_rank"]),
            lora_alpha=int(resolved["lora_alpha"]),
            lora_target_preset=footprint,
            use_dora=args.dora,
            model_variant=args.model,
            data_dir=args.data_dir,
            save_dir=str(output_dir),
            max_samples=args.max_samples,
            seed=args.seed,
            preset_name=args.preset,
            experiment_name=run_name,
        )
        row = {"footprint": f"{footprint}{args.tag}", **extract_matched_metrics(results)}
        rows.append(row)
        print(f"[Probe] {footprint}: {row}")

    summary_csv = output_dir / "probe_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["footprint", *METRIC_KEYS])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "probe_summary.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'=' * 60}\nMatched test metrics ({args.class_name}, {args.model}, seed {args.seed})")
    header = f"{'footprint':<22}" + "".join(f"{key:>14}" for key in METRIC_KEYS)
    print(header)
    for row in rows:
        cells = "".join(
            f"{row[key]:>14.4f}" if isinstance(row[key], float) else f"{'n/a':>14}"
            for key in METRIC_KEYS
        )
        print(f"{row['footprint']:<22}{cells}")
    print(f"\nSummary written to {summary_csv}")


if __name__ == "__main__":
    main()
