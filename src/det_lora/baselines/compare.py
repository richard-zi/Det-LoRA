"""
Baseline Comparison
====================

Run all methods on the same setup and generate comparison results.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from det_lora.baselines.ewc import EWCBaseline
from det_lora.baselines.finetuning import FineTuningBaseline
from det_lora.baselines.replay import ReplayBaseline
from det_lora.model.det_lora import DetLoRA
from det_lora.model.detector import RFDETRDetector
from det_lora.run_experiment import run_continual_experiment


def run_comparison(
    classes: List[str],
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    lora_rank: int = 8,
    model_variant: str = "medium",
    data_dir: str = "data/raw",
    save_dir: str = "experiments",
    synthetic: bool = False,
    methods: Optional[List[str]] = None,
    resume_dir: Optional[str] = None,
    seed: int = 42,
    comparison_name: Optional[str] = None,
) -> Dict:
    """
    Run all methods on the same setup.

    Args:
        classes: List of class names
        epochs: Epochs per task
        methods: Which methods to run (default: all)
                 Options: "det_lora", "finetuning", "ewc", "replay"

    Returns:
        Comparison results dict
    """
    if methods is None:
        methods = ["det_lora", "finetuning", "ewc", "replay"]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    comparison_dir = Path(save_dir) / (comparison_name or f"comparison_{timestamp}")
    comparison_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "config": {
            "classes": classes,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "lora_rank": lora_rank,
            "model_variant": model_variant,
            "methods": methods,
            "seed": seed,
            "output_dir": str(comparison_dir),
        }
    }

    # 1. Det-LoRA (our method)
    if "det_lora" in methods:
        print(f"\n{'='*60}")
        print("Running: Det-LoRA (Ours)")
        print(f"{'='*60}")
        all_results["det_lora"] = run_continual_experiment(
            classes=classes,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lora_rank=lora_rank,
            model_variant=model_variant,
            data_dir=data_dir,
            save_dir=str(comparison_dir),
            synthetic=synthetic,
            seed=seed,
        )

    # 2. Fine-Tuning
    if "finetuning" in methods:
        print(f"\n{'='*60}")
        print("Running: Fine-Tuning Baseline")
        print(f"{'='*60}")
        ft = FineTuningBaseline(variant=model_variant, lr=lr)
        all_results["finetuning"] = ft.run_experiment(
            classes=classes,
            epochs=epochs,
            batch_size=batch_size,
            data_dir=data_dir,
            save_dir=str(comparison_dir),
            synthetic=synthetic,
            seed=seed,
        )

    # 3. EWC
    if "ewc" in methods:
        print(f"\n{'='*60}")
        print("Running: EWC Baseline")
        print(f"{'='*60}")
        ewc = EWCBaseline(variant=model_variant, lr=lr)
        all_results["ewc"] = ewc.run_experiment(
            classes=classes,
            epochs=epochs,
            batch_size=batch_size,
            data_dir=data_dir,
            save_dir=str(comparison_dir),
            synthetic=synthetic,
            seed=seed,
        )

    # 4. Replay
    if "replay" in methods:
        print(f"\n{'='*60}")
        print("Running: Replay Baseline")
        print(f"{'='*60}")
        replay = ReplayBaseline(variant=model_variant, lr=lr)
        all_results["replay"] = replay.run_experiment(
            classes=classes,
            epochs=epochs,
            batch_size=batch_size,
            data_dir=data_dir,
            save_dir=str(comparison_dir),
            synthetic=synthetic,
            seed=seed,
        )

    # Save comparison
    all_results["output_dir"] = str(comparison_dir)
    with open(comparison_dir / "comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"{'='*60}")
    for method, results in all_results.items():
        if "final_evaluation" in results:
            if isinstance(results.get("final_evaluation"), dict):
                print(
                    f"  {method}: "
                    f"mAP@0.5={results['final_evaluation'].get('mAP@0.5', 0.0):.4f}, "
                    f"mAP@0.5:0.95={results['final_evaluation'].get('mAP@0.5:0.95', 0.0):.4f}"
                )
        elif "tasks" in results:
            final_losses = [t["final_loss"] for t in results["tasks"].values()]
            avg_loss = sum(final_losses) / len(final_losses)
            print(f"  {method}: avg final loss = {avg_loss:.4f}")
    print(f"Results: {comparison_dir}")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Det-LoRA and the generic baselines on the same setup"
    )
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--model", dest="model_variant", type=str, default="medium")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=["det_lora", "finetuning", "ewc", "replay"],
        help="Which methods to run (default: all four)",
    )
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--comparison_name", type=str, default=None)
    args = parser.parse_args()
    run_comparison(**vars(args))


if __name__ == "__main__":
    main()
