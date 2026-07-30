"""Retroactive hardening of an already trained adapter against a confusion
class that was learned later (variant A of the replay-free online formulation).

A-refresh: extend path (warm_start), i.e. own positives plus the new class as
empty-target hard negatives, with a stability anchor / teacher distillation
towards the original weights. No positives from other classes are used, so
the zero-forgetting isolation is preserved.

Usage (from the repo root):
  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --no-sync python scripts/ablations/retro_harden.py \
    --source experiments/suites/thesis_l40_main/model_nano/seed_42/det_lora/final \
    --variant nano --seed 42 --target military_aircraft --negative civilian_aircraft \
    --epochs 20 --out_dir experiments/retro/nano_seed42_refresh
"""

from __future__ import annotations

import argparse

from det_lora.train import train_adapter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="forward-only final checkpoint to harden from")
    p.add_argument("--variant", default="nano")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target", default="military_aircraft")
    p.add_argument("--negative", default="civilian_aircraft")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--stability_loss_weight", type=float, default=1e-5)
    p.add_argument("--teacher_anchor_weight", type=float, default=0.05)
    p.add_argument("--metrics_eval_every", type=int, default=1)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--data_dir", default="data/raw")
    args = p.parse_args()

    result = train_adapter(
        class_name=args.target,
        extend=True,
        extend_strategy="warm_start",
        load_dir=args.source,
        model_variant=args.variant,
        data_dir=args.data_dir,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        lora_rank=8,
        lora_alpha=16,
        stability_loss_weight=args.stability_loss_weight,
        teacher_anchor_weight=args.teacher_anchor_weight,
        metrics_eval_every=args.metrics_eval_every,
        use_hard_negatives=True,
        hard_negative_classes=[args.negative],
        use_adapter_arbitration=False,
        save_dir=args.out_dir,
        experiment_name="harden",
    )
    print("\nFINAL_CHECKPOINT_DIR:", result.get("final_checkpoint_dir"))


if __name__ == "__main__":
    main()
