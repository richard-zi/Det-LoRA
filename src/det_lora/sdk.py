"""
Det-LoRA Adapter SDK
====================

Small Python + CLI management layer for versioned adapter checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.data import DataLoader

from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.evaluator import ContinualEvaluator
from det_lora.model.det_lora import DetLoRA
from det_lora.model.detector import RFDETRDetector, get_device
from det_lora.train import collate_fn, train_adapter
from det_lora.utils import resolve_variant_settings


def _summarize_shared_quality_calibrator(calibrator: Dict[str, Any]) -> Dict[str, Any]:
    """Keep CLI checkpoint inspection compact and management-oriented."""
    if not calibrator:
        return {"enabled": False}

    return {
        "enabled": True,
        "mode": calibrator.get("mode", "unknown"),
        "feature_dim": calibrator.get("feature_dim"),
        "positive_count": calibrator.get("positive_count"),
        "negative_count": calibrator.get("negative_count"),
    }


def _load_checkpoint_metadata(checkpoint_dir: str | Path) -> Dict[str, Any]:
    """Read registry + calibration files without instantiating RF-DETR."""
    checkpoint_path = Path(checkpoint_dir)
    registry_path = checkpoint_path / "det_lora_registry.json"
    calibration_path = checkpoint_path / "adapter_calibration.json"

    if not registry_path.exists():
        raise FileNotFoundError(f"Checkpoint registry not found: {registry_path}")

    with registry_path.open() as f:
        registry = json.load(f)
    calibration = {}
    if calibration_path.exists():
        with calibration_path.open() as f:
            calibration = json.load(f)

    adapter_versions = registry.get("adapter_versions", {})
    active_versions = registry.get("active_versions", {})
    if not adapter_versions:
        adapter_versions = {
            class_name: [
                {
                    "version_id": "v1",
                    "source": "legacy",
                    "created_at": None,
                    "adapter_path": registry.get("adapter_paths", {}).get(class_name),
                }
            ]
            for class_name in registry.get("trained_classes", [])
        }
        active_versions = {class_name: "v1" for class_name in registry.get("trained_classes", [])}

    summary = {
        "checkpoint_dir": str(checkpoint_path),
        "detector_variant": registry.get("detector_variant", "medium"),
        "trained_classes": registry.get("trained_classes", []),
        "active_versions": active_versions,
        "adapter_versions": adapter_versions,
        "shared_quality_calibrator": _summarize_shared_quality_calibrator(
            calibration.get("shared_quality_calibrator", {})
        ),
    }
    return summary


class AdapterSDK:
    """High-level checkpoint management API for versioned Det-LoRA adapters."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        model_variant: Optional[str] = None,
        data_dir: str = "data/raw",
        device=None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.data_dir = data_dir
        self.metadata = _load_checkpoint_metadata(self.checkpoint_dir)
        self.model_variant = model_variant or self.metadata.get("detector_variant", "medium")
        self.device = device or get_device()

        detector = RFDETRDetector(variant=self.model_variant, device=self.device)
        self.det_lora = DetLoRA(detector=detector)
        self.det_lora.load_all(str(self.checkpoint_dir))

    @classmethod
    def inspect_checkpoint(cls, checkpoint_dir: str | Path) -> Dict[str, Any]:
        """Inspect a checkpoint without loading the detector weights."""
        return _load_checkpoint_metadata(checkpoint_dir)

    def reload(self, checkpoint_dir: str | Path) -> None:
        """Reload this SDK instance from another checkpoint."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metadata = _load_checkpoint_metadata(self.checkpoint_dir)
        self.model_variant = self.metadata.get("detector_variant", self.model_variant)
        detector = RFDETRDetector(variant=self.model_variant, device=self.device)
        self.det_lora = DetLoRA(detector=detector)
        self.det_lora.load_all(str(self.checkpoint_dir))

    def summary(self) -> Dict[str, Any]:
        """Return a concise runtime summary for scripts or CLIs."""
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "model_variant": self.model_variant,
            "classes": self.det_lora.list_classes(),
            "active_versions": dict(self.det_lora.active_versions),
            "adapter_versions": self.det_lora.list_adapter_versions(),
            "shared_quality_enabled": bool(self.det_lora.shared_quality_calibrator),
        }

    def list_adapters(self) -> Dict[str, Any]:
        """List classes, versions, and active selections."""
        return self.summary()

    def save(self, output_dir: str | Path) -> Dict[str, Any]:
        """Materialize the current in-memory checkpoint state to disk."""
        output_path = Path(output_dir)
        self.det_lora.save_all(str(output_path))
        self.reload(output_path)
        return self.summary()

    def activate_adapter_version(
        self,
        class_name: str,
        version_id: str,
        *,
        output_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Switch one class to another stored version and optionally save it."""
        result = self.det_lora.activate_adapter_version(class_name, version_id)
        if output_dir is not None:
            self.save(output_dir)
            result["output_dir"] = str(output_dir)
        return result

    def remove_adapter(
        self,
        class_name: str,
        *,
        version_id: Optional[str] = None,
        output_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Soft-remove one adapter version and optionally save the new checkpoint."""
        result = self.det_lora.remove_adapter_version(class_name, version_id=version_id)
        if output_dir is not None:
            self.save(output_dir)
            result["output_dir"] = str(output_dir)
        return result

    def add_class(self, class_name: str, **train_kwargs: Any) -> Dict[str, Any]:
        """Train and register a new class version on top of the current checkpoint."""
        preset_name = train_kwargs.pop("preset_name", None)
        model_variant = train_kwargs.get("model_variant", self.model_variant)
        resolved = resolve_variant_settings(
            variant=model_variant,
            preset_name=preset_name,
            base_defaults={
                "epochs": 50,
                "batch_size": 4,
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "lora_rank": 8,
                "lora_alpha": 16,
                "metrics_eval_every": 1,
            },
            overrides={
                "epochs": train_kwargs.pop("epochs", None),
                "batch_size": train_kwargs.pop("batch_size", None),
                "lr": train_kwargs.pop("lr", None),
                "weight_decay": train_kwargs.pop("weight_decay", None),
                "lora_rank": train_kwargs.pop("lora_rank", None),
                "lora_alpha": train_kwargs.pop("lora_alpha", None),
                "metrics_eval_every": train_kwargs.pop("metrics_eval_every", None),
            },
        )
        results = train_adapter(
            class_name=class_name,
            load_dir=str(self.checkpoint_dir),
            extend=False,
            data_dir=train_kwargs.pop("data_dir", self.data_dir),
            epochs=int(resolved["epochs"]),
            batch_size=int(resolved["batch_size"]),
            lr=float(resolved["lr"]),
            weight_decay=float(resolved["weight_decay"]),
            lora_rank=int(resolved["lora_rank"]),
            lora_alpha=int(resolved["lora_alpha"]),
            metrics_eval_every=int(resolved["metrics_eval_every"]),
            preset_name=preset_name,
            **train_kwargs,
        )
        final_dir = Path(results["output_dir"]) / "final"
        self.reload(final_dir)
        return {
            "results": results,
            "final_checkpoint": str(final_dir),
            "summary": self.summary(),
        }

    def extend_class(self, class_name: str, **train_kwargs: Any) -> Dict[str, Any]:
        """Train a new version for an existing class."""
        preset_name = train_kwargs.pop("preset_name", None)
        model_variant = train_kwargs.get("model_variant", self.model_variant)
        resolved = resolve_variant_settings(
            variant=model_variant,
            preset_name=preset_name,
            base_defaults={
                "epochs": 50,
                "batch_size": 4,
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "lora_rank": 8,
                "lora_alpha": 16,
                "metrics_eval_every": 1,
            },
            overrides={
                "epochs": train_kwargs.pop("epochs", None),
                "batch_size": train_kwargs.pop("batch_size", None),
                "lr": train_kwargs.pop("lr", None),
                "weight_decay": train_kwargs.pop("weight_decay", None),
                "lora_rank": train_kwargs.pop("lora_rank", None),
                "lora_alpha": train_kwargs.pop("lora_alpha", None),
                "metrics_eval_every": train_kwargs.pop("metrics_eval_every", None),
            },
        )
        results = train_adapter(
            class_name=class_name,
            load_dir=str(self.checkpoint_dir),
            extend=True,
            data_dir=train_kwargs.pop("data_dir", self.data_dir),
            epochs=int(resolved["epochs"]),
            batch_size=int(resolved["batch_size"]),
            lr=float(resolved["lr"]),
            weight_decay=float(resolved["weight_decay"]),
            lora_rank=int(resolved["lora_rank"]),
            lora_alpha=int(resolved["lora_alpha"]),
            metrics_eval_every=int(resolved["metrics_eval_every"]),
            preset_name=preset_name,
            **train_kwargs,
        )
        final_dir = Path(results["output_dir"]) / "final"
        self.reload(final_dir)
        return {
            "results": results,
            "final_checkpoint": str(final_dir),
            "summary": self.summary(),
        }

    def _build_test_loader(self, class_filter, *, batch_size: int, seed: int) -> DataLoader:
        dataset = load_dataset_from_raw(
            raw_dir=self.data_dir,
            class_filter=class_filter,
            split="test",
            class_id_offset=self.det_lora.detector.base_num_classes,
            img_size=self.det_lora.detector.resolution,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

    def test_adapter(
        self,
        class_name: str,
        *,
        version_id: Optional[str] = None,
        mode: str = "single+mixed",
        batch_size: int = 4,
        seed: int = 42,
        include_curves: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate one adapter version on matched single-class and/or mixed seen classes."""
        if class_name not in self.det_lora.adapter_versions:
            raise ValueError(f"Unknown class '{class_name}'")
        if mode not in {"single", "mixed", "single+mixed"}:
            raise ValueError(f"Unknown test mode '{mode}'")

        previous_version = self.det_lora.get_active_version(class_name)
        if version_id is not None and version_id != previous_version:
            self.det_lora.activate_adapter_version(class_name, version_id)
        else:
            version_id = previous_version

        evaluator = ContinualEvaluator(
            self.det_lora,
            use_shared_quality_calibrator=bool(self.det_lora.shared_quality_calibrator),
        )
        results: Dict[str, Any] = {
            "checkpoint_dir": str(self.checkpoint_dir),
            "class_name": class_name,
            "version_id": version_id,
            "mode": mode,
            "active_versions_snapshot": dict(self.det_lora.active_versions),
        }

        if mode in {"single", "single+mixed"}:
            single_loader = self._build_test_loader(class_name, batch_size=batch_size, seed=seed)
            results["single_metrics"] = evaluator.evaluate_det_lora_joint(
                dataloader=single_loader,
                class_names=[class_name],
                include_curves=include_curves,
            )

        if mode in {"mixed", "single+mixed"}:
            seen_classes = list(self.det_lora.trained_classes)
            mixed_loader = self._build_test_loader(seen_classes, batch_size=batch_size, seed=seed)
            results["mixed_metrics"] = evaluator.evaluate_det_lora_joint(
                dataloader=mixed_loader,
                class_names=seen_classes,
                include_curves=include_curves,
            )

        if previous_version is not None and previous_version != version_id:
            self.det_lora.activate_adapter_version(class_name, previous_version)
        return results


def _print_payload(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Det-LoRA adapter SDK")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List classes and adapter versions")
    list_parser.add_argument("--checkpoint_dir", required=True)

    activate_parser = subparsers.add_parser("activate", help="Activate an adapter version")
    activate_parser.add_argument("--checkpoint_dir", required=True)
    activate_parser.add_argument("--class_name", required=True)
    activate_parser.add_argument("--version", required=True)
    activate_parser.add_argument("--output_dir", required=True)
    activate_parser.add_argument("--model", default=None)

    remove_parser = subparsers.add_parser("remove", help="Soft-remove an adapter version")
    remove_parser.add_argument("--checkpoint_dir", required=True)
    remove_parser.add_argument("--class_name", required=True)
    remove_parser.add_argument("--version", default=None)
    remove_parser.add_argument("--output_dir", required=True)
    remove_parser.add_argument("--model", default=None)

    test_parser = subparsers.add_parser("test", help="Evaluate one adapter version")
    test_parser.add_argument("--checkpoint_dir", required=True)
    test_parser.add_argument("--class_name", required=True)
    test_parser.add_argument("--version", default=None)
    test_parser.add_argument(
        "--mode", default="single+mixed", choices=["single", "mixed", "single+mixed"]
    )
    test_parser.add_argument("--batch_size", type=int, default=4)
    test_parser.add_argument("--seed", type=int, default=42)
    test_parser.add_argument("--include_curves", action="store_true")
    test_parser.add_argument("--data_dir", default="data/raw")
    test_parser.add_argument("--model", default=None)

    common_train_args = {
        "epochs": {"type": int, "default": None},
        "batch_size": {"type": int, "default": None},
        "lr": {"type": float, "default": None},
        "weight_decay": {"type": float, "default": None},
        "lora_rank": {"type": int, "default": None},
        "lora_alpha": {"type": int, "default": None},
        "model": {"default": "medium"},
        "preset": {"default": None},
        "data_dir": {"default": "data/raw"},
        "save_dir": {"default": "experiments"},
        "max_samples": {"type": int, "default": None},
        "seed": {"type": int, "default": 42},
        "metrics_eval_every": {"type": int, "default": None},
    }

    add_parser = subparsers.add_parser("add", help="Train and add a new class version")
    add_parser.add_argument("--class_name", required=True)
    add_parser.add_argument("--checkpoint_dir", default=None)
    for name, kwargs in common_train_args.items():
        add_parser.add_argument(f"--{name}", **kwargs)

    extend_parser = subparsers.add_parser(
        "extend", help="Train a new version for an existing class"
    )
    extend_parser.add_argument("--class_name", required=True)
    extend_parser.add_argument("--checkpoint_dir", required=True)
    for name, kwargs in common_train_args.items():
        extend_parser.add_argument(f"--{name}", **kwargs)

    args = parser.parse_args()

    if args.command == "list":
        _print_payload(AdapterSDK.inspect_checkpoint(args.checkpoint_dir))
        return

    if args.command == "add" and args.checkpoint_dir is None:
        results = train_adapter(
            class_name=args.class_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            model_variant=args.model,
            preset_name=args.preset,
            data_dir=args.data_dir,
            save_dir=args.save_dir,
            max_samples=args.max_samples,
            seed=args.seed,
            metrics_eval_every=args.metrics_eval_every,
        )
        _print_payload(results)
        return

    sdk = AdapterSDK(
        checkpoint_dir=args.checkpoint_dir,
        model_variant=getattr(args, "model", None),
        data_dir=getattr(args, "data_dir", "data/raw"),
    )

    if args.command == "activate":
        _print_payload(
            sdk.activate_adapter_version(
                args.class_name,
                args.version,
                output_dir=args.output_dir,
            )
        )
    elif args.command == "remove":
        _print_payload(
            sdk.remove_adapter(
                args.class_name,
                version_id=args.version,
                output_dir=args.output_dir,
            )
        )
    elif args.command == "test":
        _print_payload(
            sdk.test_adapter(
                args.class_name,
                version_id=args.version,
                mode=args.mode,
                batch_size=args.batch_size,
                seed=args.seed,
                include_curves=args.include_curves,
            )
        )
    elif args.command == "add":
        _print_payload(
            sdk.add_class(
                args.class_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                model_variant=args.model,
                preset_name=args.preset,
                data_dir=args.data_dir,
                save_dir=args.save_dir,
                max_samples=args.max_samples,
                seed=args.seed,
                metrics_eval_every=args.metrics_eval_every,
            )
        )
    elif args.command == "extend":
        _print_payload(
            sdk.extend_class(
                args.class_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                model_variant=args.model,
                preset_name=args.preset,
                data_dir=args.data_dir,
                save_dir=args.save_dir,
                max_samples=args.max_samples,
                seed=args.seed,
                metrics_eval_every=args.metrics_eval_every,
            )
        )


if __name__ == "__main__":
    main()
