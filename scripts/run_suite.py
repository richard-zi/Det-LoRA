#!/usr/bin/env python3
"""
Det-LoRA Reproducible Experiment Suite Runner
=============================================

Runs multi-seed thesis experiments and writes aggregated summaries.

Examples:
    uv run python scripts/run_suite.py --phase all --seeds 42 43 44 --epochs 30
    uv run python scripts/run_suite.py --phase training --suite_name thesis_main
    uv run python scripts/run_suite.py --phase baselines --methods finetuning,ewc,replay
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from det_lora.utils import resolve_variant_settings

CORE_METRICS = [
    "mAP@0.5",
    "mAP@0.75",
    "mAP@0.5:0.95",
    "Precision@0.5",
    "Recall@0.5",
    "F1@0.5",
    "MicroPrecision@0.5",
    "MicroRecall@0.5",
    "MicroF1@0.5",
]


def log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def _load_json(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def _normalize_methods(methods_arg: Optional[str]) -> List[str]:
    if not methods_arg:
        return ["finetuning", "ewc", "replay"]
    return [method.strip() for method in methods_arg.split(",") if method.strip()]


def _extract_final_metrics(result: Dict) -> Dict[str, float]:
    metrics = result.get("final_evaluation", {}) or result.get("test_metrics", {}) or {}
    return {
        key: float(metrics[key])
        for key in CORE_METRICS
        if key in metrics and isinstance(metrics[key], (int, float))
    }


def _extract_final_ap_per_class(result: Dict) -> Dict[str, float]:
    metrics = result.get("final_evaluation", {}) or result.get("test_metrics", {}) or {}
    return {
        str(class_name): float(ap) for class_name, ap in metrics.get("AP_per_class@0.5", {}).items()
    }


def _compute_average_forgetting(result: Dict) -> Optional[float]:
    history = result.get("matched_evaluation_after_task") or result.get("evaluation_after_task", {})
    final_metrics = result.get("matched_final_evaluation") or result.get("final_evaluation", {})
    final_ap = final_metrics.get("AP_per_class@0.5", {})
    if not history or not final_ap:
        return None

    best_per_class: Dict[str, float] = {}
    for task_metrics in history.values():
        for class_name, ap in task_metrics.get("AP_per_class@0.5", {}).items():
            best_per_class[class_name] = max(best_per_class.get(class_name, 0.0), float(ap))

    forgetting_values = [
        max(0.0, best_ap - float(final_ap.get(class_name, 0.0)))
        for class_name, best_ap in best_per_class.items()
    ]
    if not forgetting_values:
        return None
    return float(mean(forgetting_values))


def _aggregate_scalars(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _build_method_summary(seed_runs: List[Dict]) -> Dict:
    metrics_by_name: Dict[str, List[float]] = {metric: [] for metric in CORE_METRICS}
    forgetting_values: List[float] = []
    ap_per_class: Dict[str, List[float]] = {}

    for seed_run in seed_runs:
        for metric_name, metric_value in seed_run.get("final_metrics", {}).items():
            metrics_by_name.setdefault(metric_name, []).append(float(metric_value))
        avg_forgetting = seed_run.get("average_forgetting")
        if avg_forgetting is not None:
            forgetting_values.append(float(avg_forgetting))
        for class_name, ap in seed_run.get("final_ap_per_class@0.5", {}).items():
            ap_per_class.setdefault(class_name, []).append(float(ap))

    return {
        "num_seeds": len(seed_runs),
        "seeds": [run["seed"] for run in seed_runs],
        "metrics": {
            metric_name: _aggregate_scalars(metric_values)
            for metric_name, metric_values in metrics_by_name.items()
            if metric_values
        },
        "average_forgetting": _aggregate_scalars(forgetting_values) if forgetting_values else {},
        "AP_per_class@0.5": {
            class_name: _aggregate_scalars(ap_values)
            for class_name, ap_values in ap_per_class.items()
            if ap_values
        },
        "runs": seed_runs,
    }


def _format_metric(summary: Dict, metric_name: str) -> str:
    metric = summary.get("metrics", {}).get(metric_name)
    if not metric:
        return "-"
    return f"{metric['mean']:.4f} +- {metric['std']:.4f}"


def _write_suite_summary_markdown(path: Path, summary: Dict) -> None:
    methods = summary.get("methods", {})
    lines = [
        "# Suite Summary",
        "",
        "| Method | Seeds | mAP@0.5 | mAP@0.75 | mAP@0.5:0.95 | Precision@0.5 | Recall@0.5 | F1@0.5 | Avg Forgetting |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for method_name, method_summary in sorted(methods.items()):
        forgetting = method_summary.get("average_forgetting", {})
        forgetting_text = "-"
        if forgetting:
            forgetting_text = f"{forgetting['mean']:.4f} +- {forgetting['std']:.4f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    method_name,
                    str(method_summary.get("num_seeds", 0)),
                    _format_metric(method_summary, "mAP@0.5"),
                    _format_metric(method_summary, "mAP@0.75"),
                    _format_metric(method_summary, "mAP@0.5:0.95"),
                    _format_metric(method_summary, "Precision@0.5"),
                    _format_metric(method_summary, "Recall@0.5"),
                    _format_metric(method_summary, "F1@0.5"),
                    forgetting_text,
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n")


def _collect_seed_run(result: Dict, seed: int) -> Dict:
    return {
        "seed": seed,
        "output_dir": result.get("config", {}).get("output_dir"),
        "final_metrics": _extract_final_metrics(result),
        "final_ap_per_class@0.5": _extract_final_ap_per_class(result),
        "average_forgetting": _compute_average_forgetting(result),
    }


def _load_or_run_det_lora(args, seed: int, seed_dir: Path) -> Dict:
    from det_lora.run_experiment import run_continual_experiment

    run_name = f"det_lora_seed_{seed}"
    run_dir = seed_dir / run_name
    results_path = run_dir / "results.json"
    if results_path.exists() and not args.force:
        log(f"Seed {seed}: existing Det-LoRA run gefunden, ueberspringe Neu-Start")
        return _load_json(results_path)

    resolved = resolve_variant_settings(
        variant=args.model,
        preset_name=args.preset,
        base_defaults={
            "epochs": 30,
            "batch_size": 4,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lora_rank": 8,
            "lora_alpha": 16,
            "metrics_eval_every": 5,
        },
        overrides={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "metrics_eval_every": args.metrics_eval_every,
        },
    )
    return run_continual_experiment(
        classes=args.classes,
        epochs=int(resolved["epochs"]),
        batch_size=int(resolved["batch_size"]),
        lr=float(resolved["lr"]),
        weight_decay=float(resolved["weight_decay"]),
        lora_rank=int(resolved["lora_rank"]),
        lora_alpha=int(resolved["lora_alpha"]),
        model_variant=args.model,
        data_dir=args.data_dir,
        save_dir=str(seed_dir),
        synthetic=args.synthetic,
        max_samples=args.max_samples,
        seed=seed,
        experiment_name=run_name,
        metrics_eval_every=int(resolved["metrics_eval_every"]),
        enable_shared_quality_calibrator=not args.disable_shared_quality_calibrator,
        preset_name=args.preset,
    )


def _load_or_run_baselines(args, seed: int, seed_dir: Path, methods: List[str]) -> Dict:
    from det_lora.baselines.compare import run_comparison

    run_name = f"comparison_seed_{seed}"
    run_dir = seed_dir / run_name
    results_path = run_dir / "comparison.json"
    if results_path.exists() and not args.force:
        log(f"Seed {seed}: bestehender Baseline-Vergleich gefunden, ueberspringe Neu-Start")
        return _load_json(results_path)

    resolved = resolve_variant_settings(
        variant=args.model,
        preset_name=args.preset,
        base_defaults={
            "epochs": 30,
            "batch_size": 4,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lora_rank": 8,
            "lora_alpha": 16,
            "metrics_eval_every": 5,
        },
        overrides={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
        },
    )
    return run_comparison(
        classes=args.classes,
        epochs=int(resolved["epochs"]),
        batch_size=int(resolved["batch_size"]),
        lr=float(resolved["lr"]),
        lora_rank=int(resolved["lora_rank"]),
        model_variant=args.model,
        data_dir=args.data_dir,
        save_dir=str(seed_dir),
        synthetic=args.synthetic,
        methods=methods,
        seed=seed,
        comparison_name=run_name,
    )


def _write_suite_outputs(
    suite_dir: Path,
    manifest: Dict,
    suite_results: Dict,
) -> None:
    summary = {
        "suite_dir": str(suite_dir),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "methods": {},
    }

    for method_name, seed_runs in suite_results.get("methods", {}).items():
        summary["methods"][method_name] = _build_method_summary(seed_runs)

    _write_json(suite_dir / "suite_manifest.json", manifest)
    _write_json(suite_dir / "suite_results.json", suite_results)
    _write_json(suite_dir / "suite_summary.json", summary)
    _write_suite_summary_markdown(suite_dir / "suite_summary.md", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Det-LoRA reproducible multi-seed suite runner")
    parser.add_argument(
        "--phase",
        required=True,
        choices=["training", "baselines", "all"],
        help="training = only Det-LoRA, baselines = only baselines, all = both",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=[
            "military_tank",
            "military_truck",
            "military_aircraft",
            "military_helicopter",
            "civilian_car",
            "civilian_aircraft",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--model", type=str, default="medium")
    parser.add_argument(
        "--preset", type=str, default=None, help="Variant-specific preset, e.g. l40_final"
    )
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--methods", type=str, default="finetuning,ewc,replay")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--suite_name", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Ignore existing per-seed runs")
    parser.add_argument(
        "--enable_shared_quality_calibrator",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--disable_shared_quality_calibrator",
        action="store_true",
        help="Disable the shared quality/objectness calibrator (enabled by default)",
    )
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suite_name = args.suite_name or f"suite_{timestamp}"
    suite_dir = Path(args.save_dir) / "suites" / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    methods = _normalize_methods(args.methods)
    manifest = {
        "suite_name": suite_name,
        "suite_dir": str(suite_dir),
        "phase": args.phase,
        "classes": args.classes,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "model": args.model,
        "preset": args.preset,
        "data_dir": args.data_dir,
        "max_samples": args.max_samples,
        "methods": methods,
        "synthetic": args.synthetic,
        "enable_shared_quality_calibrator": not args.disable_shared_quality_calibrator,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    suite_results = {
        "suite_name": suite_name,
        "suite_dir": str(suite_dir),
        "methods": {},
    }

    log(f"Starte Suite {suite_name}")
    for seed in args.seeds:
        seed_dir = suite_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        log(f"Seed {seed}: Output in {seed_dir}")

        if args.phase in {"training", "all"}:
            det_lora_result = _load_or_run_det_lora(args, seed, seed_dir)
            suite_results["methods"].setdefault("det_lora", []).append(
                _collect_seed_run(det_lora_result, seed)
            )

        if args.phase in {"baselines", "all"}:
            comparison_result = _load_or_run_baselines(args, seed, seed_dir, methods)
            for method_name, method_result in comparison_result.items():
                if method_name == "config":
                    continue
                suite_results["methods"].setdefault(method_name, []).append(
                    _collect_seed_run(method_result, seed)
                )

        _write_suite_outputs(suite_dir, manifest, suite_results)

    manifest["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_suite_outputs(suite_dir, manifest, suite_results)
    log(f"Suite abgeschlossen: {suite_dir}")


if __name__ == "__main__":
    main()
