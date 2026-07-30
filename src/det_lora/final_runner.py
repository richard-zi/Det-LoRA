"""Final thesis suite runner with auto-resume and rich artifacts."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional

from det_lora.baselines.cl_detr import CLDETRBaseline
from det_lora.baselines.ewc import EWCBaseline
from det_lora.baselines.finetuning import FineTuningBaseline, JointFineTuningBaseline
from det_lora.baselines.replay import ReplayBaseline
from det_lora.run_experiment import run_continual_experiment
from det_lora.train import train_adapter
from det_lora.utils import collect_runtime_metadata, expand_model_variants, resolve_variant_settings

DEFAULT_METHODS = ["finetuning", "ewc", "replay"]
CORE_METRICS = [
    "mAP@0.5",
    "mAP@0.75",
    "mAP@0.95",
    "mAP@0.5:0.95",
    "Precision@0.5",
    "Precision@0.95",
    "Recall@0.5",
    "Recall@0.95",
    "F1@0.5",
    "F1@0.95",
]


@dataclass(frozen=True)
class RunSpec:
    kind: str
    model: str
    seed: int
    method: str

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.model}:{self.seed}:{self.method}"


@dataclass(frozen=True)
class ExtensionSpec:
    kind: str
    model: str
    seed: int
    method: str
    target_class: str
    stage_name: str
    stage_index: int
    epochs: int
    max_samples: Optional[int]
    sample_offset: int
    source_stage_name: Optional[str]
    extend_strategy: str = "warm_start"
    version_selection_strategy: str = "anchor_latest"

    @property
    def key(self) -> str:
        return (
            f"{self.kind}:{self.model}:{self.seed}:{self.method}:"
            f"{self.target_class}:{self.stage_index}:{self.stage_name}"
        )


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def _load_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    with path.open() as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _append_event(events_path: Path, payload: Dict) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a") as handle:
        handle.write(json.dumps({"timestamp": _now(), **payload}, default=str) + "\n")


def _safe_check_output(cmd: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def capture_hardware_metadata() -> Dict:
    runtime = collect_runtime_metadata()
    metadata = {
        **runtime,
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "git_commit": _safe_check_output(["git", "rev-parse", "HEAD"]),
        "git_branch": _safe_check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python_executable": sys.executable,
        "platform_node": platform.node(),
    }
    nvidia_smi = _safe_check_output(["nvidia-smi", "-L"])
    if nvidia_smi:
        metadata["nvidia_smi"] = nvidia_smi
    return metadata


def _normalize_methods(methods) -> List[str]:
    if methods is None:
        return list(DEFAULT_METHODS)
    if isinstance(methods, str):
        methods = methods.split(",")
    return [str(method).strip() for method in methods if str(method).strip()]


def _resolve_config(config_path: Path, cli_args: argparse.Namespace) -> Dict:
    config = _load_json(config_path)
    if not config:
        raise ValueError(f"Empty or missing config: {config_path}")

    resolved = dict(config)
    for key in ("suite_name", "phase", "save_dir", "data_dir", "preset"):
        value = getattr(cli_args, key, None)
        if value is not None:
            resolved[key] = value
    if cli_args.models:
        resolved["models"] = cli_args.models
    if cli_args.seeds:
        resolved["seeds"] = cli_args.seeds
    if cli_args.max_samples is not None:
        resolved["max_samples"] = cli_args.max_samples
    if cli_args.synthetic:
        resolved["synthetic"] = True
    if cli_args.disable_shared_quality_calibrator:
        resolved["enable_shared_quality_calibrator"] = False
    if cli_args.methods:
        resolved["methods"] = _normalize_methods(cli_args.methods)

    resolved.setdefault("suite_name", "final_l40_suite")
    resolved.setdefault("phase", "all")
    resolved.setdefault("models", ["medium"])
    resolved.setdefault("classes", [])
    resolved.setdefault("seeds", [42, 43, 44])
    resolved.setdefault("preset", "l40_final")
    resolved.setdefault("data_dir", "data/raw")
    resolved.setdefault("save_dir", "experiments")
    resolved.setdefault("methods", list(DEFAULT_METHODS))
    resolved.setdefault("synthetic", False)
    resolved.setdefault("max_samples", None)
    resolved.setdefault("enable_shared_quality_calibrator", True)
    resolved.setdefault("use_adapter_arbitration", False)
    resolved["methods"] = _normalize_methods(resolved.get("methods"))
    resolved["models"] = expand_model_variants(resolved.get("models", ["medium"]))
    return resolved


def _build_run_specs(config: Dict) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for model in config["models"]:
        for seed in config["seeds"]:
            if config["phase"] in {"training", "all"}:
                specs.append(RunSpec(kind="training", model=model, seed=seed, method="det_lora"))
            if config["phase"] in {"baselines", "all"}:
                for method in config["methods"]:
                    specs.append(RunSpec(kind="baseline", model=model, seed=seed, method=method))
    return specs


def _suite_dir(config: Dict) -> Path:
    return Path(config["save_dir"]) / "suites" / config["suite_name"]


def _run_dir(suite_dir: Path, spec: RunSpec) -> Path:
    return suite_dir / f"model_{spec.model}" / f"seed_{spec.seed}" / spec.method


def _results_path(spec: RunSpec, run_dir: Path) -> Path:
    return run_dir / "results.json"


def _run_complete(spec: RunSpec, run_dir: Path, config: Dict) -> bool:
    results_path = _results_path(spec, run_dir)
    if not results_path.exists():
        return False
    if spec.kind == "training":
        return (run_dir / "final").exists()
    progress = _load_json(run_dir / "progress.json", default={})
    completed_tasks = progress.get("completed_tasks", [])
    if len(completed_tasks) >= len(config.get("classes", [])):
        return True
    results = _load_json(results_path)
    return bool(results.get("final_evaluation"))


def _collect_seed_run(result: Dict, seed: int, default_output_dir: Optional[str] = None) -> Dict:
    mixed = (
        result.get("mixed_final_evaluation")
        or result.get("final_evaluation")
        or result.get("test_metrics")
        or {}
    )
    matched = result.get("matched_final_evaluation") or {}
    forgetting = result.get("matched_forgetting") or result.get("forgetting") or {}
    avg_forgetting = None
    if forgetting:
        avg_forgetting = float(mean(float(value) for value in forgetting.values()))
    return {
        "seed": seed,
        "output_dir": (
            result.get("output_dir")
            or result.get("config", {}).get("output_dir")
            or default_output_dir
        ),
        "mixed_metrics": {key: float(mixed[key]) for key in CORE_METRICS if key in mixed},
        "matched_metrics": {key: float(matched[key]) for key in CORE_METRICS if key in matched},
        "average_forgetting": avg_forgetting,
    }


def _aggregate_scalars(values: Iterable[float]) -> Dict[str, float]:
    values = [float(value) for value in values]
    if not values:
        return {}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _fmt_metric(metrics: Dict, name: str) -> str:
    value = metrics.get(name)
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _fmt_delta(metrics: Dict, name: str) -> str:
    value = metrics.get(name)
    if value is None:
        return "-"
    return f"{float(value):+.4f}"


def _log_run_result(spec: RunSpec, seed_run: Dict) -> None:
    mixed_metrics = seed_run.get("mixed_metrics", {})
    matched_metrics = seed_run.get("matched_metrics", {})
    forgetting = seed_run.get("average_forgetting")
    forgetting_text = "-" if forgetting is None else f"{float(forgetting):.4f}"
    log(
        "Run fertig "
        f"{spec.key} | mixed mAP@0.5={_fmt_metric(mixed_metrics, 'mAP@0.5')} "
        f"mAP@0.95={_fmt_metric(mixed_metrics, 'mAP@0.95')} "
        f"mAP@0.5:0.95={_fmt_metric(mixed_metrics, 'mAP@0.5:0.95')} | "
        f"matched mAP@0.5={_fmt_metric(matched_metrics, 'mAP@0.5')} "
        f"mAP@0.95={_fmt_metric(matched_metrics, 'mAP@0.95')} | "
        f"forgetting={forgetting_text}"
    )


def _log_extension_result(spec: ExtensionSpec, seed_run: Dict) -> None:
    mixed_metrics = seed_run.get("mixed_metrics", {})
    target_metrics = seed_run.get("matched_metrics", {})
    mixed_delta = seed_run.get("mixed_extension_delta", {})
    target_delta = seed_run.get("target_extension_delta", {})
    log(
        "Extension fertig "
        f"{spec.key} | target mAP@0.5={_fmt_metric(target_metrics, 'mAP@0.5')} "
        f"mAP@0.95={_fmt_metric(target_metrics, 'mAP@0.95')} "
        f"delta@0.5={_fmt_delta(target_delta, 'mAP@0.5')} "
        f"delta@0.95={_fmt_delta(target_delta, 'mAP@0.95')} | "
        f"mixed mAP@0.5={_fmt_metric(mixed_metrics, 'mAP@0.5')} "
        f"mAP@0.95={_fmt_metric(mixed_metrics, 'mAP@0.95')} "
        f"delta@0.5={_fmt_delta(mixed_delta, 'mAP@0.5')} "
        f"delta@0.95={_fmt_delta(mixed_delta, 'mAP@0.95')}"
    )


def _build_summary(suite_dir: Path, config: Dict, suite_results: Dict) -> Dict:
    summary = {
        "suite_dir": str(suite_dir),
        "generated_at": _now(),
        "groups": {},
    }
    for group_name, seed_runs in suite_results.get("groups", {}).items():
        mixed_metrics: Dict[str, List[float]] = {}
        matched_metrics: Dict[str, List[float]] = {}
        forgetting_values: List[float] = []
        for seed_run in seed_runs:
            for name, value in seed_run.get("mixed_metrics", {}).items():
                mixed_metrics.setdefault(name, []).append(float(value))
            for name, value in seed_run.get("matched_metrics", {}).items():
                matched_metrics.setdefault(name, []).append(float(value))
            avg_forgetting = seed_run.get("average_forgetting")
            if avg_forgetting is not None:
                forgetting_values.append(float(avg_forgetting))
        summary["groups"][group_name] = {
            "num_seeds": len(seed_runs),
            "seeds": [run["seed"] for run in seed_runs],
            "mixed_metrics": {
                name: _aggregate_scalars(values) for name, values in mixed_metrics.items() if values
            },
            "matched_metrics": {
                name: _aggregate_scalars(values)
                for name, values in matched_metrics.items()
                if values
            },
            "average_forgetting": (
                _aggregate_scalars(forgetting_values) if forgetting_values else {}
            ),
            "runs": seed_runs,
        }
    return summary


def _write_summary_markdown(path: Path, summary: Dict) -> None:
    lines = [
        "# Final Suite Summary",
        "",
        "| Group | Seeds | Mixed mAP@0.5 | Mixed mAP@0.95 | Mixed mAP@0.5:0.95 | Matched mAP@0.5 | Matched mAP@0.95 | Avg Forgetting |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group_name, group in sorted(summary.get("groups", {}).items()):
        mixed50 = group.get("mixed_metrics", {}).get("mAP@0.5", {})
        mixed95 = group.get("mixed_metrics", {}).get("mAP@0.95", {})
        mixed5095 = group.get("mixed_metrics", {}).get("mAP@0.5:0.95", {})
        matched50 = group.get("matched_metrics", {}).get("mAP@0.5", {})
        matched95 = group.get("matched_metrics", {}).get("mAP@0.95", {})
        forgetting = group.get("average_forgetting", {})

        def fmt(metric: Dict) -> str:
            if not metric:
                return "-"
            return f"{metric['mean']:.4f} +- {metric['std']:.4f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    group_name,
                    str(group.get("num_seeds", 0)),
                    fmt(mixed50),
                    fmt(mixed95),
                    fmt(mixed5095),
                    fmt(matched50),
                    fmt(matched95),
                    fmt(forgetting),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def _resolved_hparams(config: Dict, model: str) -> Dict[str, float | int]:
    return resolve_variant_settings(
        variant=model,
        preset_name=config.get("preset"),
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
            "epochs": config.get("epochs"),
            "batch_size": config.get("batch_size"),
            "lr": config.get("lr"),
            "weight_decay": config.get("weight_decay"),
            "lora_rank": config.get("lora_rank"),
            "lora_alpha": config.get("lora_alpha"),
            "metrics_eval_every": config.get("metrics_eval_every"),
        },
    )


def _resolved_extension_hparams(config: Dict, model: str) -> Dict[str, float | int]:
    resolved = dict(_resolved_hparams(config, model))
    extension_config = config.get("extension", {})
    for key in (
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "lora_rank",
        "lora_alpha",
        "metrics_eval_every",
    ):
        value = extension_config.get(key)
        if value is not None:
            resolved[key] = value
    return resolved


def _build_extension_stage_plan(config: Dict, model: str) -> List[Dict]:
    extension_config = config.get("extension", {})
    default_hparams = _resolved_extension_hparams(config, model)
    raw_stages = extension_config.get("stages") or []
    if not raw_stages:
        return [
            {
                "name": str(extension_config.get("stage_name") or "extension"),
                "epochs": int(extension_config.get("epochs", default_hparams["epochs"])),
                "max_samples": extension_config.get("max_samples"),
                "disjoint_max_samples": extension_config.get("max_samples"),
                "sample_offset": int(extension_config.get("sample_offset", 0)),
                "disjoint_sample_offset": int(extension_config.get("sample_offset", 0)),
                "source_stage_name": None,
            }
        ]

    stages: List[Dict] = []
    previous_stage_name: Optional[str] = None
    next_disjoint_offset = int(extension_config.get("sample_offset", 0))
    for stage_index, stage_config in enumerate(raw_stages):
        stage_name = str(stage_config.get("name") or f"stage_{stage_index + 1}").strip()
        stage_name = stage_name.replace(" ", "_")
        stage_max_samples = stage_config.get("max_samples", extension_config.get("max_samples"))
        disjoint_max_samples = stage_config.get("disjoint_max_samples", stage_max_samples)
        sample_offset = int(
            stage_config.get("sample_offset", extension_config.get("sample_offset", 0))
        )
        stages.append(
            {
                "name": stage_name,
                "epochs": int(stage_config.get("epochs", default_hparams["epochs"])),
                "max_samples": stage_max_samples,
                "disjoint_max_samples": disjoint_max_samples,
                "sample_offset": sample_offset,
                "disjoint_sample_offset": next_disjoint_offset,
                "source_stage_name": previous_stage_name,
            }
        )
        if disjoint_max_samples is not None:
            next_disjoint_offset += int(disjoint_max_samples)
        previous_stage_name = stage_name
    return stages


def _run_training(spec: RunSpec, config: Dict, run_dir: Path) -> Dict:
    hparams = _resolved_hparams(config, spec.model)
    if run_dir.exists():
        return run_continual_experiment(
            classes=config["classes"],
            epochs=int(hparams["epochs"]),
            batch_size=int(hparams["batch_size"]),
            lr=float(hparams["lr"]),
            weight_decay=float(hparams["weight_decay"]),
            lora_rank=int(hparams["lora_rank"]),
            lora_alpha=int(hparams["lora_alpha"]),
            model_variant=spec.model,
            data_dir=config["data_dir"],
            resume_dir=str(run_dir),
            synthetic=bool(config.get("synthetic", False)),
            max_samples=config.get("max_samples"),
            seed=spec.seed,
            metrics_eval_every=int(hparams["metrics_eval_every"]),
            enable_shared_quality_calibrator=bool(
                config.get("enable_shared_quality_calibrator", True)
            ),
            use_adapter_arbitration=bool(config.get("use_adapter_arbitration", False)),
            use_hard_negatives=bool(config.get("use_hard_negatives", True)),
            symmetric_hard_negatives=bool(config.get("symmetric_hard_negatives", False)),
            lora_target_preset=config.get("lora_target_preset", "default"),
            use_dora=bool(config.get("use_dora", False)),
            use_shared_adapter=bool(config.get("use_shared_adapter", False)),
            shared_drift_weight=float(config.get("shared_drift_weight", 1.0)),
            preset_name=config.get("preset"),
        )

    return run_continual_experiment(
        classes=config["classes"],
        epochs=int(hparams["epochs"]),
        batch_size=int(hparams["batch_size"]),
        lr=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),
        lora_rank=int(hparams["lora_rank"]),
        lora_alpha=int(hparams["lora_alpha"]),
        model_variant=spec.model,
        data_dir=config["data_dir"],
        save_dir=str(run_dir.parent),
        synthetic=bool(config.get("synthetic", False)),
        max_samples=config.get("max_samples"),
        seed=spec.seed,
        experiment_name=spec.method,
        metrics_eval_every=int(hparams["metrics_eval_every"]),
        enable_shared_quality_calibrator=bool(config.get("enable_shared_quality_calibrator", True)),
        use_adapter_arbitration=bool(config.get("use_adapter_arbitration", False)),
        use_hard_negatives=bool(config.get("use_hard_negatives", True)),
        symmetric_hard_negatives=bool(config.get("symmetric_hard_negatives", False)),
        lora_target_preset=config.get("lora_target_preset", "default"),
        use_dora=bool(config.get("use_dora", False)),
        use_shared_adapter=bool(config.get("use_shared_adapter", False)),
        shared_drift_weight=float(config.get("shared_drift_weight", 1.0)),
        preset_name=config.get("preset"),
    )


def _run_baseline(spec: RunSpec, config: Dict, run_dir: Path) -> Dict:
    hparams = _resolved_hparams(config, spec.model)
    common_kwargs = {
        "classes": config["classes"],
        "epochs": int(hparams["epochs"]),
        "batch_size": int(hparams["batch_size"]),
        "data_dir": config["data_dir"],
        "save_dir": str(run_dir.parent),
        "synthetic": bool(config.get("synthetic", False)),
        "resume_dir": str(run_dir),
        "seed": spec.seed,
        "metrics_eval_every": int(hparams["metrics_eval_every"]),
    }
    if spec.method == "finetuning":
        runner = FineTuningBaseline(
            variant=spec.model,
            lr=float(hparams["lr"]),
            weight_decay=float(hparams["weight_decay"]),
        )
    elif spec.method == "joint_finetuning":
        runner = JointFineTuningBaseline(
            variant=spec.model,
            lr=float(hparams["lr"]),
            weight_decay=float(hparams["weight_decay"]),
        )
        return runner.run_experiment(**common_kwargs)
    elif spec.method == "ewc":
        runner = EWCBaseline(
            variant=spec.model,
            lr=float(hparams["lr"]),
            weight_decay=float(hparams["weight_decay"]),
        )
    elif spec.method == "replay":
        runner = ReplayBaseline(
            variant=spec.model,
            lr=float(hparams["lr"]),
            weight_decay=float(hparams["weight_decay"]),
        )
    elif spec.method == "cl_detr":
        runner = CLDETRBaseline(
            variant=spec.model,
            lr=float(hparams["lr"]),
            weight_decay=float(hparams["weight_decay"]),
            calibration_epochs=int(config.get("cl_detr_calibration_epochs", 5)),
            dkd_top_k=int(config.get("cl_detr_top_k", 10)),
            dkd_iou_lambda=float(config.get("cl_detr_iou_lambda", 0.7)),
        )
    else:
        raise ValueError(f"Unsupported baseline method '{spec.method}'")
    return runner.run_experiment(**common_kwargs)


def _run_extension_training(
    spec: ExtensionSpec,
    config: Dict,
    run_dir: Path,
    source_run_dir: Path,
) -> Dict:
    extension_config = config.get("extension", {})
    extension_data_dir = extension_config.get("data_dir", config["data_dir"])
    extension_test_data_dir = extension_config.get("test_data_dir", config["data_dir"])
    extension_arbitration_data_dir = extension_config.get(
        "arbitration_data_dir",
        extension_test_data_dir,
    )
    hparams = _resolved_extension_hparams(config, spec.model)
    symmetric_hn = bool(
        extension_config.get(
            "symmetric_hard_negatives", config.get("symmetric_hard_negatives", False)
        )
    )
    extension_hard_negative_classes = (
        [c for c in config["classes"] if c != spec.target_class] if symmetric_hn else None
    )
    result = train_adapter(
        class_name=spec.target_class,
        epochs=int(spec.epochs),
        batch_size=int(hparams["batch_size"]),
        lr=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),
        lora_rank=int(hparams["lora_rank"]),
        lora_alpha=int(hparams["lora_alpha"]),
        model_variant=spec.model,
        data_dir=extension_data_dir,
        test_data_dir=extension_test_data_dir,
        arbitration_data_dir=extension_arbitration_data_dir,
        save_dir=str(run_dir.parent),
        experiment_name=run_dir.name,
        extend=True,
        synthetic=bool(config.get("synthetic", False)),
        max_samples=spec.max_samples,
        sample_offset=spec.sample_offset,
        load_dir=str(source_run_dir / "final"),
        seed=spec.seed + int(extension_config.get("seed_offset", 1000)),
        extend_strategy=spec.extend_strategy,
        version_selection_strategy=spec.version_selection_strategy,
        metrics_eval_every=int(hparams["metrics_eval_every"]),
        stability_loss_weight=float(extension_config.get("stability_loss_weight", 1e-5)),
        teacher_anchor_weight=float(extension_config.get("teacher_anchor_weight", 0.05)),
        use_hard_negatives=bool(extension_config.get("use_hard_negatives", True)),
        hard_negative_classes=extension_hard_negative_classes,
        use_adapter_arbitration=bool(extension_config.get("use_adapter_arbitration", True)),
        preset_name=config.get("preset"),
    )
    result["stage_name"] = spec.stage_name
    result["stage_index"] = spec.stage_index
    result["stage_epochs"] = spec.epochs
    result["stage_max_samples"] = spec.max_samples
    result["stage_sample_offset"] = spec.sample_offset
    result["source_stage_name"] = spec.source_stage_name
    result["extend_strategy"] = spec.extend_strategy
    result["version_selection_strategy"] = spec.version_selection_strategy
    result["source_run_dir"] = str(source_run_dir)
    result["source_checkpoint_dir"] = str(source_run_dir / "final")
    return result


def _run_extension_baseline(
    spec: ExtensionSpec,
    config: Dict,
    run_dir: Path,
    source_run_dir: Path,
) -> Dict:
    extension_config = config.get("extension", {})
    extension_data_dir = extension_config.get("data_dir", config["data_dir"])
    extension_test_data_dir = extension_config.get("test_data_dir", config["data_dir"])
    hparams = _resolved_extension_hparams(config, spec.model)
    source_checkpoint_task = (
        spec.target_class if spec.source_stage_name is not None else config["classes"][-1]
    )
    common_kwargs = {
        "class_name": spec.target_class,
        "seen_classes": config["classes"],
        "load_dir": str(source_run_dir),
        "load_task_name": source_checkpoint_task,
        "epochs": int(spec.epochs),
        "batch_size": int(hparams["batch_size"]),
        "data_dir": extension_data_dir,
        "test_data_dir": extension_test_data_dir,
        "save_dir": str(run_dir.parent),
        "synthetic": bool(config.get("synthetic", False)),
        "seed": spec.seed,
        "max_samples": spec.max_samples,
        "sample_offset": spec.sample_offset,
        "extension_seed_offset": int(extension_config.get("seed_offset", 1000)),
        "metrics_eval_every": int(hparams["metrics_eval_every"]),
        "experiment_name": run_dir.name,
    }
    if spec.method == "finetuning":
        runner = FineTuningBaseline(
            variant=spec.model,
            lr=float(hparams["lr"]),
            weight_decay=float(hparams["weight_decay"]),
        )
    elif spec.method == "ewc":
        runner = EWCBaseline(
            variant=spec.model,
            lr=float(hparams["lr"]),
            weight_decay=float(hparams["weight_decay"]),
        )
    elif spec.method == "replay":
        runner = ReplayBaseline(
            variant=spec.model,
            lr=float(hparams["lr"]),
            weight_decay=float(hparams["weight_decay"]),
        )
    elif spec.method == "cl_detr":
        runner = CLDETRBaseline(
            variant=spec.model,
            lr=float(hparams["lr"]),
            weight_decay=float(hparams["weight_decay"]),
            calibration_epochs=int(config.get("cl_detr_calibration_epochs", 5)),
            dkd_top_k=int(config.get("cl_detr_top_k", 10)),
            dkd_iou_lambda=float(config.get("cl_detr_iou_lambda", 0.7)),
        )
    else:
        raise ValueError(f"Unsupported baseline method '{spec.method}'")
    result = runner.extend_experiment(**common_kwargs)
    result["stage_name"] = spec.stage_name
    result["stage_index"] = spec.stage_index
    result["stage_epochs"] = spec.epochs
    result["stage_max_samples"] = spec.max_samples
    result["stage_sample_offset"] = spec.sample_offset
    result["source_stage_name"] = spec.source_stage_name
    result["source_run_dir"] = str(source_run_dir)
    result["source_checkpoint_task"] = source_checkpoint_task
    return result


def _run_with_log(log_path: Path, fn, *args, **kwargs):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        tee = _Tee(sys.stdout, handle)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            return fn(*args, **kwargs)


def _group_name(spec: RunSpec) -> str:
    return f"{spec.model}:{spec.method}"


def _extension_suite_dir(suite_dir: Path) -> Path:
    return suite_dir / "extend"


def _build_extension_run_specs(config: Dict) -> List[ExtensionSpec]:
    extension_config = config.get("extension", {})
    if not extension_config.get("enabled", False):
        return []

    target_classes = extension_config.get("classes") or []
    if not target_classes and config.get("classes"):
        target_classes = [config["classes"][0]]

    specs: List[ExtensionSpec] = []
    configured_strategies = extension_config.get("det_lora_strategies")
    if configured_strategies:
        det_lora_strategies = [str(strategy) for strategy in configured_strategies]
    else:
        det_lora_strategies = [str(extension_config.get("det_lora_strategy", "warm_start"))]
    version_selection_strategy = str(
        extension_config.get("version_selection_strategy", "anchor_latest")
    )
    for target_class in target_classes:
        for model in config["models"]:
            stage_plan = _build_extension_stage_plan(config, model)
            for stage_index, stage_config in enumerate(stage_plan):
                for seed in config["seeds"]:
                    if config["phase"] in {"training", "all"}:
                        for strategy in det_lora_strategies:
                            method = (
                                "det_lora" if not configured_strategies else f"det_lora_{strategy}"
                            )
                            max_samples = stage_config["max_samples"]
                            sample_offset = int(stage_config["sample_offset"])
                            if strategy == "grow_freeze" and stage_index > 0:
                                max_samples = stage_config["disjoint_max_samples"]
                                sample_offset = int(stage_config["disjoint_sample_offset"])
                            specs.append(
                                ExtensionSpec(
                                    kind="extension",
                                    model=model,
                                    seed=seed,
                                    method=method,
                                    target_class=target_class,
                                    stage_name=stage_config["name"],
                                    stage_index=stage_index,
                                    epochs=int(stage_config["epochs"]),
                                    max_samples=max_samples,
                                    sample_offset=sample_offset,
                                    source_stage_name=stage_config["source_stage_name"],
                                    extend_strategy=strategy,
                                    version_selection_strategy=version_selection_strategy,
                                )
                            )
                    if config["phase"] in {"baselines", "all"}:
                        for method in config["methods"]:
                            specs.append(
                                ExtensionSpec(
                                    kind="extension",
                                    model=model,
                                    seed=seed,
                                    method=method,
                                    target_class=target_class,
                                    stage_name=stage_config["name"],
                                    stage_index=stage_index,
                                    epochs=int(stage_config["epochs"]),
                                    max_samples=stage_config["max_samples"],
                                    sample_offset=int(stage_config["sample_offset"]),
                                    source_stage_name=stage_config["source_stage_name"],
                                )
                            )
    return specs


def _extension_run_dir(extension_dir: Path, spec: ExtensionSpec) -> Path:
    return (
        extension_dir
        / f"model_{spec.model}"
        / f"seed_{spec.seed}"
        / spec.method
        / spec.target_class
        / spec.stage_name
    )


def _extension_group_name(spec: ExtensionSpec) -> str:
    return f"{spec.model}:{spec.method}:{spec.target_class}:{spec.stage_name}"


def _extension_source_run_dir(suite_dir: Path, extension_dir: Path, spec: ExtensionSpec) -> Path:
    source_method = "det_lora" if spec.method.startswith("det_lora") else spec.method
    if spec.source_stage_name is None:
        return suite_dir / f"model_{spec.model}" / f"seed_{spec.seed}" / source_method
    return (
        extension_dir
        / f"model_{spec.model}"
        / f"seed_{spec.seed}"
        / spec.method
        / spec.target_class
        / spec.source_stage_name
    )


def _extension_results_path(run_dir: Path) -> Path:
    return run_dir / "results.json"


def _extension_run_complete(run_dir: Path) -> bool:
    results_path = _extension_results_path(run_dir)
    if not results_path.exists():
        return False
    results = _load_json(results_path, default={})
    final_checkpoint_dir = results.get("final_checkpoint_dir")
    if final_checkpoint_dir is not None:
        return Path(final_checkpoint_dir).exists()
    final_checkpoint_task = results.get("final_checkpoint_task")
    if final_checkpoint_task is not None:
        return (run_dir / "checkpoints" / final_checkpoint_task).exists()
    if (run_dir / "final").exists():
        return True
    return bool(
        results.get("mixed_final_evaluation")
        or results.get("test_mixed_metrics")
        or results.get("test_metrics")
        or results.get("final_evaluation")
    )


def _collect_extension_seed_run(result: Dict, spec: ExtensionSpec) -> Dict:
    mixed_metrics = (
        result.get("mixed_final_evaluation")
        or result.get("test_mixed_metrics")
        or result.get("test_metrics")
        or {}
    )
    target_metrics = (
        result.get("matched_final_evaluation") or result.get("test_target_metrics") or {}
    )
    pre_mixed_metrics = (
        result.get("pre_extend_mixed_metrics") or result.get("pre_extend_metrics") or {}
    )
    pre_target_metrics = result.get("pre_extend_target_metrics") or {}

    return {
        "seed": spec.seed,
        "target_class": spec.target_class,
        "stage_name": spec.stage_name,
        "stage_index": spec.stage_index,
        "stage_epochs": spec.epochs,
        "stage_max_samples": spec.max_samples,
        "stage_sample_offset": spec.sample_offset,
        "extend_strategy": spec.extend_strategy,
        "version_selection_strategy": spec.version_selection_strategy,
        "source_stage_name": spec.source_stage_name,
        "output_dir": result.get("output_dir"),
        "final_checkpoint_dir": result.get("final_checkpoint_dir"),
        "source_experiment_dir": result.get("source_experiment_dir"),
        "source_run_dir": result.get("source_run_dir"),
        "mixed_metrics": {
            key: float(mixed_metrics[key]) for key in CORE_METRICS if key in mixed_metrics
        },
        "matched_metrics": {
            key: float(target_metrics[key]) for key in CORE_METRICS if key in target_metrics
        },
        "pre_mixed_metrics": {
            key: float(pre_mixed_metrics[key]) for key in CORE_METRICS if key in pre_mixed_metrics
        },
        "pre_matched_metrics": {
            key: float(pre_target_metrics[key]) for key in CORE_METRICS if key in pre_target_metrics
        },
        "mixed_extension_delta": result.get("mixed_extension_delta") or {},
        "target_extension_delta": result.get("target_extension_delta") or {},
    }


def _build_extension_summary(extension_dir: Path, config: Dict, suite_results: Dict) -> Dict:
    summary = {
        "suite_dir": str(extension_dir),
        "source_suite_dir": str(_suite_dir(config)),
        "generated_at": _now(),
        "groups": {},
    }
    for group_name, seed_runs in suite_results.get("groups", {}).items():
        mixed_metrics: Dict[str, List[float]] = {}
        target_metrics: Dict[str, List[float]] = {}
        pre_mixed_metrics: Dict[str, List[float]] = {}
        pre_target_metrics: Dict[str, List[float]] = {}
        mixed_delta: Dict[str, List[float]] = {}
        target_delta: Dict[str, List[float]] = {}

        for seed_run in seed_runs:
            for name, value in seed_run.get("mixed_metrics", {}).items():
                mixed_metrics.setdefault(name, []).append(float(value))
            for name, value in seed_run.get("matched_metrics", {}).items():
                target_metrics.setdefault(name, []).append(float(value))
            for name, value in seed_run.get("pre_mixed_metrics", {}).items():
                pre_mixed_metrics.setdefault(name, []).append(float(value))
            for name, value in seed_run.get("pre_matched_metrics", {}).items():
                pre_target_metrics.setdefault(name, []).append(float(value))
            for name, value in seed_run.get("mixed_extension_delta", {}).items():
                if isinstance(value, (int, float)):
                    mixed_delta.setdefault(name, []).append(float(value))
            for name, value in seed_run.get("target_extension_delta", {}).items():
                if isinstance(value, (int, float)):
                    target_delta.setdefault(name, []).append(float(value))

        summary["groups"][group_name] = {
            "num_seeds": len(seed_runs),
            "seeds": [run["seed"] for run in seed_runs],
            "mixed_metrics": {
                name: _aggregate_scalars(values) for name, values in mixed_metrics.items() if values
            },
            "target_metrics": {
                name: _aggregate_scalars(values)
                for name, values in target_metrics.items()
                if values
            },
            "pre_mixed_metrics": {
                name: _aggregate_scalars(values)
                for name, values in pre_mixed_metrics.items()
                if values
            },
            "pre_target_metrics": {
                name: _aggregate_scalars(values)
                for name, values in pre_target_metrics.items()
                if values
            },
            "mixed_extension_delta": {
                name: _aggregate_scalars(values) for name, values in mixed_delta.items() if values
            },
            "target_extension_delta": {
                name: _aggregate_scalars(values) for name, values in target_delta.items() if values
            },
            "runs": seed_runs,
        }
    return summary


def _write_extension_summary_markdown(path: Path, summary: Dict) -> None:
    lines = [
        "# Extension Suite Summary",
        "",
        "| Group | Seeds | Target mAP@0.5 | Target mAP@0.95 | Target mAP@0.5:0.95 | Mixed mAP@0.5 | Mixed mAP@0.95 | Mixed mAP@0.5:0.95 | Target Delta mAP@0.5 | Target Delta mAP@0.95 | Mixed Delta mAP@0.5 | Mixed Delta mAP@0.95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for group_name, group in sorted(summary.get("groups", {}).items()):

        def fmt(metric: Dict) -> str:
            if not metric:
                return "-"
            return f"{metric['mean']:.4f} +- {metric['std']:.4f}"

        target_metrics = group.get("target_metrics", {})
        mixed_metrics = group.get("mixed_metrics", {})
        target_delta = group.get("target_extension_delta", {})
        mixed_delta = group.get("mixed_extension_delta", {})

        lines.append(
            "| "
            + " | ".join(
                [
                    group_name,
                    str(group.get("num_seeds", 0)),
                    fmt(target_metrics.get("mAP@0.5")),
                    fmt(target_metrics.get("mAP@0.95")),
                    fmt(target_metrics.get("mAP@0.5:0.95")),
                    fmt(mixed_metrics.get("mAP@0.5")),
                    fmt(mixed_metrics.get("mAP@0.95")),
                    fmt(mixed_metrics.get("mAP@0.5:0.95")),
                    fmt(target_delta.get("mAP@0.5")),
                    fmt(target_delta.get("mAP@0.95")),
                    fmt(mixed_delta.get("mAP@0.5")),
                    fmt(mixed_delta.get("mAP@0.95")),
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n")


def run_final_suite(config: Dict) -> Dict:
    suite_dir = _suite_dir(config)
    suite_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = suite_dir / "suite_manifest.json"
    state_path = suite_dir / "suite_state.json"
    results_path = suite_dir / "suite_results.json"
    summary_path = suite_dir / "suite_summary.json"
    events_path = suite_dir / "suite_events.jsonl"

    specs = _build_run_specs(config)
    run_plan = {
        spec.key: {
            "kind": spec.kind,
            "model": spec.model,
            "seed": spec.seed,
            "method": spec.method,
            "status": "pending",
            "run_dir": str(_run_dir(suite_dir, spec)),
            "results_path": str(_results_path(spec, _run_dir(suite_dir, spec))),
        }
        for spec in specs
    }

    state = _load_json(state_path, default={})
    state.setdefault("suite_name", config["suite_name"])
    state.setdefault("status", "running")
    state.setdefault("started_at", _now())
    state.setdefault("finished_at", None)
    state.setdefault("runs", {})
    for key, payload in run_plan.items():
        state["runs"].setdefault(key, payload)

    manifest = {
        **config,
        "suite_dir": str(suite_dir),
        "started_at": state["started_at"],
    }
    _write_json(manifest_path, manifest)
    _write_json(suite_dir / "resolved_config.json", config)
    _write_json(suite_dir / "hardware.json", capture_hardware_metadata())
    _write_json(state_path, state)

    suite_results = _load_json(
        results_path,
        default={"suite_name": config["suite_name"], "suite_dir": str(suite_dir), "groups": {}},
    )

    log(f"Final-Suite startet: {config['suite_name']} | runs={len(specs)} | dir={suite_dir}")
    for run_index, spec in enumerate(specs, start=1):
        run_dir = _run_dir(suite_dir, spec)
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        state["runs"][spec.key]["run_dir"] = str(run_dir)
        state["runs"][spec.key]["results_path"] = str(_results_path(spec, run_dir))

        if _run_complete(spec, run_dir, config):
            log(f"[{run_index}/{len(specs)}] Skip completed {spec.key}")
            state["runs"][spec.key]["status"] = "completed"
            _append_event(events_path, {"event": "skip_completed", "run": spec.key})
            result = _load_json(_results_path(spec, run_dir))
        else:
            log(f"[{run_index}/{len(specs)}] Starting {spec.key} -> {run_dir}")
            state["runs"][spec.key]["status"] = "running"
            state["runs"][spec.key]["last_started_at"] = _now()
            _write_json(state_path, state)
            _append_event(events_path, {"event": "run_started", "run": spec.key})
            log_path = run_dir / "run.log"
            if spec.kind == "training":
                result = _run_with_log(log_path, _run_training, spec, config, run_dir)
            else:
                result = _run_with_log(log_path, _run_baseline, spec, config, run_dir)
            state["runs"][spec.key]["status"] = "completed"
            state["runs"][spec.key]["last_finished_at"] = _now()
            _append_event(events_path, {"event": "run_completed", "run": spec.key})

        group_name = _group_name(spec)
        runs = suite_results["groups"].setdefault(group_name, [])
        current_seed_run = _collect_seed_run(result, spec.seed, default_output_dir=str(run_dir))
        _log_run_result(spec, current_seed_run)
        runs = [run for run in runs if run.get("seed") != spec.seed]
        runs.append(current_seed_run)
        runs.sort(key=lambda entry: entry["seed"])
        suite_results["groups"][group_name] = runs

        _write_json(state_path, state)
        _write_json(results_path, suite_results)

    state["status"] = "completed"
    state["finished_at"] = _now()
    _write_json(state_path, state)

    summary = _build_summary(suite_dir, config, suite_results)
    _write_json(summary_path, summary)
    _write_summary_markdown(suite_dir / "suite_summary.md", summary)
    _append_event(events_path, {"event": "suite_completed", "suite_name": config["suite_name"]})
    manifest["finished_at"] = _now()
    _write_json(manifest_path, manifest)

    extension_specs = _build_extension_run_specs(config)
    if extension_specs:
        log(f"Starting extension suite for {config['suite_name']}")
        extension_dir = _extension_suite_dir(suite_dir)
        extension_dir.mkdir(parents=True, exist_ok=True)
        extension_state_path = extension_dir / "suite_state.json"
        extension_results_path = extension_dir / "suite_results.json"
        extension_summary_path = extension_dir / "suite_summary.json"
        extension_manifest_path = extension_dir / "suite_manifest.json"
        extension_events_path = extension_dir / "suite_events.jsonl"

        extension_manifest = {
            **config,
            "suite_name": f"{config['suite_name']}_extend",
            "suite_dir": str(extension_dir),
            "source_suite_dir": str(suite_dir),
            "source_suite_name": config["suite_name"],
            "stages": config.get("extension", {}).get("stages", []),
            "started_at": _now(),
            "mode": "extension",
        }
        extension_suite_results = _load_json(
            extension_results_path,
            default={
                "suite_name": extension_manifest["suite_name"],
                "suite_dir": str(extension_dir),
                "source_suite_dir": str(suite_dir),
                "groups": {},
            },
        )
        extension_state = _load_json(extension_state_path, default={})
        extension_state.setdefault("suite_name", extension_manifest["suite_name"])
        extension_state.setdefault("status", "running")
        extension_state.setdefault("started_at", extension_manifest["started_at"])
        extension_state.setdefault("finished_at", None)
        extension_state.setdefault("runs", {})
        for spec in extension_specs:
            run_dir = _extension_run_dir(extension_dir, spec)
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            extension_state["runs"].setdefault(
                spec.key,
                {
                    "kind": spec.kind,
                    "model": spec.model,
                    "seed": spec.seed,
                    "method": spec.method,
                    "target_class": spec.target_class,
                    "extend_strategy": spec.extend_strategy,
                    "run_dir": str(run_dir),
                    "results_path": str(_extension_results_path(run_dir)),
                    "status": "pending",
                },
            )

        _write_json(extension_manifest_path, extension_manifest)
        _write_json(extension_state_path, extension_state)
        _write_json(extension_dir / "resolved_config.json", config)
        _write_json(extension_dir / "hardware.json", capture_hardware_metadata())
        _append_event(
            extension_events_path,
            {"event": "extension_suite_started", "suite_name": extension_manifest["suite_name"]},
        )

        log(f"Extension-Suite startet: runs={len(extension_specs)} | dir={extension_dir}")
        for run_index, spec in enumerate(extension_specs, start=1):
            run_dir = _extension_run_dir(extension_dir, spec)
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            source_run_dir = _extension_source_run_dir(suite_dir, extension_dir, spec)
            extension_state["runs"][spec.key]["run_dir"] = str(run_dir)
            extension_state["runs"][spec.key]["results_path"] = str(
                _extension_results_path(run_dir)
            )
            extension_state["runs"][spec.key]["stage_name"] = spec.stage_name
            extension_state["runs"][spec.key]["stage_index"] = spec.stage_index
            extension_state["runs"][spec.key]["epochs"] = spec.epochs
            extension_state["runs"][spec.key]["max_samples"] = spec.max_samples
            extension_state["runs"][spec.key]["sample_offset"] = spec.sample_offset
            extension_state["runs"][spec.key]["extend_strategy"] = spec.extend_strategy
            extension_state["runs"][spec.key]["source_stage_name"] = spec.source_stage_name

            if _extension_run_complete(run_dir):
                log(f"[extend {run_index}/{len(extension_specs)}] Skip completed {spec.key}")
                extension_state["runs"][spec.key]["status"] = "completed"
                _append_event(extension_events_path, {"event": "skip_completed", "run": spec.key})
                result = _load_json(_extension_results_path(run_dir))
            else:
                log(
                    f"[extend {run_index}/{len(extension_specs)}] Starting {spec.key} "
                    f"source={source_run_dir} -> {run_dir}"
                )
                extension_state["runs"][spec.key]["status"] = "running"
                extension_state["runs"][spec.key]["last_started_at"] = _now()
                _write_json(extension_state_path, extension_state)
                _append_event(extension_events_path, {"event": "run_started", "run": spec.key})
                log_path = run_dir / "run.log"
                if spec.kind == "extension" and spec.method.startswith("det_lora"):
                    result = _run_with_log(
                        log_path,
                        _run_extension_training,
                        spec,
                        config,
                        run_dir,
                        source_run_dir,
                    )
                else:
                    result = _run_with_log(
                        log_path,
                        _run_extension_baseline,
                        spec,
                        config,
                        run_dir,
                        source_run_dir,
                    )
                extension_state["runs"][spec.key]["status"] = "completed"
                extension_state["runs"][spec.key]["last_finished_at"] = _now()
                _append_event(extension_events_path, {"event": "run_completed", "run": spec.key})

            group_name = _extension_group_name(spec)
            runs = extension_suite_results["groups"].setdefault(group_name, [])
            current_seed_run = _collect_extension_seed_run(result, spec)
            _log_extension_result(spec, current_seed_run)
            runs = [run for run in runs if run.get("seed") != spec.seed]
            runs.append(current_seed_run)
            runs.sort(key=lambda entry: entry["seed"])
            extension_suite_results["groups"][group_name] = runs

            _write_json(extension_state_path, extension_state)
            _write_json(extension_results_path, extension_suite_results)

        extension_state["status"] = "completed"
        extension_state["finished_at"] = _now()
        _write_json(extension_state_path, extension_state)
        extension_manifest["finished_at"] = extension_state["finished_at"]
        _write_json(extension_manifest_path, extension_manifest)

        extension_summary = _build_extension_summary(extension_dir, config, extension_suite_results)
        _write_json(extension_summary_path, extension_summary)
        _write_extension_summary_markdown(extension_dir / "suite_summary.md", extension_summary)
        _append_event(
            extension_events_path,
            {"event": "suite_completed", "suite_name": extension_manifest["suite_name"]},
        )
        log(f"Extension-Suite abgeschlossen: {extension_dir}")

    return summary


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run final thesis suites with auto-resume")
    parser.add_argument("--config", required=True, help="JSON config for the final suite")
    parser.add_argument("--suite_name", default=None)
    parser.add_argument("--phase", choices=["training", "baselines", "all"], default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--methods", default=None, help="Comma-separated baseline methods")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--disable_shared_quality_calibrator",
        action="store_true",
        help="Disable the shared quality/objectness calibrator",
    )
    args = parser.parse_args(argv)

    config = _resolve_config(Path(args.config), args)
    summary = run_final_suite(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
