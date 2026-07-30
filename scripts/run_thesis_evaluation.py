#!/usr/bin/env python
"""Generate thesis-ready evaluation artifacts from completed cluster suites."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JOINT_METHOD_ORDER = ["det_lora", "joint_finetuning"]
MAIN_METHOD_ORDER = ["det_lora", "finetuning", "ewc", "replay"]
EXTENSION_METHOD_ORDER = [
    "det_lora_warm_start",
    "det_lora_grow_freeze",
    "finetuning",
    "ewc",
    "replay",
]
METHOD_LABELS = {
    "det_lora": "Det-LoRA",
    "joint_finetuning": "Joint FT",
    "finetuning": "Fine-Tuning",
    "ewc": "EWC",
    "replay": "Replay",
    "det_lora_warm_start": "Det-LoRA Warm",
    "det_lora_grow_freeze": "Det-LoRA Grow",
}
METHOD_COLORS = {
    "det_lora": "#0072B2",
    "joint_finetuning": "#D55E00",
    "finetuning": "#CC79A7",
    "ewc": "#E69F00",
    "replay": "#009E73",
    "det_lora_warm_start": "#56B4E9",
    "det_lora_grow_freeze": "#0072B2",
}
INFERENCE_POSTPROCESS = {
    "score_threshold": 0.5,
    "relative_score_margin": 0.15,
    "nms_iou": 0.55,
    "max_detections": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis evaluation artifacts")
    parser.add_argument(
        "--joint_suite_dir",
        type=Path,
        default=Path("experiments/suites/thesis_l40_joint_baseline"),
    )
    parser.add_argument(
        "--main_suite_dir",
        type=Path,
        default=Path("experiments/suites/thesis_l40_main"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("experiments/analysis/thesis_l40_evaluation"),
    )
    parser.add_argument(
        "--main_config", type=Path, default=Path("configs/iterations/iteration1_base.json")
    )
    parser.add_argument("--joint_config", type=Path, default=Path("configs/baselines/joint.json"))
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--raw_split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--raw_max_samples", type=int, default=260)
    parser.add_argument("--skip_inference", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_key(prefix: str, metric_name: str, suffix: str) -> str:
    return f"{prefix}_{metric_name}_{suffix}"


def flatten_metric_block(prefix: str, block: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    if isinstance(block, (int, float)):
        row[f"{prefix}_mean"] = float(block)
        return row
    if not isinstance(block, dict):
        return row
    for metric_name, stats in sorted(block.items()):
        if isinstance(stats, (int, float)):
            row[f"{prefix}_{metric_name}"] = float(stats)
            continue
        for suffix in ("mean", "std", "min", "max"):
            if suffix in stats:
                row[metric_key(prefix, metric_name, suffix)] = stats[suffix]
    return row


def parse_main_group_name(group_name: str) -> Dict[str, str]:
    model, method = group_name.split(":")
    return {"model": model, "method": method}


def parse_extension_group_name(group_name: str) -> Dict[str, str]:
    model, method, target_class, stage_name = group_name.split(":")
    return {
        "model": model,
        "method": method,
        "target_class": target_class,
        "stage_name": stage_name,
    }


def build_summary_rows(summary: Dict[str, Any], summary_type: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group_name, group in sorted(summary.get("groups", {}).items()):
        if summary_type == "extension":
            row = parse_extension_group_name(group_name)
            row.update(flatten_metric_block("target", group.get("target_metrics", {})))
            row.update(flatten_metric_block("pre_target", group.get("pre_target_metrics", {})))
            row.update(
                flatten_metric_block("target_delta", group.get("target_extension_delta", {}))
            )
            row.update(flatten_metric_block("mixed_delta", group.get("mixed_extension_delta", {})))
        else:
            row = parse_main_group_name(group_name)
            row.update(flatten_metric_block("matched", group.get("matched_metrics", {})))
        row["group_name"] = group_name
        row["num_seeds"] = group.get("num_seeds", 0)
        row["seeds"] = ",".join(str(seed) for seed in group.get("seeds", []))
        row.update(flatten_metric_block("mixed", group.get("mixed_metrics", {})))
        row.update(flatten_metric_block("forgetting", group.get("average_forgetting", {})))
        rows.append(row)
    return rows


def build_run_rows(summary: Dict[str, Any], summary_type: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group_name, group in sorted(summary.get("groups", {}).items()):
        if summary_type == "extension":
            base = parse_extension_group_name(group_name)
        else:
            base = parse_main_group_name(group_name)
        for run in group.get("runs", []):
            row = dict(base)
            row["group_name"] = group_name
            row["seed"] = run.get("seed")
            row["output_dir"] = run.get("output_dir")
            for metric_name, value in sorted(run.get("mixed_metrics", {}).items()):
                row[f"mixed_{metric_name}"] = value
            if summary_type == "extension":
                for metric_name, value in sorted(run.get("matched_metrics", {}).items()):
                    row[f"target_{metric_name}"] = value
                for metric_name, value in sorted(run.get("target_extension_delta", {}).items()):
                    row[f"target_delta_{metric_name}"] = value
                for metric_name, value in sorted(run.get("mixed_extension_delta", {}).items()):
                    row[f"mixed_delta_{metric_name}"] = value
            else:
                for metric_name, value in sorted(run.get("matched_metrics", {}).items()):
                    row[f"matched_{metric_name}"] = value
            row["average_forgetting"] = run.get("average_forgetting")
            rows.append(row)
    return rows


def order_for_methods(rows: Sequence[Dict[str, Any]], preferred_order: Sequence[str]) -> List[str]:
    present = {row["method"] for row in rows}
    ordered = [method for method in preferred_order if method in present]
    remainder = sorted(present.difference(ordered))
    return ordered + remainder


def plot_grouped_metric(
    rows: Sequence[Dict[str, Any]],
    *,
    metric_mean_key: str,
    metric_std_key: str,
    model_order: Sequence[str],
    method_order: Sequence[str],
    output_stem: Path,
    title: str,
    ylabel: str,
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    methods = order_for_methods(rows, method_order)
    x = np.arange(len(model_order))
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, method in enumerate(methods):
        method_rows = {row["model"]: row for row in rows if row["method"] == method}
        means = [method_rows.get(model, {}).get(metric_mean_key, np.nan) for model in model_order]
        stds = [method_rows.get(model, {}).get(metric_std_key, 0.0) for model in model_order]
        offset = (idx - (len(methods) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_order)
    ax.set_xlabel("Model variant")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_extension_delta(
    rows: Sequence[Dict[str, Any]],
    *,
    stage_name: str,
    delta_mean_key: str,
    delta_std_key: str,
    target_classes: Sequence[str],
    model_order: Sequence[str],
    method_order: Sequence[str],
    output_stem: Path,
    title: str,
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    methods = order_for_methods(rows, method_order)
    stage_rows = [row for row in rows if row["stage_name"] == stage_name]
    fig, axes = plt.subplots(
        len(target_classes), 1, figsize=(12, 4 * len(target_classes)), sharex=True
    )
    if len(target_classes) == 1:
        axes = [axes]

    x = np.arange(len(model_order))
    width = 0.8 / max(1, len(methods))
    for axis, target_class in zip(axes, target_classes):
        class_rows = [row for row in stage_rows if row["target_class"] == target_class]
        for idx, method in enumerate(methods):
            method_rows = {row["model"]: row for row in class_rows if row["method"] == method}
            means = [
                method_rows.get(model, {}).get(delta_mean_key, np.nan) for model in model_order
            ]
            stds = [method_rows.get(model, {}).get(delta_std_key, 0.0) for model in model_order]
            offset = (idx - (len(methods) - 1) / 2) * width
            axis.bar(
                x + offset,
                means,
                width=width,
                yerr=stds,
                capsize=4,
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method),
            )
        axis.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        axis.set_ylabel(target_class.replace("_", " "))
        axis.grid(axis="y", alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(model_order)
    axes[-1].set_xlabel("Model variant")
    axes[0].legend(ncol=min(3, len(methods)))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def select_best_run(
    rows: Sequence[Dict[str, Any]],
    *,
    method: str,
    metric_key_name: str,
    stage_name: str | None = None,
    target_class: str | None = None,
) -> Dict[str, Any]:
    candidates = [row for row in rows if row["method"] == method]
    if stage_name is not None:
        candidates = [row for row in candidates if row.get("stage_name") == stage_name]
    if target_class is not None:
        candidates = [row for row in candidates if row.get("target_class") == target_class]
    if not candidates:
        raise ValueError(
            f"No candidates found for method={method} stage={stage_name} target={target_class}"
        )
    return max(candidates, key=lambda row: float(row.get(metric_key_name, float("-inf"))))


def run_joint_inference_job(
    *,
    checkpoint_dir: Path,
    model: str,
    classes: Sequence[str],
    output_dir: Path,
    raw_dir: Path,
    raw_split: str,
    raw_max_samples: int,
    version_selection: Sequence[str] | None = None,
) -> Dict[str, Any]:
    command = [
        sys.executable,
        "scripts/run_joint_inference.py",
        "--checkpoint",
        str(checkpoint_dir),
        "--model",
        model,
        "--classes",
        *classes,
        "--raw_dir",
        str(raw_dir),
        "--raw_split",
        raw_split,
        "--raw_one_per_class",
        "--raw_max_samples",
        str(raw_max_samples),
        "--score_threshold",
        str(INFERENCE_POSTPROCESS["score_threshold"]),
        "--relative_score_margin",
        str(INFERENCE_POSTPROCESS["relative_score_margin"]),
        "--nms_iou",
        str(INFERENCE_POSTPROCESS["nms_iou"]),
        "--max_detections",
        str(INFERENCE_POSTPROCESS["max_detections"]),
        "--output_dir",
        str(output_dir),
    ]
    if version_selection:
        command.extend(["--version_selection", *version_selection])
    subprocess.run(command, check=True)
    report_path = output_dir / "report.json"
    report = load_json(report_path)
    report["command"] = command
    report_path.write_text(json.dumps(report, indent=2))
    return report


def write_markdown_report(
    path: Path,
    *,
    joint_best_det_lora: Dict[str, Any],
    joint_best_joint_ft: Dict[str, Any],
    extension_best_jobs: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> None:
    lines = [
        "# Thesis Evaluation Report",
        "",
        f"Generated in `{output_dir}`.",
        "",
        "## Best Joint Comparison Runs",
        "",
        f"- Det-LoRA: model `{joint_best_det_lora['model']}`, seed `{joint_best_det_lora['seed']}`, mixed mAP@0.5:0.95 `{float(joint_best_det_lora['mixed_mAP@0.5:0.95']):.4f}`",
        f"- Joint Fine-Tuning: model `{joint_best_joint_ft['model']}`, seed `{joint_best_joint_ft['seed']}`, mixed mAP@0.5:0.95 `{float(joint_best_joint_ft['mixed_mAP@0.5:0.95']):.4f}`",
        "",
        "## Generated Inference Jobs",
        "",
    ]
    for job in extension_best_jobs:
        lines.append(
            f"- `{job['name']}` -> model `{job['model']}`, checkpoint `{job['checkpoint_dir']}`"
        )
    lines.extend(
        [
            "",
            "## Directories",
            "",
            f"- Tables: `{output_dir / 'tables'}`",
            f"- Plots: `{output_dir / 'plots'}`",
            f"- Inference: `{output_dir / 'inference'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    joint_config = load_json(args.joint_config)
    main_config = load_json(args.main_config)

    joint_summary = load_json(args.joint_suite_dir / "suite_summary.json")
    main_summary = load_json(args.main_suite_dir / "suite_summary.json")
    extension_dir = args.main_suite_dir / "extend"
    extension_summary = load_json(extension_dir / "suite_summary.json")

    joint_summary_rows = build_summary_rows(joint_summary, "main")
    joint_run_rows = build_run_rows(joint_summary, "main")
    main_summary_rows = build_summary_rows(main_summary, "main")
    main_run_rows = build_run_rows(main_summary, "main")
    extension_summary_rows = build_summary_rows(extension_summary, "extension")
    extension_run_rows = build_run_rows(extension_summary, "extension")

    tables_dir = args.output_dir / "tables"
    write_csv(tables_dir / "joint_summary.csv", joint_summary_rows)
    write_csv(tables_dir / "joint_runs.csv", joint_run_rows)
    write_csv(tables_dir / "main_summary.csv", main_summary_rows)
    write_csv(tables_dir / "main_runs.csv", main_run_rows)
    write_csv(tables_dir / "extension_summary.csv", extension_summary_rows)
    write_csv(tables_dir / "extension_runs.csv", extension_run_rows)

    model_order = list(main_config["models"])
    plots_dir = args.output_dir / "plots"
    plot_grouped_metric(
        joint_summary_rows,
        metric_mean_key="mixed_mAP@0.5_mean",
        metric_std_key="mixed_mAP@0.5_std",
        model_order=model_order,
        method_order=JOINT_METHOD_ORDER,
        output_stem=plots_dir / "joint_mAP50_by_model",
        title="Joint Training Comparison: mAP@0.5 by Model",
        ylabel="mAP@0.5",
    )
    plot_grouped_metric(
        joint_summary_rows,
        metric_mean_key="mixed_mAP@0.5:0.95_mean",
        metric_std_key="mixed_mAP@0.5:0.95_std",
        model_order=model_order,
        method_order=JOINT_METHOD_ORDER,
        output_stem=plots_dir / "joint_mAP5095_by_model",
        title="Joint Training Comparison: mAP@0.5:0.95 by Model",
        ylabel="mAP@0.5:0.95",
    )
    plot_grouped_metric(
        main_summary_rows,
        metric_mean_key="mixed_mAP@0.5_mean",
        metric_std_key="mixed_mAP@0.5_std",
        model_order=model_order,
        method_order=MAIN_METHOD_ORDER,
        output_stem=plots_dir / "continual_mAP50_by_model",
        title="Continual Main Suite: mAP@0.5 by Model",
        ylabel="mAP@0.5",
    )
    plot_grouped_metric(
        main_summary_rows,
        metric_mean_key="mixed_mAP@0.5:0.95_mean",
        metric_std_key="mixed_mAP@0.5:0.95_std",
        model_order=model_order,
        method_order=MAIN_METHOD_ORDER,
        output_stem=plots_dir / "continual_mAP5095_by_model",
        title="Continual Main Suite: mAP@0.5:0.95 by Model",
        ylabel="mAP@0.5:0.95",
    )

    target_classes = list(main_config["extension"]["classes"])
    for stage_name in ("stage_1", "stage_2"):
        plot_extension_delta(
            extension_summary_rows,
            stage_name=stage_name,
            delta_mean_key="target_delta_mAP@0.5_mean",
            delta_std_key="target_delta_mAP@0.5_std",
            target_classes=target_classes,
            model_order=model_order,
            method_order=EXTENSION_METHOD_ORDER,
            output_stem=plots_dir / f"{stage_name}_target_delta_mAP50",
            title=f"Extension {stage_name}: Target Delta mAP@0.5",
        )
        plot_extension_delta(
            extension_summary_rows,
            stage_name=stage_name,
            delta_mean_key="mixed_delta_mAP@0.5_mean",
            delta_std_key="mixed_delta_mAP@0.5_std",
            target_classes=target_classes,
            model_order=model_order,
            method_order=EXTENSION_METHOD_ORDER,
            output_stem=plots_dir / f"{stage_name}_mixed_delta_mAP50",
            title=f"Extension {stage_name}: Mixed Delta mAP@0.5",
        )

    joint_best_det_lora = select_best_run(
        joint_run_rows, method="det_lora", metric_key_name="mixed_mAP@0.5:0.95"
    )
    joint_best_joint_ft = select_best_run(
        joint_run_rows, method="joint_finetuning", metric_key_name="mixed_mAP@0.5:0.95"
    )

    inference_jobs: List[Dict[str, Any]] = []
    if not args.skip_inference:
        inference_dir = args.output_dir / "inference"
        main_checkpoint_dir = Path(joint_best_det_lora["output_dir"]) / "final"
        main_job_name = "joint_best_det_lora"
        inference_jobs.append(
            {
                "name": main_job_name,
                "model": joint_best_det_lora["model"],
                "checkpoint_dir": str(main_checkpoint_dir),
                "report": run_joint_inference_job(
                    checkpoint_dir=main_checkpoint_dir,
                    model=joint_best_det_lora["model"],
                    classes=main_config["classes"],
                    output_dir=inference_dir / main_job_name,
                    raw_dir=args.raw_dir,
                    raw_split=args.raw_split,
                    raw_max_samples=args.raw_max_samples,
                ),
            }
        )
        for target_class in target_classes:
            best_extension = select_best_run(
                extension_run_rows,
                method="det_lora_grow_freeze",
                metric_key_name="target_mAP@0.5:0.95",
                stage_name="stage_2",
                target_class=target_class,
            )
            checkpoint_dir = Path(best_extension["output_dir"]) / "final"
            job_name = f"extension_{target_class}_stage2_best"
            inference_jobs.append(
                {
                    "name": job_name,
                    "model": best_extension["model"],
                    "checkpoint_dir": str(checkpoint_dir),
                    "report": run_joint_inference_job(
                        checkpoint_dir=checkpoint_dir,
                        model=best_extension["model"],
                        classes=main_config["classes"],
                        output_dir=inference_dir / job_name,
                        raw_dir=args.raw_dir,
                        raw_split=args.raw_split,
                        raw_max_samples=args.raw_max_samples,
                        version_selection=[f"{target_class}=anchor_latest"],
                    ),
                }
            )

    write_markdown_report(
        args.output_dir / "evaluation_report.md",
        joint_best_det_lora=joint_best_det_lora,
        joint_best_joint_ft=joint_best_joint_ft,
        extension_best_jobs=inference_jobs,
        output_dir=args.output_dir,
    )

    manifest = {
        "joint_suite_dir": str(args.joint_suite_dir),
        "main_suite_dir": str(args.main_suite_dir),
        "extension_suite_dir": str(extension_dir),
        "output_dir": str(args.output_dir),
        "generated_tables": sorted(str(path) for path in tables_dir.glob("*.csv")),
        "generated_plots": sorted(str(path) for path in plots_dir.glob("*")),
        "generated_inference": inference_jobs,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
