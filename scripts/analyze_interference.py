#!/usr/bin/env python3
"""
Det-LoRA Interference Analysis
==============================

Diagnoses class interference between a task checkpoint and a later final checkpoint.

Outputs:
- matched/mixed metrics with and without calibration
- encoder proposal top-k overlap statistics for a focus class
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.evaluator import ContinualEvaluator, aggregate_classwise_metrics
from det_lora.model.det_lora import DetLoRA
from det_lora.model.detector import RFDETRDetector
from det_lora.train import collate_fn


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def _make_loader(
    data_dir: str,
    split: str,
    class_id_offset: int,
    resolution: int,
    batch_size: int,
    seed: int,
    class_filter: Optional[str] = None,
) -> DataLoader:
    dataset = load_dataset_from_raw(
        raw_dir=data_dir,
        class_filter=class_filter,
        split=split,
        class_id_offset=class_id_offset,
        img_size=resolution,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )


def _load_det_lora(checkpoint_dir: str, variant: str) -> DetLoRA:
    detector = RFDETRDetector(variant=variant)
    det_lora = DetLoRA(detector=detector)
    det_lora.load_all(checkpoint_dir)
    return det_lora


def _disable_calibration(det_lora: DetLoRA) -> None:
    det_lora._adapter_calibrators = {}
    det_lora._score_banks = {}


def _evaluate_mixed(
    det_lora: DetLoRA,
    dataloader: DataLoader,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    return ContinualEvaluator(det_lora).evaluate_det_lora_joint(
        dataloader=dataloader,
        class_names=class_names,
    )


def _evaluate_matched(
    checkpoint_dir: str,
    variant: str,
    data_dir: str,
    split: str,
    class_names: Sequence[str],
    class_id_offset: int,
    resolution: int,
    batch_size: int,
    seed: int,
    disable_calibration: bool,
) -> Dict[str, Any]:
    det_lora = _load_det_lora(checkpoint_dir, variant)
    if disable_calibration:
        _disable_calibration(det_lora)

    evaluator = ContinualEvaluator(det_lora)
    per_class = {}
    for class_name in class_names:
        loader = _make_loader(
            data_dir=data_dir,
            split=split,
            class_id_offset=class_id_offset,
            resolution=resolution,
            batch_size=batch_size,
            seed=seed,
            class_filter=class_name,
        )
        per_class[class_name] = evaluator.evaluate_det_lora_joint(
            dataloader=loader,
            class_names=[class_name],
        )
    return aggregate_classwise_metrics(per_class)


@torch.no_grad()
def _collect_proposal_trace(
    det_lora: DetLoRA,
    dataloader: DataLoader,
    class_name: str,
    max_batches: Optional[int] = None,
) -> List[Dict[str, Any]]:
    det_lora.set_eval_mode()
    det_lora.load_adapter_for_eval(class_name)
    class_id = det_lora.get_class_id(class_name)
    num_queries = int(det_lora.detector._get_inner_model().num_queries)
    device = det_lora.device

    traces: List[Dict[str, Any]] = []
    image_index = 0
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        outputs = det_lora.forward(pixel_values=batch["pixel_values"].to(device))
        enc_outputs = outputs.get("enc_outputs")
        if enc_outputs is None:
            raise RuntimeError("RF-DETR wrapper does not expose enc_outputs for proposal analysis")

        enc_logits = enc_outputs["pred_logits"]
        topk = min(num_queries, enc_logits.shape[1])
        model_topk = torch.topk(enc_logits.max(-1).values, k=topk, dim=1).indices.cpu()
        focus_topk = torch.topk(enc_logits[..., class_id], k=topk, dim=1).indices.cpu()

        for sample_idx in range(enc_logits.shape[0]):
            traces.append(
                {
                    "image_index": image_index,
                    "model_topk": model_topk[sample_idx].tolist(),
                    "focus_topk": focus_topk[sample_idx].tolist(),
                }
            )
            image_index += 1

    det_lora.unload_adapter()
    return traces


def _summarize_distribution(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _compare_trace_key(
    reference: List[Dict[str, Any]],
    candidate: List[Dict[str, Any]],
    reference_key: str,
    candidate_key: str,
) -> Dict[str, Any]:
    candidate_by_image = {entry["image_index"]: entry[candidate_key] for entry in candidate}
    overlap_scores: List[float] = []
    jaccard_scores: List[float] = []

    for ref_entry in reference:
        ref_topk = ref_entry[reference_key]
        cand_topk = candidate_by_image.get(ref_entry["image_index"])
        if cand_topk is None:
            continue

        ref_set = set(ref_topk)
        cand_set = set(cand_topk)
        if not ref_set:
            continue

        intersection = len(ref_set & cand_set)
        union = len(ref_set | cand_set)
        overlap_scores.append(intersection / len(ref_set))
        jaccard_scores.append(intersection / union if union else 1.0)

    return {
        "overlap_at_k": _summarize_distribution(overlap_scores),
        "jaccard_at_k": _summarize_distribution(jaccard_scores),
        "num_images": len(overlap_scores),
    }


def _build_report(args) -> Dict[str, Any]:
    focus_loader = _make_loader(
        data_dir=args.data_dir,
        split=args.split,
        class_id_offset=args.class_id_offset,
        resolution=args.resolution,
        batch_size=args.batch_size,
        seed=args.seed,
        class_filter=args.focus_class,
    )
    mixed_loader = _make_loader(
        data_dir=args.data_dir,
        split=args.split,
        class_id_offset=args.class_id_offset,
        resolution=args.resolution,
        batch_size=args.batch_size,
        seed=args.seed,
        class_filter=None,
    )

    task_det_lora = _load_det_lora(args.task_checkpoint, args.model)
    final_with_cal = _load_det_lora(args.final_checkpoint, args.model)
    final_without_cal = _load_det_lora(args.final_checkpoint, args.model)
    _disable_calibration(final_without_cal)

    if args.classes:
        class_names = list(args.classes)
    else:
        class_names = list(final_with_cal.trained_classes)

    task_focus_metrics = _evaluate_mixed(task_det_lora, focus_loader, [args.focus_class])
    final_focus_metrics_with_cal = _evaluate_mixed(final_with_cal, focus_loader, [args.focus_class])
    final_focus_metrics_without_cal = _evaluate_mixed(
        final_without_cal, focus_loader, [args.focus_class]
    )

    final_mixed_with_cal = _evaluate_mixed(final_with_cal, mixed_loader, class_names)
    final_mixed_without_cal = _evaluate_mixed(final_without_cal, mixed_loader, class_names)

    final_matched_with_cal = _evaluate_matched(
        checkpoint_dir=args.final_checkpoint,
        variant=args.model,
        data_dir=args.data_dir,
        split=args.split,
        class_names=class_names,
        class_id_offset=args.class_id_offset,
        resolution=args.resolution,
        batch_size=args.batch_size,
        seed=args.seed,
        disable_calibration=False,
    )
    final_matched_without_cal = _evaluate_matched(
        checkpoint_dir=args.final_checkpoint,
        variant=args.model,
        data_dir=args.data_dir,
        split=args.split,
        class_names=class_names,
        class_id_offset=args.class_id_offset,
        resolution=args.resolution,
        batch_size=args.batch_size,
        seed=args.seed,
        disable_calibration=True,
    )

    task_trace = _collect_proposal_trace(
        task_det_lora,
        focus_loader,
        args.focus_class,
        max_batches=args.max_batches,
    )
    final_trace = _collect_proposal_trace(
        final_with_cal,
        focus_loader,
        args.focus_class,
        max_batches=args.max_batches,
    )

    proposal_overlap = {
        "task_model_vs_final_model": _compare_trace_key(
            task_trace,
            final_trace,
            reference_key="model_topk",
            candidate_key="model_topk",
        ),
        "task_model_vs_final_focus": _compare_trace_key(
            task_trace,
            final_trace,
            reference_key="model_topk",
            candidate_key="focus_topk",
        ),
        "final_model_vs_final_focus": _compare_trace_key(
            final_trace,
            final_trace,
            reference_key="model_topk",
            candidate_key="focus_topk",
        ),
    }

    return {
        "config": {
            "task_checkpoint": args.task_checkpoint,
            "final_checkpoint": args.final_checkpoint,
            "focus_class": args.focus_class,
            "classes": class_names,
            "data_dir": args.data_dir,
            "split": args.split,
            "batch_size": args.batch_size,
            "resolution": args.resolution,
            "seed": args.seed,
            "max_batches": args.max_batches,
        },
        "metrics": {
            "task_focus": task_focus_metrics,
            "final_focus_with_calibration": final_focus_metrics_with_cal,
            "final_focus_without_calibration": final_focus_metrics_without_cal,
            "final_mixed_with_calibration": final_mixed_with_cal,
            "final_mixed_without_calibration": final_mixed_without_cal,
            "final_matched_with_calibration": final_matched_with_cal,
            "final_matched_without_calibration": final_matched_without_cal,
        },
        "proposal_overlap": proposal_overlap,
    }


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    metrics = report["metrics"]
    overlap = report["proposal_overlap"]
    lines = [
        "# Interference Analysis",
        "",
        f"Focus class: `{report['config']['focus_class']}`",
        "",
        "## Metrics",
        "",
        f"- Task checkpoint focus mAP@0.5: {metrics['task_focus'].get('mAP@0.5', 0.0):.4f}",
        f"- Final focus mAP@0.5 with calibration: {metrics['final_focus_with_calibration'].get('mAP@0.5', 0.0):.4f}",
        f"- Final focus mAP@0.5 without calibration: {metrics['final_focus_without_calibration'].get('mAP@0.5', 0.0):.4f}",
        f"- Final mixed mAP@0.5 with calibration: {metrics['final_mixed_with_calibration'].get('mAP@0.5', 0.0):.4f}",
        f"- Final mixed mAP@0.5 without calibration: {metrics['final_mixed_without_calibration'].get('mAP@0.5', 0.0):.4f}",
        "",
        "## Proposal Overlap",
        "",
        f"- Task-model vs final-model overlap@k mean: {overlap['task_model_vs_final_model']['overlap_at_k']['mean']:.4f}",
        f"- Task-model vs final-focus overlap@k mean: {overlap['task_model_vs_final_focus']['overlap_at_k']['mean']:.4f}",
        f"- Final-model vs final-focus overlap@k mean: {overlap['final_model_vs_final_focus']['overlap_at_k']['mean']:.4f}",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Det-LoRA proposal interference")
    parser.add_argument("--task-checkpoint", required=True)
    parser.add_argument("--final-checkpoint", required=True)
    parser.add_argument("--focus-class", required=True)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=576)
    parser.add_argument("--class-id-offset", type=int, default=91)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = _build_report(args)
    _write_json(output_dir / "interference_report.json", report)
    _write_markdown(output_dir / "interference_report.md", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
