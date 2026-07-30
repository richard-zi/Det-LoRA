#!/usr/bin/env python
"""Run Det-LoRA joint inference with demo-oriented postprocessing."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, Subset

from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.arbitration import (
    apply_adapter_arbitration,
    simplify_joint_predictions_for_display,
)
from det_lora.evaluation.evaluator import collect_det_lora_joint_predictions
from det_lora.model.det_lora import DetLoRA
from det_lora.model.detector import RFDETRDetector
from det_lora.train import _det_lora_class_id_mapping, collate_fn

COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (255, 178, 29),
    (145, 30, 180),
    (70, 240, 240),
]


class ImageInferenceDataset(Dataset):
    """Small unlabeled image dataset for inference-only runs."""

    def __init__(self, image_paths: Sequence[Path], img_size: int):
        self.image_paths = [Path(path) for path in image_paths]
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        pixel_values = TF.to_tensor(image)
        pixel_values = TF.normalize(
            pixel_values,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return {
            "pixel_values": pixel_values,
            "labels": {
                "labels": torch.empty((0,), dtype=torch.long),
                "boxes": torch.empty((0, 4), dtype=torch.float32),
            },
            "sample_id": idx,
        }


def parse_version_selection(values: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--version_selection entries must use class=strategy")
        class_name, strategy = value.split("=", 1)
        if strategy not in {"all", "anchor_latest", "latest"}:
            raise ValueError("version selection strategy must be all, anchor_latest, or latest")
        result[class_name] = strategy
    return result


def resolve_image_paths(
    args: argparse.Namespace, detector: RFDETRDetector
) -> tuple[List[Path], Any]:
    if args.images:
        image_paths = [Path(path) for path in args.images]
        return image_paths, ImageInferenceDataset(image_paths, detector.resolution)

    if args.image_dir:
        image_dir = Path(args.image_dir)
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        return image_paths, ImageInferenceDataset(image_paths, detector.resolution)

    raise ValueError("Provide --images or --image_dir")


def resolve_sample_dataset(
    args: argparse.Namespace,
    detector: RFDETRDetector,
    det_lora: DetLoRA,
) -> tuple[List[Path], Any]:
    class_mapping = _det_lora_class_id_mapping(det_lora, args.classes)
    dataset = load_dataset_from_raw(
        raw_dir=args.raw_dir,
        class_filter=args.classes,
        split=args.raw_split,
        img_size=detector.resolution,
        seed=args.seed,
        max_samples=args.raw_max_samples,
        class_id_mapping=class_mapping,
    )
    if args.raw_one_per_class:
        selected = []
        seen = set()
        for idx, (_, annots) in enumerate(dataset.samples):
            sample_classes = {annot["class_name"] for annot in annots}
            for class_name in args.classes:
                if class_name in sample_classes and class_name not in seen:
                    selected.append(idx)
                    seen.add(class_name)
                    break
            if seen == set(args.classes):
                break
        dataset = Subset(dataset, selected)
        image_paths = [
            Path(args.raw_dir) / "Images" / dataset.dataset.samples[idx][0] for idx in selected
        ]
        return image_paths, dataset

    image_paths = [Path(args.raw_dir) / "Images" / filename for filename, _ in dataset.samples]
    return image_paths, dataset


def draw_prediction(
    image_path: Path,
    prediction: Dict[str, np.ndarray],
    id_to_name: Dict[int, str],
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except OSError:
        font = ImageFont.load_default()

    order = np.argsort(prediction["scores"])[::-1]
    for rank, pred_idx in enumerate(order):
        class_id = int(prediction["labels"][pred_idx])
        class_name = id_to_name.get(class_id, str(class_id))
        color = COLORS[rank % len(COLORS)]
        x1, y1, x2, y2 = prediction["boxes"][pred_idx]
        box = [x1 * width, y1 * height, x2 * width, y2 * height]
        draw.rectangle(box, outline=color, width=4)
        label = f"{class_name.replace('military_', '')} {float(prediction['scores'][pred_idx]):.2f}"
        text_y = max(0.0, box[1])
        text_box = draw.textbbox((box[0], text_y), label, font=font)
        draw.rectangle(
            [text_box[0] - 2, text_box[1] - 2, text_box[2] + 2, text_box[3] + 2],
            fill=color,
        )
        draw.text((box[0], text_y), label, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def prediction_rows(
    prediction: Dict[str, np.ndarray],
    id_to_name: Dict[int, str],
) -> List[Dict[str, Any]]:
    order = np.argsort(prediction["scores"])[::-1]
    return [
        {
            "class": id_to_name.get(
                int(prediction["labels"][idx]), str(int(prediction["labels"][idx]))
            ),
            "score": float(prediction["scores"][idx]),
            "box": [float(value) for value in prediction["boxes"][idx].tolist()],
        }
        for idx in order
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Det-LoRA joint inference")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--model", default="small", choices=("nano", "small", "base", "medium", "large")
    )
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--images", nargs="*", default=[])
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--raw_dir", default=None)
    parser.add_argument("--raw_split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--raw_max_samples", type=int, default=260)
    parser.add_argument("--raw_one_per_class", action="store_true")
    parser.add_argument("--arbitration_state", type=Path, default=None)
    parser.add_argument("--version_selection", nargs="*", default=[])
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--relative_score_margin", type=float, default=0.25)
    parser.add_argument("--nms_iou", type=float, default=0.55)
    parser.add_argument("--max_detections", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1013)
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/inference"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = RFDETRDetector(variant=args.model)
    det_lora = DetLoRA(detector=detector)
    det_lora.load_all(str(args.checkpoint))

    if args.arbitration_state:
        state = json.loads(args.arbitration_state.read_text())
        if "arbitration_state" in state:
            state = state["arbitration_state"]
        if "state" in state:
            state = state["state"]
        det_lora.set_adapter_arbitration_state(state)

    if args.raw_dir:
        image_paths, dataset = resolve_sample_dataset(args, detector, det_lora)
    else:
        image_paths, dataset = resolve_image_paths(args, detector)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    version_selection = parse_version_selection(args.version_selection)
    predictions, _, _ = collect_det_lora_joint_predictions(
        det_lora,
        dataloader,
        args.classes,
        version_selection_by_class=version_selection or None,
    )
    arbitrated = apply_adapter_arbitration(predictions, det_lora.adapter_arbitration_state)
    display_predictions = simplify_joint_predictions_for_display(
        arbitrated,
        score_threshold=args.score_threshold,
        relative_score_margin=args.relative_score_margin,
        iou_threshold=args.nms_iou,
        max_detections_per_image=args.max_detections,
        class_agnostic=True,
    )

    id_to_name = {det_lora.get_class_id(class_name): class_name for class_name in args.classes}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for idx, image_path in enumerate(image_paths):
        output_path = args.output_dir / f"{idx:03d}_{image_path.stem}.jpg"
        draw_prediction(image_path, display_predictions[idx], id_to_name, output_path)
        items.append(
            {
                "image": str(image_path),
                "output": str(output_path),
                "raw_prediction_count": int(predictions[idx]["scores"].shape[0]),
                "display_prediction_count": int(display_predictions[idx]["scores"].shape[0]),
                "predictions": prediction_rows(display_predictions[idx], id_to_name),
            }
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "classes": args.classes,
        "version_selection": version_selection,
        "postprocess": {
            "score_threshold": args.score_threshold,
            "relative_score_margin": args.relative_score_margin,
            "nms_iou": args.nms_iou,
            "max_detections": args.max_detections,
        },
        "items": items,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"report": str(report_path), "num_images": len(items)}, indent=2))


if __name__ == "__main__":
    main()
