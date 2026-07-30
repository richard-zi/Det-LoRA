"""
Dataset Preprocessing
======================

Converts the Mendeley Military & Civilian Vehicles dataset
to a standardized format for Det-LoRA training.

Usage:
    uv run python -m det_lora.data.preprocess
    uv run python -m det_lora.data.preprocess --raw_dir data/raw --output data/processed
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from det_lora.data.dataset import (
    CLASS_TO_ID,
    CLASSES,
    load_annotations_from_csv,
)


def create_splits(
    annotations: Dict[str, List[Dict]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """Split annotations into train and val sets."""
    import random

    random.seed(seed)
    filenames = sorted(annotations.keys())
    random.shuffle(filenames)

    split_idx = int(len(filenames) * (1 - val_ratio))
    train_files = filenames[:split_idx]
    val_files = filenames[split_idx:]

    train_annots = {f: annotations[f] for f in train_files}
    val_annots = {f: annotations[f] for f in val_files}

    return train_annots, val_annots


def save_coco_format(
    annotations: Dict[str, List[Dict]],
    images_dir: Path,
    output_path: Path,
    class_filter: str = None,
    class_id_offset: int = 80,
) -> Dict:
    """
    Save annotations in COCO JSON format.

    Args:
        annotations: filename -> list of annotations
        images_dir: Path to images
        output_path: Output JSON path
        class_filter: Only include this class
        class_id_offset: Offset for class IDs
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Categories
    if class_filter:
        coco["categories"].append(
            {
                "id": CLASS_TO_ID[class_filter] + class_id_offset,
                "name": class_filter,
            }
        )
    else:
        for class_id, class_name in CLASSES.items():
            coco["categories"].append(
                {
                    "id": class_id + class_id_offset,
                    "name": class_name,
                }
            )

    ann_id = 0
    for img_id, (filename, annots) in enumerate(annotations.items()):
        img_path = images_dir / filename
        if not img_path.exists():
            continue

        if class_filter:
            annots = [a for a in annots if a["class_name"] == class_filter]
            if not annots:
                continue

        # Get image dimensions from first annotation
        img_w = annots[0]["width"]
        img_h = annots[0]["height"]

        coco["images"].append(
            {
                "id": img_id,
                "file_name": filename,
                "width": img_w,
                "height": img_h,
            }
        )

        for annot in annots:
            w = annot["xmax"] - annot["xmin"]
            h = annot["ymax"] - annot["ymin"]
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": annot["class_id"] + class_id_offset,
                    "bbox": [annot["xmin"], annot["ymin"], w, h],  # xywh
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f)

    return {
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "categories": len(coco["categories"]),
    }


def process_dataset(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """Process raw Mendeley dataset into COCO format."""
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    images_dir = raw_path / "Images"
    labels_dir = raw_path / "Labels" / "CSV Format"

    print("Loading annotations...")
    train_csv = load_annotations_from_csv(labels_dir / "train_labels.csv")
    test_csv = load_annotations_from_csv(labels_dir / "test_labels.csv")
    all_annots = {**train_csv, **test_csv}

    # Class distribution
    class_counts: Dict[str, int] = defaultdict(int)
    for annots in all_annots.values():
        for a in annots:
            class_counts[a["class_name"]] += 1

    print(f"\nFound {len(all_annots)} images, {sum(class_counts.values())} annotations")
    for name, count in sorted(class_counts.items()):
        print(f"  {name}: {count}")

    # Split
    train_annots, val_annots = create_splits(all_annots, val_ratio, seed)
    print(f"\nSplit: {len(train_annots)} train, {len(val_annots)} val")

    # Save all-class COCO JSON
    print("\nSaving all-class dataset...")
    stats = save_coco_format(train_annots, images_dir, output_path / "train.json")
    print(f"  Train: {stats['images']} images, {stats['annotations']} annotations")

    stats = save_coco_format(val_annots, images_dir, output_path / "val.json")
    print(f"  Val: {stats['images']} images, {stats['annotations']} annotations")

    # Save per-class COCO JSON
    print("\nSaving per-class datasets...")
    for class_name in CLASSES.values():
        for split_name, split_annots in [("train", train_annots), ("val", val_annots)]:
            stats = save_coco_format(
                split_annots,
                images_dir,
                output_path / "per_class" / class_name / f"{split_name}.json",
                class_filter=class_name,
            )
            print(f"  {class_name}/{split_name}: {stats['images']} images")

    # Save metadata
    meta = {
        "classes": dict(CLASSES),
        "class_to_id": CLASS_TO_ID,
        "total_images": len(all_annots),
        "total_annotations": sum(class_counts.values()),
        "class_counts": dict(class_counts),
        "train_images": len(train_annots),
        "val_images": len(val_annots),
        "val_ratio": val_ratio,
        "seed": seed,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Mendeley dataset")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    process_dataset(args.raw_dir, args.output, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
