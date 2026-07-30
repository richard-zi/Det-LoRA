"""
Detection Dataset for Det-LoRA
================================

Provides PyTorch datasets for the Mendeley Military Vehicles dataset
in COCO-compatible format for RT-DETRv2 training.

Output format per sample:
- image: PIL Image or tensor [3, H, W]
- target: {"class_labels": tensor, "boxes": tensor [N, 4] in cxcywh normalized}
"""

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

# Class mapping (same as preprocessing)
CLASSES = {
    0: "military_tank",
    1: "military_truck",
    2: "military_aircraft",
    3: "military_helicopter",
    4: "civilian_car",
    5: "civilian_aircraft",
}

CLASS_TO_ID = {v: k for k, v in CLASSES.items()}

CSV_CLASS_MAPPING = {
    "military tank": "military_tank",
    "military truck": "military_truck",
    "military aircraft": "military_aircraft",
    "military helicopter": "military_helicopter",
    "civilian car": "civilian_car",
    "civilian aircraft": "civilian_aircraft",
}


def load_annotations_from_csv(csv_path: Path) -> Dict[str, List[Dict]]:
    """Parse CSV label file into annotations dict."""
    annotations: Dict[str, List[Dict]] = defaultdict(list)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            class_name = CSV_CLASS_MAPPING.get(row["class"], row["class"])

            if class_name not in CLASS_TO_ID:
                continue

            annotations[filename].append(
                {
                    "class_name": class_name,
                    "class_id": CLASS_TO_ID[class_name],
                    "xmin": int(row["xmin"]),
                    "ymin": int(row["ymin"]),
                    "xmax": int(row["xmax"]),
                    "ymax": int(row["ymax"]),
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                }
            )

    return dict(annotations)


def xyxy_to_cxcywh_normalized(
    xmin: int, ymin: int, xmax: int, ymax: int, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert xyxy absolute to cxcywh normalized [0,1]."""
    cx = ((xmin + xmax) / 2) / img_w
    cy = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    )


class DetectionDataset(Dataset):
    """
    Detection dataset for Mendeley Military Vehicles.

    Loads images and annotations, optionally filtering by class.
    Outputs pixel_values tensor + targets dict with 'labels' and 'boxes'.

    Args:
        images_dir: Path to images directory
        annotations: Dict of filename -> list of annotation dicts
        class_filter: If set, only include this class (for incremental training)
        class_id_offset: Offset for class IDs (e.g., 90 for COCO base)
        img_size: Target image resolution (square resize)
        max_samples: Limit number of samples (for debugging)
        sample_offset: Skip filtered samples before max_samples is applied
    """

    def __init__(
        self,
        images_dir: Path,
        annotations: Dict[str, List[Dict]],
        class_filter: Optional[str | Sequence[str]] = None,
        class_id_offset: int = 90,
        img_size: int = 576,
        max_samples: Optional[int] = None,
        sample_offset: int = 0,
        class_id_mapping: Optional[Dict[str, int]] = None,
    ):
        self.images_dir = Path(images_dir)
        self.class_filter = _normalize_class_filter(class_filter)
        self.class_id_offset = class_id_offset
        self.img_size = img_size
        self.class_id_mapping = dict(class_id_mapping) if class_id_mapping is not None else None

        # Filter and prepare samples
        self.samples: List[Tuple[str, List[Dict]]] = []
        for filename, annots in annotations.items():
            img_path = self.images_dir / filename
            if not img_path.exists():
                continue

            if self.class_filter:
                filtered = [a for a in annots if a["class_name"] in self.class_filter]
                if not filtered:
                    continue
                annots = filtered

            self.samples.append((filename, annots))

        if sample_offset < 0:
            raise ValueError("sample_offset must be non-negative")
        if sample_offset:
            self.samples = self.samples[sample_offset:]
        if max_samples:
            self.samples = self.samples[:max_samples]

        if self.class_id_mapping is not None:
            sample_classes = {
                annot["class_name"] for _, sample_annots in self.samples for annot in sample_annots
            }
            missing_classes = sorted(sample_classes - set(self.class_id_mapping))
            if missing_classes:
                raise ValueError(
                    "class_id_mapping is missing model IDs for: " + ", ".join(missing_classes)
                )

        # Build class mapping for filtered classes
        if self.class_filter:
            self.class_names = sorted(self.class_filter)
        else:
            seen = set()
            self.class_names = []
            for _, annots in self.samples:
                for a in annots:
                    if a["class_name"] not in seen:
                        seen.add(a["class_name"])
                        self.class_names.append(a["class_name"])
            self.class_names.sort()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filename, annots = self.samples[idx]
        img_path = self.images_dir / filename

        # Load image
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # Build target
        boxes = []
        class_labels = []

        for annot in annots:
            cx, cy, w, h = xyxy_to_cxcywh_normalized(
                annot["xmin"],
                annot["ymin"],
                annot["xmax"],
                annot["ymax"],
                img_w,
                img_h,
            )
            boxes.append([cx, cy, w, h])

            if self.class_id_mapping is not None:
                class_labels.append(self.class_id_mapping[annot["class_name"]])
            else:
                class_labels.append(annot["class_id"] + self.class_id_offset)

        # Resize image to target size and convert to tensor
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        pixel_values = TF.to_tensor(image)  # [3, H, W], normalized [0, 1]
        # Normalize with ImageNet mean/std (standard for DINOv2 backbone)
        pixel_values = TF.normalize(
            pixel_values,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        target = {
            "labels": torch.tensor(class_labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
        }

        return {"pixel_values": pixel_values, "labels": target, "sample_id": idx}


def load_dataset_from_raw(
    raw_dir: str = "data/raw",
    class_filter: Optional[str | Sequence[str]] = None,
    split: str = "train",
    class_id_offset: int = 90,
    img_size: int = 576,
    val_ratio: float = 0.2,
    seed: int = 42,
    max_samples: Optional[int] = None,
    sample_offset: int = 0,
    class_id_mapping: Optional[Dict[str, int]] = None,
) -> DetectionDataset:
    """
    Convenience function to load dataset from raw Mendeley data.

    Args:
        raw_dir: Path to raw dataset (with Images/ and Labels/ subdirs)
        class_filter: Only include this class
        split: "train", "val", or "test"
        class_id_offset: Offset for class IDs
        processor: AutoImageProcessor
        val_ratio: Validation split ratio
        seed: Random seed
        max_samples: Limit samples
        sample_offset: Skip this many filtered samples before applying max_samples
        class_id_mapping: Explicit class-name to model-label mapping
    """
    raw_path = Path(raw_dir)
    images_dir = raw_path / "Images"
    labels_dir = raw_path / "Labels" / "CSV Format"

    # Load annotations
    train_annots = load_annotations_from_csv(labels_dir / "train_labels.csv")

    # Keep the official test split untouched and derive validation only from
    # the official training split. This avoids train/test leakage.
    if split == "test":
        test_annots = load_annotations_from_csv(labels_dir / "test_labels.csv")
        filtered_annots = test_annots
    elif split in {"train", "val"}:
        train_files = sorted(train_annots.keys())
        rng = random.Random(seed)
        rng.shuffle(train_files)
        split_idx = int(len(train_files) * (1 - val_ratio))
        if split == "train":
            selected_files = train_files[:split_idx]
        else:
            selected_files = train_files[split_idx:]
        filtered_annots = {f: train_annots[f] for f in selected_files if f in train_annots}
    else:
        raise ValueError(f"Unknown split '{split}'. Choose from: train, val, test")

    return DetectionDataset(
        images_dir=images_dir,
        annotations=filtered_annots,
        class_filter=class_filter,
        class_id_offset=class_id_offset,
        img_size=img_size,
        max_samples=max_samples,
        sample_offset=sample_offset,
        class_id_mapping=class_id_mapping,
    )


def _normalize_class_filter(
    class_filter: Optional[str | Sequence[str]],
) -> Optional[Tuple[str, ...]]:
    """Normalize a class filter into a stable tuple."""
    if class_filter is None:
        return None
    if isinstance(class_filter, str):
        return (class_filter,)
    normalized = tuple(dict.fromkeys(cls for cls in class_filter if cls))
    return normalized or None
