"""
Inference & Prediction Visualization
======================================

Draw bounding box predictions on images for qualitative evaluation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from det_lora.model.det_lora import DetLoRA

# Color palette for different classes (colorblind-friendly)
CLASS_COLORS = [
    (230, 25, 75),  # red
    (60, 180, 75),  # green
    (0, 130, 200),  # blue
    (255, 178, 29),  # orange
    (145, 30, 180),  # purple
    (70, 240, 240),  # cyan
    (240, 50, 230),  # magenta
    (188, 143, 143),  # brown
    (128, 128, 0),  # olive
    (0, 0, 128),  # navy
]


def predict(
    det_lora: DetLoRA,
    image: Image.Image,
    confidence: float = 0.3,
    img_size: Optional[int] = None,
) -> List[Dict]:
    """
    Run inference on a single PIL image.

    Args:
        det_lora: Trained DetLoRA model
        image: PIL Image (RGB)
        confidence: Minimum confidence threshold
        img_size: Resize to this resolution (uses detector default if None)

    Returns:
        List of dicts: {"box": [x1,y1,x2,y2] absolute, "label": int, "score": float}
    """
    det_lora.set_eval_mode()
    device = det_lora.device

    if img_size is None:
        img_size = det_lora.detector.resolution

    orig_w, orig_h = image.size

    # Preprocess
    img_resized = image.resize((img_size, img_size), Image.BILINEAR)
    pixel_values = TF.to_tensor(img_resized)
    pixel_values = TF.normalize(pixel_values, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pixel_values = pixel_values.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = det_lora.forward(pixel_values=pixel_values)

    logits = outputs["pred_logits"][0]  # [num_queries, num_classes]
    boxes = outputs["pred_boxes"][0]  # [num_queries, 4] cxcywh normalized

    # Sigmoid for RF-DETR
    probs = logits.sigmoid()
    scores, labels = probs.max(-1)
    mask = scores > confidence

    predictions = []
    for i in torch.where(mask)[0]:
        cx, cy, w, h = boxes[i].cpu().tolist()
        # Convert normalized cxcywh to absolute xyxy
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        predictions.append(
            {
                "box": [x1, y1, x2, y2],
                "label": int(labels[i].item()),
                "score": float(scores[i].item()),
            }
        )

    # Sort by score descending
    predictions.sort(key=lambda p: p["score"], reverse=True)
    return predictions


def draw_predictions(
    image: Image.Image,
    predictions: List[Dict],
    class_names: Optional[Dict[int, str]] = None,
    line_width: int = 3,
) -> Image.Image:
    """
    Draw bounding boxes with labels on an image.

    Args:
        image: PIL Image (RGB)
        predictions: List of {"box": [x1,y1,x2,y2], "label": int, "score": float}
        class_names: Optional mapping of label_id -> display name
        line_width: Box line width

    Returns:
        New PIL Image with boxes drawn
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for pred in predictions:
        x1, y1, x2, y2 = pred["box"]
        label = pred["label"]
        score = pred["score"]

        color = CLASS_COLORS[label % len(CLASS_COLORS)]
        name = class_names.get(label, f"class_{label}") if class_names else f"class_{label}"
        text = f"{name} {score:.2f}"

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # Draw label background
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
            fill=color,
        )
        draw.text((x1, y1), text, fill=(255, 255, 255), font=font)

    return img


def visualize_comparison(
    image: Image.Image,
    predictions: List[Dict],
    ground_truth: List[Dict],
    class_names: Optional[Dict[int, str]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side comparison: predictions vs ground truth.

    Args:
        image: Original PIL Image
        predictions: Predicted boxes
        ground_truth: GT boxes (same format as predictions, score optional)
        class_names: Label ID to name mapping
        output_path: Save figure to this path if provided

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    pred_img = draw_predictions(image, predictions, class_names)
    gt_img = draw_predictions(image, ground_truth, class_names)

    ax1.imshow(pred_img)
    ax1.set_title("Predictions", fontsize=14)
    ax1.axis("off")

    ax2.imshow(gt_img)
    ax2.set_title("Ground Truth", fontsize=14)
    ax2.axis("off")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison to {output_path}")

    return fig


def save_prediction_grid(
    images: List[Image.Image],
    predictions_list: List[List[Dict]],
    output_path: str,
    class_names: Optional[Dict[int, str]] = None,
    cols: int = 4,
    figsize_per_image: float = 4.0,
) -> None:
    """
    Save a grid of prediction images for thesis figures.

    Args:
        images: List of PIL Images
        predictions_list: List of prediction lists (one per image)
        output_path: Output file path
        class_names: Label ID to name mapping
        cols: Number of columns in the grid
        figsize_per_image: Size per image in the grid
    """
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(figsize_per_image * cols, figsize_per_image * rows)
    )

    if rows == 1:
        axes = [axes] if cols == 1 else [axes]
    axes_flat = np.array(axes).flatten()

    for i, (img, preds) in enumerate(zip(images, predictions_list)):
        drawn = draw_predictions(img, preds, class_names)
        axes_flat[i].imshow(drawn)
        axes_flat[i].axis("off")

    # Hide empty axes
    for i in range(n, len(axes_flat)):
        axes_flat[i].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prediction grid ({n} images) to {output_path}")
