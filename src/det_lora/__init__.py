"""
Det-LoRA: Parameter-Efficient Continual Learning for Object Detection
=====================================================================

A LoRA-based method for class-incremental and data-incremental
continual learning on the RF-DETR object detector.

Author: Richard Zimmermann
Thesis: Det-LoRA: Parameter-Efficient Class-Incremental Object Detection
with RF-DETR (FOM Hochschule, 2026)
"""

from det_lora.model.det_lora import DetLoRA
from det_lora.model.detector import RFDETRDetector

__all__ = ["DetLoRA", "RFDETRDetector", "AdapterSDK"]


def __getattr__(name: str):
    if name == "AdapterSDK":
        from det_lora.sdk import AdapterSDK

        return AdapterSDK
    raise AttributeError(f"module 'det_lora' has no attribute {name!r}")
