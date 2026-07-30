"""Unit tests for the CL-DETR baseline (DKD label merging)."""

import torch

from det_lora.baselines.cl_detr import build_dkd_targets


def _gt(label, box):
    return {"labels": torch.tensor([label]), "boxes": torch.tensor([box])}


def test_dkd_merges_confident_old_class_pseudo_labels():
    # q0 is confident on old-class column 2; q1 is background.
    logits = torch.tensor([[[-5.0, -5.0, 3.0, -5.0], [-5.0, -5.0, -5.0, -5.0]]])
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]])
    gt = [_gt(93, [0.9, 0.9, 0.1, 0.1])]
    merged = build_dkd_targets(logits, boxes, gt, old_class_ids=[2], top_k=10)
    labels = merged[0]["labels"].tolist()
    assert 93 in labels and 2 in labels
    assert merged[0]["boxes"].shape[0] == 2


def test_dkd_noop_without_old_classes():
    logits = torch.zeros((1, 2, 4))
    boxes = torch.zeros((1, 2, 4))
    gt = [_gt(91, [0.5, 0.5, 0.2, 0.2])]
    merged = build_dkd_targets(logits, boxes, gt, old_class_ids=[])
    assert merged[0]["labels"].tolist() == [91]


def test_dkd_drops_pseudo_overlapping_new_gt():
    # Pseudo box identical to the new-class GT box -> IoU=1 > lambda -> dropped.
    logits = torch.tensor([[[-5.0, -5.0, 3.0, -5.0]]])
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
    gt = [_gt(93, [0.5, 0.5, 0.2, 0.2])]
    merged = build_dkd_targets(logits, boxes, gt, old_class_ids=[2], iou_lambda=0.7)
    assert merged[0]["labels"].tolist() == [93]


def test_dkd_ignores_low_confidence_predictions():
    # All scores below the floor -> no pseudo-labels added.
    logits = torch.full((1, 3, 4), -5.0)
    boxes = torch.rand((1, 3, 4))
    gt = [_gt(93, [0.5, 0.5, 0.2, 0.2])]
    merged = build_dkd_targets(logits, boxes, gt, old_class_ids=[2], score_floor=0.3)
    assert merged[0]["labels"].tolist() == [93]
