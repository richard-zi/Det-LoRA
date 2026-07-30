import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from det_lora.baselines.checkpoint import prepare_detector_for_checkpoint_load
from det_lora.baselines.ewc import EWCBaseline
from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.arbitration import (
    apply_adapter_arbitration,
    fit_adapter_arbitration_state,
    simplify_joint_predictions_for_display,
)
from det_lora.evaluation.evaluator import (
    ContinualEvaluator,
    _select_adapter_versions,
    aggregate_classwise_metrics,
    apply_shared_quality_calibrator,
    collect_det_lora_joint_predictions,
    extract_shared_quality_features,
    fit_shared_quality_calibrator,
    summarize_mixed_confusion,
)
from det_lora.evaluation.metrics import compute_map
from det_lora.model.det_lora import DetLoRA
from det_lora.model.detector import RFDETRDetector
from det_lora.sdk import AdapterSDK
from det_lora.train import (
    _compute_teacher_anchor_loss,
    _extract_distillation_targets,
    _make_adapter_hard_negatives,
    collate_fn,
    train_one_epoch,
)


def _read_official_filenames(csv_path: Path) -> set[str]:
    with csv_path.open() as f:
        return {row["filename"] for row in csv.DictReader(f)}


def test_official_split_has_no_train_test_leakage():
    raw_dir = Path("data/raw")
    labels_dir = raw_dir / "Labels" / "CSV Format"

    official_train = _read_official_filenames(labels_dir / "train_labels.csv")
    official_test = _read_official_filenames(labels_dir / "test_labels.csv")

    train_ds = load_dataset_from_raw(raw_dir=str(raw_dir), split="train", seed=123)
    val_ds = load_dataset_from_raw(raw_dir=str(raw_dir), split="val", seed=123)
    test_ds = load_dataset_from_raw(raw_dir=str(raw_dir), split="test", seed=123)

    train_files = {filename for filename, _ in train_ds.samples}
    val_files = {filename for filename, _ in val_ds.samples}
    test_files = {filename for filename, _ in test_ds.samples}

    assert train_files.isdisjoint(val_files)
    assert train_files | val_files <= official_train
    assert test_files <= official_test


def test_sample_offset_selects_deterministic_window():
    raw_dir = Path("data/raw")
    first_window = load_dataset_from_raw(
        raw_dir=str(raw_dir),
        class_filter="military_tank",
        split="train",
        seed=123,
        max_samples=3,
    )
    shifted_window = load_dataset_from_raw(
        raw_dir=str(raw_dir),
        class_filter="military_tank",
        split="train",
        seed=123,
        max_samples=2,
        sample_offset=1,
    )

    assert shifted_window.samples == first_window.samples[1:3]


def test_ewc_penalty_ignores_new_head_rows_after_class_expansion():
    class DetectorStub:
        device = torch.device("cpu")

        def __init__(self):
            self.model = nn.Module()
            self.model.class_embed = nn.Parameter(torch.tensor([[1.0], [2.0], [3.0]]))

    baseline = EWCBaseline.__new__(EWCBaseline)
    baseline.detector = DetectorStub()
    baseline._fisher = {"class_embed": torch.ones(2, 1)}
    baseline._optimal_params = {"class_embed": torch.zeros(2, 1)}

    penalty = baseline._ewc_penalty()

    # Only the shared prefix with the pre-expansion head should contribute.
    assert torch.isclose(penalty, torch.tensor(5.0))


def test_ewc_overlap_slices_use_shared_prefix_shape():
    overlap = EWCBaseline._overlap_slices(
        torch.zeros(93, 256),
        torch.zeros(92, 256),
        torch.zeros(92, 256),
    )

    assert overlap == (slice(0, 92), slice(0, 256))


def test_prepare_detector_for_checkpoint_load_expands_seen_classes_in_order():
    expanded_classes = []

    class DetectorStub:
        def expand_classification_head(self, class_name):
            expanded_classes.append(class_name)

    prepare_detector_for_checkpoint_load(
        DetectorStub(),
        ["military_tank", "military_truck", "military_aircraft"],
    )

    assert expanded_classes == [
        "military_tank",
        "military_truck",
        "military_aircraft",
    ]


def test_adapter_hard_negatives_wrap_seen_classes_as_empty_targets(monkeypatch):
    class TinyDataset:
        def __init__(self, class_name: str):
            self.class_name = class_name

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {
                "pixel_values": torch.zeros(3, 16, 16),
                "labels": {
                    "labels": torch.tensor([90 + idx], dtype=torch.long),
                    "boxes": torch.ones(1, 4),
                },
                "sample_id": idx,
            }

    calls = []

    def fake_load_dataset_from_raw(**kwargs):
        calls.append(kwargs)
        return TinyDataset(kwargs["class_filter"])

    monkeypatch.setattr("det_lora.train.load_dataset_from_raw", fake_load_dataset_from_raw)
    det_lora = type(
        "DetLoRAStub",
        (),
        {
            "trained_classes": ["military_tank", "military_aircraft", "military_helicopter"],
            "get_class_id": lambda self, class_name: {
                "military_tank": 91,
                "military_aircraft": 92,
                "military_helicopter": 93,
            }[class_name],
        },
    )()
    detector = type("DetectorStub", (), {"base_num_classes": 90})()

    datasets, counts = _make_adapter_hard_negatives(
        det_lora=det_lora,
        raw_dir="data/raw",
        target_class="military_aircraft",
        detector=detector,
        img_size=64,
        seed=7,
        max_samples_per_class=8,
    )

    assert [call["class_filter"] for call in calls] == [
        "military_tank",
        "military_helicopter",
    ]
    assert counts == {"military_tank": 2, "military_helicopter": 2}
    assert len(datasets) == 2
    sample = datasets[0][0]
    assert sample["labels"]["labels"].numel() == 0
    assert sample["labels"]["boxes"].shape == (0, 4)


def test_explicit_class_id_mapping_overrides_dataset_class_order():
    dataset = load_dataset_from_raw(
        raw_dir="data/raw",
        class_filter="military_aircraft",
        split="test",
        class_id_mapping={"military_aircraft": 92},
        max_samples=1,
    )

    labels = dataset[0]["labels"]["labels"].tolist()
    assert labels
    assert set(labels) == {92}


def test_explicit_class_id_mapping_fails_for_missing_class():
    with pytest.raises(ValueError, match="military_aircraft"):
        load_dataset_from_raw(
            raw_dir="data/raw",
            class_filter="military_aircraft",
            split="test",
            class_id_mapping={"military_tank": 91},
            max_samples=1,
        )


def test_anchor_latest_version_selection_keeps_memory_anchor_and_latest_extension():
    det_lora = type(
        "VersionedDetLoRA",
        (),
        {
            "_adapter_versions": {
                "tank": [
                    {"version_id": "v1"},
                    {"version_id": "v2"},
                    {"version_id": "v3"},
                    {"version_id": "v4"},
                ]
            }
        },
    )()

    assert _select_adapter_versions(det_lora, "tank", "anchor_latest") == ["v1", "v4"]
    assert _select_adapter_versions(det_lora, "tank", "all") == ["v1", "v2", "v3", "v4"]
    assert _select_adapter_versions(det_lora, "tank", "latest") == ["v4"]


def test_compute_map_ignores_predictions_outside_target_subset():
    predictions = [
        {
            "boxes": [[0.1, 0.1, 0.4, 0.4]],
            "scores": [0.95],
            "labels": [1],
        }
    ]
    ground_truths = [
        {
            "boxes": [[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.8, 0.8]],
            "labels": [1, 2],
        }
    ]

    unrestricted = compute_map(predictions, ground_truths)
    restricted = compute_map(predictions, ground_truths, target_class_ids=[1])

    assert unrestricted["mAP@0.5"] < restricted["mAP@0.5"]
    assert restricted["mAP@0.5"] == 1.0


def test_compute_map_reports_detection_metrics():
    predictions = [
        {
            "boxes": [[0.1, 0.1, 0.4, 0.4]],
            "scores": [0.95],
            "labels": [1],
        }
    ]
    ground_truths = [
        {
            "boxes": [[0.1, 0.1, 0.4, 0.4]],
            "labels": [1],
        }
    ]

    metrics = compute_map(predictions, ground_truths, target_class_ids=[1])

    assert metrics["mAP@0.5"] == 1.0
    assert metrics["mAP@0.75"] == 1.0
    assert metrics["mAP@0.95"] == 1.0
    assert metrics["Precision@0.5"] == 1.0
    assert metrics["Recall@0.5"] == 1.0
    assert metrics["F1@0.5"] == 1.0
    assert metrics["Precision@0.95"] == 1.0
    assert metrics["Recall@0.95"] == 1.0
    assert metrics["F1@0.95"] == 1.0
    assert metrics["TP@0.5"] == 1
    assert metrics["FP@0.5"] == 0
    assert metrics["FN@0.5"] == 0
    assert metrics["MicroPrecision@0.5"] == 1.0
    assert metrics["MicroRecall@0.5"] == 1.0
    assert metrics["MicroPrecision@0.95"] == 1.0
    assert metrics["MicroRecall@0.95"] == 1.0


def test_compute_map_can_export_precision_recall_curves():
    predictions = [
        {
            "boxes": [[0.1, 0.1, 0.4, 0.4]],
            "scores": [0.95],
            "labels": [1],
        }
    ]
    ground_truths = [
        {
            "boxes": [[0.1, 0.1, 0.4, 0.4]],
            "labels": [1],
        }
    ]

    metrics = compute_map(predictions, ground_truths, target_class_ids=[1], include_curves=True)
    curve = metrics["PR_curve_per_class@0.5"][1]
    curve95 = metrics["PR_curve_per_class@0.95"][1]

    assert curve["precision"] == [1.0]
    assert curve["recall"] == [1.0]
    assert curve["score_thresholds"] == [0.95]
    assert curve["ap"] == 1.0
    assert curve["num_gt"] == 1.0
    assert curve["num_pred"] == 1.0
    assert curve95["precision"] == [1.0]
    assert curve95["recall"] == [1.0]


def test_aggregate_classwise_metrics_keeps_precision_recall_curves():
    curve = {
        "precision": [1.0],
        "recall": [1.0],
        "score_thresholds": [0.95],
        "ap": 1.0,
        "tp": 1.0,
        "fp": 0.0,
        "fn": 0.0,
        "num_gt": 1.0,
        "num_pred": 1.0,
    }
    per_class_metrics = {
        "tank": {
            "mAP@0.5": 1.0,
            "mAP@0.75": 1.0,
            "mAP@0.5:0.95": 1.0,
            "mAP@0.95": 1.0,
            "Precision@0.5": 1.0,
            "Precision@0.95": 1.0,
            "Recall@0.5": 1.0,
            "Recall@0.95": 1.0,
            "F1@0.5": 1.0,
            "F1@0.95": 1.0,
            "MicroPrecision@0.5": 1.0,
            "MicroPrecision@0.95": 1.0,
            "MicroRecall@0.5": 1.0,
            "MicroRecall@0.95": 1.0,
            "MicroF1@0.5": 1.0,
            "MicroF1@0.95": 1.0,
            "TP@0.5": 1,
            "TP@0.95": 1,
            "FP@0.5": 0,
            "FP@0.95": 0,
            "FN@0.5": 0,
            "FN@0.95": 0,
            "AP_per_class@0.5": {"tank": 1.0},
            "AP_per_class@0.75": {"tank": 1.0},
            "AP_per_class@0.95": {"tank": 1.0},
            "Precision_per_class@0.5": {"tank": 1.0},
            "Precision_per_class@0.95": {"tank": 1.0},
            "Recall_per_class@0.5": {"tank": 1.0},
            "Recall_per_class@0.95": {"tank": 1.0},
            "F1_per_class@0.5": {"tank": 1.0},
            "F1_per_class@0.95": {"tank": 1.0},
            "PR_curve_per_class@0.5": {"tank": curve},
            "PR_curve_per_class@0.95": {"tank": curve},
        }
    }

    aggregated = aggregate_classwise_metrics(per_class_metrics)

    assert aggregated["PR_curve_per_class@0.5"]["tank"]["precision"] == [1.0]
    assert aggregated["PR_curve_per_class@0.5"]["tank"]["recall"] == [1.0]
    assert aggregated["PR_curve_per_class@0.95"]["tank"]["precision"] == [1.0]
    assert aggregated["mAP@0.95"] == 1.0


class _ToyDetectorModel(nn.Module):
    def eval(self):
        return self


class _ToyDetector:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = _ToyDetectorModel()

    def forward(self, pixel_values):
        return {
            "pred_logits": torch.tensor([[[0.0, 8.0]]], dtype=torch.float32),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]]], dtype=torch.float32),
        }


def test_standard_detector_evaluation_exports_precision_recall_curves():
    evaluator = ContinualEvaluator()
    batch = {
        "pixel_values": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
        "labels": [
            {
                "labels": torch.tensor([1], dtype=torch.int64),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            }
        ],
    }

    metrics = evaluator.evaluate_standard_detector(
        detector=_ToyDetector(),
        dataloader=[batch],
        class_names=["tank"],
        class_ids=[1],
        include_curves=True,
    )

    assert metrics["mAP@0.5"] == 1.0
    assert metrics["mAP@0.95"] == 1.0
    assert metrics["PR_curve_per_class@0.5"]["tank"]["precision"] == [1.0]
    assert metrics["PR_curve_per_class@0.5"]["tank"]["recall"] == [1.0]
    assert metrics["PR_curve_per_class@0.95"]["tank"]["precision"] == [1.0]


def test_collate_fn_preserves_sample_ids():
    batch = [
        {
            "pixel_values": torch.zeros((3, 8, 8), dtype=torch.float32),
            "labels": {
                "class_labels": torch.tensor([1], dtype=torch.int64),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            },
            "sample_id": 3,
        },
        {
            "pixel_values": torch.ones((3, 8, 8), dtype=torch.float32),
            "labels": {
                "class_labels": torch.tensor([2], dtype=torch.int64),
                "boxes": torch.tensor([[0.4, 0.4, 0.1, 0.1]], dtype=torch.float32),
            },
            "sample_id": 9,
        },
    ]

    collated = collate_fn(batch)

    assert collated["pixel_values"].shape == (2, 3, 8, 8)
    assert collated["sample_ids"].tolist() == [3, 9]


def test_teacher_anchor_loss_is_zero_for_identical_outputs():
    student_logits = torch.tensor([[[0.1, -0.2], [0.3, 0.4]]], dtype=torch.float32)
    student_boxes = torch.tensor(
        [[[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.1, 0.1]]], dtype=torch.float32
    )

    identical = _compute_teacher_anchor_loss(
        student_logits,
        student_boxes,
        student_logits.clone(),
        student_boxes.clone(),
    )
    shifted = _compute_teacher_anchor_loss(
        student_logits,
        student_boxes,
        student_logits + 0.5,
        student_boxes + 0.1,
    )

    assert identical["teacher_anchor_loss"].item() == pytest.approx(0.0)
    assert identical["teacher_cls_loss"].item() == pytest.approx(0.0)
    assert identical["teacher_box_loss"].item() == pytest.approx(0.0)
    assert shifted["teacher_anchor_loss"].item() > 0.0


def test_extract_distillation_targets_keeps_selected_classes_and_multiscale_outputs():
    outputs = {
        "pred_logits": torch.arange(1 * 2 * 5, dtype=torch.float32).reshape(1, 2, 5),
        "pred_boxes": torch.ones((1, 2, 4), dtype=torch.float32),
        "enc_outputs": {
            "pred_logits": torch.arange(1 * 2 * 5, dtype=torch.float32).reshape(1, 2, 5) + 10,
            "pred_boxes": torch.zeros((1, 2, 4), dtype=torch.float32),
        },
        "aux_outputs": [
            {
                "pred_logits": torch.arange(1 * 2 * 5, dtype=torch.float32).reshape(1, 2, 5) + 20,
                "pred_boxes": torch.full((1, 2, 4), 2.0, dtype=torch.float32),
            }
        ],
    }

    targets = _extract_distillation_targets(outputs, class_ids=[1, 4])

    assert targets["pred_logits"].shape == (1, 2, 2)
    assert targets["enc_outputs"]["pred_logits"].shape == (1, 2, 2)
    assert len(targets["aux_outputs"]) == 1
    assert targets["aux_outputs"][0]["pred_logits"].shape == (1, 2, 2)
    assert torch.equal(targets["pred_logits"][0, 0], torch.tensor([1.0, 4.0]))


def test_forgetting_is_computed_from_recorded_history():
    evaluator = ContinualEvaluator()
    evaluator.history = {
        0: {
            "metrics": {
                "mAP@0.5": 0.6,
                "mAP@0.5:0.95": 0.3,
                "AP_per_class@0.5": {"tank": 0.8},
            },
            "class_names": ["tank"],
        },
        1: {
            "metrics": {
                "mAP@0.5": 0.55,
                "mAP@0.5:0.95": 0.28,
                "AP_per_class@0.5": {"tank": 0.7, "truck": 0.6},
            },
            "class_names": ["tank", "truck"],
        },
    }

    forgetting = evaluator.compute_forgetting()

    assert forgetting["tank"] == pytest.approx(0.1)
    assert forgetting["truck"] == 0.0


class _DummyModel:
    def eval(self):
        return self


class _DummyDetector:
    def __init__(self, score: float):
        self.device = torch.device("cpu")
        self.model = _DummyModel()
        self._score = score

    def forward(self, pixel_values):
        score = torch.tensor(self._score, dtype=torch.float32)
        class_logit = torch.logit(score, eps=1e-6)
        return {
            "pred_logits": torch.tensor([[[0.0, class_logit.item()]]], dtype=torch.float32),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]]], dtype=torch.float32),
        }


class _ForwardEchoModel:
    def __call__(self, _samples):
        return {
            "pred_logits": torch.tensor([[[0.1, 0.2]]], dtype=torch.float32),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]]], dtype=torch.float32),
            "enc_outputs": {
                "pred_logits": torch.tensor([[[0.3, 0.4]]], dtype=torch.float32),
                "pred_boxes": torch.tensor([[[0.4, 0.4, 0.1, 0.1]]], dtype=torch.float32),
            },
            "aux_outputs": [
                {
                    "pred_logits": torch.tensor([[[0.5, 0.6]]], dtype=torch.float32),
                    "pred_boxes": torch.tensor([[[0.3, 0.3, 0.1, 0.1]]], dtype=torch.float32),
                }
            ],
        }


class _HeadStateInnerModel(nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.class_embed = nn.Linear(2, out_features)
        self.transformer = type(
            "TransformerStub",
            (),
            {"enc_out_class_embed": nn.ModuleList([nn.Linear(2, out_features) for _ in range(2)])},
        )()


class _HeadStateDetector:
    def __init__(self, out_features: int):
        self.device = torch.device("cpu")
        self.inner = _HeadStateInnerModel(out_features)
        self.model = self.inner
        self.criterion = None

    def _get_inner_model(self):
        return self.inner

    def _rebuild_criterion(self):
        return None


class _TeacherTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))


class _TeacherTrainDetLoRA:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = _TeacherTrainModel()

    def forward(self, pixel_values, targets=None):
        batch_size = pixel_values.shape[0]
        num_queries = 3900 if self.model.training else 300
        pred_logits = (
            torch.ones((batch_size, num_queries, 2), dtype=torch.float32) * self.model.anchor
        )
        pred_boxes = torch.ones((batch_size, num_queries, 4), dtype=torch.float32) * (
            self.model.anchor + 0.5
        )
        outputs = {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "enc_outputs": {
                "pred_logits": pred_logits + 0.25,
                "pred_boxes": pred_boxes + 0.1,
            },
            "aux_outputs": [
                {
                    "pred_logits": pred_logits + 0.5,
                    "pred_boxes": pred_boxes + 0.2,
                }
            ],
        }
        if targets is not None:
            outputs["loss"] = (pred_logits.mean() + pred_boxes.mean()) * 0 + self.model.anchor.pow(
                2
            )
            outputs["loss_dict"] = {"loss_ce": self.model.anchor.pow(2)}
        return outputs

    def get_trainable_params(self):
        return [self.model.anchor]

    def orthogonal_loss(self):
        return torch.tensor(0.0)

    def stability_loss(self):
        return torch.tensor(0.0)


class _JointEvalDetector:
    def __init__(self):
        self.device = torch.device("cpu")
        self.active_class = None

    def extract_shared_encoder_context(self, pixel_values):
        return {"pixel_values": pixel_values.clone()}

    def _make_outputs(self, pixel_values, class_name):
        batch_size = pixel_values.shape[0]
        pred_logits = torch.full((batch_size, 2, 93), -10.0, dtype=torch.float32)
        pred_boxes = torch.tensor(
            [
                [[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]],
            ],
            dtype=torch.float32,
        ).repeat(batch_size, 1, 1)
        if class_name == "tank":
            pred_logits[:, 0, 91] = 2.0
            pred_logits[:, 1, 91] = -1.0
        elif class_name == "truck":
            pred_logits[:, 0, 92] = 1.5
            pred_logits[:, 1, 92] = -0.5
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

    def forward_from_shared_encoder_context(self, context):
        return self._make_outputs(context["pixel_values"], self.active_class)


class _JointEvalDetLoRA:
    def __init__(self):
        self.device = torch.device("cpu")
        self.detector = _JointEvalDetector()
        self._prepared = []

    def set_eval_mode(self):
        return None

    def prepare_eval_adapter_cache(self, class_names):
        self._prepared = list(class_names)

    def activate_cached_eval_adapter(self, class_name):
        self.detector.active_class = class_name

    def clear_eval_adapter_cache(self):
        self.detector.active_class = None
        self._prepared = []

    def load_adapter_for_eval(self, class_name):
        self.detector.active_class = class_name

    def unload_adapter(self):
        self.detector.active_class = None

    def forward(self, pixel_values, targets=None):
        return self.detector._make_outputs(pixel_values, self.detector.active_class)

    def calibrate_scores(self, class_name, scores):
        return scores

    def get_class_id(self, class_name):
        return {"tank": 91, "truck": 92}[class_name]


def test_train_one_epoch_teacher_anchor_uses_eval_query_shape():
    det_lora = _TeacherTrainDetLoRA()
    optimizer = torch.optim.SGD(det_lora.get_trainable_params(), lr=0.1)
    batch = [
        {
            "pixel_values": torch.zeros((3, 8, 8), dtype=torch.float32),
            "labels": {
                "class_labels": torch.tensor([1], dtype=torch.int64),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            },
            "sample_id": 0,
        }
    ]
    teacher_cache = {
        0: {
            "pred_logits": torch.zeros((300, 2), dtype=torch.float32),
            "pred_boxes": torch.zeros((300, 4), dtype=torch.float32),
            "enc_outputs": {
                "pred_logits": torch.zeros((300, 2), dtype=torch.float32),
                "pred_boxes": torch.zeros((300, 4), dtype=torch.float32),
            },
            "aux_outputs": [
                {
                    "pred_logits": torch.zeros((300, 2), dtype=torch.float32),
                    "pred_boxes": torch.zeros((300, 4), dtype=torch.float32),
                }
            ],
        }
    }

    metrics = train_one_epoch(
        det_lora=det_lora,
        dataloader=[collate_fn(batch)],
        optimizer=optimizer,
        epoch=1,
        use_orthogonal_loss=False,
        teacher_cache=teacher_cache,
        teacher_anchor_weight=0.1,
    )

    assert metrics["teacher_anchor_loss"] >= 0.0
    assert metrics["teacher_encoder_loss"] >= 0.0
    assert metrics["teacher_aux_loss"] >= 0.0


def test_evaluator_uses_ranking_based_ap_by_default():
    batch = {
        "pixel_values": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
        "labels": [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
            }
        ],
    }

    default_metrics = ContinualEvaluator().evaluate_standard_detector(
        _DummyDetector(score=0.2),
        [batch],
        class_names=["truck"],
        class_ids=[1],
    )
    strict_metrics = ContinualEvaluator(confidence_threshold=0.3).evaluate_standard_detector(
        _DummyDetector(score=0.2),
        [batch],
        class_names=["truck"],
        class_ids=[1],
    )

    assert default_metrics["mAP@0.5"] == 1.0
    assert strict_metrics["mAP@0.5"] == 0.0


def test_collect_det_lora_joint_predictions_shared_encoder_matches_legacy_path():
    det_lora = _JointEvalDetLoRA()
    batch = {
        "pixel_values": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
        "labels": [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
                "labels": torch.tensor([91], dtype=torch.int64),
            }
        ],
    }

    shared_predictions, shared_ground_truths, shared_target_ids = (
        collect_det_lora_joint_predictions(
            det_lora,
            [batch],
            ["tank", "truck"],
            use_shared_encoder_cache=True,
        )
    )
    legacy_predictions, legacy_ground_truths, legacy_target_ids = (
        collect_det_lora_joint_predictions(
            det_lora,
            [batch],
            ["tank", "truck"],
            use_shared_encoder_cache=False,
        )
    )

    assert shared_target_ids == legacy_target_ids == [91, 92]
    assert len(shared_predictions) == len(legacy_predictions) == 1
    assert len(shared_ground_truths) == len(legacy_ground_truths) == 1
    for key in ("boxes", "labels"):
        np.testing.assert_allclose(shared_ground_truths[0][key], legacy_ground_truths[0][key])
    for key in ("boxes", "scores", "labels", "query_ids"):
        np.testing.assert_allclose(shared_predictions[0][key], legacy_predictions[0][key])


def test_detector_forward_preserves_encoder_outputs():
    detector = object.__new__(RFDETRDetector)
    detector.model = _ForwardEchoModel()
    detector.criterion = lambda outputs, targets: {"loss_ce": torch.tensor(1.0)}
    detector.prepare_input = lambda pixel_values: pixel_values

    result = RFDETRDetector.forward(
        detector,
        pixel_values=torch.zeros((1, 3, 8, 8), dtype=torch.float32),
    )

    assert "enc_outputs" in result
    assert "aux_outputs" in result
    assert torch.equal(
        result["enc_outputs"]["pred_logits"],
        torch.tensor([[[0.3, 0.4]]], dtype=torch.float32),
    )


def test_aggregate_classwise_metrics_preserves_per_class_ap():
    aggregated = aggregate_classwise_metrics(
        {
            "tank": {
                "mAP@0.5": 0.8,
                "mAP@0.75": 0.5,
                "mAP@0.5:0.95": 0.4,
                "Precision@0.5": 0.7,
                "Recall@0.5": 0.6,
                "F1@0.5": 0.646153846,
                "MicroPrecision@0.5": 0.7,
                "MicroRecall@0.5": 0.6,
                "MicroF1@0.5": 0.646153846,
                "TP@0.5": 6,
                "FP@0.5": 2,
                "FN@0.5": 4,
                "AP_per_class@0.5": {"tank": 0.8},
                "AP_per_class@0.75": {"tank": 0.5},
                "Precision_per_class@0.5": {"tank": 0.7},
                "Recall_per_class@0.5": {"tank": 0.6},
                "F1_per_class@0.5": {"tank": 0.646153846},
            },
            "truck": {
                "mAP@0.5": 0.4,
                "mAP@0.75": 0.2,
                "mAP@0.5:0.95": 0.1,
                "Precision@0.5": 0.5,
                "Recall@0.5": 0.25,
                "F1@0.5": 0.333333333,
                "MicroPrecision@0.5": 0.5,
                "MicroRecall@0.5": 0.25,
                "MicroF1@0.5": 0.333333333,
                "TP@0.5": 2,
                "FP@0.5": 2,
                "FN@0.5": 6,
                "AP_per_class@0.5": {"truck": 0.4},
                "AP_per_class@0.75": {"truck": 0.2},
                "Precision_per_class@0.5": {"truck": 0.5},
                "Recall_per_class@0.5": {"truck": 0.25},
                "F1_per_class@0.5": {"truck": 0.333333333},
            },
        }
    )

    assert aggregated["mAP@0.5"] == pytest.approx(0.6)
    assert aggregated["TP@0.5"] == 8
    assert aggregated["FP@0.5"] == 4
    assert aggregated["FN@0.5"] == 10
    assert aggregated["AP_per_class@0.5"] == {"tank": 0.8, "truck": 0.4}


def test_summarize_mixed_confusion_reports_ap_gap():
    matched_history = {
        1: {
            "metrics": {
                "mAP@0.5": 0.7,
                "AP_per_class@0.5": {"tank": 0.8, "truck": 0.6},
            },
            "class_names": ["tank", "truck"],
        }
    }
    mixed_history = {
        1: {
            "metrics": {
                "mAP@0.5": 0.5,
                "AP_per_class@0.5": {"tank": 0.55, "truck": 0.45},
            },
            "class_names": ["tank", "truck"],
        }
    }

    summary = summarize_mixed_confusion(matched_history, mixed_history)

    assert summary["1"]["mAP@0.5_gap"] == pytest.approx(0.2)
    assert summary["1"]["AP_per_class@0.5_gap"]["tank"] == pytest.approx(0.25)
    assert summary["1"]["AP_per_class@0.5_gap"]["truck"] == pytest.approx(0.15)


def test_extract_shared_quality_features_combines_embeddings_and_geometry():
    prediction = {
        "boxes": np.array([[0.1, 0.1, 0.4, 0.4]], dtype=np.float32),
        "scores": np.array([0.8], dtype=np.float32),
        "labels": np.array([1], dtype=np.int64),
        "quality_features": np.array([[0.2, -0.1, 0.3]], dtype=np.float32),
    }

    features = extract_shared_quality_features(prediction)

    assert features.shape == (1, 7)
    np.testing.assert_allclose(features[0, :3], np.array([0.2, -0.1, 0.3], dtype=np.float32))
    assert features[0, 3] == pytest.approx(0.8)


def test_shared_quality_calibrator_learns_class_agnostic_tp_signal():
    predictions = [
        {
            "boxes": np.array(
                [
                    [0.10, 0.10, 0.40, 0.40],
                    [0.10, 0.10, 0.40, 0.40],
                    [0.55, 0.55, 0.85, 0.85],
                    [0.55, 0.55, 0.85, 0.85],
                ],
                dtype=np.float32,
            ),
            "scores": np.array([0.72, 0.70, 0.74, 0.71], dtype=np.float32),
            "labels": np.array([1, 2, 1, 2], dtype=np.int64),
            "quality_features": np.array(
                [
                    [2.0, 1.5],
                    [-2.0, -1.5],
                    [-2.0, -1.5],
                    [2.0, 1.5],
                ],
                dtype=np.float32,
            ),
        }
    ]
    ground_truths = [
        {
            "boxes": np.array(
                [
                    [0.10, 0.10, 0.40, 0.40],
                    [0.55, 0.55, 0.85, 0.85],
                ],
                dtype=np.float32,
            ),
            "labels": np.array([1, 2], dtype=np.int64),
        }
    ]

    calibrator = fit_shared_quality_calibrator(
        predictions,
        ground_truths,
        target_class_ids=[1, 2],
        steps=250,
        lr=0.05,
    )
    adjusted = apply_shared_quality_calibrator(predictions, calibrator)

    assert not calibrator.get("identity", False)
    assert adjusted[0]["scores"][0] > adjusted[0]["scores"][1]
    assert adjusted[0]["scores"][3] > adjusted[0]["scores"][2]


def test_adapter_arbitration_uses_class_prototypes_for_overlapping_experts():
    predictions = [
        {
            "boxes": np.array(
                [
                    [0.10, 0.10, 0.40, 0.40],
                    [0.10, 0.10, 0.40, 0.40],
                    [0.55, 0.55, 0.85, 0.85],
                    [0.55, 0.55, 0.85, 0.85],
                ],
                dtype=np.float32,
            ),
            "scores": np.array([0.70, 0.74, 0.73, 0.71], dtype=np.float32),
            "labels": np.array([1, 2, 1, 2], dtype=np.int64),
            "quality_features": np.array(
                [
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        }
    ]
    ground_truths = [
        {
            "boxes": np.array(
                [
                    [0.10, 0.10, 0.40, 0.40],
                    [0.55, 0.55, 0.85, 0.85],
                ],
                dtype=np.float32,
            ),
            "labels": np.array([1, 2], dtype=np.int64),
        }
    ]

    state = fit_adapter_arbitration_state(
        predictions,
        ground_truths,
        target_class_ids=[1, 2],
        prototype_weight_grid=(0.0, 1.5),
        loser_penalty_grid=(0.0, 1.0),
    )
    adjusted = apply_adapter_arbitration(predictions, state)

    assert state["prototype_counts"] == {"1": 1, "2": 1}
    assert not state.get("identity", False)
    assert adjusted[0]["scores"][0] > adjusted[0]["scores"][1]
    assert adjusted[0]["scores"][3] > adjusted[0]["scores"][2]


def test_adapter_arbitration_classifier_learns_from_cross_adapter_false_positives():
    predictions = [
        {
            "boxes": np.array(
                [
                    [0.10, 0.10, 0.40, 0.40],
                    [0.10, 0.10, 0.40, 0.40],
                ],
                dtype=np.float32,
            ),
            "scores": np.array([0.90, 0.70], dtype=np.float32),
            "labels": np.array([1, 2], dtype=np.int64),
            "quality_features": np.array(
                [
                    [0.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        },
        {
            "boxes": np.array([[0.55, 0.55, 0.85, 0.85]], dtype=np.float32),
            "scores": np.array([0.90], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
            "quality_features": np.array([[1.0, 0.0]], dtype=np.float32),
        },
    ]
    ground_truths = [
        {
            "boxes": np.array([[0.10, 0.10, 0.40, 0.40]], dtype=np.float32),
            "labels": np.array([2], dtype=np.int64),
        },
        {
            "boxes": np.array([[0.55, 0.55, 0.85, 0.85]], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
        },
    ]

    state = fit_adapter_arbitration_state(
        predictions,
        ground_truths,
        target_class_ids=[1, 2],
        prototype_weight_grid=(0.0,),
        classifier_weight_grid=(1.0,),
        loser_penalty_grid=(0.0, 1.5),
        class_bias_grid=(0.0,),
    )
    adjusted = apply_adapter_arbitration(predictions, state)

    assert state["region_classifier"]["train_count"] == 3
    assert adjusted[0]["scores"][1] > adjusted[0]["scores"][0]


def test_simplify_joint_predictions_for_display_keeps_readable_object_winners():
    predictions = [
        {
            "boxes": np.array(
                [
                    [0.10, 0.10, 0.40, 0.40],
                    [0.11, 0.11, 0.41, 0.41],
                    [0.60, 0.60, 0.80, 0.80],
                    [0.20, 0.60, 0.30, 0.70],
                ],
                dtype=np.float32,
            ),
            "scores": np.array([0.90, 0.85, 0.70, 0.20], dtype=np.float32),
            "labels": np.array([1, 2, 2, 1], dtype=np.int64),
            "query_ids": np.array([0, 0, 1, 2], dtype=np.int64),
        }
    ]

    simplified = simplify_joint_predictions_for_display(
        predictions,
        score_threshold=0.5,
        iou_threshold=0.5,
        max_detections_per_image=2,
    )

    assert simplified[0]["scores"].tolist() == pytest.approx([0.90, 0.70])
    assert simplified[0]["labels"].tolist() == [1, 2]


def test_simplify_joint_predictions_for_display_can_use_relative_score_floor():
    predictions = [
        {
            "boxes": np.array(
                [
                    [0.10, 0.10, 0.30, 0.30],
                    [0.60, 0.60, 0.80, 0.80],
                ],
                dtype=np.float32,
            ),
            "scores": np.array([0.95, 0.65], dtype=np.float32),
            "labels": np.array([1, 2], dtype=np.int64),
        }
    ]

    simplified = simplify_joint_predictions_for_display(
        predictions,
        score_threshold=0.5,
        relative_score_margin=0.2,
        max_detections_per_image=2,
    )

    assert simplified[0]["scores"].tolist() == pytest.approx([0.95])
    assert simplified[0]["labels"].tolist() == [1]


def test_det_lora_calibrator_fits_and_applies_score_transform():
    det_lora = object.__new__(DetLoRA)
    det_lora.detector = type("DetectorStub", (), {"device": torch.device("cpu")})()
    det_lora._adapter_calibrators = {}
    det_lora._score_banks = {}
    det_lora.record_score_bank(
        "tank",
        positive_scores=[0.91, 0.88, 0.84, 0.79, 0.76, 0.73],
        negative_scores=[0.34, 0.28, 0.21, 0.19, 0.12, 0.08],
    )
    calibrator = det_lora.fit_calibrator("tank", steps=150, lr=0.08)

    raw = torch.tensor([0.85, 0.2], dtype=torch.float32)
    calibrated = det_lora.calibrate_scores("tank", raw)

    assert calibrator["positive_count"] == 6.0
    assert calibrator["negative_count"] == 6.0
    assert calibrated[0].item() > calibrated[1].item()
    assert calibrated[0].item() > 0.85
    assert calibrated[1].item() < 0.2


def test_det_lora_builds_cumulative_head_state_from_base_and_rows():
    detector = _HeadStateDetector(out_features=4)
    detector.base_num_classes = 2
    detector.added_classes = ["tank", "truck"]

    det_lora = object.__new__(DetLoRA)
    det_lora.detector = detector
    det_lora.trained_classes = ["tank", "truck"]
    det_lora._base_head_state = {
        "class_embed": {
            "weight": torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
            "bias": torch.tensor([0.1, 0.2], dtype=torch.float32),
        },
        "enc_out_class_embed": [
            {
                "weight": torch.tensor([[3.0, 3.0], [4.0, 4.0]], dtype=torch.float32),
                "bias": torch.tensor([0.3, 0.4], dtype=torch.float32),
            },
            {
                "weight": torch.tensor([[5.0, 5.0], [6.0, 6.0]], dtype=torch.float32),
                "bias": torch.tensor([0.5, 0.6], dtype=torch.float32),
            },
        ],
    }
    det_lora._class_head_rows = {
        "tank": {
            "class_embed": {
                "weight": torch.tensor([[7.0, 7.0]], dtype=torch.float32),
                "bias": torch.tensor([0.7], dtype=torch.float32),
            },
            "enc_out_class_embed": [
                {
                    "weight": torch.tensor([[8.0, 8.0]], dtype=torch.float32),
                    "bias": torch.tensor([0.8], dtype=torch.float32),
                },
                {
                    "weight": torch.tensor([[9.0, 9.0]], dtype=torch.float32),
                    "bias": torch.tensor([0.9], dtype=torch.float32),
                },
            ],
        },
        "truck": {
            "class_embed": {
                "weight": torch.tensor([[10.0, 10.0]], dtype=torch.float32),
                "bias": torch.tensor([1.0], dtype=torch.float32),
            },
            "enc_out_class_embed": [
                {
                    "weight": torch.tensor([[11.0, 11.0]], dtype=torch.float32),
                    "bias": torch.tensor([1.1], dtype=torch.float32),
                },
                {
                    "weight": torch.tensor([[12.0, 12.0]], dtype=torch.float32),
                    "bias": torch.tensor([1.2], dtype=torch.float32),
                },
            ],
        },
    }
    det_lora._class_head_states = {}

    tank_state = det_lora._build_cumulative_head_state("tank")
    truck_state = det_lora._build_cumulative_head_state("truck")

    assert tank_state["class_embed"]["weight"].shape == (3, 2)
    assert truck_state["class_embed"]["weight"].shape == (4, 2)
    assert torch.equal(
        tank_state["class_embed"]["weight"][-1],
        torch.tensor([7.0, 7.0], dtype=torch.float32),
    )
    assert torch.equal(
        truck_state["class_embed"]["weight"][-1],
        torch.tensor([10.0, 10.0], dtype=torch.float32),
    )
    assert torch.equal(
        truck_state["enc_out_class_embed"][1]["bias"][-1],
        torch.tensor(1.2, dtype=torch.float32),
    )


def test_det_lora_restores_per_class_head_snapshots():
    detector = _HeadStateDetector(out_features=3)
    det_lora = object.__new__(DetLoRA)
    det_lora.detector = detector
    det_lora._class_head_states = {}
    det_lora._global_head_state = None

    with torch.no_grad():
        detector.inner.class_embed.weight.fill_(1.0)
        detector.inner.class_embed.bias.fill_(0.5)
        for layer in detector.inner.transformer.enc_out_class_embed:
            layer.weight.fill_(2.0)
            layer.bias.fill_(1.5)
    class_head = det_lora._capture_head_state()

    detector.inner.class_embed = nn.Linear(2, 4)
    detector.inner.transformer.enc_out_class_embed = nn.ModuleList(
        [nn.Linear(2, 4) for _ in range(2)]
    )
    with torch.no_grad():
        detector.inner.class_embed.weight.fill_(9.0)
        detector.inner.class_embed.bias.fill_(8.0)
        for layer in detector.inner.transformer.enc_out_class_embed:
            layer.weight.fill_(7.0)
            layer.bias.fill_(6.0)
    global_head = det_lora._capture_head_state()

    det_lora._class_head_states["tank"] = class_head
    det_lora._global_head_state = global_head

    det_lora._apply_head_state(det_lora._class_head_states["tank"])
    assert detector.inner.class_embed.out_features == 3
    assert torch.all(detector.inner.class_embed.weight == 1.0)
    assert torch.all(detector.inner.transformer.enc_out_class_embed[0].weight == 2.0)
    assert detector.inner.class_embed.weight.requires_grad is False

    det_lora._restore_global_head_state()
    assert detector.inner.class_embed.out_features == 4
    assert torch.all(detector.inner.class_embed.weight == 9.0)
    assert torch.all(detector.inner.transformer.enc_out_class_embed[0].weight == 7.0)


def test_det_lora_save_all_persists_shared_quality_calibrator(tmp_path):
    detector = _HeadStateDetector(out_features=3)
    detector.variant = "stub"
    detector.added_classes = ["tank"]
    detector.base_num_classes = 91
    source_adapter_dir = tmp_path / "source_adapters" / "lora_tank"
    source_adapter_dir.mkdir(parents=True)
    (source_adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")
    (source_adapter_dir / "README.md").write_text("adapter")
    det_lora = object.__new__(DetLoRA)
    det_lora.detector = detector
    det_lora.default_rank = 8
    det_lora.default_alpha = 16
    det_lora.trained_classes = ["tank"]
    det_lora.current_class = None
    det_lora._peft_applied = False
    det_lora._head_mask = type("MaskStub", (), {"frozen_indices": {91}})()
    det_lora._adapter_paths = {"tank": str(source_adapter_dir)}
    det_lora._adapter_calibrators = {}
    det_lora._score_banks = {}
    det_lora._shared_quality_calibrator = {
        "weight": [0.2] * 9,
        "bias": -0.05,
        "mean": [0.0] * 9,
        "std": [1.0] * 9,
        "positive_count": 5,
        "negative_count": 6,
    }
    det_lora._adapter_arbitration_state = {
        "identity": False,
        "prototype_weight": 1.0,
        "class_prototypes": {"91": [1.0, 0.0]},
        "prototype_counts": {"91": 3},
    }
    det_lora._class_head_states = {}
    det_lora._global_head_state = None

    det_lora.save_all(str(tmp_path))

    with open(tmp_path / "adapter_calibration.json") as f:
        payload = json.load(f)
    with open(tmp_path / "det_lora_registry.json") as f:
        registry = json.load(f)

    assert payload["shared_quality_calibrator"]["bias"] == pytest.approx(-0.05)
    assert payload["adapter_arbitration_state"]["prototype_weight"] == pytest.approx(1.0)
    assert registry["active_versions"]["tank"] == "v1"
    assert registry["adapter_versions"]["tank"][0]["version_id"] == "v1"
    assert (
        registry["adapter_versions"]["tank"][0]["adapter_path"]
        == "adapter_versions/tank/v1/adapter"
    )
    assert registry["adapter_paths"]["tank"] == "adapters/lora_tank"
    assert (tmp_path / "adapters" / "lora_tank" / "adapter_model.safetensors").exists()
    assert (
        tmp_path / "adapter_versions" / "tank" / "v1" / "adapter" / "adapter_model.safetensors"
    ).exists()


def test_det_lora_load_all_resolves_relative_adapter_paths(tmp_path):
    detector = _HeadStateDetector(out_features=3)
    detector.variant = "stub"
    detector.added_classes = ["tank"]
    detector.base_num_classes = 91
    source_adapter_dir = tmp_path / "source_adapters" / "lora_tank"
    source_adapter_dir.mkdir(parents=True)
    (source_adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")

    det_lora = object.__new__(DetLoRA)
    det_lora.detector = detector
    det_lora.default_rank = 8
    det_lora.default_alpha = 16
    det_lora.trained_classes = ["tank"]
    det_lora.current_class = None
    det_lora._peft_applied = False
    det_lora._head_mask = type("MaskStub", (), {"frozen_indices": {91}})()
    det_lora._adapter_paths = {"tank": str(source_adapter_dir)}
    det_lora._adapter_calibrators = {}
    det_lora._score_banks = {}
    det_lora._shared_quality_calibrator = {"identity": False, "bias": 0.3}
    det_lora._adapter_arbitration_state = {
        "identity": False,
        "prototype_weight": 0.75,
        "class_prototypes": {"91": [1.0, 0.0]},
    }
    det_lora._class_head_states = {}
    det_lora._global_head_state = None

    det_lora.save_all(str(tmp_path))

    load_detector = _HeadStateDetector(out_features=3)
    load_detector.variant = "stub"
    load_detector.added_classes = ["tank"]
    load_detector.base_num_classes = 91
    load_det_lora = object.__new__(DetLoRA)
    load_det_lora.detector = load_detector
    load_det_lora.default_rank = 8
    load_det_lora.default_alpha = 16
    load_det_lora._head_mask = type(
        "MaskStub",
        (),
        {
            "__init__": lambda self: setattr(self, "frozen_indices", set()),
            "freeze_class": lambda self, idx: self.frozen_indices.add(idx),
            "remove_hooks": lambda self: None,
            "register_hooks": lambda self, inner: None,
        },
    )()
    load_det_lora._peft_applied = False

    load_det_lora.load_all(str(tmp_path))

    assert load_det_lora.adapters["tank"] == str(
        tmp_path / "adapter_versions" / "tank" / "v1" / "adapter"
    )
    assert load_det_lora.shared_quality_calibrator["bias"] == pytest.approx(0.3)
    assert load_det_lora.adapter_arbitration_state["prototype_weight"] == pytest.approx(0.75)
    assert load_det_lora.active_versions["tank"] == "v1"


def test_det_lora_load_all_backfills_legacy_class_head_snapshots(tmp_path):
    legacy_detector = _HeadStateDetector(out_features=4)
    legacy_detector.variant = "stub"
    legacy_detector.added_classes = ["tank", "truck"]
    legacy_detector.base_num_classes = 91
    with torch.no_grad():
        legacy_detector.inner.class_embed.weight.fill_(1.25)
        legacy_detector.inner.class_embed.bias.fill_(0.5)
        for idx, layer in enumerate(legacy_detector.inner.transformer.enc_out_class_embed):
            layer.weight.fill_(2.0 + idx)
            layer.bias.fill_(0.25 + idx)

    legacy_det_lora = object.__new__(DetLoRA)
    legacy_det_lora.detector = legacy_detector
    legacy_head_state = legacy_det_lora._capture_head_state()

    torch.save(legacy_head_state, tmp_path / "head_weights.pt")
    with open(tmp_path / "det_lora_registry.json", "w") as f:
        json.dump(
            {
                "trained_classes": ["tank", "truck"],
                "current_class": None,
                "frozen_head_indices": [91, 92],
                "added_classes": ["tank", "truck"],
                "base_num_classes": 91,
                "default_rank": 8,
                "default_alpha": 16,
                "adapter_paths": {},
                "peft_applied": False,
                "class_head_snapshots": [],
            },
            f,
            indent=2,
        )
    with open(tmp_path / "adapter_calibration.json", "w") as f:
        json.dump({}, f)

    load_detector = _HeadStateDetector(out_features=4)
    load_detector.variant = "stub"
    load_detector.added_classes = ["tank", "truck"]
    load_detector.base_num_classes = 91
    load_det_lora = object.__new__(DetLoRA)
    load_det_lora.detector = load_detector
    load_det_lora.default_rank = 8
    load_det_lora.default_alpha = 16
    load_det_lora._head_mask = type(
        "MaskStub",
        (),
        {
            "__init__": lambda self: setattr(self, "frozen_indices", set()),
            "freeze_class": lambda self, idx: self.frozen_indices.add(idx),
            "remove_hooks": lambda self: None,
            "register_hooks": lambda self, inner: None,
        },
    )()
    load_det_lora._peft_applied = False

    load_det_lora.load_all(str(tmp_path))

    assert set(load_det_lora._class_head_states) == {"tank", "truck"}
    assert load_det_lora.active_versions == {"tank": "v1", "truck": "v1"}
    assert torch.equal(
        load_det_lora._class_head_states["tank"]["class_embed"]["weight"],
        legacy_head_state["class_embed"]["weight"],
    )
    assert torch.equal(
        load_det_lora._class_head_states["truck"]["enc_out_class_embed"][1]["bias"],
        legacy_head_state["enc_out_class_embed"][1]["bias"],
    )


def test_det_lora_save_all_materializes_missing_class_head_snapshots(tmp_path):
    detector = _HeadStateDetector(out_features=4)
    detector.variant = "stub"
    detector.added_classes = ["tank", "truck"]
    detector.base_num_classes = 91

    det_lora = object.__new__(DetLoRA)
    det_lora.detector = detector
    det_lora.default_rank = 8
    det_lora.default_alpha = 16
    det_lora.trained_classes = ["tank", "truck"]
    det_lora.current_class = None
    det_lora._peft_applied = False
    det_lora._head_mask = type("MaskStub", (), {"frozen_indices": {91, 92}})()
    det_lora._adapter_paths = {}
    det_lora._adapter_calibrators = {}
    det_lora._score_banks = {}
    det_lora._class_head_states = {}
    det_lora._global_head_state = None

    det_lora.save_all(str(tmp_path))

    with open(tmp_path / "det_lora_registry.json") as f:
        registry = json.load(f)

    assert registry["class_head_snapshots"] == ["tank", "truck"]
    assert registry["active_versions"] == {"tank": "v1", "truck": "v1"}
    assert (tmp_path / "class_heads" / "tank.pt").exists()
    assert (tmp_path / "class_heads" / "truck.pt").exists()


def test_det_lora_activate_adapter_version_switches_active_runtime_state():
    detector = _HeadStateDetector(out_features=3)
    detector.variant = "stub"
    detector.base_num_classes = 2
    detector.added_classes = ["tank"]

    det_lora = object.__new__(DetLoRA)
    det_lora.detector = detector
    det_lora.default_rank = 8
    det_lora.default_alpha = 16
    det_lora.current_class = None
    det_lora._peft_applied = False
    det_lora._head_mask = type("MaskStub", (), {"frozen_indices": set()})()
    det_lora._base_head_state = {
        "class_embed": {
            "weight": torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
            "bias": torch.tensor([0.1, 0.2], dtype=torch.float32),
        },
        "enc_out_class_embed": [
            {
                "weight": torch.tensor([[3.0, 3.0], [4.0, 4.0]], dtype=torch.float32),
                "bias": torch.tensor([0.3, 0.4], dtype=torch.float32),
            },
            {
                "weight": torch.tensor([[5.0, 5.0], [6.0, 6.0]], dtype=torch.float32),
                "bias": torch.tensor([0.5, 0.6], dtype=torch.float32),
            },
        ],
    }
    det_lora._adapter_versions = {
        "tank": [
            {
                "version_id": "v1",
                "source": "add",
                "created_at": None,
                "adapter_path": "/tmp/tank_v1",
            },
            {
                "version_id": "v2",
                "source": "extend",
                "created_at": None,
                "adapter_path": "/tmp/tank_v2",
            },
        ]
    }
    det_lora._active_versions = {"tank": "v2"}
    det_lora._versioned_class_head_rows = {
        "tank": {
            "v1": {
                "class_embed": {
                    "weight": torch.tensor([[7.0, 7.0]], dtype=torch.float32),
                    "bias": torch.tensor([0.7], dtype=torch.float32),
                },
                "enc_out_class_embed": [
                    {
                        "weight": torch.tensor([[8.0, 8.0]], dtype=torch.float32),
                        "bias": torch.tensor([0.8], dtype=torch.float32),
                    },
                    {
                        "weight": torch.tensor([[9.0, 9.0]], dtype=torch.float32),
                        "bias": torch.tensor([0.9], dtype=torch.float32),
                    },
                ],
            },
            "v2": {
                "class_embed": {
                    "weight": torch.tensor([[10.0, 10.0]], dtype=torch.float32),
                    "bias": torch.tensor([1.0], dtype=torch.float32),
                },
                "enc_out_class_embed": [
                    {
                        "weight": torch.tensor([[11.0, 11.0]], dtype=torch.float32),
                        "bias": torch.tensor([1.1], dtype=torch.float32),
                    },
                    {
                        "weight": torch.tensor([[12.0, 12.0]], dtype=torch.float32),
                        "bias": torch.tensor([1.2], dtype=torch.float32),
                    },
                ],
            },
        }
    }
    det_lora._versioned_adapter_calibrators = {
        "tank": {
            "v1": {"temperature": 1.0, "bias": 0.1, "positive_count": 1, "negative_count": 1},
            "v2": {"temperature": 1.0, "bias": 0.3, "positive_count": 1, "negative_count": 1},
        }
    }
    det_lora._versioned_score_banks = {
        "tank": {
            "v1": {"positive_scores": [0.8], "negative_scores": [0.2]},
            "v2": {"positive_scores": [0.9], "negative_scores": [0.3]},
        }
    }
    det_lora._adapter_paths = {}
    det_lora._adapter_calibrators = {}
    det_lora._score_banks = {}
    det_lora._class_head_rows = {}
    det_lora._class_head_states = {}
    det_lora._global_head_state = None
    det_lora._loaded_eval_adapters = {}
    det_lora._stability_anchor = {}
    det_lora._shared_quality_calibrator = {}

    result = det_lora.activate_adapter_version("tank", "v1")

    assert result["active_version"] == "v1"
    assert det_lora.get_active_version("tank") == "v1"
    assert det_lora.adapters["tank"] == "/tmp/tank_v1"
    assert det_lora.calibrators["tank"]["bias"] == pytest.approx(0.1)
    assert torch.equal(
        det_lora._class_head_rows["tank"]["class_embed"]["weight"],
        torch.tensor([[7.0, 7.0]], dtype=torch.float32),
    )


def test_det_lora_remove_adapter_version_removes_class_when_last_version_is_deleted():
    detector = _HeadStateDetector(out_features=4)
    detector.variant = "stub"
    detector.base_num_classes = 2
    detector.added_classes = ["tank", "truck"]

    det_lora = object.__new__(DetLoRA)
    det_lora.detector = detector
    det_lora.default_rank = 8
    det_lora.default_alpha = 16
    det_lora.current_class = None
    det_lora._peft_applied = False
    det_lora._head_mask = type("MaskStub", (), {"frozen_indices": set()})()
    det_lora._base_head_state = {
        "class_embed": {
            "weight": torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
            "bias": torch.tensor([0.1, 0.2], dtype=torch.float32),
        },
        "enc_out_class_embed": [
            {
                "weight": torch.tensor([[3.0, 3.0], [4.0, 4.0]], dtype=torch.float32),
                "bias": torch.tensor([0.3, 0.4], dtype=torch.float32),
            },
            {
                "weight": torch.tensor([[5.0, 5.0], [6.0, 6.0]], dtype=torch.float32),
                "bias": torch.tensor([0.5, 0.6], dtype=torch.float32),
            },
        ],
    }
    tank_row = {
        "class_embed": {
            "weight": torch.tensor([[7.0, 7.0]], dtype=torch.float32),
            "bias": torch.tensor([0.7], dtype=torch.float32),
        },
        "enc_out_class_embed": [
            {
                "weight": torch.tensor([[8.0, 8.0]], dtype=torch.float32),
                "bias": torch.tensor([0.8], dtype=torch.float32),
            },
            {
                "weight": torch.tensor([[9.0, 9.0]], dtype=torch.float32),
                "bias": torch.tensor([0.9], dtype=torch.float32),
            },
        ],
    }
    truck_row = {
        "class_embed": {
            "weight": torch.tensor([[10.0, 10.0]], dtype=torch.float32),
            "bias": torch.tensor([1.0], dtype=torch.float32),
        },
        "enc_out_class_embed": [
            {
                "weight": torch.tensor([[11.0, 11.0]], dtype=torch.float32),
                "bias": torch.tensor([1.1], dtype=torch.float32),
            },
            {
                "weight": torch.tensor([[12.0, 12.0]], dtype=torch.float32),
                "bias": torch.tensor([1.2], dtype=torch.float32),
            },
        ],
    }
    det_lora._adapter_versions = {
        "tank": [
            {
                "version_id": "v1",
                "source": "add",
                "created_at": None,
                "adapter_path": "/tmp/tank_v1",
            }
        ],
        "truck": [
            {
                "version_id": "v1",
                "source": "add",
                "created_at": None,
                "adapter_path": "/tmp/truck_v1",
            }
        ],
    }
    det_lora._active_versions = {"tank": "v1", "truck": "v1"}
    det_lora._versioned_class_head_rows = {
        "tank": {"v1": tank_row},
        "truck": {"v1": truck_row},
    }
    det_lora._versioned_adapter_calibrators = {
        "tank": {"v1": {"temperature": 1.0, "bias": 0.1, "positive_count": 1, "negative_count": 1}},
        "truck": {
            "v1": {"temperature": 1.0, "bias": 0.2, "positive_count": 1, "negative_count": 1}
        },
    }
    det_lora._versioned_score_banks = {
        "tank": {"v1": {"positive_scores": [0.8], "negative_scores": [0.2]}},
        "truck": {"v1": {"positive_scores": [0.85], "negative_scores": [0.25]}},
    }
    det_lora.trained_classes = ["tank", "truck"]
    det_lora._adapter_paths = {}
    det_lora._adapter_calibrators = {}
    det_lora._score_banks = {}
    det_lora._class_head_rows = {}
    det_lora._class_head_states = {}
    det_lora._global_head_state = None
    det_lora._loaded_eval_adapters = {}
    det_lora._stability_anchor = {}
    det_lora._shared_quality_calibrator = {}

    result = det_lora.remove_adapter_version("tank", version_id="v1")

    assert result["active_version"] is None
    assert det_lora.trained_classes == ["truck"]
    assert det_lora.detector.added_classes == ["truck"]
    assert "tank" not in det_lora.adapter_versions
    assert det_lora.adapters["truck"] == "/tmp/truck_v1"
    assert det_lora._global_head_state["class_embed"]["weight"].shape[0] == 3


def test_adapter_sdk_inspect_checkpoint_reports_versions(tmp_path):
    detector = _HeadStateDetector(out_features=3)
    detector.variant = "stub"
    detector.added_classes = ["tank"]
    detector.base_num_classes = 91
    source_adapter_dir = tmp_path / "source_adapters" / "lora_tank"
    source_adapter_dir.mkdir(parents=True)
    (source_adapter_dir / "adapter_model.safetensors").write_bytes(b"adapter")

    det_lora = object.__new__(DetLoRA)
    det_lora.detector = detector
    det_lora.default_rank = 8
    det_lora.default_alpha = 16
    det_lora.trained_classes = ["tank"]
    det_lora.current_class = None
    det_lora._peft_applied = False
    det_lora._head_mask = type("MaskStub", (), {"frozen_indices": {91}})()
    det_lora._adapter_paths = {"tank": str(source_adapter_dir)}
    det_lora._adapter_calibrators = {
        "tank": {"temperature": 1.0, "bias": 0.2, "positive_count": 1, "negative_count": 1}
    }
    det_lora._score_banks = {"tank": {"positive_scores": [0.8], "negative_scores": [0.2]}}
    det_lora._shared_quality_calibrator = {"identity": False, "bias": 0.3}
    det_lora._class_head_states = {}
    det_lora._class_head_rows = {}
    det_lora._global_head_state = None

    det_lora.save_all(str(tmp_path))
    summary = AdapterSDK.inspect_checkpoint(tmp_path)

    assert summary["detector_variant"] == "stub"
    assert summary["active_versions"] == {"tank": "v1"}
    assert summary["adapter_versions"]["tank"][0]["version_id"] == "v1"
    assert summary["shared_quality_calibrator"]["enabled"] is True
    assert summary["shared_quality_calibrator"]["positive_count"] is None
