"""Tests for the post-hoc cross-adapter conflict gate (third design iteration)."""

import numpy as np
import pytest

from det_lora.evaluation.conflict_gate import apply_conflict_gate, fit_pair_gate
from det_lora.model.det_lora import DetLoRA
from tests.test_data_and_metrics import _HeadStateDetector

BOX = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)


def _make_training_predictions(n_per_class=70, dim=4, seed=0):
    """Synthetic conflict objects: both class adapters fire on the same box; the
    cross-adapter embedding cleanly separates the true class."""
    rng = np.random.default_rng(seed)
    predictions, ground_truths = [], []
    for true_class in (91, 92):
        for _ in range(n_per_class):
            if true_class == 91:
                emb91 = np.array([1.0, 1.0, 0.0, 0.0]) + rng.normal(0, 0.05, dim)
                emb92 = np.zeros(dim) + rng.normal(0, 0.05, dim)
            else:
                emb91 = np.zeros(dim) + rng.normal(0, 0.05, dim)
                emb92 = np.array([1.0, 1.0, 0.0, 0.0]) + rng.normal(0, 0.05, dim)
            predictions.append(
                {
                    "scores": np.array([0.8, 0.7], dtype=np.float32),
                    "boxes": np.stack([BOX, BOX]),
                    "labels": np.array([91, 92]),
                    "quality_features": np.stack([emb91, emb92]).astype(np.float32),
                }
            )
            ground_truths.append({"boxes": BOX[None, :], "labels": np.array([true_class])})
    return predictions, ground_truths


def test_fit_pair_gate_builds_classifier_for_confusable_pair():
    preds, gts = _make_training_predictions()
    state = fit_pair_gate(preds, gts, class_ids=[91, 92])
    assert "91,92" in state["pairs"]
    entry = state["pairs"]["91,92"]
    assert entry["classes"] == [91, 92]
    assert entry["means"][0].shape[0] == 8  # concat of two 4-dim embeddings


def test_gate_suppresses_higher_scoring_wrong_class():
    preds, gts = _make_training_predictions()
    state = fit_pair_gate(preds, gts, class_ids=[91, 92])

    # An object whose embedding says class 91, but the 92 adapter fires HIGHER.
    conflict = {
        "scores": np.array([0.5, 0.9], dtype=np.float32),  # [91-det, 92-det]
        "boxes": np.stack([BOX, BOX]),
        "labels": np.array([91, 92]),
        "quality_features": np.stack(
            [
                np.array([1.0, 1.0, 0.0, 0.0]),  # 91 adapter sees an in-class object
                np.zeros(4),  # 92 adapter sees an out-of-class object
            ]
        ).astype(np.float32),
    }
    gated = apply_conflict_gate([conflict], state, penalty=0.0)[0]
    # The (wrong) higher-scoring class-92 detection is suppressed; class 91 is kept.
    assert gated["scores"][1] == pytest.approx(0.0)
    assert gated["scores"][0] == pytest.approx(0.5)


def test_gate_leaves_non_conflicting_detections_untouched():
    preds, gts = _make_training_predictions()
    state = fit_pair_gate(preds, gts, class_ids=[91, 92])
    # Only one class fires -> no conflict -> scores unchanged.
    single = {
        "scores": np.array([0.8], dtype=np.float32),
        "boxes": BOX[None, :],
        "labels": np.array([91]),
        "quality_features": np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    }
    gated = apply_conflict_gate([single], state, penalty=0.0)[0]
    assert gated["scores"][0] == pytest.approx(0.8)


def test_det_lora_persists_and_restores_conflict_gate(tmp_path):
    state = {
        "pairs": {
            "91,92": {
                "classes": [91, 92],
                "means": [np.zeros(8, np.float32), np.ones(8, np.float32)],
                "precisions": [np.eye(8, dtype=np.float32), np.eye(8, dtype=np.float32)],
                "counts": [70, 70],
            }
        },
        "floor": 0.1,
        "cluster_iou": 0.5,
        "resolve_tau": 0.3,
        "penalty": 0.5,
    }

    def _stub(detector):
        d = object.__new__(DetLoRA)
        d.detector = detector
        d.default_rank = 8
        d.default_alpha = 16
        d.trained_classes = ["tank"]
        d.current_class = None
        d._peft_applied = False
        d._adapter_paths = {}
        d._class_head_states = {}
        d._global_head_state = None
        return d

    save_detector = _HeadStateDetector(out_features=3)
    save_detector.variant = "stub"
    save_detector.added_classes = ["tank"]
    save_detector.base_num_classes = 91
    saver = _stub(save_detector)
    saver._head_mask = type("MaskStub", (), {"frozen_indices": {91}})()
    saver._conflict_gate = state
    saver.save_all(str(tmp_path))
    assert (tmp_path / "conflict_gate.pt").exists()

    load_detector = _HeadStateDetector(out_features=3)
    load_detector.variant = "stub"
    load_detector.added_classes = ["tank"]
    load_detector.base_num_classes = 91
    loader = _stub(load_detector)
    loader._head_mask = type(
        "MaskStub",
        (),
        {
            "__init__": lambda self: setattr(self, "frozen_indices", set()),
            "freeze_class": lambda self, idx: self.frozen_indices.add(idx),
            "remove_hooks": lambda self: None,
            "register_hooks": lambda self, inner: None,
        },
    )()
    loader.load_all(str(tmp_path))
    assert "91,92" in loader.conflict_gate["pairs"]
    np.testing.assert_allclose(loader.conflict_gate["pairs"]["91,92"]["means"][1], np.ones(8))
