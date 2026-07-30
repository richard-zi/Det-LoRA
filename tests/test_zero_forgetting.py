"""
Tests for true zero forgetting in Det-LoRA.

Verifies that:
1. Only the NEW head neuron is trainable per task (all others gradient-masked)
2. COCO weights (neurons 0-90) remain byte-identical across all tasks
3. Previously trained class neurons remain unchanged after subsequent tasks
4. Backbone, bbox_embed stay frozen
5. Gradient mask actually zeros out gradients during backward pass
6. AdamW weight decay doesn't drift frozen neurons (param group fix)
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from det_lora.model.det_lora import DetLoRA
from det_lora.model.detector import RFDETRDetector

BASE_HEAD_SIZE = 91  # COCO (90 classes + 1 no-object)


@pytest.fixture(scope="module")
def detector():
    """Load RF-DETR once for all tests on CPU for portability."""
    return RFDETRDetector(variant="medium", device=torch.device("cpu"))


@pytest.fixture(scope="module")
def coco_snapshot(detector):
    """Snapshot of original COCO head weights (first 91 neurons only)."""
    inner = detector._get_inner_model()
    return {
        "class_embed_weight": inner.class_embed.weight.data[:BASE_HEAD_SIZE].clone(),
        "class_embed_bias": (
            inner.class_embed.bias.data[:BASE_HEAD_SIZE].clone()
            if inner.class_embed.bias is not None
            else None
        ),
        "enc_out": [
            layer.weight.data[:BASE_HEAD_SIZE].clone()
            for layer in inner.transformer.enc_out_class_embed
        ],
    }


def _cleanup(det_lora, detector):
    """Remove LoRA and reset for next test."""
    det_lora._head_mask.remove_hooks()
    detector.model.zero_grad(set_to_none=True)
    if det_lora._peft_applied:
        detector.model = det_lora.model.unload()
        det_lora._peft_applied = False


class TestHeadFreezing:
    """Test that only the new neuron is trainable at each task."""

    def test_task1_only_new_neuron_trainable(self, detector):
        det_lora = DetLoRA(detector)
        det_lora.add_class("tank")

        inner = detector._get_inner_model()
        head_size = inner.class_embed.out_features
        frozen = det_lora._head_mask.frozen_indices
        trainable = [i for i in range(head_size) if i not in frozen]

        assert trainable == [head_size - 1]
        assert all(i in frozen for i in range(BASE_HEAD_SIZE))

        _cleanup(det_lora, detector)

    def test_three_tasks_sequential(self, detector):
        det_lora = DetLoRA(detector)
        classes = ["alpha", "beta", "gamma"]

        for task_idx, cls in enumerate(classes):
            det_lora.add_class(cls)
            inner = detector._get_inner_model()
            head_size = inner.class_embed.out_features
            frozen = det_lora._head_mask.frozen_indices
            trainable = [i for i in range(head_size) if i not in frozen]

            # Only the last neuron should be trainable
            assert trainable == [
                head_size - 1
            ], f"Task {task_idx + 1} ({cls}): expected [{head_size - 1}], got {trainable}"
            # All COCO neurons frozen
            assert all(i in frozen for i in range(BASE_HEAD_SIZE))
            # All previous custom neurons frozen
            for prev_idx in range(BASE_HEAD_SIZE, BASE_HEAD_SIZE + task_idx):
                assert prev_idx in frozen

            # Simulate finalize
            det_lora._head_mask.remove_hooks()
            if det_lora._peft_applied:
                detector.model = det_lora.model.unload()
                det_lora._peft_applied = False
            class_idx = detector.base_num_classes + len(det_lora.trained_classes)
            det_lora._head_mask.freeze_class(class_idx)
            det_lora.trained_classes.append(cls)
            det_lora.current_class = None


class TestCocoWeightsUnchanged:
    """Test that COCO head weights are never modified by head expansion."""

    def test_coco_weights_identical_after_add_class(self, detector, coco_snapshot):
        det_lora = DetLoRA(detector)
        det_lora.add_class("test_cls")

        inner = detector._get_inner_model()

        # class_embed: first 91 rows must match original
        curr_w = inner.class_embed.weight.data[:BASE_HEAD_SIZE]
        assert torch.equal(coco_snapshot["class_embed_weight"], curr_w), (
            f"class_embed weight changed! "
            f"max diff={(curr_w - coco_snapshot['class_embed_weight']).abs().max().item():.2e}"
        )

        if coco_snapshot["class_embed_bias"] is not None:
            curr_b = inner.class_embed.bias.data[:BASE_HEAD_SIZE]
            assert torch.equal(coco_snapshot["class_embed_bias"], curr_b)

        # enc_out_class_embed layers
        for i, orig_w in enumerate(coco_snapshot["enc_out"]):
            curr_layer_w = inner.transformer.enc_out_class_embed[i].weight.data[:BASE_HEAD_SIZE]
            assert torch.equal(orig_w, curr_layer_w), f"enc_out_class_embed[{i}] weight changed!"

        _cleanup(det_lora, detector)


class TestParameterFreezing:
    """Test that backbone and bbox_embed remain frozen."""

    def test_backbone_frozen(self, detector):
        det_lora = DetLoRA(detector)
        det_lora.add_class("freeze_test")

        for name, param in detector.model.named_parameters():
            if "backbone" in name:
                assert not param.requires_grad, f"Backbone param {name} is trainable!"

        _cleanup(det_lora, detector)

    def test_bbox_embed_frozen(self, detector):
        det_lora = DetLoRA(detector)
        det_lora.add_class("bbox_test")

        for name, param in detector.model.named_parameters():
            if "bbox_embed" in name:
                assert not param.requires_grad, f"bbox_embed param {name} is trainable!"

        _cleanup(det_lora, detector)

    def test_only_lora_and_head_trainable(self, detector):
        det_lora = DetLoRA(detector)
        det_lora.add_class("groups_test")

        trainable_groups = set()
        for name, param in detector.model.named_parameters():
            if param.requires_grad:
                if "lora" in name.lower():
                    trainable_groups.add("lora")
                elif "class_embed" in name:
                    trainable_groups.add("class_embed")
                else:
                    trainable_groups.add(name)

        assert trainable_groups == {"lora", "class_embed"}

        _cleanup(det_lora, detector)


class TestGradientMask:
    """Test that gradient masking actually zeros frozen neuron gradients."""

    def test_gradient_zeros_for_frozen_neurons(self, detector):
        det_lora = DetLoRA(detector)
        det_lora.add_class("grad_test")

        inner = detector._get_inner_model()

        dummy_input = torch.randn(1, 3, 576, 576, device=detector.device)
        samples = detector.prepare_input(dummy_input)
        outputs = detector.model(samples)
        loss = outputs["pred_logits"].sum()
        loss.backward()

        # Frozen neurons must have zero gradients
        if inner.class_embed.weight.grad is not None:
            for idx in det_lora._head_mask.frozen_indices:
                if idx < inner.class_embed.weight.grad.shape[0]:
                    grad_norm = inner.class_embed.weight.grad[idx].abs().max().item()
                    assert (
                        grad_norm == 0.0
                    ), f"Frozen neuron {idx} has non-zero gradient: {grad_norm:.2e}"

            # New neuron must have NON-zero gradient
            new_idx = inner.class_embed.out_features - 1
            new_grad_norm = inner.class_embed.weight.grad[new_idx].abs().max().item()
            assert (
                new_grad_norm > 0.0
            ), f"New neuron {new_idx} has zero gradient - it should be trainable!"

        _cleanup(det_lora, detector)


class TestStackedLoRA:
    """Test that extend_class (Stacked LoRA) freezes correctly."""

    def test_extend_only_target_neuron_trainable(self, detector):
        det_lora = DetLoRA(detector)

        # Train tank first
        det_lora.add_class("stacked_tank")
        det_lora._head_mask.remove_hooks()
        if det_lora._peft_applied:
            detector.model = det_lora.model.unload()
            det_lora._peft_applied = False
        class_idx = detector.base_num_classes + len(det_lora.trained_classes)
        det_lora._head_mask.freeze_class(class_idx)
        det_lora.trained_classes.append("stacked_tank")
        det_lora.current_class = None

        # Train truck
        det_lora.add_class("stacked_truck")
        det_lora._head_mask.remove_hooks()
        if det_lora._peft_applied:
            detector.model = det_lora.model.unload()
            det_lora._peft_applied = False
        class_idx = detector.base_num_classes + len(det_lora.trained_classes)
        det_lora._head_mask.freeze_class(class_idx)
        det_lora.trained_classes.append("stacked_truck")
        det_lora.current_class = None

        # Now EXTEND tank (Stacked LoRA)
        det_lora.extend_class("stacked_tank")

        inner = detector._get_inner_model()
        head_size = inner.class_embed.out_features
        frozen = det_lora._head_mask.frozen_indices
        tank_idx = detector.base_num_classes + det_lora.trained_classes.index("stacked_tank")
        trainable = [i for i in range(head_size) if i not in frozen]

        # Only tank neuron should be trainable
        assert trainable == [tank_idx], f"Expected only [{tank_idx}] (tank), got {trainable}"
        # COCO frozen
        assert all(i in frozen for i in range(BASE_HEAD_SIZE))
        # Truck neuron also frozen
        truck_idx = detector.base_num_classes + det_lora.trained_classes.index("stacked_truck")
        assert truck_idx in frozen

        _cleanup(det_lora, detector)

    def test_extend_warm_starts_existing_adapter_and_tracks_stability(self, detector, tmp_path):
        det_lora = DetLoRA(detector)
        class_name = "retained_extension_class"

        det_lora.add_class(class_name)
        det_lora.finalize_task(save_dir=str(tmp_path))

        det_lora.extend_class(class_name)

        assert det_lora._peft_applied is True
        assert det_lora.stability_loss().item() == pytest.approx(0.0)

        first_trainable = next(param for param in det_lora.get_trainable_params())
        with torch.no_grad():
            first_trainable.add_(0.01)

        assert det_lora.stability_loss().item() > 0.0
        class_idx = detector.base_num_classes + det_lora.trained_classes.index(class_name)
        det_lora.finalize_task(save_dir=str(tmp_path / "extended"))
        assert class_idx in det_lora._head_mask.frozen_indices

        _cleanup(det_lora, detector)


class TestWeightIntegrityAfterTrainingStep:
    """Test that frozen weights don't change after optimizer.step()."""

    def test_coco_unchanged_with_correct_param_groups(self, detector, coco_snapshot):
        """With weight_decay=0 on head, COCO weights must be byte-identical."""
        det_lora = DetLoRA(detector)
        det_lora.add_class("optim_test")

        inner = detector._get_inner_model()
        pre_step_weight = inner.class_embed.weight.data[:BASE_HEAD_SIZE].clone()

        # Build optimizer with correct param groups (same as train.py)
        lora_params = []
        head_params = []
        for name, param in det_lora.model.named_parameters():
            if not param.requires_grad:
                continue
            if "class_embed" in name or "enc_out_class" in name:
                head_params.append(param)
            else:
                lora_params.append(param)

        optimizer = AdamW(
            [
                {"params": lora_params, "weight_decay": 1e-4},
                {"params": head_params, "weight_decay": 0.0},
            ],
            lr=1e-3,
        )

        # Forward + backward + step
        dummy_input = torch.randn(1, 3, 576, 576, device=detector.device)
        samples = detector.prepare_input(dummy_input)
        outputs = detector.model(samples)
        loss = outputs["pred_logits"].sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # COCO weights must be EXACTLY unchanged
        post_step_weight = inner.class_embed.weight.data[:BASE_HEAD_SIZE]
        assert torch.equal(pre_step_weight, post_step_weight), (
            f"COCO weights changed after optimizer.step()! "
            f"max diff={(post_step_weight - pre_step_weight).abs().max().item():.2e}"
        )

        _cleanup(det_lora, detector)

    def test_adamw_with_wd_would_drift_without_fix(self, detector):
        """Demonstrate that naive AdamW (wd on head) WOULD drift COCO weights."""
        det_lora = DetLoRA(detector)
        det_lora.add_class("drift_test")

        inner = detector._get_inner_model()
        pre_step_weight = inner.class_embed.weight.data[:BASE_HEAD_SIZE].clone()

        # WRONG: weight_decay on ALL trainable params (the old bug)
        all_trainable = det_lora.get_trainable_params()
        optimizer = AdamW(all_trainable, lr=1e-3, weight_decay=1e-2)

        dummy_input = torch.randn(1, 3, 576, 576, device=detector.device)
        samples = detector.prepare_input(dummy_input)
        outputs = detector.model(samples)
        loss = outputs["pred_logits"].sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        post_step_weight = inner.class_embed.weight.data[:BASE_HEAD_SIZE]
        max_diff = (post_step_weight - pre_step_weight).abs().max().item()

        # With wd > 0 on head, weights WILL drift (this proves the bug existed)
        assert (
            max_diff > 0
        ), "Expected COCO weight drift with naive AdamW+wd, but weights didn't change"

        _cleanup(det_lora, detector)


class TestLoraTargetPresets:
    """Test that named adapter footprints attach LoRA exactly where intended."""

    @staticmethod
    def _lora_wrapped_module_names(model: nn.Module) -> set:
        return {name for name, module in model.named_modules() if hasattr(module, "lora_A")}

    def test_unknown_preset_rejected(self, detector):
        with pytest.raises(ValueError, match="lora_target_preset"):
            DetLoRA(detector, lora_target_preset="does_not_exist")

    def test_default_preset_leaves_box_path_frozen(self, detector):
        det_lora = DetLoRA(detector)
        det_lora.add_class("preset_default_probe")

        wrapped = self._lora_wrapped_module_names(det_lora.model)
        assert wrapped, "No LoRA modules attached at all"
        assert not any("bbox_embed" in name for name in wrapped)
        assert not any("sampling_offsets" in name for name in wrapped)

        _cleanup(det_lora, detector)

    def test_localization_box_preset_adapts_box_path(self, detector):
        det_lora = DetLoRA(detector, lora_target_preset="localization_box")
        det_lora.add_class("preset_box_probe")

        wrapped = self._lora_wrapped_module_names(det_lora.model)
        for expected_suffix in (
            "cross_attn.sampling_offsets",
            "cross_attn.attention_weights",
            "bbox_embed.layers.0",
            "bbox_embed.layers.2",
        ):
            assert any(name.endswith(expected_suffix) for name in wrapped), (
                f"Expected a LoRA-wrapped module ending in '{expected_suffix}', "
                f"got: {sorted(wrapped)}"
            )
        # Backbone and encoder-output box heads must stay untouched
        assert not any("backbone" in name for name in wrapped)
        assert not any("enc_out_bbox_embed" in name for name in wrapped)
        # Base weights stay frozen; only LoRA deltas train
        inner_bbox_layer = [
            module
            for name, module in det_lora.model.named_modules()
            if name.endswith("bbox_embed.layers.0") and hasattr(module, "base_layer")
        ][0]
        assert not inner_bbox_layer.base_layer.weight.requires_grad

        _cleanup(det_lora, detector)


class TestClLoraSharedAdapter:
    """CL-LoRA mode: task-shared adapter with fixed orthogonal down-projection."""

    def test_dora_combination_rejected(self, detector):
        with pytest.raises(ValueError, match="shared"):
            DetLoRA(detector, use_shared_adapter=True, use_dora=True)

    def test_shared_down_is_orthogonal_and_frozen(self, detector):
        det_lora = DetLoRA(detector, use_shared_adapter=True)
        det_lora.add_class("cl_probe")

        shared_modules = list(det_lora._iter_shared_lora_modules())
        assert shared_modules, "No shared adapter modules attached"
        for _, module in shared_modules:
            down = module.lora_A["shared"].weight
            up = module.lora_B["shared"].weight
            gram = down @ down.T
            identity = torch.eye(gram.shape[0], device=gram.device)
            assert torch.allclose(gram, identity, atol=1e-4), "down-projection not orthonormal"
            assert not down.requires_grad, "shared down-projection must stay frozen"
            assert up.requires_grad, "shared up-projection must be trainable"
            assert torch.all(up == 0), "shared up-projection must start at zero"

        # Class adapter and shared adapter are both active
        first_module = shared_modules[0][1]
        assert set(first_module.active_adapters) == {"default", "shared"}

        _cleanup(det_lora, detector)

    def test_drift_loss_zero_without_anchor_positive_after_drift(self, detector):
        det_lora = DetLoRA(detector, use_shared_adapter=True)
        det_lora.add_class("cl_drift_probe")

        assert det_lora.shared_drift_loss().item() == 0.0

        # Simulate a trained shared adapter, re-anchor, then drift
        for _, module in det_lora._iter_shared_lora_modules():
            with torch.no_grad():
                module.lora_B["shared"].weight.add_(0.5)
        det_lora._capture_shared_anchor()
        assert det_lora.shared_drift_loss().item() == 0.0
        for _, module in det_lora._iter_shared_lora_modules():
            with torch.no_grad():
                module.lora_B["shared"].weight.add_(0.1)
        assert det_lora.shared_drift_loss().item() > 0.0

        _cleanup(det_lora, detector)

    def test_extension_keeps_shared_frozen(self, detector, tmp_path):
        det_lora = DetLoRA(detector, use_shared_adapter=True)
        det_lora.add_class("cl_extend_probe")
        det_lora.finalize_task(save_dir=str(tmp_path / "adapters"))
        assert det_lora._shared_adapter_dir is not None
        assert (tmp_path / "adapters" / "_shared" / "shared").exists()

        det_lora.extend_class("cl_extend_probe")
        shared_params = [
            param for name, param in det_lora.model.named_parameters() if ".shared." in name
        ]
        assert shared_params, "shared adapter missing during Track-B extension"
        assert all(
            not p.requires_grad for p in shared_params
        ), "Track B must not train the shared adapter"

        _cleanup(det_lora, detector)
