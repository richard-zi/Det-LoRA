"""
Det-LoRA: LoRA-based Continual Learning for Object Detection
=============================================================

Architecture: Independent Per-Class LoRA + Gradient-Masked Head

For each new class:
1. Apply LoRA to decoder attention layers on the FROZEN base model
2. Expand classification head by 1 neuron (gradient-masked for old neurons)
3. Train LoRA + new head neuron
4. Save LoRA adapter to disk + unload() (base model restored to original)
5. Head accumulates trained neurons across tasks

Parameter overwriting is strongly reduced because:
- Base model is NEVER modified (LoRA applied and removed, not merged)
- Old head neurons are gradient-masked (gradients zeroed during backprop)
- Each adapter is independently trained on the same frozen base
- At inference: per-class adapter activation, then calibrated score merging
"""

import json
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from det_lora.model.detector import LORA_TARGET_PRESETS, RFDETRDetector


class _HeadGradientMask:
    """Zeros out gradients for frozen class indices in the classification head."""

    def __init__(self):
        self.frozen_indices: Set[int] = set()
        self._hooks = []

    def freeze_class(self, class_idx: int) -> None:
        self.frozen_indices.add(class_idx)

    def register_hooks(self, inner_model: nn.Module) -> None:
        self.remove_hooks()
        self._register_on_layer(inner_model.class_embed)
        for layer in inner_model.transformer.enc_out_class_embed:
            self._register_on_layer(layer)

    def _register_on_layer(self, layer: nn.Linear) -> None:
        if layer.weight.requires_grad:
            self._hooks.append(layer.weight.register_hook(self._mask_grad))
        if layer.bias is not None and layer.bias.requires_grad:
            self._hooks.append(layer.bias.register_hook(self._mask_grad))

    def _mask_grad(self, grad: torch.Tensor) -> torch.Tensor:
        if not self.frozen_indices:
            return grad
        grad = grad.clone()
        for idx in self.frozen_indices:
            if idx < grad.shape[0]:
                grad[idx] = 0
        return grad

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []


class DetLoRA:
    """
    Det-LoRA: Parameter-Efficient Continual Learning for Object Detection.

    Each class gets an independent LoRA adapter trained on the frozen base model.
    The base model is NEVER modified. Head neurons accumulate across tasks.

    Args:
        detector: RFDETRDetector instance
        default_rank: Default LoRA rank
        default_alpha: Default LoRA alpha scaling
    """

    def __init__(
        self,
        detector: RFDETRDetector,
        default_rank: int = 8,
        default_alpha: int = 16,
        lora_target_preset: str = "default",
        use_dora: bool = False,
        use_shared_adapter: bool = False,
    ):
        if lora_target_preset not in LORA_TARGET_PRESETS:
            raise ValueError(
                f"Unknown lora_target_preset '{lora_target_preset}'. "
                f"Choose from: {sorted(LORA_TARGET_PRESETS)}"
            )
        if use_shared_adapter and use_dora:
            raise ValueError(
                "use_shared_adapter requires vanilla LoRA: the CL-LoRA shared adapter "
                "fixes an orthogonal down-projection, which DoRA's magnitude "
                "decomposition would invalidate."
            )
        self.detector = detector
        self.default_rank = default_rank
        self.default_alpha = default_alpha
        self.lora_target_preset = lora_target_preset
        self.use_dora = use_dora
        self.use_shared_adapter = use_shared_adapter
        self._shared_adapter_dir: Optional[str] = None
        self._shared_anchor: Dict[str, tuple] = {}

        self.trained_classes: List[str] = []
        self.current_class: Optional[str] = None
        self._peft_applied = False
        self._head_mask = _HeadGradientMask()
        self._adapter_paths: Dict[str, str] = {}  # class_name -> saved adapter path
        self._adapter_calibrators: Dict[str, Dict[str, float]] = {}
        self._score_banks: Dict[str, Dict[str, List[float]]] = {}
        self._shared_quality_calibrator: Dict[str, Any] = {}
        self._adapter_arbitration_state: Dict[str, Any] = {}
        self._conflict_gate: Dict[str, Any] = {}
        self._adapter_versions: Dict[str, List[Dict[str, Any]]] = {}
        self._active_versions: Dict[str, str] = {}
        self._versioned_class_head_rows: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._versioned_adapter_calibrators: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._versioned_score_banks: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        self._base_head_state: Optional[Dict[str, Any]] = None
        self._class_head_rows: Dict[str, Dict[str, Any]] = {}
        self._class_head_states: Dict[str, Dict[str, Any]] = {}
        self._global_head_state: Optional[Dict[str, Any]] = None
        self._loaded_eval_adapters: Dict[str, str] = {}
        self._stability_anchor: Dict[str, torch.Tensor] = {}
        self._checkpoint_dir: Optional[str] = None

        self.detector.freeze_all()
        self._base_head_state = self._capture_head_state()

    @property
    def model(self) -> nn.Module:
        return self.detector.model

    @property
    def device(self) -> torch.device:
        return self.detector.device

    @property
    def adapters(self) -> Dict[str, str]:
        """Saved adapters: {class_name -> path}."""
        return self._adapter_paths

    @property
    def calibrators(self) -> Dict[str, Dict[str, float]]:
        """Per-adapter score calibration parameters."""
        return self._adapter_calibrators

    @property
    def score_banks(self) -> Dict[str, Dict[str, List[float]]]:
        """Compact positive/negative score samples per class."""
        return self._score_banks

    @property
    def shared_quality_calibrator(self) -> Dict[str, Any]:
        """Return the fitted shared quality/objectness calibrator state."""
        return self._shared_quality_calibrator

    @property
    def adapter_arbitration_state(self) -> Dict[str, Any]:
        """Return the fitted compact adapter-arbitration state."""
        return self._adapter_arbitration_state

    @property
    def conflict_gate(self) -> Dict[str, Any]:
        """Return the fitted post-hoc cross-adapter conflict-gate state."""
        return self._conflict_gate

    @property
    def adapter_versions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Version metadata per class."""
        return self._adapter_versions

    @property
    def active_versions(self) -> Dict[str, str]:
        """Currently active version per class."""
        return self._active_versions

    def set_shared_quality_calibrator(self, calibrator: Dict[str, Any]) -> None:
        """Replace the current shared quality/objectness calibrator."""
        self._shared_quality_calibrator = calibrator

    def set_conflict_gate(self, gate_state: Dict[str, Any]) -> None:
        """Replace the current post-hoc cross-adapter conflict-gate state."""
        self._conflict_gate = gate_state

    def set_adapter_arbitration_state(self, state: Dict[str, Any]) -> None:
        """Replace the compact adapter-arbitration state."""
        self._adapter_arbitration_state = state

    def list_classes(self) -> List[Dict[str, Any]]:
        """Summarize the currently known classes and their active versions."""
        ordered_classes = list(dict.fromkeys(self.trained_classes + sorted(self._adapter_versions)))
        result = []
        for class_name in ordered_classes:
            versions = self._adapter_versions.get(class_name, [])
            result.append(
                {
                    "class_name": class_name,
                    "active_version": self._active_versions.get(class_name),
                    "num_versions": len(versions),
                    "versions": [entry["version_id"] for entry in versions],
                }
            )
        return result

    def list_adapter_versions(
        self, class_name: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return adapter version metadata, marking the active version per class."""
        classes = [class_name] if class_name is not None else sorted(self._adapter_versions)
        listed: Dict[str, List[Dict[str, Any]]] = {}
        for name in classes:
            listed[name] = [
                {
                    **entry,
                    "active": entry["version_id"] == self._active_versions.get(name),
                }
                for entry in self._adapter_versions.get(name, [])
            ]
        return listed

    def get_active_version(self, class_name: str) -> Optional[str]:
        """Return the active version ID for a class."""
        return self._active_versions.get(class_name)

    def _ensure_runtime_state(self) -> None:
        """Backfill optional attributes for legacy checkpoints and test stubs."""
        if not hasattr(self, "lora_target_preset"):
            self.lora_target_preset = "default"
        if not hasattr(self, "use_dora"):
            self.use_dora = False
        if not hasattr(self, "use_shared_adapter"):
            self.use_shared_adapter = False
        if not hasattr(self, "_shared_adapter_dir"):
            self._shared_adapter_dir = None
        if not hasattr(self, "_shared_anchor"):
            self._shared_anchor = {}
        if not hasattr(self, "_adapter_calibrators"):
            self._adapter_calibrators = {}
        if not hasattr(self, "_score_banks"):
            self._score_banks = {}
        if not hasattr(self, "_shared_quality_calibrator"):
            self._shared_quality_calibrator = {}
        if not hasattr(self, "_adapter_arbitration_state"):
            self._adapter_arbitration_state = {}
        if not hasattr(self, "_conflict_gate"):
            self._conflict_gate = {}
        if not hasattr(self, "_adapter_versions"):
            self._adapter_versions = {}
        if not hasattr(self, "_active_versions"):
            self._active_versions = {}
        if not hasattr(self, "_versioned_class_head_rows"):
            self._versioned_class_head_rows = {}
        if not hasattr(self, "_versioned_adapter_calibrators"):
            self._versioned_adapter_calibrators = {}
        if not hasattr(self, "_versioned_score_banks"):
            self._versioned_score_banks = {}
        if not hasattr(self, "_base_head_state"):
            self._base_head_state = None
        if not hasattr(self, "_class_head_rows"):
            self._class_head_rows = {}
        if not hasattr(self, "_class_head_states"):
            self._class_head_states = {}
        if not hasattr(self, "_global_head_state"):
            self._global_head_state = None
        if not hasattr(self, "_loaded_eval_adapters"):
            self._loaded_eval_adapters = {}
        if not hasattr(self, "_stability_anchor"):
            self._stability_anchor = {}
        if not hasattr(self, "_checkpoint_dir"):
            self._checkpoint_dir = None

    def get_calibrator(self, class_name: str) -> Dict[str, float]:
        """Return a class calibrator or the identity transform."""
        calibrator = self._adapter_calibrators.get(class_name)
        if calibrator:
            return calibrator
        return {
            "temperature": 1.0,
            "bias": 0.0,
            "positive_count": 0.0,
            "negative_count": 0.0,
        }

    def _apply_probability_calibration(
        self,
        scores: torch.Tensor,
        temperature: float,
        bias: float,
    ) -> torch.Tensor:
        """Apply the learned temperature+bias transform in probability space."""
        logits = torch.logit(scores.clamp(1e-6, 1 - 1e-6))
        return torch.sigmoid(logits / temperature + bias)

    def calibrate_scores(self, class_name: str, scores: torch.Tensor) -> torch.Tensor:
        """Apply a per-class temperature+bias calibration in probability space."""
        calibrator = self.get_calibrator(class_name)
        temperature = max(float(calibrator.get("temperature", 1.0)), 1e-3)
        bias = float(calibrator.get("bias", 0.0))
        return self._apply_probability_calibration(scores, temperature, bias)

    def record_score_bank(
        self,
        class_name: str,
        positive_scores: Optional[List[float]] = None,
        negative_scores: Optional[List[float]] = None,
        max_bank_size: int = 2048,
    ) -> Dict[str, List[float]]:
        """Append compact score samples used for per-adapter calibration."""
        bank = self._score_banks.setdefault(
            class_name,
            {"positive_scores": [], "negative_scores": []},
        )
        if positive_scores:
            bank["positive_scores"].extend(float(score) for score in positive_scores)
            bank["positive_scores"] = self._compress_scores(
                bank["positive_scores"],
                limit=max_bank_size,
            )
        if negative_scores:
            bank["negative_scores"].extend(float(score) for score in negative_scores)
            bank["negative_scores"] = self._compress_scores(
                bank["negative_scores"],
                limit=max_bank_size,
            )
        self._sync_active_version_state(class_name)
        return bank

    def fit_calibrator(
        self,
        class_name: str,
        steps: int = 250,
        lr: float = 0.05,
    ) -> Dict[str, float]:
        """
        Fit a lightweight temperature+bias calibrator from stored score samples.

        The calibrator learns to separate matched positives from negatives
        accumulated on later tasks, without replaying old images.
        """
        bank = self._score_banks.get(class_name, {})
        positives = [float(score) for score in bank.get("positive_scores", [])]
        negatives = [float(score) for score in bank.get("negative_scores", [])]

        if len(positives) < 4 or len(negatives) < 4:
            calibrator = {
                "temperature": 1.0,
                "bias": 0.0,
                "positive_count": float(len(positives)),
                "negative_count": float(len(negatives)),
            }
            self._adapter_calibrators[class_name] = calibrator
            return calibrator

        pos = torch.tensor(positives, dtype=torch.float32, device=self.device).clamp(1e-4, 1 - 1e-4)
        neg = torch.tensor(negatives, dtype=torch.float32, device=self.device).clamp(1e-4, 1 - 1e-4)
        scores = torch.cat([pos, neg], dim=0)
        targets = torch.cat(
            [
                torch.ones_like(pos),
                torch.zeros_like(neg),
            ],
            dim=0,
        )
        weights = torch.cat(
            [
                torch.full_like(pos, 0.5 / max(len(positives), 1)),
                torch.full_like(neg, 0.5 / max(len(negatives), 1)),
            ],
            dim=0,
        )
        score_logits = torch.logit(scores)

        log_temperature = nn.Parameter(torch.zeros((), device=self.device))
        bias = nn.Parameter(torch.zeros((), device=self.device))
        optimizer = torch.optim.Adam([log_temperature, bias], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
            logits = score_logits / temperature + bias
            loss = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                weight=weights,
                reduction="sum",
            )
            # Keep the calibrator close to identity unless the data justifies otherwise.
            loss = loss + 1e-3 * (log_temperature.pow(2) + bias.pow(2))
            loss.backward()
            optimizer.step()

        fitted_temperature = float(
            torch.exp(log_temperature).clamp(0.05, 20.0).detach().cpu().item()
        )
        fitted_bias = float(bias.detach().cpu().item())

        calibrator = {
            "temperature": fitted_temperature,
            "bias": fitted_bias,
            "positive_count": float(len(positives)),
            "negative_count": float(len(negatives)),
        }
        self._adapter_calibrators[class_name] = calibrator
        self._sync_active_version_state(class_name)
        return calibrator

    def _ensure_version_store(self) -> None:
        """Backfill the versioned adapter state containers."""
        self._ensure_runtime_state()

    def _next_version_id(self, class_name: str) -> str:
        """Allocate the next monotonically increasing version for one class."""
        existing = self._adapter_versions.get(class_name, [])
        max_version = 0
        for entry in existing:
            version_id = str(entry.get("version_id", ""))
            if version_id.startswith("v"):
                try:
                    max_version = max(max_version, int(version_id[1:]))
                except ValueError:
                    continue
        return f"v{max_version + 1}"

    def _get_version_entry(self, class_name: str, version_id: str) -> Dict[str, Any]:
        """Resolve one version metadata entry."""
        for entry in self._adapter_versions.get(class_name, []):
            if entry.get("version_id") == version_id:
                return entry
        raise ValueError(f"Unknown adapter version '{class_name}:{version_id}'")

    def _record_adapter_version(
        self,
        class_name: str,
        version_id: str,
        adapter_path: Optional[str],
        source: str,
        row_state: Dict[str, Any],
    ) -> None:
        """Register one new adapter version and make it active."""
        self._adapter_versions.setdefault(class_name, [])
        self._adapter_versions[class_name].append(
            {
                "version_id": version_id,
                "source": source,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "adapter_path": adapter_path,
            }
        )
        self._versioned_class_head_rows.setdefault(class_name, {})[version_id] = row_state
        self._versioned_adapter_calibrators.setdefault(class_name, {})[version_id] = deepcopy(
            self._adapter_calibrators.get(class_name, self.get_calibrator(class_name))
        )
        self._versioned_score_banks.setdefault(class_name, {})[version_id] = deepcopy(
            self._score_banks.get(
                class_name,
                {"positive_scores": [], "negative_scores": []},
            )
        )
        self._active_versions[class_name] = version_id

    def _sync_active_version_state(self, class_name: str) -> None:
        """Persist the active runtime state back into the active version store."""
        self._ensure_runtime_state()
        version_id = self._active_versions.get(class_name)
        if not version_id:
            return
        entry = self._get_version_entry(class_name, version_id)
        entry["adapter_path"] = self._adapter_paths.get(class_name, entry.get("adapter_path"))
        if class_name in self._class_head_rows:
            self._versioned_class_head_rows.setdefault(class_name, {})[version_id] = deepcopy(
                self._class_head_rows[class_name]
            )
        self._versioned_adapter_calibrators.setdefault(class_name, {})[version_id] = deepcopy(
            self._adapter_calibrators.get(class_name, self.get_calibrator(class_name))
        )
        self._versioned_score_banks.setdefault(class_name, {})[version_id] = deepcopy(
            self._score_banks.get(
                class_name,
                {"positive_scores": [], "negative_scores": []},
            )
        )

    def _rebuild_active_snapshots(self) -> None:
        """Recompute cumulative active head snapshots from base + active rows."""
        self._class_head_states = {}
        if self._base_head_state is None:
            self._global_head_state = None
            return

        state = self._clone_head_state(self._base_head_state)
        for class_name in self.trained_classes:
            row_state = self._class_head_rows.get(class_name)
            if row_state is None:
                continue
            state = self._append_class_head_row_state(state, row_state)
            self._class_head_states[class_name] = self._clone_head_state(state)
        self._global_head_state = self._clone_head_state(state)

    def _refresh_head_mask_for_active_classes(self) -> None:
        """Keep all active finalized head rows frozen when no task is running."""
        self._head_mask.frozen_indices = set(
            range(self.detector.base_num_classes + len(self.trained_classes))
        )

    def _rebuild_active_runtime_from_versions(self) -> None:
        """Materialize active runtime maps from the version registry."""
        self._ensure_runtime_state()
        ordered_classes = list(
            dict.fromkeys(getattr(self, "trained_classes", []) + sorted(self._adapter_versions))
        )
        active_classes = [
            class_name
            for class_name in ordered_classes
            if class_name in self._active_versions
            and self._active_versions[class_name]
            and self._adapter_versions.get(class_name)
        ]

        self.trained_classes = active_classes
        self.detector.added_classes = list(active_classes)
        self._adapter_paths = {}
        self._adapter_calibrators = {}
        self._score_banks = {}
        self._class_head_rows = {}

        for class_name in active_classes:
            version_id = self._active_versions[class_name]
            entry = self._get_version_entry(class_name, version_id)
            if entry.get("adapter_path") is not None:
                self._adapter_paths[class_name] = str(entry["adapter_path"])
            self._adapter_calibrators[class_name] = deepcopy(
                self._versioned_adapter_calibrators.get(class_name, {}).get(
                    version_id,
                    self.get_calibrator(class_name),
                )
            )
            self._score_banks[class_name] = deepcopy(
                self._versioned_score_banks.get(class_name, {}).get(
                    version_id,
                    {"positive_scores": [], "negative_scores": []},
                )
            )
            row_state = self._versioned_class_head_rows.get(class_name, {}).get(version_id)
            if row_state is None:
                raise ValueError(
                    f"Missing class head row for active version '{class_name}:{version_id}'"
                )
            self._class_head_rows[class_name] = deepcopy(row_state)

        self._rebuild_active_snapshots()
        self._refresh_head_mask_for_active_classes()
        if self.current_class is None and self._global_head_state is not None:
            self._apply_head_state(self._global_head_state)

    def _bootstrap_version_store_from_active_state(self) -> None:
        """Upgrade pre-versioned in-memory state into a minimal version registry."""
        self._ensure_runtime_state()
        for class_name in self.trained_classes:
            if self._adapter_versions.get(class_name):
                continue
            version_id = self._active_versions.get(class_name, "v1")
            row_state = self._class_head_rows.get(class_name)
            if row_state is None:
                head_state = self._global_head_state or self._capture_head_state()
                class_idx = self.detector.base_num_classes + self.trained_classes.index(class_name)
                if head_state["class_embed"]["weight"].shape[0] <= class_idx:
                    class_idx = (
                        head_state["class_embed"]["weight"].shape[0]
                        - len(self.trained_classes)
                        + self.trained_classes.index(class_name)
                    )
                if head_state["class_embed"]["weight"].shape[0] > class_idx:
                    row_state = self._extract_class_head_row_state(head_state, class_idx)
            self._adapter_versions[class_name] = [
                {
                    "version_id": version_id,
                    "source": "legacy",
                    "created_at": None,
                    "adapter_path": self._adapter_paths.get(class_name),
                }
            ]
            self._active_versions[class_name] = version_id
            if row_state is not None:
                self._class_head_rows[class_name] = deepcopy(row_state)
                self._versioned_class_head_rows.setdefault(class_name, {})[version_id] = deepcopy(
                    row_state
                )
            self._versioned_adapter_calibrators.setdefault(class_name, {})[version_id] = deepcopy(
                self._adapter_calibrators.get(class_name, self.get_calibrator(class_name))
            )
            self._versioned_score_banks.setdefault(class_name, {})[version_id] = deepcopy(
                self._score_banks.get(
                    class_name,
                    {"positive_scores": [], "negative_scores": []},
                )
            )

    def activate_adapter_version(self, class_name: str, version_id: str) -> Dict[str, Any]:
        """Activate a stored version of one class and rebuild the active runtime state."""
        self._ensure_runtime_state()
        self._get_version_entry(class_name, version_id)
        if self._loaded_eval_adapters or self._peft_applied:
            self.unload_adapter()
        self._active_versions[class_name] = version_id
        self._rebuild_active_runtime_from_versions()
        return {
            "class_name": class_name,
            "active_version": version_id,
            "trained_classes": list(self.trained_classes),
        }

    def remove_adapter_version(
        self,
        class_name: str,
        version_id: Optional[str] = None,
        reassign_active: bool = True,
    ) -> Dict[str, Any]:
        """Soft-remove one adapter version from the active checkpoint state."""
        self._ensure_runtime_state()
        current_active = self._active_versions.get(class_name)
        target_version = version_id or current_active
        if target_version is None:
            raise ValueError(f"Class '{class_name}' has no active version to remove")
        if self._loaded_eval_adapters or self._peft_applied:
            self.unload_adapter()

        versions = self._adapter_versions.get(class_name, [])
        kept_versions = [entry for entry in versions if entry.get("version_id") != target_version]
        if len(kept_versions) == len(versions):
            raise ValueError(f"Unknown adapter version '{class_name}:{target_version}'")

        self._adapter_versions[class_name] = kept_versions
        self._versioned_class_head_rows.get(class_name, {}).pop(target_version, None)
        self._versioned_adapter_calibrators.get(class_name, {}).pop(target_version, None)
        self._versioned_score_banks.get(class_name, {}).pop(target_version, None)

        if not kept_versions:
            self._adapter_versions.pop(class_name, None)
            self._active_versions.pop(class_name, None)
            self._versioned_class_head_rows.pop(class_name, None)
            self._versioned_adapter_calibrators.pop(class_name, None)
            self._versioned_score_banks.pop(class_name, None)
        elif current_active == target_version and reassign_active:
            self._active_versions[class_name] = kept_versions[-1]["version_id"]

        self._rebuild_active_runtime_from_versions()
        return {
            "class_name": class_name,
            "removed_version": target_version,
            "active_version": self._active_versions.get(class_name),
            "remaining_versions": [
                entry["version_id"] for entry in self._adapter_versions.get(class_name, [])
            ],
            "trained_classes": list(self.trained_classes),
        }

    def _compress_scores(self, scores: List[float], limit: int) -> List[float]:
        """Deterministically compress a score bank while preserving its range."""
        normalized = sorted(float(score) for score in scores)
        if len(normalized) <= limit:
            return normalized
        if limit <= 1:
            return [normalized[-1]]
        return [
            normalized[int(round(i * (len(normalized) - 1) / (limit - 1)))] for i in range(limit)
        ]

    def _capture_head_state(self) -> Dict[str, Any]:
        """Snapshot the current classification/proposal head state onto CPU."""
        inner = self.detector._get_inner_model()
        return {
            "class_embed": {
                key: value.detach().cpu().clone()
                for key, value in inner.class_embed.state_dict().items()
            },
            "enc_out_class_embed": [
                {key: value.detach().cpu().clone() for key, value in layer.state_dict().items()}
                for layer in inner.transformer.enc_out_class_embed
            ],
        }

    def _clone_head_state(self, head_state: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-clone a serialized head state so class snapshots stay independent."""
        return {
            "class_embed": {
                key: value.detach().cpu().clone()
                for key, value in head_state["class_embed"].items()
            },
            "enc_out_class_embed": [
                {key: value.detach().cpu().clone() for key, value in layer_state.items()}
                for layer_state in head_state["enc_out_class_embed"]
            ],
        }

    def _extract_base_head_state(self, head_state: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only the frozen COCO/base rows of a serialized head state."""
        base_num_classes = self.detector.base_num_classes
        class_embed = {
            key: value[:base_num_classes].detach().cpu().clone()
            for key, value in head_state["class_embed"].items()
        }
        enc_out_class_embed = [
            {
                key: value[:base_num_classes].detach().cpu().clone()
                for key, value in layer_state.items()
            }
            for layer_state in head_state["enc_out_class_embed"]
        ]
        return {
            "class_embed": class_embed,
            "enc_out_class_embed": enc_out_class_embed,
        }

    def _extract_class_head_row_state(
        self, head_state: Dict[str, Any], class_idx: int
    ) -> Dict[str, Any]:
        """Extract the single appended head row associated with one incremental class."""
        return {
            "class_embed": {
                key: value[class_idx : class_idx + 1].detach().cpu().clone()
                for key, value in head_state["class_embed"].items()
            },
            "enc_out_class_embed": [
                {
                    key: value[class_idx : class_idx + 1].detach().cpu().clone()
                    for key, value in layer_state.items()
                }
                for layer_state in head_state["enc_out_class_embed"]
            ],
        }

    def _append_class_head_row_state(
        self,
        head_state: Dict[str, Any],
        row_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Append one incremental class row onto a cloned cumulative head state."""
        updated = self._clone_head_state(head_state)
        for key, value in row_state["class_embed"].items():
            updated["class_embed"][key] = torch.cat([updated["class_embed"][key], value], dim=0)
        for idx, layer_row_state in enumerate(row_state["enc_out_class_embed"]):
            for key, value in layer_row_state.items():
                updated["enc_out_class_embed"][idx][key] = torch.cat(
                    [updated["enc_out_class_embed"][idx][key], value],
                    dim=0,
                )
        return updated

    def _backfill_class_head_rows_from_snapshots(self) -> None:
        """Derive row-wise head states from older cumulative snapshots when needed."""
        for idx, class_name in enumerate(self.trained_classes):
            if class_name in self._class_head_rows:
                continue
            snapshot = self._class_head_states.get(class_name)
            if snapshot is None:
                continue
            class_idx = self.detector.base_num_classes + idx
            if snapshot["class_embed"]["weight"].shape[0] <= class_idx:
                class_idx = (
                    snapshot["class_embed"]["weight"].shape[0] - len(self.trained_classes) + idx
                )
            if class_idx < 0 or snapshot["class_embed"]["weight"].shape[0] <= class_idx:
                continue
            self._class_head_rows[class_name] = self._extract_class_head_row_state(
                snapshot, class_idx
            )

    def _build_cumulative_head_state(self, class_name: str) -> Optional[Dict[str, Any]]:
        """
        Reconstruct the cumulative head state for one class from base + per-class rows.

        The reconstructed head keeps absolute class IDs stable because it appends
        every previously seen incremental row up to the requested class in
        training order.
        """
        if self._base_head_state is None or class_name not in self.trained_classes:
            return None

        state = self._clone_head_state(self._base_head_state)
        for seen_class in self.trained_classes:
            row_state = self._class_head_rows.get(seen_class)
            if row_state is None:
                return None
            state = self._append_class_head_row_state(state, row_state)
            if seen_class == class_name:
                return state
        return None

    def _resolve_eval_head_state(self, class_name: str) -> Optional[Dict[str, Any]]:
        """Prefer reconstructed residual heads, fall back to stored snapshots."""
        head_state = self._build_cumulative_head_state(class_name)
        if head_state is not None:
            return head_state
        return self._class_head_states.get(class_name)

    def _ensure_class_head_snapshots(self) -> None:
        """
        Backfill missing per-class head snapshots from the best available global state.

        Older checkpoints predate explicit `class_heads/` snapshots. They still carry a
        final accumulated `head_weights.pt`, which is enough to keep later save/load
        cycles self-contained even if we cannot reconstruct the exact historic heads.
        """
        if self._global_head_state is None:
            return
        for class_name in self.trained_classes:
            if class_name in self._class_head_states:
                continue
            self._class_head_states[class_name] = self._clone_head_state(self._global_head_state)

    def _make_linear_from_state(self, state: Dict[str, torch.Tensor]) -> nn.Linear:
        """Recreate a Linear layer from a serialized state dict."""
        weight = state["weight"]
        out_features, in_features = weight.shape
        bias_tensor = state.get("bias")
        layer = nn.Linear(
            in_features,
            out_features,
            bias=bias_tensor is not None,
        ).to(self.device)
        layer.load_state_dict(state)
        layer.weight.requires_grad = False
        if layer.bias is not None:
            layer.bias.requires_grad = False
        return layer

    def _apply_head_state(self, head_state: Dict[str, Any]) -> None:
        """Swap in a previously captured classification/proposal head state."""
        inner = self.detector._get_inner_model()
        inner.class_embed = self._make_linear_from_state(head_state["class_embed"])
        for idx, state in enumerate(head_state["enc_out_class_embed"]):
            inner.transformer.enc_out_class_embed[idx] = self._make_linear_from_state(state)
        self.detector.criterion = self.detector._rebuild_criterion()

    def _restore_global_head_state(self) -> None:
        """Restore the accumulated shared head after per-class evaluation."""
        if self._global_head_state is not None:
            self._apply_head_state(self._global_head_state)

    def add_class(
        self,
        class_name: str,
        rank: Optional[int] = None,
        alpha: Optional[int] = None,
    ) -> str:
        """
        Add a new class for training.

        1. Expands head by 1 neuron
        2. Applies fresh LoRA on the frozen base
        3. Gradient mask protects old head neurons
        4. After training, call finalize_task() to save adapter and restore base
        """
        rank = rank or self.default_rank
        alpha = alpha or self.default_alpha

        # Expand head (adds new neuron, keeps old ones)
        self.detector.expand_classification_head(class_name)

        # Freeze ALL existing neurons (COCO + previously trained classes)
        # Only the brand new neuron (last one) will be trainable
        inner = self.detector._get_inner_model()
        new_head_size = inner.class_embed.out_features
        for idx in range(new_head_size - 1):
            self._head_mask.freeze_class(idx)

        # Freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        # Apply fresh LoRA (+ trainable CL-LoRA shared adapter in Track A)
        self._apply_lora(rank, alpha)
        self._attach_shared_adapter(trainable=True)

        # Unfreeze head (gradient mask protects old neurons)
        self._unfreeze_head()
        inner = self.detector._get_inner_model()
        self._head_mask.register_hooks(inner)

        self.current_class = class_name
        self._stability_anchor = {}

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        adapter_name = f"lora_{class_name}"
        print(
            f"[Det-LoRA] Added class '{class_name}' "
            f"(rank={rank}, trainable={trainable:,}, "
            f"frozen_neurons={sorted(self._head_mask.frozen_indices)})"
        )
        return adapter_name

    def finalize_task(self, save_dir: Optional[str] = None) -> None:
        """
        Finalize: save LoRA adapter to disk, restore base model.

        After this:
        - LoRA adapter saved to disk (can be loaded for inference)
        - Base model restored to original (unload, NOT merge)
        - Head retains trained neuron for this class
        - Ready for next class
        """
        if self.current_class is None:
            raise ValueError("No active task to finalize")

        # Drop the gradient hooks first so a failure below cannot leave stale
        # hooks behind that would double-fire on the next add_class().
        self._head_mask.remove_hooks()

        class_name = self.current_class
        is_extension = class_name in self.trained_classes
        version_id = self._next_version_id(class_name)
        if class_name in self.trained_classes:
            class_idx = self.detector.base_num_classes + self.trained_classes.index(class_name)
        else:
            class_idx = self.detector.base_num_classes + len(self.trained_classes)

        # Preserve the exact head state that produced this class before later
        # head expansions can interfere with its proposal/classification path.
        class_head_state = self._capture_head_state()
        self._class_head_states[class_name] = class_head_state
        class_row_state = self._extract_class_head_row_state(class_head_state, class_idx)
        self._class_head_rows[class_name] = class_row_state
        self._global_head_state = class_head_state

        adapter_path = self._adapter_paths.get(class_name)
        if self._peft_applied:
            # Save LoRA adapter weights
            if save_dir:
                adapter_path = str(Path(save_dir) / class_name / version_id)
                if self.use_shared_adapter:
                    # Keep the class adapter dir clean of the shared adapter;
                    # the evolving shared state is persisted separately.
                    self.model.save_pretrained(adapter_path, selected_adapters=["default"])
                    self._save_shared_adapter(Path(save_dir))
                else:
                    self.model.save_pretrained(adapter_path)
                self._adapter_paths[class_name] = adapter_path
                print(f"[Det-LoRA] Saved LoRA adapter to {adapter_path}")

            # Unload LoRA WITHOUT merging - base model restored to original
            self.detector.model = self.model.unload()
            self._peft_applied = False

        # Freeze this class's head neuron for future tasks
        self._head_mask.freeze_class(class_idx)

        # Track
        if class_name not in self.trained_classes:
            self.trained_classes.append(class_name)
        self.current_class = None
        self._stability_anchor = {}

        if adapter_path is None:
            adapter_path = self._adapter_paths.get(class_name)
        self._record_adapter_version(
            class_name=class_name,
            version_id=version_id,
            adapter_path=adapter_path,
            source="extend" if is_extension else "add",
            row_state=deepcopy(class_row_state),
        )
        self._rebuild_active_runtime_from_versions()

        # Rebuild criterion
        self.detector.criterion = self.detector._rebuild_criterion()

        print(
            f"[Det-LoRA] Finalized '{class_name}': "
            f"adapter saved, base restored, head[{class_idx}] frozen"
        )

    def _prepare_class_extension(self, class_name: str) -> int:
        """Unfreeze only the target class head row for extension training."""
        if class_name not in self.trained_classes:
            raise ValueError(f"Class '{class_name}' not trained yet.")

        class_idx = self.detector.base_num_classes + self.trained_classes.index(class_name)

        # Freeze ALL neurons, then unfreeze only the target class
        inner = self.detector._get_inner_model()
        head_size = inner.class_embed.out_features
        for idx in range(head_size):
            self._head_mask.freeze_class(idx)
        self._head_mask.frozen_indices.discard(class_idx)

        for param in self.model.parameters():
            param.requires_grad = False
        return class_idx

    def extend_class(
        self,
        class_name: str,
        rank: Optional[int] = None,
        alpha: Optional[int] = None,
    ) -> str:
        """
        Extend existing class with new data (Data-Incremental).

        Unfreezes the class's head neuron and warm-starts from the active LoRA
        version for that class.
        """
        rank = rank or self.default_rank
        alpha = alpha or self.default_alpha
        class_idx = self._prepare_class_extension(class_name)

        adapter_path = self._adapter_paths.get(class_name)
        if adapter_path and Path(adapter_path).exists():
            from peft import PeftModel

            self._reset_peft_metadata()
            if not hasattr(self.detector.model, "get_input_embeddings"):
                self.detector.model.get_input_embeddings = lambda: nn.Embedding(1, 1)
            self.detector.model = PeftModel.from_pretrained(
                self.detector.model,
                adapter_path,
                is_trainable=True,
            )
            self._peft_applied = True
        else:
            self._apply_lora(rank, alpha)
        # Track B keeps the cross-class shared adapter frozen: a data-
        # incremental extension must only touch the class-specific adapter.
        self._attach_shared_adapter(trainable=False)
        self._unfreeze_head()
        inner = self.detector._get_inner_model()
        self._head_mask.register_hooks(inner)
        self._capture_stability_anchor()

        self.current_class = class_name
        print(f"[Det-LoRA] Extending '{class_name}' (head[{class_idx}] unfrozen)")
        return f"lora_{class_name}"

    def extend_class_with_fresh_adapter(
        self,
        class_name: str,
        rank: Optional[int] = None,
        alpha: Optional[int] = None,
    ) -> str:
        """Extend an existing class by adding a fresh LoRA version."""
        rank = rank or self.default_rank
        alpha = alpha or self.default_alpha
        class_idx = self._prepare_class_extension(class_name)

        self._apply_lora(rank, alpha)
        self._attach_shared_adapter(trainable=False)
        self._unfreeze_head()
        inner = self.detector._get_inner_model()
        self._head_mask.register_hooks(inner)
        self._capture_stability_anchor()

        self.current_class = class_name
        print(
            f"[Det-LoRA] Extending '{class_name}' with fresh adapter "
            f"(head[{class_idx}] unfrozen)"
        )
        return f"lora_{class_name}"

    def get_class_id(self, class_name: str) -> int:
        """Resolve an incrementally added class name to its absolute class ID."""
        return self.detector.get_class_id(class_name)

    def _reset_peft_metadata(self) -> None:
        """Clear stale PEFT markers before attaching or reloading an adapter."""
        if hasattr(self.detector.model, "peft_config"):
            try:
                delattr(self.detector.model, "peft_config")
            except (AttributeError, TypeError):
                try:
                    self.detector.model.peft_config = {}
                except AttributeError:
                    pass
        if hasattr(self.detector.model, "_hf_peft_config_loaded"):
            self.detector.model._hf_peft_config_loaded = False

    def _apply_lora(self, rank: int, alpha: int) -> None:
        if not self._peft_applied:
            self._reset_peft_metadata()
        if not hasattr(self.detector.model, "get_input_embeddings"):
            self.detector.model.get_input_embeddings = lambda: nn.Embedding(1, 1)

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=LORA_TARGET_PRESETS[self.lora_target_preset],
            lora_dropout=0.05,
            bias="none",
            use_dora=self.use_dora,
        )
        self.detector.model = get_peft_model(self.detector.model, lora_config)
        self._peft_applied = True

    def _build_shared_lora_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.default_rank,
            lora_alpha=self.default_alpha,
            target_modules=LORA_TARGET_PRESETS[self.lora_target_preset],
            lora_dropout=0.05,
            bias="none",
        )

    def _attach_shared_adapter(self, trainable: bool) -> None:
        """Attach the CL-LoRA task-shared adapter alongside the class adapter.

        On first use the shared adapter is created with a FIXED random
        orthogonal down-projection (rows of lora_A orthonormal) and a
        zero-initialized, trainable up-projection (lora_B) -- following
        CL-LoRA (He et al., CVPR 2025). On later tasks the evolving shared
        state is reloaded from disk. Track B (data-incremental extension)
        attaches it frozen so extensions only touch the class adapter.
        """
        if not self.use_shared_adapter:
            return
        if self._shared_adapter_dir and Path(self._shared_adapter_dir).exists():
            self.model.load_adapter(
                self._shared_adapter_dir, adapter_name="shared", is_trainable=trainable
            )
        else:
            self.model.add_adapter("shared", self._build_shared_lora_config())
            self._init_shared_orthogonal_down()
        active = ["default", "shared"]
        self.model.base_model.set_adapter(active, inference_mode=False)
        self._freeze_shared_down()
        if not trainable:
            for name, param in self.model.named_parameters():
                if ".shared." in name:
                    param.requires_grad = False
        self._capture_shared_anchor()

    def _iter_shared_lora_modules(self):
        for module_name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and "shared" in module.lora_A:
                yield module_name, module

    def _init_shared_orthogonal_down(self) -> None:
        """Fixed random orthogonal down-projection: rows of lora_A orthonormal."""
        for _, module in self._iter_shared_lora_modules():
            weight = module.lora_A["shared"].weight  # (r, k)
            rank, in_features = weight.shape
            gaussian = torch.randn(rank, in_features, device=weight.device)
            # Orthonormal rows via the right-singular vectors of a Gaussian draw
            _, _, vh = torch.linalg.svd(gaussian, full_matrices=False)
            with torch.no_grad():
                weight.copy_(vh[:rank])
            nn.init.zeros_(module.lora_B["shared"].weight)

    def _freeze_shared_down(self) -> None:
        for _, module in self._iter_shared_lora_modules():
            module.lora_A["shared"].weight.requires_grad = False

    def _capture_shared_anchor(self) -> None:
        """Snapshot the shared up-projection plus per-dimension importance.

        Importance follows CL-LoRA's gradient-reassignment intuition: output
        dimensions whose up-projection rows carry large L2 norm encode the
        knowledge of previous tasks and must drift least. Empty before the
        first task (nothing to preserve yet).
        """
        self._shared_anchor = {}
        for module_name, module in self._iter_shared_lora_modules():
            weight = module.lora_B["shared"].weight  # (d, r)
            if float(weight.abs().sum()) == 0.0:
                continue  # fresh shared adapter: nothing learned yet
            row_norms = weight.norm(dim=1)
            importance = row_norms / row_norms.mean().clamp_min(1e-8)
            self._shared_anchor[module_name] = (
                weight.detach().clone(),
                importance.detach().clone(),
            )

    def shared_drift_loss(self) -> torch.Tensor:
        """Importance-weighted L2 anchor of the shared up-projection."""
        if not self._shared_anchor:
            return torch.tensor(0.0, device=self.device)
        total = torch.tensor(0.0, device=self.device)
        matched = 0
        for module_name, module in self._iter_shared_lora_modules():
            anchor = self._shared_anchor.get(module_name)
            if anchor is None:
                continue
            previous_weight, importance = anchor
            drift = module.lora_B["shared"].weight - previous_weight
            total = total + (importance[:, None] * drift.pow(2)).mean()
            matched += 1
        if matched == 0:
            return torch.tensor(0.0, device=self.device)
        return total / matched

    def _save_shared_adapter(self, adapters_root: Path) -> None:
        """Persist the evolving shared adapter next to the class adapters."""
        if not (self.use_shared_adapter and self._peft_applied):
            return
        if "shared" not in getattr(self.model, "peft_config", {}):
            return
        shared_root = adapters_root / "_shared"
        shared_root.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(shared_root), selected_adapters=["shared"])
        # Non-default adapter names are saved into a subdirectory by PEFT.
        self._shared_adapter_dir = str(shared_root / "shared")

    def _unfreeze_head(self) -> None:
        inner = self.detector._get_inner_model()
        inner.class_embed.weight.requires_grad = True
        if inner.class_embed.bias is not None:
            inner.class_embed.bias.requires_grad = True
        for layer in inner.transformer.enc_out_class_embed:
            layer.weight.requires_grad = True
            if layer.bias is not None:
                layer.bias.requires_grad = True

    def get_trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def _capture_stability_anchor(self) -> None:
        """Snapshot current trainable weights for extension regularization."""
        self._stability_anchor = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def orthogonal_loss(self) -> torch.Tensor:
        """
        Independent-adapter Det-LoRA currently trains exactly one active LoRA
        at a time. Orthogonal regularization is therefore a no-op.
        """
        return torch.tensor(0.0, device=self.device)

    def enable_merge_consistency(self) -> int:
        """Prepare DuET-style merge-consistency anchors from saved adapters.

        Loads the LoRA deltas (B @ A) of all previously finalized class
        adapters from disk and stores their elementwise mean per target
        module. ``merge_consistency_loss`` then penalizes sign conflicts of
        the currently trained delta against this anchor, so later task
        arithmetic (adapter merging) faces fewer TIES-style sign conflicts.
        Replay-free: only reads previously saved adapter weights, no data.

        Returns the number of anchored modules (0 if no previous adapters).
        """
        from safetensors import safe_open

        anchor_sums: Dict[str, torch.Tensor] = {}
        anchor_counts: Dict[str, int] = {}
        for class_name, adapter_path in self._adapter_paths.items():
            weights_file = Path(adapter_path) / "adapter_model.safetensors"
            if not weights_file.exists():
                continue
            pairs: Dict[str, Dict[str, torch.Tensor]] = {}
            with safe_open(str(weights_file), framework="pt") as f:
                for key in f.keys():
                    if ".lora_A." in key:
                        module_path = key.split(".lora_A.")[0]
                        pairs.setdefault(module_path, {})["A"] = f.get_tensor(key)
                    elif ".lora_B." in key:
                        module_path = key.split(".lora_B.")[0]
                        pairs.setdefault(module_path, {})["B"] = f.get_tensor(key)
            for module_path, ab in pairs.items():
                if "A" not in ab or "B" not in ab:
                    continue
                delta = (ab["B"] @ ab["A"]).to(self.device)
                if module_path in anchor_sums:
                    anchor_sums[module_path] = anchor_sums[module_path] + delta
                    anchor_counts[module_path] += 1
                else:
                    anchor_sums[module_path] = delta
                    anchor_counts[module_path] = 1

        self._merge_anchor_deltas = {
            module_path: total / anchor_counts[module_path]
            for module_path, total in anchor_sums.items()
        }
        return len(self._merge_anchor_deltas)

    def merge_consistency_loss(self) -> torch.Tensor:
        """Sign-conflict penalty of the active LoRA delta vs. anchor deltas."""
        anchors = getattr(self, "_merge_anchor_deltas", None)
        if not anchors:
            return torch.tensor(0.0, device=self.device)
        total = torch.tensor(0.0, device=self.device)
        matched_modules = 0
        for module_name, module in self.model.named_modules():
            if not hasattr(module, "lora_A"):
                continue
            adapter_names = [n for n in module.lora_A.keys()]
            if not adapter_names:
                continue
            adapter_name = adapter_names[0]
            anchor = None
            for anchor_path, anchor_delta in anchors.items():
                if module_name.endswith(anchor_path) or anchor_path.endswith(module_name):
                    anchor = anchor_delta
                    break
            if anchor is None:
                continue
            delta = module.lora_B[adapter_name].weight @ module.lora_A[adapter_name].weight
            conflict_mask = anchor.sign()
            total = total + torch.relu(-(delta * conflict_mask)).mean()
            matched_modules += 1
        if matched_modules == 0:
            return torch.tensor(0.0, device=self.device)
        return total / matched_modules

    def stability_loss(self) -> torch.Tensor:
        """
        Penalize large deviations from the loaded extension starting point.

        This is only active for `extend_class`, where we warm-start from the
        previously learned adapter and want to keep that knowledge anchored.
        """
        if not self._stability_anchor:
            return torch.tensor(0.0, device=self.device)

        total = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            anchor = self._stability_anchor.get(name)
            if anchor is None:
                continue
            total = total + torch.sum((param - anchor.to(param.device)) ** 2)
        return total

    def set_eval_mode(self) -> None:
        self.model.eval()

    def set_train_mode(self) -> None:
        self.model.train()

    def _eval_adapter_name(self, class_name: str) -> str:
        """Stable PEFT adapter handle for cached evaluation adapters."""
        return f"eval_{class_name}"

    def prepare_eval_adapter_cache(self, class_names: List[str]) -> None:
        """
        Load all requested eval adapters once and keep them resident in memory.

        This enables cheap `set_adapter(...)` switching during joint inference,
        so the expensive part becomes the decoder pass rather than repeated
        disk-backed adapter loads.
        """
        self._ensure_runtime_state()
        if not class_names:
            return

        requested = [class_name for class_name in class_names if class_name in self._adapter_paths]
        if not requested:
            return

        if self._peft_applied:
            self.detector.model = self.model.unload()
            self._peft_applied = False
        self._loaded_eval_adapters = {}

        primary = requested[0]
        primary_name = self._eval_adapter_name(primary)
        primary_path = self._adapter_paths[primary]
        from peft import PeftModel

        self._reset_peft_metadata()
        if not hasattr(self.detector.model, "get_input_embeddings"):
            self.detector.model.get_input_embeddings = lambda: nn.Embedding(1, 1)
        self.detector.model = PeftModel.from_pretrained(
            self.detector.model,
            primary_path,
            adapter_name=primary_name,
            is_trainable=False,
        )
        self._peft_applied = True
        self._loaded_eval_adapters[primary] = primary_name

        for class_name in requested[1:]:
            adapter_name = self._eval_adapter_name(class_name)
            adapter_path = self._adapter_paths[class_name]
            self.model.load_adapter(
                adapter_path,
                adapter_name=adapter_name,
                is_trainable=False,
            )
            self._loaded_eval_adapters[class_name] = adapter_name

        if self.use_shared_adapter and self._shared_adapter_dir:
            self.model.load_adapter(
                self._shared_adapter_dir, adapter_name="shared", is_trainable=False
            )

        self.model.eval()

    def activate_cached_eval_adapter(self, class_name: str) -> None:
        """Switch to an already cached eval adapter without reloading from disk."""
        self._ensure_runtime_state()
        adapter_name = self._loaded_eval_adapters.get(class_name)
        if adapter_name is None:
            raise ValueError(f"Adapter '{class_name}' not prepared in eval cache")

        head_state = self._resolve_eval_head_state(class_name)
        if head_state is not None:
            self._apply_head_state(head_state)
        else:
            self._restore_global_head_state()
        if self.use_shared_adapter and "shared" in getattr(self.model, "peft_config", {}):
            self.model.base_model.set_adapter([adapter_name, "shared"], inference_mode=True)
        else:
            self.model.set_adapter(adapter_name)
        self.model.eval()

    def clear_eval_adapter_cache(self) -> None:
        """Unload cached eval adapters and restore the plain base model."""
        self._ensure_runtime_state()
        if self._peft_applied:
            self.detector.model = self.model.unload()
            self._peft_applied = False
        self._loaded_eval_adapters = {}
        self._restore_global_head_state()

    def load_adapter_for_eval(self, class_name: str) -> None:
        """Load a specific class's LoRA adapter for per-class evaluation."""
        self._ensure_runtime_state()
        if class_name not in self._adapter_paths:
            raise ValueError(f"No saved adapter for '{class_name}'")

        cached_name = self._loaded_eval_adapters.get(class_name)
        if cached_name is not None:
            self.activate_cached_eval_adapter(class_name)
            return

        head_state = self._resolve_eval_head_state(class_name)
        if head_state is not None:
            self._apply_head_state(head_state)
        else:
            self._restore_global_head_state()

        adapter_path = self._adapter_paths[class_name]
        if not Path(adapter_path).exists():
            raise FileNotFoundError(
                f"Adapter path for '{class_name}' not found: {Path(adapter_path).resolve()}. "
                "Checkpoint may have been moved; re-run load_all() on its new location."
            )

        # Remove current LoRA if any
        if self._peft_applied:
            self.detector.model = self.model.unload()
            self._peft_applied = False

        # Load adapter from saved checkpoint
        from peft import PeftModel

        self._reset_peft_metadata()
        if not hasattr(self.detector.model, "get_input_embeddings"):
            self.detector.model.get_input_embeddings = lambda: nn.Embedding(1, 1)
        self.detector.model = PeftModel.from_pretrained(
            self.detector.model,
            adapter_path,
            is_trainable=False,
        )
        self._peft_applied = True
        if self.use_shared_adapter and self._shared_adapter_dir:
            self.model.load_adapter(
                self._shared_adapter_dir, adapter_name="shared", is_trainable=False
            )
            self.model.base_model.set_adapter(["default", "shared"], inference_mode=True)
        self.model.eval()

    def unload_adapter(self) -> None:
        """Remove LoRA adapter, restore base model."""
        if self._loaded_eval_adapters:
            self.clear_eval_adapter_cache()
            return
        if self._peft_applied:
            self.detector.model = self.model.unload()
            self._peft_applied = False
        self._restore_global_head_state()

    def forward(
        self,
        pixel_values: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, Any]:
        return self.detector.forward(pixel_values=pixel_values, targets=targets)

    def save_all(self, save_dir: str) -> None:
        """Save complete state: head weights + adapter paths + registry."""
        self._ensure_runtime_state()
        self._bootstrap_version_store_from_active_state()
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = str(path)
        for class_name in list(self.trained_classes):
            self._sync_active_version_state(class_name)
        self._rebuild_active_snapshots()
        self._global_head_state = self._capture_head_state()
        if self._base_head_state is None and self._global_head_state is not None:
            self._base_head_state = self._extract_base_head_state(self._global_head_state)
        self._ensure_class_head_snapshots()

        # Save PEFT model if currently in training
        if self._peft_applied:
            self.model.save_pretrained(str(path / "peft_model"))

        # Save head weights (accumulated across all tasks)
        torch.save(self._global_head_state, str(path / "head_weights.pt"))
        if self._base_head_state is not None:
            torch.save(self._base_head_state, str(path / "base_head.pt"))

        class_heads_dir = path / "class_heads"
        class_heads_dir.mkdir(parents=True, exist_ok=True)
        for class_name, head_state in self._class_head_states.items():
            torch.save(head_state, class_heads_dir / f"{class_name}.pt")

        class_head_rows_dir = path / "class_head_rows"
        class_head_rows_dir.mkdir(parents=True, exist_ok=True)
        for class_name, row_state in self._class_head_rows.items():
            torch.save(row_state, class_head_rows_dir / f"{class_name}.pt")

        packaged_adapter_paths: Dict[str, str] = {}
        packaged_versions: Dict[str, List[Dict[str, Any]]] = {}
        version_root = path / "adapter_versions"
        adapters_dir = path / "adapters"
        for class_name, versions in self._adapter_versions.items():
            packaged_versions[class_name] = []
            for entry in versions:
                version_id = entry["version_id"]
                version_base = version_root / class_name / version_id
                packaged_entry = {
                    key: value for key, value in entry.items() if key != "adapter_path"
                }
                adapter_path = entry.get("adapter_path")
                if adapter_path:
                    source_path = Path(adapter_path)
                    if source_path.exists():
                        target_path = version_base / "adapter"
                        if target_path.exists():
                            shutil.rmtree(target_path)
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(source_path, target_path)
                        packaged_entry["adapter_path"] = str(target_path.relative_to(path))
                    else:
                        packaged_entry["adapter_path"] = adapter_path
                row_state = self._versioned_class_head_rows.get(class_name, {}).get(version_id)
                if row_state is not None:
                    row_path = version_base / "class_row.pt"
                    row_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(row_state, row_path)
                    packaged_entry["class_row_path"] = str(row_path.relative_to(path))
                packaged_versions[class_name].append(packaged_entry)

            active_version = self._active_versions.get(class_name)
            if active_version is None:
                continue
            active_entry = next(
                (
                    entry
                    for entry in packaged_versions[class_name]
                    if entry.get("version_id") == active_version
                ),
                None,
            )
            if active_entry is None:
                continue
            active_source_path = active_entry.get("adapter_path")
            if active_source_path:
                source_path = path / active_source_path
                if source_path.exists():
                    target_path = adapters_dir / f"lora_{class_name}"
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(source_path, target_path)
                    packaged_adapter_paths[class_name] = str(target_path.relative_to(path))
                else:
                    packaged_adapter_paths[class_name] = active_source_path

        # Package the evolving CL-LoRA shared adapter alongside class adapters
        packaged_shared_path = None
        if self.use_shared_adapter and self._shared_adapter_dir:
            shared_source = Path(self._shared_adapter_dir)
            if shared_source.exists():
                shared_target = adapters_dir / "_shared" / "shared"
                if shared_target.exists():
                    shutil.rmtree(shared_target)
                shared_target.parent.mkdir(parents=True, exist_ok=True)
                if shared_source.resolve() != shared_target.resolve():
                    shutil.copytree(shared_source, shared_target)
                packaged_shared_path = str(shared_target.relative_to(path))

        # Save registry
        registry = {
            "trained_classes": self.trained_classes,
            "current_class": self.current_class,
            "frozen_head_indices": sorted(self._head_mask.frozen_indices),
            "added_classes": self.detector.added_classes,
            "base_num_classes": self.detector.base_num_classes,
            "detector_variant": self.detector.variant,
            "default_rank": self.default_rank,
            "default_alpha": self.default_alpha,
            "lora_target_preset": self.lora_target_preset,
            "use_dora": self.use_dora,
            "use_shared_adapter": self.use_shared_adapter,
            "shared_adapter_path": packaged_shared_path,
            "adapter_paths": packaged_adapter_paths,
            "adapter_versions": packaged_versions,
            "active_versions": self._active_versions,
            "peft_applied": self._peft_applied,
            "class_head_snapshots": sorted(self._class_head_states.keys()),
            "class_head_rows": sorted(self._class_head_rows.keys()),
        }
        with open(path / "det_lora_registry.json", "w") as f:
            json.dump(registry, f, indent=2)

        calibration_state = {
            "calibrators": self._adapter_calibrators,
            "score_banks": self._score_banks,
            "shared_quality_calibrator": self._shared_quality_calibrator,
            "adapter_arbitration_state": self._adapter_arbitration_state,
            "versioned_calibrators": self._versioned_adapter_calibrators,
            "versioned_score_banks": self._versioned_score_banks,
        }
        with open(path / "adapter_calibration.json", "w") as f:
            json.dump(calibration_state, f, indent=2)

        # The conflict gate carries dense per-pair covariance matrices, so it is
        # persisted as a binary tensor file rather than in the calibration JSON.
        if self._conflict_gate:
            torch.save(self._conflict_gate, str(path / "conflict_gate.pt"))

        print(f"[Det-LoRA] Saved to {path}")

    def load_all(self, load_dir: str) -> None:
        """Load complete state from checkpoint."""
        self._ensure_runtime_state()
        path = Path(load_dir)
        self._checkpoint_dir = str(path)
        self._stability_anchor = {}
        self._adapter_versions = {}
        self._active_versions = {}
        self._versioned_class_head_rows = {}
        self._versioned_adapter_calibrators = {}
        self._versioned_score_banks = {}
        self._base_head_state = None
        self._class_head_rows = {}
        self._class_head_states = {}
        self._global_head_state = None
        self._loaded_eval_adapters = {}

        if self._peft_applied:
            self._head_mask.remove_hooks()
            self.detector.model = self.model.unload()
            self._peft_applied = False

        with open(path / "det_lora_registry.json") as f:
            registry = json.load(f)

        self.lora_target_preset = registry.get("lora_target_preset", "default")
        self.use_dora = bool(registry.get("use_dora", False))
        self.use_shared_adapter = bool(registry.get("use_shared_adapter", False))
        shared_relative_path = registry.get("shared_adapter_path")
        if shared_relative_path:
            shared_path = Path(str(shared_relative_path))
            if not shared_path.is_absolute():
                shared_path = path / shared_path
            self._shared_adapter_dir = str(shared_path)
        else:
            self._shared_adapter_dir = None
        self._shared_anchor = {}

        # Expand head for all classes
        for class_name in registry["added_classes"]:
            if class_name not in self.detector.added_classes:
                self.detector.expand_classification_head(class_name)

        # Load head weights
        head_path = path / "head_weights.pt"
        if head_path.exists():
            head_state = torch.load(str(head_path), map_location="cpu")
            self._apply_head_state(head_state)
            self._global_head_state = head_state
        base_head_path = path / "base_head.pt"
        if base_head_path.exists():
            self._base_head_state = torch.load(str(base_head_path), map_location="cpu")
        elif self._global_head_state is not None:
            self._base_head_state = self._extract_base_head_state(self._global_head_state)

        # Restore state
        self.trained_classes = registry.get("trained_classes", [])
        self.current_class = registry.get("current_class")
        self._adapter_paths = {}
        for class_name, adapter_path in registry.get("adapter_paths", {}).items():
            adapter_path_str = str(adapter_path)
            resolved_path = Path(adapter_path_str)
            if not resolved_path.is_absolute():
                resolved_path = path / resolved_path
            self._adapter_paths[class_name] = str(resolved_path)
        self._adapter_calibrators = {}
        self._score_banks = {}
        self._shared_quality_calibrator = {}
        self._adapter_arbitration_state = {}
        self._conflict_gate = {}
        for idx in registry.get("frozen_head_indices", []):
            self._head_mask.freeze_class(idx)

        calibration_path = path / "adapter_calibration.json"
        if calibration_path.exists():
            with open(calibration_path) as f:
                calibration_state = json.load(f)
            self._adapter_calibrators = calibration_state.get("calibrators", {})
            self._score_banks = calibration_state.get("score_banks", {})
            self._shared_quality_calibrator = calibration_state.get(
                "shared_quality_calibrator",
                {},
            )
            self._adapter_arbitration_state = calibration_state.get(
                "adapter_arbitration_state",
                {},
            )
            self._versioned_adapter_calibrators = calibration_state.get(
                "versioned_calibrators",
                {},
            )
            self._versioned_score_banks = calibration_state.get(
                "versioned_score_banks",
                {},
            )

        conflict_gate_path = path / "conflict_gate.pt"
        if conflict_gate_path.exists():
            self._conflict_gate = torch.load(str(conflict_gate_path), weights_only=False)

        class_heads_dir = path / "class_heads"
        for class_name in registry.get("class_head_snapshots", []):
            snapshot_path = class_heads_dir / f"{class_name}.pt"
            if snapshot_path.exists():
                self._class_head_states[class_name] = torch.load(
                    str(snapshot_path),
                    map_location="cpu",
                )
        class_head_rows_dir = path / "class_head_rows"
        for class_name in registry.get("class_head_rows", []):
            row_path = class_head_rows_dir / f"{class_name}.pt"
            if row_path.exists():
                self._class_head_rows[class_name] = torch.load(
                    str(row_path),
                    map_location="cpu",
                )

        adapter_versions = registry.get("adapter_versions", {})
        if adapter_versions:
            parsed_versions: Dict[str, List[Dict[str, Any]]] = {}
            parsed_rows: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for class_name, versions in adapter_versions.items():
                parsed_versions[class_name] = []
                parsed_rows[class_name] = {}
                for entry in versions:
                    parsed_entry = dict(entry)
                    adapter_path = parsed_entry.get("adapter_path")
                    if adapter_path is not None:
                        resolved_path = Path(str(adapter_path))
                        if not resolved_path.is_absolute():
                            resolved_path = path / resolved_path
                        parsed_entry["adapter_path"] = str(resolved_path)
                    row_path = parsed_entry.get("class_row_path")
                    if row_path is not None:
                        resolved_row_path = Path(str(row_path))
                        if not resolved_row_path.is_absolute():
                            resolved_row_path = path / resolved_row_path
                        if resolved_row_path.exists():
                            parsed_rows[class_name][parsed_entry["version_id"]] = torch.load(
                                str(resolved_row_path),
                                map_location="cpu",
                            )
                    parsed_versions[class_name].append(parsed_entry)
            self._adapter_versions = parsed_versions
            self._active_versions = registry.get(
                "active_versions",
                {
                    class_name: versions[-1]["version_id"]
                    for class_name, versions in parsed_versions.items()
                    if versions
                },
            )
            self._versioned_class_head_rows = parsed_rows
            self._rebuild_active_runtime_from_versions()
        else:
            self._backfill_class_head_rows_from_snapshots()
            self._ensure_class_head_snapshots()
            self._bootstrap_version_store_from_active_state()

        peft_model_dir = path / "peft_model"
        if registry.get("peft_applied") and peft_model_dir.exists():
            from peft import PeftModel

            self._reset_peft_metadata()
            if not hasattr(self.detector.model, "get_input_embeddings"):
                self.detector.model.get_input_embeddings = lambda: nn.Embedding(1, 1)
            self.detector.model = PeftModel.from_pretrained(
                self.detector.model,
                str(peft_model_dir),
                is_trainable=True,
            )
            self._peft_applied = True
        else:
            self._peft_applied = False

        if self._peft_applied and self.current_class is not None:
            self._unfreeze_head()
            self._head_mask.register_hooks(self.detector._get_inner_model())
        elif self._global_head_state is not None:
            self._apply_head_state(self._global_head_state)

        self.detector.criterion = self.detector._rebuild_criterion()
        print(f"[Det-LoRA] Loaded {len(self.trained_classes)} classes from {path}")

    def summary(self) -> str:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        lines = [
            "=" * 60,
            "Det-LoRA Summary",
            "=" * 60,
            f"Base model: RF-DETR {self.detector.variant}",
            f"Total params: {total:,}",
            f"Trainable params: {trainable:,} ({100*trainable/max(total,1):.2f}%)",
            f"PEFT active: {self._peft_applied}",
            f"Trained classes: {', '.join(self.trained_classes) or 'none'}",
            f"Current task: {self.current_class or 'none'}",
            f"Frozen head neurons: {sorted(self._head_mask.frozen_indices)}",
            f"Saved adapters: {list(self._adapter_paths.keys())}",
            "=" * 60,
        ]
        return "\n".join(lines)
