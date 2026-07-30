"""
CL-DETR Baseline (detector-specific IOD)
========================================

Adaptation of *Continual Detection Transformer* (Liu et al., CVPR 2023,
https://arxiv.org/abs/2304.03110) onto the RF-DETR pipeline used in this thesis.

CL-DETR = Experience Replay + two detector-specific additions:

1. **Detector Knowledge Distillation (DKD)** — a frozen copy of the model from the
   previous phase predicts on every training image. The top-K most confident
   *foreground* predictions for the OLD classes are selected, predictions that
   overlap a new-class ground-truth box too much (IoU > lambda) are dropped, and the
   survivors are merged with the available ground-truth labels into a single label
   set. The model is then trained with the standard DETR loss on the merged labels
   (eq. 3-5 in the paper). Background predictions are deliberately ignored because
   they are imbalanced and can contradict the new-class evidence.

2. **Two-step phase training** — a main step trains with DKD on new data + replay
   exemplars, followed by a short calibration step on the balanced exemplar buffer
   with the plain DETR loss (Sec. 3.4).

Documented adaptations vs. the original (declared as a limitation in the thesis):
  - Base detector is RF-DETR (frozen DINOv2 backbone, sigmoid per-class scoring)
    instead of Deformable/UP-DETR; benchmark is the Mendeley Military Vehicles set
    instead of COCO.
  - Under this thesis' single-class-per-phase protocol, the distribution-preserving
    exemplar SELECTION (Algorithm 2) reduces to a balanced per-class buffer, so it is
    realised as balanced sampling (inherited from ReplayBaseline). The DKD loss and
    the two-step calibration are implemented faithfully.
"""

import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from det_lora.baselines.replay import ReplayBaseline
from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.evaluator import ContinualEvaluator
from det_lora.train import SyntheticDetectionDataset, collate_fn, evaluate_loss
from det_lora.utils import collect_runtime_metadata, set_global_seed

DKD_TOP_K = 10
DKD_IOU_LAMBDA = 0.7
DKD_SCORE_FLOOR = 0.3


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def build_dkd_targets(
    teacher_logits: torch.Tensor,
    teacher_boxes: torch.Tensor,
    gt_targets: List[Dict[str, torch.Tensor]],
    old_class_ids: List[int],
    *,
    top_k: int = DKD_TOP_K,
    iou_lambda: float = DKD_IOU_LAMBDA,
    score_floor: float = DKD_SCORE_FLOOR,
) -> List[Dict[str, torch.Tensor]]:
    """Merge ground-truth labels with old-model pseudo-labels (CL-DETR eq. 3-5).

    Args:
        teacher_logits: [B, Q, C] raw logits of the frozen previous model.
        teacher_boxes:  [B, Q, 4] predicted boxes (cxcywh, normalised).
        gt_targets:     per-image dicts with 'labels' and 'boxes' (new-class GT).
        old_class_ids:  absolute class IDs the old model is allowed to pseudo-label.
    Returns:
        per-image merged target dicts (GT + filtered pseudo-labels).
    """
    if not old_class_ids:
        return gt_targets

    device = teacher_logits.device
    old_ids = torch.tensor(old_class_ids, device=device, dtype=torch.long)
    merged: List[Dict[str, torch.Tensor]] = []

    for img_idx, gt in enumerate(gt_targets):
        # Per-class sigmoid scores restricted to old classes (RF-DETR scores per-class).
        old_scores = teacher_logits[img_idx][:, old_ids].sigmoid()  # [Q, |old|]
        best_score, best_col = old_scores.max(dim=-1)  # [Q]

        # Foreground = confident enough; then keep the top-K most confident.
        foreground = best_score > score_floor
        if foreground.any():
            fg_idx = foreground.nonzero(as_tuple=True)[0]
            if fg_idx.numel() > top_k:
                top = torch.topk(best_score[fg_idx], top_k).indices
                fg_idx = fg_idx[top]
        else:
            fg_idx = best_score.new_empty((0,), dtype=torch.long)

        gt_labels = gt["labels"].to(device)
        gt_boxes = gt["boxes"].to(device)

        if fg_idx.numel() > 0:
            pseudo_boxes = teacher_boxes[img_idx][fg_idx]  # [P, 4] cxcywh
            pseudo_labels = old_ids[best_col[fg_idx]]
            # Drop pseudo boxes overlapping a new-class GT box too much (eq. 3).
            if gt_boxes.numel() > 0:
                iou = box_iou(_cxcywh_to_xyxy(pseudo_boxes), _cxcywh_to_xyxy(gt_boxes))
                keep = (
                    (iou.max(dim=1).values <= iou_lambda)
                    if iou.numel()
                    else torch.ones(pseudo_boxes.shape[0], dtype=torch.bool, device=device)
                )
                pseudo_boxes = pseudo_boxes[keep]
                pseudo_labels = pseudo_labels[keep]
            merged.append(
                {
                    "labels": torch.cat([gt_labels, pseudo_labels], dim=0),
                    "boxes": torch.cat([gt_boxes, pseudo_boxes], dim=0),
                }
            )
        else:
            merged.append({"labels": gt_labels, "boxes": gt_boxes})

    return merged


class CLDETRBaseline(ReplayBaseline):
    """Detector-specific IOD baseline: DKD + replay + two-step calibration."""

    def __init__(
        self,
        variant: str = "medium",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        buffer_per_class: int = 50,
        calibration_epochs: int = 5,
        dkd_top_k: int = DKD_TOP_K,
        dkd_iou_lambda: float = DKD_IOU_LAMBDA,
    ):
        super().__init__(
            variant=variant,
            lr=lr,
            weight_decay=weight_decay,
            buffer_per_class=buffer_per_class,
        )
        self.calibration_epochs = calibration_epochs
        self.dkd_top_k = dkd_top_k
        self.dkd_iou_lambda = dkd_iou_lambda
        self._teacher_model = None
        print(
            f"[CL-DETR] DKD top_k={dkd_top_k}, IoU_lambda={dkd_iou_lambda}, "
            f"calibration_epochs={calibration_epochs}"
        )

    # --- teacher (frozen previous-phase model) ------------------------------
    def _snapshot_teacher(self) -> None:
        """Freeze a deep copy of the current model as the DKD teacher."""
        self._teacher_model = copy.deepcopy(self.detector.model)
        self._teacher_model.eval()
        for param in self._teacher_model.parameters():
            param.requires_grad = False

    def _drop_teacher(self) -> None:
        self._teacher_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _teacher_predict(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        samples = self.detector.prepare_input(pixel_values)
        out = self._teacher_model(samples)
        return {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}

    # --- shared loss / training helpers -------------------------------------
    def _batch_targets(self, batch) -> List[Dict[str, torch.Tensor]]:
        targets = []
        for label in batch["labels"]:
            lbl_key = "labels" if "labels" in label else "class_labels"
            targets.append(
                {
                    "labels": label[lbl_key].to(self.detector.device),
                    "boxes": label["boxes"].to(self.detector.device),
                }
            )
        return targets

    def _dkd_loss(self, pixel_values, targets, old_class_ids):
        """Standard DETR loss on GT merged with old-model pseudo-labels."""
        if self._teacher_model is not None and old_class_ids:
            teacher = self._teacher_predict(pixel_values)
            targets = build_dkd_targets(
                teacher["pred_logits"],
                teacher["pred_boxes"],
                targets,
                old_class_ids,
                top_k=self.dkd_top_k,
                iou_lambda=self.dkd_iou_lambda,
            )
        return self.detector.forward(pixel_values=pixel_values, targets=targets)["loss"]

    def _calibration_step(self, params, optimizer_lr: float) -> None:
        """Short fine-tune on the balanced exemplar buffer with the plain loss (Sec 3.4)."""
        if self.calibration_epochs <= 0 or not self.replay_datasets:
            return
        cal_ds = ConcatDataset(self.replay_datasets)
        cal_loader = DataLoader(
            cal_ds, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0
        )
        optimizer = AdamW(params, lr=optimizer_lr, weight_decay=self.weight_decay)
        for epoch in range(1, self.calibration_epochs + 1):
            self.detector.model.train()
            for batch in tqdm(cal_loader, desc=f"Calibration {epoch}", leave=False):
                pixel_values = batch["pixel_values"].to(self.detector.device)
                targets = self._batch_targets(batch)
                loss = self.detector.forward(pixel_values=pixel_values, targets=targets)["loss"]
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

    # --- Track A: class-incremental ----------------------------------------
    def run_experiment(
        self,
        classes: List[str],
        epochs: int = 30,
        batch_size: int = 4,
        data_dir: str = "data/raw",
        save_dir: str = "experiments",
        synthetic: bool = False,
        resume_dir: Optional[str] = None,
        seed: int = 42,
        metrics_eval_every: int = 2,
    ) -> Dict:
        from det_lora.baselines.checkpoint import (
            is_better_validation_checkpoint,
            load_model_checkpoint,
            load_progress,
            save_model_checkpoint,
            save_progress,
        )

        set_global_seed(seed)
        if resume_dir:
            experiment_dir = Path(resume_dir)
        else:
            experiment_dir = Path(save_dir) / f"baseline_cl_detr_{time.strftime('%Y%m%d_%H%M%S')}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        progress = load_progress(experiment_dir)
        completed = set(progress.get("completed_tasks", []))
        resolution = self.detector.resolution
        results_path = experiment_dir / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
        else:
            results = {
                "method": "cl_detr",
                "buffer_per_class": self.buffer_per_class,
                "dkd_top_k": self.dkd_top_k,
                "dkd_iou_lambda": self.dkd_iou_lambda,
                "calibration_epochs": self.calibration_epochs,
                "tasks": {},
                "evaluation_after_task": {},
                "config": {
                    "seed": seed,
                    "eval_split": "test",
                    "runtime": collect_runtime_metadata(),
                },
            }

        evaluator = ContinualEvaluator()

        for task_idx, class_name in enumerate(classes):
            if class_name in completed:
                print(f"[CL-DETR] Task {task_idx + 1}/{len(classes)}: {class_name} - SKIPPED")
                self.detector.expand_classification_head(class_name)
                load_model_checkpoint(self.detector.model, experiment_dir, class_name)
                if not synthetic:
                    self._load_state(experiment_dir, class_name)
                continue
            print(f"\n[CL-DETR] Task {task_idx + 1}/{len(classes)}: {class_name}")

            # DKD teacher = model state with all previously learned classes.
            old_class_ids = [self.detector.get_class_id(c) for c in classes[:task_idx]]
            if task_idx > 0 and not synthetic:
                self._snapshot_teacher()

            self.detector.expand_classification_head(class_name)

            if synthetic:
                new_ds = SyntheticDetectionDataset(100, self.detector.get_num_classes(), resolution)
                val_ds = SyntheticDetectionDataset(50, self.detector.get_num_classes(), resolution)
            else:
                new_ds = load_dataset_from_raw(
                    raw_dir=data_dir,
                    class_filter=class_name,
                    split="train",
                    class_id_offset=self.detector.base_num_classes,
                    img_size=resolution,
                    seed=seed,
                )
                val_ds = load_dataset_from_raw(
                    raw_dir=data_dir,
                    class_filter=class_name,
                    split="val",
                    class_id_offset=self.detector.base_num_classes,
                    img_size=resolution,
                    seed=seed,
                )

            if self.replay_datasets:
                train_ds = ConcatDataset([new_ds] + self.replay_datasets)
                print(f"  New: {len(new_ds)}, Replay: {sum(len(r) for r in self.replay_datasets)}")
            else:
                train_ds = new_ds

            loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
            )

            params = [p for p in self.detector.model.parameters() if p.requires_grad]
            optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=self.lr * 0.01)

            history = []
            best_epoch = 0
            best_val_loss = float("inf")
            best_val_map50 = float("-inf")
            class_ids = [self.detector.get_class_id(class_name)]
            for epoch in range(1, epochs + 1):
                self.detector.model.train()
                total_loss = 0
                for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
                    pixel_values = batch["pixel_values"].to(self.detector.device)
                    targets = self._batch_targets(batch)
                    loss = self._dkd_loss(pixel_values, targets, old_class_ids)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()

                scheduler.step()
                avg_loss = total_loss / len(loader)
                val_metrics = evaluate_loss(self.detector, val_loader)
                entry = {"epoch": epoch, "loss": avg_loss, "val_loss": val_metrics["loss"]}
                val_detection_metrics = None
                if (
                    not synthetic
                    and metrics_eval_every > 0
                    and (epoch % metrics_eval_every == 0 or epoch == epochs)
                ):
                    val_detection_metrics = evaluator.evaluate_standard_detector(
                        detector=self.detector,
                        dataloader=val_loader,
                        class_names=[class_name],
                        class_ids=class_ids,
                    )
                    entry.update({f"val_{k}": v for k, v in val_detection_metrics.items()})
                history.append(entry)
                current_val_map50 = (
                    float(val_detection_metrics["mAP@0.5"])
                    if val_detection_metrics is not None
                    else float("-inf")
                )
                if is_better_validation_checkpoint(
                    current_val_map50, val_metrics["loss"], best_val_map50, best_val_loss
                ):
                    best_epoch = epoch
                    best_val_loss = val_metrics["loss"]
                    best_val_map50 = max(best_val_map50, current_val_map50)
                    save_model_checkpoint(
                        self.detector.model,
                        experiment_dir,
                        class_name,
                        checkpoint_root="best_checkpoints",
                    )
                if epoch % 10 == 0 or epoch == epochs:
                    metric_text = ""
                    if val_detection_metrics is not None:
                        metric_text = f" | Val mAP@0.5: {val_detection_metrics['mAP@0.5']:.4f}"
                    print(
                        f"  Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | "
                        f"Val: {val_metrics['loss']:.4f}{metric_text}"
                    )

            load_model_checkpoint(
                self.detector.model,
                experiment_dir,
                class_name,
                checkpoint_root="best_checkpoints",
            )
            self._drop_teacher()

            # Add current task to the (balanced) exemplar buffer, then calibrate.
            self._register_replay_buffer(
                class_name=class_name,
                seed=seed,
                resolution=resolution,
                dataset=new_ds,
                max_samples=None,
                sample_offset=0,
                raw_dir=data_dir,
            )
            if not synthetic:
                self._calibration_step(params, optimizer_lr=self.lr * 0.1)

            results["tasks"][class_name] = {
                "history": history,
                "final_loss": history[-1]["loss"],
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_map50": None if best_val_map50 == float("-inf") else best_val_map50,
            }
            if not synthetic:
                self._save_state(experiment_dir, class_name)
                seen_classes = classes[: task_idx + 1]
                eval_ds = load_dataset_from_raw(
                    raw_dir=data_dir,
                    class_filter=seen_classes,
                    split="test",
                    class_id_offset=self.detector.base_num_classes,
                    img_size=resolution,
                    seed=seed,
                )
                eval_loader = DataLoader(
                    eval_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=0,
                )
                metrics = evaluator.evaluate_standard_detector(
                    detector=self.detector,
                    dataloader=eval_loader,
                    class_names=seen_classes,
                    class_ids=[self.detector.get_class_id(c) for c in seen_classes],
                    task_idx=task_idx,
                    include_curves=True,
                )
                results["evaluation_after_task"][class_name] = metrics

            save_model_checkpoint(self.detector.model, experiment_dir, class_name)
            completed.add(class_name)
            save_progress(experiment_dir, completed, class_name)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

        if evaluator.history:
            results["final_evaluation"] = evaluator.history[max(evaluator.history.keys())][
                "metrics"
            ]
            evaluator.save_results(str(experiment_dir / "evaluation.json"))

        results["output_dir"] = str(experiment_dir)
        results["final_checkpoint_task"] = classes[-1] if classes else None
        results["final_checkpoint_dir"] = (
            str(experiment_dir / "checkpoints" / classes[-1]) if classes else None
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[CL-DETR] Results saved to {experiment_dir}")
        return results

    # --- Track B: data-incremental extension --------------------------------
    def extend_experiment(
        self,
        class_name: str,
        *,
        seen_classes: List[str],
        load_dir: str,
        load_task_name: str,
        epochs: int = 10,
        batch_size: int = 4,
        data_dir: str = "data/raw",
        test_data_dir: Optional[str] = None,
        save_dir: str = "experiments",
        synthetic: bool = False,
        seed: int = 42,
        max_samples: Optional[int] = None,
        sample_offset: int = 0,
        extension_seed_offset: int = 1000,
        metrics_eval_every: int = 2,
        experiment_name: Optional[str] = None,
    ) -> Dict:
        """Data-incremental extension. DKD teacher = the loaded pre-extension model,
        distilling ALL seen classes (retention); then calibration on the buffer."""
        from det_lora.baselines.checkpoint import (
            is_better_validation_checkpoint,
            load_model_checkpoint,
            prepare_detector_for_checkpoint_load,
            save_model_checkpoint,
        )

        set_global_seed(seed)
        experiment_dir = Path(save_dir) / (
            experiment_name or f"extend_cl_detr_{class_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)

        prepare_detector_for_checkpoint_load(self.detector, seen_classes)
        if not load_model_checkpoint(self.detector.model, Path(load_dir), load_task_name):
            raise FileNotFoundError(
                f"Source checkpoint not found for {load_task_name} in {load_dir}"
            )
        if not synthetic and not self._load_state(Path(load_dir), load_task_name):
            raise FileNotFoundError(f"Replay state not found for {load_task_name} in {load_dir}")

        # Teacher = pre-extension model; distil all seen classes (data-incremental retention).
        old_class_ids = [self.detector.get_class_id(c) for c in seen_classes]
        if not synthetic:
            self._snapshot_teacher()

        resolution = self.detector.resolution
        test_seed = seed
        train_seed = seed + extension_seed_offset
        evaluation_data_dir = test_data_dir or data_dir
        evaluator = ContinualEvaluator()

        target_test_loader = mixed_test_loader = None
        pre_extend_target_metrics = pre_extend_mixed_metrics = None
        if not synthetic:
            target_test_ds = load_dataset_from_raw(
                raw_dir=evaluation_data_dir,
                class_filter=class_name,
                split="test",
                class_id_offset=self.detector.base_num_classes,
                img_size=resolution,
                seed=test_seed,
            )
            target_test_loader = DataLoader(
                target_test_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
            pre_extend_target_metrics = evaluator.evaluate_standard_detector(
                detector=self.detector,
                dataloader=target_test_loader,
                class_names=[class_name],
                class_ids=[self.detector.get_class_id(class_name)],
                include_curves=True,
            )
            mixed_test_ds = load_dataset_from_raw(
                raw_dir=evaluation_data_dir,
                class_filter=seen_classes,
                split="test",
                class_id_offset=self.detector.base_num_classes,
                img_size=resolution,
                seed=test_seed,
            )
            mixed_test_loader = DataLoader(
                mixed_test_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
            pre_extend_mixed_metrics = evaluator.evaluate_standard_detector(
                detector=self.detector,
                dataloader=mixed_test_loader,
                class_names=seen_classes,
                class_ids=[self.detector.get_class_id(c) for c in seen_classes],
                include_curves=True,
            )

        if synthetic:
            new_ds = SyntheticDetectionDataset(100, self.detector.get_num_classes(), resolution)
            val_ds = SyntheticDetectionDataset(50, self.detector.get_num_classes(), resolution)
        else:
            new_ds = load_dataset_from_raw(
                raw_dir=data_dir,
                class_filter=class_name,
                split="train",
                class_id_offset=self.detector.base_num_classes,
                img_size=resolution,
                seed=train_seed,
                max_samples=max_samples,
                sample_offset=sample_offset,
            )
            val_ds = load_dataset_from_raw(
                raw_dir=data_dir,
                class_filter=class_name,
                split="val",
                class_id_offset=self.detector.base_num_classes,
                img_size=resolution,
                seed=train_seed,
            )

        if self.replay_datasets:
            train_ds = ConcatDataset([new_ds] + self.replay_datasets)
            print(f"  New: {len(new_ds)}, Replay: {sum(len(r) for r in self.replay_datasets)}")
        else:
            train_ds = new_ds
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
        )

        params = [p for p in self.detector.model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=self.lr * 0.01)

        history = []
        best_epoch = 0
        best_val_loss = float("inf")
        best_val_map50 = float("-inf")
        target_class_ids = [self.detector.get_class_id(class_name)]
        for epoch in range(1, epochs + 1):
            self.detector.model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
                pixel_values = batch["pixel_values"].to(self.detector.device)
                targets = self._batch_targets(batch)
                loss = self._dkd_loss(pixel_values, targets, old_class_ids)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            val_metrics = evaluate_loss(self.detector, val_loader)
            entry = {"epoch": epoch, "loss": avg_loss, "val_loss": val_metrics["loss"]}
            val_detection_metrics = None
            if (
                not synthetic
                and metrics_eval_every > 0
                and (epoch % metrics_eval_every == 0 or epoch == epochs)
            ):
                val_detection_metrics = evaluator.evaluate_standard_detector(
                    detector=self.detector,
                    dataloader=val_loader,
                    class_names=[class_name],
                    class_ids=target_class_ids,
                )
                entry.update({f"val_{k}": v for k, v in val_detection_metrics.items()})
            history.append(entry)
            current_val_map50 = (
                float(val_detection_metrics["mAP@0.5"])
                if val_detection_metrics is not None
                else float("-inf")
            )
            if is_better_validation_checkpoint(
                current_val_map50, val_metrics["loss"], best_val_map50, best_val_loss
            ):
                best_epoch = epoch
                best_val_loss = val_metrics["loss"]
                best_val_map50 = max(best_val_map50, current_val_map50)
                save_model_checkpoint(
                    self.detector.model,
                    experiment_dir,
                    class_name,
                    checkpoint_root="best_checkpoints",
                )

        load_model_checkpoint(
            self.detector.model, experiment_dir, class_name, checkpoint_root="best_checkpoints"
        )
        self._drop_teacher()

        self._register_replay_buffer(
            class_name=class_name,
            seed=train_seed,
            resolution=resolution,
            dataset=new_ds,
            max_samples=max_samples,
            sample_offset=sample_offset,
            raw_dir=data_dir,
        )
        if not synthetic:
            self._calibration_step(params, optimizer_lr=self.lr * 0.1)
            self._save_state(experiment_dir, class_name)

        if synthetic:
            test_target_metrics = {}
            test_mixed_metrics = {}
        else:
            test_target_metrics = evaluator.evaluate_standard_detector(
                detector=self.detector,
                dataloader=target_test_loader,
                class_names=[class_name],
                class_ids=[self.detector.get_class_id(class_name)],
                include_curves=True,
            )
            test_mixed_metrics = evaluator.evaluate_standard_detector(
                detector=self.detector,
                dataloader=mixed_test_loader,
                class_names=seen_classes,
                class_ids=[self.detector.get_class_id(c) for c in seen_classes],
                include_curves=True,
            )

        save_model_checkpoint(self.detector.model, experiment_dir, class_name)

        target_extension_delta = mixed_extension_delta = None
        keys = ("mAP@0.5", "mAP@0.75", "mAP@0.95", "mAP@0.5:0.95")
        if pre_extend_target_metrics is not None and test_target_metrics:
            target_extension_delta = {
                k: float(test_target_metrics.get(k, 0.0) - pre_extend_target_metrics.get(k, 0.0))
                for k in keys
            }
        if pre_extend_mixed_metrics is not None and test_mixed_metrics:
            mixed_extension_delta = {
                k: float(test_mixed_metrics.get(k, 0.0) - pre_extend_mixed_metrics.get(k, 0.0))
                for k in keys
            }

        results = {
            "method": "cl_detr",
            "mode": "extend",
            "buffer_per_class": self.buffer_per_class,
            "target_class": class_name,
            "seen_classes": seen_classes,
            "source_experiment_dir": load_dir,
            "source_checkpoint_task": load_task_name,
            "output_dir": str(experiment_dir),
            "train_data_dir": data_dir,
            "test_data_dir": evaluation_data_dir,
            "final_checkpoint_task": class_name,
            "final_checkpoint_dir": str(experiment_dir / "checkpoints" / class_name),
            "history": history,
            "pre_extend_target_metrics": pre_extend_target_metrics,
            "pre_extend_mixed_metrics": pre_extend_mixed_metrics,
            "test_target_metrics": test_target_metrics,
            "test_mixed_metrics": test_mixed_metrics,
            "final_evaluation": test_mixed_metrics,
            "target_extension_delta": target_extension_delta,
            "mixed_extension_delta": mixed_extension_delta,
            "config": {
                "seed": seed,
                "train_seed": train_seed,
                "max_samples": max_samples,
                "sample_offset": sample_offset,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_map50": None if best_val_map50 == float("-inf") else best_val_map50,
                "eval_split": "test",
                "runtime": collect_runtime_metadata(),
            },
        }
        with open(experiment_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        with open(experiment_dir / "evaluation.json", "w") as f:
            json.dump(
                {
                    "method": "cl_detr",
                    "mode": "extend",
                    "pre_extend_target_metrics": pre_extend_target_metrics,
                    "pre_extend_mixed_metrics": pre_extend_mixed_metrics,
                    "test_target_metrics": test_target_metrics,
                    "test_mixed_metrics": test_mixed_metrics,
                    "target_extension_delta": target_extension_delta,
                    "mixed_extension_delta": mixed_extension_delta,
                    "history": history,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"[CL-DETR] Extension results saved to {experiment_dir}")
        return results
