"""
Elastic Weight Consolidation (EWC) Baseline
=============================================

Kirkpatrick et al. 2017 - Overcoming catastrophic forgetting in neural networks.
After each task, compute Fisher Information Matrix to identify important weights.
Penalize changes to important weights during subsequent tasks.
"""

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.evaluator import ContinualEvaluator
from det_lora.model.detector import RFDETRDetector
from det_lora.train import SyntheticDetectionDataset, collate_fn, evaluate_loss
from det_lora.utils import collect_runtime_metadata, set_global_seed


class EWCBaseline:
    """
    Elastic Weight Consolidation baseline.

    After each task, computes the diagonal Fisher Information Matrix
    and penalizes deviations from optimal weights of previous tasks.
    """

    def __init__(
        self,
        variant: str = "medium",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        ewc_lambda: float = 1000.0,
    ):
        self.detector = RFDETRDetector(variant=variant)
        self.lr = lr
        self.weight_decay = weight_decay
        self.ewc_lambda = ewc_lambda

        # Freeze backbone only
        for name, param in self.detector.model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # EWC storage
        self._fisher: Dict[str, torch.Tensor] = {}
        self._optimal_params: Dict[str, torch.Tensor] = {}

        trainable = sum(p.numel() for p in self.detector.model.parameters() if p.requires_grad)
        print(f"[EWC] Trainable params: {trainable:,}, lambda={ewc_lambda}")

    @staticmethod
    def _overlap_slices(*tensors: torch.Tensor) -> Optional[tuple[slice, ...]]:
        """Return the shared prefix view across tensors with possibly expanded heads."""
        if not tensors:
            return None
        ndim = tensors[0].ndim
        if any(tensor.ndim != ndim for tensor in tensors[1:]):
            return None
        return tuple(slice(0, min(tensor.shape[dim] for tensor in tensors)) for dim in range(ndim))

    def _compute_fisher(self, dataloader: DataLoader, num_samples: int = 200) -> None:
        """Compute diagonal Fisher Information Matrix from data."""
        self.detector.model.eval()
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.detector.model.named_parameters()
            if p.requires_grad
        }

        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            pixel_values = batch["pixel_values"].to(self.detector.device)
            targets = []
            for label in batch["labels"]:
                lbl_key = "labels" if "labels" in label else "class_labels"
                targets.append(
                    {
                        "labels": label[lbl_key].to(self.detector.device),
                        "boxes": label["boxes"].to(self.detector.device),
                    }
                )
            out = self.detector.forward(pixel_values=pixel_values, targets=targets)
            loss = out["loss"]
            self.detector.model.zero_grad()
            loss.backward()

            for n, p in self.detector.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
            count += pixel_values.shape[0]

        # Normalize
        for n in fisher:
            fisher[n] /= max(count, 1)

        # Accumulate with previous Fisher (multi-task)
        for n in fisher:
            if n in self._fisher:
                overlap = self._overlap_slices(self._fisher[n], fisher[n])
                if overlap is None:
                    fisher[n] = fisher[n]
                else:
                    fisher[n][overlap] += self._fisher[n][overlap]
                self._fisher[n] = fisher[n]
            else:
                self._fisher[n] = fisher[n]

        # Store current optimal params
        for n, p in self.detector.model.named_parameters():
            if p.requires_grad:
                self._optimal_params[n] = p.data.clone()

        print(f"[EWC] Fisher computed from {count} samples")

    def _ewc_penalty(self) -> torch.Tensor:
        """Compute EWC penalty: sum F_i * (theta_i - theta_star_i)^2."""
        if not self._fisher:
            return torch.tensor(0.0, device=self.detector.device)
        penalty = torch.tensor(0.0, device=self.detector.device)
        for n, p in self.detector.model.named_parameters():
            if n in self._fisher and p.requires_grad:
                optimal = self._optimal_params.get(n)
                if optimal is None:
                    continue
                overlap = self._overlap_slices(p, self._fisher[n], optimal)
                if overlap is None:
                    continue
                penalty += (self._fisher[n][overlap] * (p[overlap] - optimal[overlap]).pow(2)).sum()
        return penalty

    def _serialize_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "fisher": {name: tensor.detach().cpu() for name, tensor in self._fisher.items()},
            "optimal_params": {
                name: tensor.detach().cpu() for name, tensor in self._optimal_params.items()
            },
        }

    def _save_state(self, experiment_dir: Path, task_name: str) -> None:
        from det_lora.baselines.checkpoint import save_state

        save_state(experiment_dir, task_name, self._serialize_state())

    def _load_state(self, experiment_dir: Path, task_name: str) -> bool:
        from det_lora.baselines.checkpoint import load_state

        state = load_state(experiment_dir, task_name)
        if not state:
            return False

        self._fisher = {
            name: tensor.to(self.detector.device)
            for name, tensor in state.get("fisher", {}).items()
        }
        self._optimal_params = {
            name: tensor.to(self.detector.device)
            for name, tensor in state.get("optimal_params", {}).items()
        }
        return True

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
        """Run EWC experiment with checkpoint/resume."""
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
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            experiment_dir = Path(save_dir) / f"baseline_ewc_{timestamp}"
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
                "method": "ewc",
                "ewc_lambda": self.ewc_lambda,
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
                print(f"[EWC] Task {task_idx + 1}/{len(classes)}: {class_name} - SKIPPED")
                self.detector.expand_classification_head(class_name)
                load_model_checkpoint(self.detector.model, experiment_dir, class_name)
                if not synthetic:
                    self._load_state(experiment_dir, class_name)
                continue
            print(f"\n[EWC] Task {task_idx + 1}/{len(classes)}: {class_name}")

            self.detector.expand_classification_head(class_name)

            if synthetic:
                ds = SyntheticDetectionDataset(100, self.detector.get_num_classes(), resolution)
                val_ds = SyntheticDetectionDataset(50, self.detector.get_num_classes(), resolution)
            else:
                ds = load_dataset_from_raw(
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
            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
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
                total_ewc = 0
                for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
                    pixel_values = batch["pixel_values"].to(self.detector.device)
                    targets = []
                    for label in batch["labels"]:
                        lbl_key = "labels" if "labels" in label else "class_labels"
                        targets.append(
                            {
                                "labels": label[lbl_key].to(self.detector.device),
                                "boxes": label["boxes"].to(self.detector.device),
                            }
                        )
                    out = self.detector.forward(pixel_values=pixel_values, targets=targets)
                    loss = out["loss"]

                    ewc_loss = self._ewc_penalty() * self.ewc_lambda
                    total = loss + ewc_loss

                    optimizer.zero_grad()
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    total_ewc += ewc_loss.item()

                scheduler.step()
                val_metrics = evaluate_loss(self.detector, val_loader)
                avg_loss = total_loss / len(loader)
                avg_ewc = total_ewc / len(loader)
                entry = {
                    "epoch": epoch,
                    "loss": avg_loss,
                    "ewc_loss": avg_ewc,
                    "val_loss": val_metrics["loss"],
                }
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
                    entry.update(
                        {f"val_{key}": value for key, value in val_detection_metrics.items()}
                    )
                history.append(entry)
                current_val_map50 = (
                    float(val_detection_metrics["mAP@0.5"])
                    if val_detection_metrics is not None
                    else float("-inf")
                )
                if is_better_validation_checkpoint(
                    current_val_map50,
                    val_metrics["loss"],
                    best_val_map50,
                    best_val_loss,
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
                        f"EWC: {avg_ewc:.4f} | Val: {val_metrics['loss']:.4f}{metric_text}"
                    )

            load_model_checkpoint(
                self.detector.model,
                experiment_dir,
                class_name,
                checkpoint_root="best_checkpoints",
            )

            # Compute Fisher after this task
            self._compute_fisher(loader)
            if not synthetic:
                self._save_state(experiment_dir, class_name)

            results["tasks"][class_name] = {
                "history": history,
                "final_loss": history[-1]["loss"],
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_map50": None if best_val_map50 == float("-inf") else best_val_map50,
            }

            if not synthetic:
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
                class_ids = [self.detector.get_class_id(cls) for cls in seen_classes]
                metrics = evaluator.evaluate_standard_detector(
                    detector=self.detector,
                    dataloader=eval_loader,
                    class_names=seen_classes,
                    class_ids=class_ids,
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
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

        results["output_dir"] = str(experiment_dir)
        results["final_checkpoint_task"] = classes[-1] if classes else None
        results["final_checkpoint_dir"] = (
            str(experiment_dir / "checkpoints" / classes[-1]) if classes else None
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[EWC] Results saved to {experiment_dir}")
        return results

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
        """Extend an already trained class with EWC regularization."""
        from det_lora.baselines.checkpoint import (
            is_better_validation_checkpoint,
            load_model_checkpoint,
            prepare_detector_for_checkpoint_load,
            save_model_checkpoint,
        )

        set_global_seed(seed)
        experiment_dir = Path(save_dir) / (
            experiment_name or f"extend_{class_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)

        prepare_detector_for_checkpoint_load(self.detector, seen_classes)
        if not load_model_checkpoint(self.detector.model, Path(load_dir), load_task_name):
            raise FileNotFoundError(
                f"Source checkpoint not found for {load_task_name} in {load_dir}"
            )
        if not synthetic and not self._load_state(Path(load_dir), load_task_name):
            raise FileNotFoundError(f"EWC state not found for {load_task_name} in {load_dir}")

        resolution = self.detector.resolution
        test_seed = seed
        train_seed = seed + extension_seed_offset
        evaluation_data_dir = test_data_dir or data_dir

        evaluator = ContinualEvaluator()
        pre_extend_target_metrics = None
        pre_extend_mixed_metrics = None
        test_target_metrics = None
        test_mixed_metrics = None

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
                class_ids=[self.detector.get_class_id(cls) for cls in seen_classes],
                include_curves=True,
            )

        if synthetic:
            train_dataset = SyntheticDetectionDataset(
                100, self.detector.get_num_classes(), resolution
            )
            val_dataset = SyntheticDetectionDataset(50, self.detector.get_num_classes(), resolution)
        else:
            train_dataset = load_dataset_from_raw(
                raw_dir=data_dir,
                class_filter=class_name,
                split="train",
                class_id_offset=self.detector.base_num_classes,
                img_size=resolution,
                seed=train_seed,
                max_samples=max_samples,
                sample_offset=sample_offset,
            )
            val_dataset = load_dataset_from_raw(
                raw_dir=data_dir,
                class_filter=class_name,
                split="val",
                class_id_offset=self.detector.base_num_classes,
                img_size=resolution,
                seed=train_seed,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
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
            total_ewc = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
                pixel_values = batch["pixel_values"].to(self.detector.device)
                targets = []
                for label in batch["labels"]:
                    lbl_key = "labels" if "labels" in label else "class_labels"
                    targets.append(
                        {
                            "labels": label[lbl_key].to(self.detector.device),
                            "boxes": label["boxes"].to(self.detector.device),
                        }
                    )
                out = self.detector.forward(pixel_values=pixel_values, targets=targets)
                loss = out["loss"]
                ewc_loss = self._ewc_penalty() * self.ewc_lambda
                total = loss + ewc_loss

                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_ewc += ewc_loss.item()

            scheduler.step()
            val_metrics = evaluate_loss(self.detector, val_loader)
            avg_loss = total_loss / len(train_loader)
            avg_ewc = total_ewc / len(train_loader)
            entry = {
                "epoch": epoch,
                "loss": avg_loss,
                "ewc_loss": avg_ewc,
                "val_loss": val_metrics["loss"],
            }
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
                entry.update({f"val_{key}": value for key, value in val_detection_metrics.items()})
            history.append(entry)
            current_val_map50 = (
                float(val_detection_metrics["mAP@0.5"])
                if val_detection_metrics is not None
                else float("-inf")
            )
            if is_better_validation_checkpoint(
                current_val_map50,
                val_metrics["loss"],
                best_val_map50,
                best_val_loss,
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
                    f"EWC: {avg_ewc:.4f} | Val: {val_metrics['loss']:.4f}{metric_text}"
                )

        load_model_checkpoint(
            self.detector.model,
            experiment_dir,
            class_name,
            checkpoint_root="best_checkpoints",
        )
        self._compute_fisher(train_loader)
        if not synthetic:
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
                class_ids=[self.detector.get_class_id(cls) for cls in seen_classes],
                include_curves=True,
            )

        save_model_checkpoint(self.detector.model, experiment_dir, class_name)

        target_extension_delta = None
        mixed_extension_delta = None
        if pre_extend_target_metrics is not None and test_target_metrics is not None:
            target_extension_delta = {
                key: float(
                    test_target_metrics.get(key, 0.0) - pre_extend_target_metrics.get(key, 0.0)
                )
                for key in ("mAP@0.5", "mAP@0.75", "mAP@0.95", "mAP@0.5:0.95")
            }
        if pre_extend_mixed_metrics is not None and test_mixed_metrics is not None:
            mixed_extension_delta = {
                key: float(
                    test_mixed_metrics.get(key, 0.0) - pre_extend_mixed_metrics.get(key, 0.0)
                )
                for key in ("mAP@0.5", "mAP@0.75", "mAP@0.95", "mAP@0.5:0.95")
            }

        results = {
            "method": "ewc",
            "mode": "extend",
            "ewc_lambda": self.ewc_lambda,
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
        evaluation_payload = {
            "method": "ewc",
            "mode": "extend",
            "pre_extend_target_metrics": pre_extend_target_metrics,
            "pre_extend_mixed_metrics": pre_extend_mixed_metrics,
            "test_target_metrics": test_target_metrics,
            "test_mixed_metrics": test_mixed_metrics,
            "target_extension_delta": target_extension_delta,
            "mixed_extension_delta": mixed_extension_delta,
            "history": history,
        }
        with open(experiment_dir / "evaluation.json", "w") as f:
            json.dump(evaluation_payload, f, indent=2, default=str)
        print(f"[EWC] Extension results saved to {experiment_dir}")
        return results
