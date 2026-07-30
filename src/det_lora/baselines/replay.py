"""
Experience Replay Baseline
============================

Maintains a buffer of exemplars from previous tasks and mixes
them with new task data during training.
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from det_lora.data.dataset import load_dataset_from_raw
from det_lora.evaluation.evaluator import ContinualEvaluator
from det_lora.model.detector import RFDETRDetector
from det_lora.train import SyntheticDetectionDataset, collate_fn, evaluate_loss
from det_lora.utils import collect_runtime_metadata, set_global_seed


class ReplayBaseline:
    """
    Experience Replay baseline.

    Stores a fixed number of exemplars per class from previous tasks.
    During training on a new task, mixes replay data with new data.
    """

    def __init__(
        self,
        variant: str = "medium",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        buffer_per_class: int = 50,
    ):
        self.detector = RFDETRDetector(variant=variant)
        self.lr = lr
        self.weight_decay = weight_decay
        self.buffer_per_class = buffer_per_class

        # Freeze backbone only
        for name, param in self.detector.model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Replay buffer: list of datasets from previous tasks
        self.replay_datasets: List[Subset] = []
        self.replay_buffer_metadata: List[Dict[str, object]] = []

        trainable = sum(p.numel() for p in self.detector.model.parameters() if p.requires_grad)
        print(f"[Replay] Trainable params: {trainable:,}, buffer={buffer_per_class}/class")

    def _sample_buffer(self, dataset, n: int) -> Subset:
        """Sample n random indices from a dataset for the replay buffer."""
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        return Subset(dataset, indices[:n])

    def _serialize_state(self) -> Dict[str, object]:
        buffer_state = []
        for metadata, subset in zip(self.replay_buffer_metadata, self.replay_datasets):
            buffer_state.append(
                {
                    "metadata": metadata,
                    "indices": list(subset.indices),
                }
            )
        return {"buffers": buffer_state}

    def _save_state(self, experiment_dir: Path, task_name: str) -> None:
        from det_lora.baselines.checkpoint import save_state

        save_state(experiment_dir, task_name, self._serialize_state())

    def _load_state(self, experiment_dir: Path, task_name: str) -> bool:
        from det_lora.baselines.checkpoint import load_state

        state = load_state(experiment_dir, task_name)
        if not state:
            return False

        self.replay_datasets = []
        self.replay_buffer_metadata = []
        for buffer_entry in state.get("buffers", []):
            metadata = buffer_entry.get("metadata", {})
            indices = buffer_entry.get("indices", [])
            dataset_args = metadata.get("dataset_args", {})
            dataset = load_dataset_from_raw(**dataset_args)
            self.replay_datasets.append(Subset(dataset, indices))
            self.replay_buffer_metadata.append(metadata)
        return True

    def _register_replay_buffer(
        self,
        *,
        class_name: str,
        seed: int,
        resolution: int,
        dataset,
        max_samples: Optional[int],
        sample_offset: int,
        raw_dir: str,
    ) -> None:
        buffer = self._sample_buffer(dataset, self.buffer_per_class)
        self.replay_datasets.append(buffer)
        self.replay_buffer_metadata.append(
            {
                "class_name": class_name,
                "dataset_args": {
                    "raw_dir": raw_dir,
                    "class_filter": class_name,
                    "split": "train",
                    "class_id_offset": self.detector.base_num_classes,
                    "img_size": resolution,
                    "seed": seed,
                    "max_samples": max_samples,
                    "sample_offset": sample_offset,
                },
            }
        )
        print(f"  Added {len(buffer)} samples to replay buffer")

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
        """Run replay experiment with checkpoint/resume."""
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
            experiment_dir = Path(save_dir) / f"baseline_replay_{timestamp}"
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
                "method": "replay",
                "buffer_per_class": self.buffer_per_class,
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
                print(f"[Replay] Task {task_idx + 1}/{len(classes)}: {class_name} - SKIPPED")
                self.detector.expand_classification_head(class_name)
                load_model_checkpoint(self.detector.model, experiment_dir, class_name)
                if not synthetic:
                    self._load_state(experiment_dir, class_name)
                continue
            print(f"\n[Replay] Task {task_idx + 1}/{len(classes)}: {class_name}")

            self.detector.expand_classification_head(class_name)

            # New task dataset
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

            # Combine with replay buffer
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
                        f"Val: {val_metrics['loss']:.4f}{metric_text}"
                    )

            load_model_checkpoint(
                self.detector.model,
                experiment_dir,
                class_name,
                checkpoint_root="best_checkpoints",
            )

            # Add current task data to replay buffer
            self._register_replay_buffer(
                class_name=class_name,
                seed=seed,
                resolution=resolution,
                dataset=new_ds,
                max_samples=None,
                sample_offset=0,
                raw_dir=data_dir,
            )

            results["tasks"][class_name] = {
                "history": history,
                "final_loss": history[-1]["loss"],
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_map50": None if best_val_map50 == float("-inf") else best_val_map50,
            }
            if not synthetic:
                self._save_state(experiment_dir, class_name)

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
        print(f"[Replay] Results saved to {experiment_dir}")
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
        """Extend an already trained class with experience replay."""
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
            raise FileNotFoundError(f"Replay state not found for {load_task_name} in {load_dir}")

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
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
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
                    f"Val: {val_metrics['loss']:.4f}{metric_text}"
                )

        load_model_checkpoint(
            self.detector.model,
            experiment_dir,
            class_name,
            checkpoint_root="best_checkpoints",
        )

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
            "method": "replay",
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
        evaluation_payload = {
            "method": "replay",
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
        print(f"[Replay] Extension results saved to {experiment_dir}")
        return results
