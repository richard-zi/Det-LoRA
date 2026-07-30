import json
from pathlib import Path

import det_lora.final_runner as run_final_runner
from det_lora.final_runner import ExtensionSpec, _build_run_specs, run_final_suite
from det_lora.utils import resolve_variant_settings


def test_resolve_variant_settings_prefers_cli_overrides():
    resolved = resolve_variant_settings(
        variant="large",
        preset_name="l40_final",
        base_defaults={
            "epochs": 30,
            "batch_size": 4,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "lora_rank": 8,
            "lora_alpha": 16,
            "metrics_eval_every": 5,
        },
        overrides={"batch_size": 2, "epochs": 12},
    )
    assert resolved["batch_size"] == 2
    assert resolved["epochs"] == 12
    assert resolved["lr"] == 8e-5
    assert resolved["lora_rank"] == 12
    assert resolved["lora_alpha"] == 24


def test_build_run_specs_for_all_phase():
    specs = _build_run_specs(
        {
            "phase": "all",
            "models": ["nano", "medium"],
            "seeds": [1, 2],
            "methods": ["finetuning", "ewc"],
        }
    )
    keys = {spec.key for spec in specs}
    assert "training:nano:1:det_lora" in keys
    assert "training:medium:2:det_lora" in keys
    assert "baseline:nano:1:finetuning" in keys
    assert "baseline:medium:2:ewc" in keys
    assert len(specs) == 12


def test_run_baseline_dispatches_joint_finetuning(monkeypatch, tmp_path):
    captured = {}

    class DummyJointFineTuningBaseline:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def run_experiment(self, **kwargs):
            captured["run"] = kwargs
            return {
                "method": "joint_finetuning",
                "output_dir": str(tmp_path),
                "final_checkpoint_task": "joint",
                "final_checkpoint_dir": str(tmp_path / "checkpoints" / "joint"),
                "final_evaluation": {"mAP@0.5": 0.5, "mAP@0.95": 0.1},
            }

    monkeypatch.setattr(
        "det_lora.final_runner.JointFineTuningBaseline",
        DummyJointFineTuningBaseline,
    )

    spec = run_final_runner.RunSpec(
        kind="baseline",
        model="medium",
        seed=7,
        method="joint_finetuning",
    )
    config = {
        "classes": ["class_a", "class_b"],
        "data_dir": "data/raw",
        "synthetic": True,
        "preset": "l40_final",
    }

    result = run_final_runner._run_baseline(spec, config, tmp_path / "joint")

    assert captured["init"]["variant"] == "medium"
    assert captured["run"]["classes"] == ["class_a", "class_b"]
    assert captured["run"]["resume_dir"] == str(tmp_path / "joint")
    assert captured["run"]["metrics_eval_every"] == 2
    assert result["method"] == "joint_finetuning"


def test_run_final_suite_resumes_completed_runs(monkeypatch, tmp_path):
    training_calls = []
    baseline_calls = []

    def fake_training(spec, config, run_dir):
        training_calls.append(spec.key)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "final").mkdir(parents=True, exist_ok=True)
        result = {
            "output_dir": str(run_dir),
            "mixed_final_evaluation": {"mAP@0.5": 0.5, "mAP@0.5:0.95": 0.3},
            "matched_final_evaluation": {"mAP@0.5": 0.6},
            "matched_forgetting": {"class_a": 0.0},
        }
        (run_dir / "results.json").write_text(json.dumps(result))
        return result

    def fake_baseline(spec, config, run_dir):
        baseline_calls.append(spec.key)
        run_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "output_dir": str(run_dir),
            "final_evaluation": {"mAP@0.5": 0.25, "mAP@0.5:0.95": 0.1},
            "forgetting": {"class_a": 0.2},
        }
        (run_dir / "results.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr("det_lora.final_runner._run_training", fake_training)
    monkeypatch.setattr("det_lora.final_runner._run_baseline", fake_baseline)

    config = {
        "suite_name": "resume_suite",
        "phase": "all",
        "models": ["medium"],
        "classes": ["class_a"],
        "seeds": [7],
        "preset": "l40_final",
        "data_dir": "data/raw",
        "save_dir": str(tmp_path),
        "methods": ["finetuning"],
        "synthetic": True,
        "max_samples": 2,
        "enable_shared_quality_calibrator": True,
    }

    run_final_suite(config)
    assert training_calls == ["training:medium:7:det_lora"]
    assert baseline_calls == ["baseline:medium:7:finetuning"]

    run_final_suite(config)
    assert training_calls == ["training:medium:7:det_lora"]
    assert baseline_calls == ["baseline:medium:7:finetuning"]

    summary_path = Path(config["save_dir"]) / "suites" / config["suite_name"] / "suite_summary.json"
    assert summary_path.exists()


def test_run_final_suite_runs_extension_suite(monkeypatch, tmp_path):
    call_log = []
    stage_run_dirs = {}

    def _write_result(run_dir, payload):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "results.json").write_text(json.dumps(payload))

    def fake_training(spec, config, run_dir):
        call_log.append(f"main:{spec.kind}:{spec.method}")
        (run_dir / "final").mkdir(parents=True, exist_ok=True)
        result = {
            "output_dir": str(run_dir),
            "mixed_final_evaluation": {"mAP@0.5": 0.5, "mAP@0.5:0.95": 0.3},
            "matched_final_evaluation": {"mAP@0.5": 0.6, "mAP@0.5:0.95": 0.4},
            "matched_forgetting": {"class_a": 0.0},
        }
        _write_result(run_dir, result)
        return result

    def fake_baseline(spec, config, run_dir):
        call_log.append(f"main:{spec.kind}:{spec.method}")
        result = {
            "output_dir": str(run_dir),
            "final_evaluation": {"mAP@0.5": 0.25, "mAP@0.5:0.95": 0.1},
            "forgetting": {"class_a": 0.2},
        }
        _write_result(run_dir, result)
        return result

    def fake_extension_training(spec, config, run_dir, source_run_dir):
        call_log.append(
            f"extend:{spec.stage_name}:{spec.kind}:{spec.method}:{spec.target_class}:{source_run_dir.name}"
        )
        if spec.stage_name == "stage_2":
            assert source_run_dir == stage_run_dirs[(spec.method, "stage_1")]
        (run_dir / "final").mkdir(parents=True, exist_ok=True)
        result = {
            "output_dir": str(run_dir),
            "source_experiment_dir": str(source_run_dir / "final"),
            "final_checkpoint_dir": str(run_dir / "final"),
            "mixed_final_evaluation": {"mAP@0.5": 0.7, "mAP@0.5:0.95": 0.4},
            "matched_final_evaluation": {"mAP@0.5": 0.8, "mAP@0.5:0.95": 0.5},
            "pre_extend_metrics": {"mAP@0.5": 0.5, "mAP@0.5:0.95": 0.2},
            "pre_extend_target_metrics": {"mAP@0.5": 0.45, "mAP@0.5:0.95": 0.15},
            "mixed_extension_delta": {"mAP@0.5": 0.2, "mAP@0.5:0.95": 0.2},
            "target_extension_delta": {"mAP@0.5": 0.35, "mAP@0.5:0.95": 0.35},
        }
        _write_result(run_dir, result)
        stage_run_dirs[(spec.method, spec.stage_name)] = run_dir
        return result

    def fake_extension_baseline(spec, config, run_dir, source_run_dir):
        call_log.append(
            f"extend:{spec.stage_name}:{spec.kind}:{spec.method}:{spec.target_class}:{source_run_dir.name}"
        )
        if spec.stage_name == "stage_2":
            assert source_run_dir == stage_run_dirs[(spec.method, "stage_1")]
        result = {
            "output_dir": str(run_dir),
            "source_experiment_dir": str(source_run_dir),
            "final_checkpoint_dir": str(run_dir / "checkpoints" / spec.target_class),
            "test_mixed_metrics": {"mAP@0.5": 0.3, "mAP@0.5:0.95": 0.12},
            "test_target_metrics": {"mAP@0.5": 0.4, "mAP@0.5:0.95": 0.18},
            "pre_extend_mixed_metrics": {"mAP@0.5": 0.2, "mAP@0.5:0.95": 0.1},
            "pre_extend_target_metrics": {"mAP@0.5": 0.25, "mAP@0.5:0.95": 0.11},
            "mixed_extension_delta": {"mAP@0.5": 0.1, "mAP@0.5:0.95": 0.02},
            "target_extension_delta": {"mAP@0.5": 0.15, "mAP@0.5:0.95": 0.07},
        }
        _write_result(run_dir, result)
        stage_run_dirs[(spec.method, spec.stage_name)] = run_dir
        return result

    monkeypatch.setattr("det_lora.final_runner._run_training", fake_training)
    monkeypatch.setattr("det_lora.final_runner._run_baseline", fake_baseline)
    monkeypatch.setattr("det_lora.final_runner._run_extension_training", fake_extension_training)
    monkeypatch.setattr("det_lora.final_runner._run_extension_baseline", fake_extension_baseline)

    config = {
        "suite_name": "extend_suite",
        "phase": "all",
        "models": ["medium"],
        "classes": ["class_a"],
        "seeds": [7],
        "preset": "l40_final",
        "data_dir": "data/raw",
        "save_dir": str(tmp_path),
        "methods": ["finetuning"],
        "synthetic": True,
        "max_samples": 2,
        "enable_shared_quality_calibrator": True,
        "extension": {
            "enabled": True,
            "classes": ["class_a"],
            "stages": [
                {"name": "stage_1", "epochs": 2, "max_samples": 4},
                {"name": "stage_2", "epochs": 2, "max_samples": 8},
            ],
            "seed_offset": 5,
        },
    }

    run_final_suite(config)

    assert call_log == [
        "main:training:det_lora",
        "main:baseline:finetuning",
        "extend:stage_1:extension:det_lora:class_a:det_lora",
        "extend:stage_1:extension:finetuning:class_a:finetuning",
        "extend:stage_2:extension:det_lora:class_a:stage_1",
        "extend:stage_2:extension:finetuning:class_a:stage_1",
    ]

    suite_dir = Path(config["save_dir"]) / "suites" / config["suite_name"]
    extend_dir = suite_dir / "extend"
    assert (extend_dir / "suite_summary.json").exists()
    assert (extend_dir / "suite_summary.md").exists()
    summary = json.loads((extend_dir / "suite_summary.json").read_text())
    assert "medium:det_lora:class_a:stage_1" in summary["groups"]
    assert "medium:det_lora:class_a:stage_2" in summary["groups"]


def test_extension_specs_include_grow_freeze_strategy():
    config = {
        "suite_name": "extend_strategy_suite",
        "phase": "training",
        "models": ["medium"],
        "classes": ["class_a"],
        "seeds": [7],
        "preset": "l40_final",
        "data_dir": "data/raw",
        "save_dir": "experiments",
        "methods": [],
        "synthetic": True,
        "extension": {
            "enabled": True,
            "classes": ["class_a"],
            "det_lora_strategies": ["warm_start", "grow_freeze"],
            "version_selection_strategy": "anchor_latest",
            "stages": [
                {"name": "stage_1", "epochs": 2, "max_samples": 4},
                {"name": "stage_2", "epochs": 2, "max_samples": 8, "disjoint_max_samples": 4},
            ],
        },
    }

    specs = run_final_runner._build_extension_run_specs(config)
    by_method_stage = {(spec.method, spec.stage_name): spec for spec in specs}

    assert by_method_stage[("det_lora_warm_start", "stage_2")].max_samples == 8
    assert by_method_stage[("det_lora_warm_start", "stage_2")].sample_offset == 0
    assert by_method_stage[("det_lora_grow_freeze", "stage_2")].max_samples == 4
    assert by_method_stage[("det_lora_grow_freeze", "stage_2")].sample_offset == 4
    assert by_method_stage[("det_lora_grow_freeze", "stage_2")].extend_strategy == "grow_freeze"
    assert (
        by_method_stage[("det_lora_grow_freeze", "stage_2")].version_selection_strategy
        == "anchor_latest"
    )


def test_extension_suite_uses_separate_test_data_dir(monkeypatch, tmp_path):
    captured = {}

    def fake_train_adapter(**kwargs):
        captured["train_adapter"] = kwargs
        return {
            "output_dir": str(tmp_path / "extend"),
            "source_experiment_dir": kwargs["load_dir"],
            "final_checkpoint_dir": str(tmp_path / "extend" / "final"),
            "mixed_final_evaluation": {"mAP@0.5": 0.7, "mAP@0.5:0.95": 0.4},
            "matched_final_evaluation": {"mAP@0.5": 0.8, "mAP@0.5:0.95": 0.5},
            "mixed_extension_delta": {"mAP@0.5": 0.2, "mAP@0.5:0.95": 0.2},
            "target_extension_delta": {"mAP@0.5": 0.35, "mAP@0.5:0.95": 0.35},
        }

    class DummyFineTuningBaseline:
        def __init__(self, *args, **kwargs):
            pass

        def extend_experiment(self, **kwargs):
            captured["baseline_extend"] = kwargs
            return {
                "output_dir": str(tmp_path / "baseline_extend"),
                "source_experiment_dir": kwargs["load_dir"],
                "final_checkpoint_dir": str(
                    tmp_path / "baseline_extend" / "checkpoints" / "class_a"
                ),
                "test_mixed_metrics": {"mAP@0.5": 0.3, "mAP@0.5:0.95": 0.12},
                "test_target_metrics": {"mAP@0.5": 0.4, "mAP@0.5:0.95": 0.18},
                "pre_extend_mixed_metrics": {"mAP@0.5": 0.2, "mAP@0.5:0.95": 0.1},
                "pre_extend_target_metrics": {"mAP@0.5": 0.25, "mAP@0.5:0.95": 0.11},
                "mixed_extension_delta": {"mAP@0.5": 0.1, "mAP@0.5:0.95": 0.02},
                "target_extension_delta": {"mAP@0.5": 0.15, "mAP@0.5:0.95": 0.07},
            }

    monkeypatch.setattr("det_lora.final_runner.train_adapter", fake_train_adapter)
    monkeypatch.setattr("det_lora.final_runner.FineTuningBaseline", DummyFineTuningBaseline)

    config = {
        "suite_name": "extend_paths",
        "phase": "training",
        "models": ["medium"],
        "classes": ["class_a"],
        "seeds": [7],
        "preset": "l40_final",
        "data_dir": "data/raw",
        "save_dir": str(tmp_path),
        "methods": ["finetuning"],
        "synthetic": True,
        "max_samples": 2,
        "enable_shared_quality_calibrator": True,
        "extension": {
            "enabled": True,
            "classes": ["class_a"],
            "data_dir": "data/extension/raw",
            "test_data_dir": "data/raw",
            "stages": [{"name": "stage_1", "epochs": 2, "max_samples": 4}],
            "seed_offset": 5,
        },
    }

    training_spec = ExtensionSpec(
        kind="extension",
        model="medium",
        seed=7,
        method="det_lora",
        target_class="class_a",
        stage_name="stage_1",
        stage_index=0,
        epochs=2,
        max_samples=4,
        sample_offset=0,
        source_stage_name=None,
    )
    baseline_spec = ExtensionSpec(
        kind="extension",
        model="medium",
        seed=7,
        method="finetuning",
        target_class="class_a",
        stage_name="stage_1",
        stage_index=0,
        epochs=2,
        max_samples=4,
        sample_offset=3,
        source_stage_name=None,
    )
    source_run_dir = tmp_path / "source_run"
    run_dir = tmp_path / "run_dir"

    train_result = run_final_runner._run_extension_training(
        training_spec, config, run_dir, source_run_dir
    )
    assert captured["train_adapter"]["data_dir"] == "data/extension/raw"
    assert captured["train_adapter"]["test_data_dir"] == "data/raw"
    assert captured["train_adapter"]["sample_offset"] == 0
    assert captured["train_adapter"]["version_selection_strategy"] == "anchor_latest"
    assert captured["train_adapter"]["use_hard_negatives"] is True

    baseline_result = run_final_runner._run_extension_baseline(
        baseline_spec, config, run_dir, source_run_dir
    )
    assert captured["baseline_extend"]["data_dir"] == "data/extension/raw"
    assert captured["baseline_extend"]["test_data_dir"] == "data/raw"
    assert captured["baseline_extend"]["sample_offset"] == 3
    assert train_result["stage_name"] == "stage_1"
    assert baseline_result["stage_name"] == "stage_1"
