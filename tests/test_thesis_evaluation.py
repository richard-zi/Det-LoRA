from scripts.run_thesis_evaluation import (
    build_run_rows,
    build_summary_rows,
    flatten_metric_block,
    parse_extension_group_name,
    parse_main_group_name,
    select_best_run,
)


def test_parse_main_group_name():
    assert parse_main_group_name("large:det_lora") == {
        "model": "large",
        "method": "det_lora",
    }


def test_parse_extension_group_name():
    assert parse_extension_group_name("small:replay:military_tank:stage_2") == {
        "model": "small",
        "method": "replay",
        "target_class": "military_tank",
        "stage_name": "stage_2",
    }


def test_build_summary_rows_flattens_extension_metrics():
    summary = {
        "groups": {
            "small:det_lora_grow_freeze:military_tank:stage_2": {
                "num_seeds": 3,
                "seeds": [42, 43, 44],
                "mixed_metrics": {"mAP@0.5": {"mean": 0.7, "std": 0.1}},
                "target_metrics": {"mAP@0.5": {"mean": 0.8, "std": 0.05}},
                "target_extension_delta": {"mAP@0.5": {"mean": 0.02, "std": 0.01}},
                "mixed_extension_delta": {"mAP@0.5": {"mean": -0.01, "std": 0.02}},
                "runs": [],
            }
        }
    }

    rows = build_summary_rows(summary, "extension")

    assert rows == [
        {
            "model": "small",
            "method": "det_lora_grow_freeze",
            "target_class": "military_tank",
            "stage_name": "stage_2",
            "group_name": "small:det_lora_grow_freeze:military_tank:stage_2",
            "num_seeds": 3,
            "seeds": "42,43,44",
            "mixed_mAP@0.5_mean": 0.7,
            "mixed_mAP@0.5_std": 0.1,
            "target_mAP@0.5_mean": 0.8,
            "target_mAP@0.5_std": 0.05,
            "target_delta_mAP@0.5_mean": 0.02,
            "target_delta_mAP@0.5_std": 0.01,
            "mixed_delta_mAP@0.5_mean": -0.01,
            "mixed_delta_mAP@0.5_std": 0.02,
        }
    ]


def test_build_run_rows_and_select_best_run():
    summary = {
        "groups": {
            "small:det_lora:military_tank:stage_2": {
                "runs": [
                    {
                        "seed": 42,
                        "output_dir": "run_a",
                        "mixed_metrics": {"mAP@0.5:0.95": 0.4},
                        "matched_metrics": {"mAP@0.5:0.95": 0.6},
                        "target_extension_delta": {"mAP@0.5": 0.01},
                        "mixed_extension_delta": {"mAP@0.5": -0.02},
                    },
                    {
                        "seed": 43,
                        "output_dir": "run_b",
                        "mixed_metrics": {"mAP@0.5:0.95": 0.5},
                        "matched_metrics": {"mAP@0.5:0.95": 0.7},
                        "target_extension_delta": {"mAP@0.5": 0.03},
                        "mixed_extension_delta": {"mAP@0.5": -0.01},
                    },
                ]
            }
        }
    }

    rows = build_run_rows(summary, "extension")
    best = select_best_run(
        rows,
        method="det_lora",
        metric_key_name="target_mAP@0.5:0.95",
        stage_name="stage_2",
        target_class="military_tank",
    )

    assert len(rows) == 2
    assert best["seed"] == 43
    assert best["output_dir"] == "run_b"


def test_flatten_metric_block_accepts_scalar_average():
    assert flatten_metric_block("forgetting", 0.0) == {"forgetting_mean": 0.0}
