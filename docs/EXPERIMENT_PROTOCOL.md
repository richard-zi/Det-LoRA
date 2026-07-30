# Experiment protocol

This is the benchmark protocol used for the thesis experiments.

## Main class-incremental benchmark

- Classes: `military_tank`, `military_truck`, `military_aircraft`, `military_helicopter`, `civilian_car`, `civilian_aircraft`
- Seeds: `42`, `43`, `44`
- Models: `RF-DETR nano`, `small`, `base`, `medium`, `large`
- Evaluation split: official `test`
- Shared quality/objectness calibrator: `on` by default
- Merge reranker: `off` by default for AP-focused reporting
- Selection during single-class training: validation `mAP@0.5` with validation-loss tie-break
- Report both:
  - matched class-wise retention metrics
  - mixed seen-class joint-inference metrics
- Report at least:
  - `mAP@0.5`
  - `mAP@0.75`
  - `mAP@0.5:0.95`
  - `Precision@0.5`
  - `Recall@0.5`
  - `F1@0.5`
  - average forgetting

Run:

```bash
uv run python scripts/run_final_suite.py \
  --config configs/iterations/iteration1_base.json
```

The thesis config also runs a separate post-benchmark extension suite.
It reuses the final checkpoints from the main run, extends the configured
target class in multiple stages with additional images, and stores the extra
artifacts under `experiments/suites/<suite_name>/extend/`.
Each stage can grow the image budget for that class, which allows reporting
stage-by-stage improvements instead of a single one-shot extension.
The staged extension images are read from `data/extension/raw/`.
The held-out test split for extension evaluation still comes from `data/raw`
via `extension.test_data_dir`.

For smaller direct multi-seed runs without the final orchestration layer:

```bash
uv run python scripts/run_suite.py \
  --phase training \
  --classes military_tank military_truck military_aircraft \
  --seeds 42 43 44 \
  --model medium \
  --preset l40_final
```

For the thesis-default full suite across all RF-DETR variants, use the config-driven runner above. Restrict to a single model only for explicit ablations, smoke tests, or recovery runs.

Artifacts:

- `suite_manifest.json`: exact suite configuration
- `suite_state.json`: resumable suite status
- `suite_results.json`: raw per-run results
- `suite_events.jsonl`: append-only event log
- `resolved_config.json`: final config after overrides
- `hardware.json`: machine, git and GPU context
- `suite_summary.json`: machine-readable aggregate summary
- `suite_summary.md`: compact report table
- `PR_curve_per_class@0.5` and `PR_curve_per_class@0.95`: per-class precision-recall curves stored in the run evaluation payloads
- `mAP@0.95`: scalar AP at IoU 0.95 for a stricter localization check
- `extend/`: separate post-run class-extension suite with its own `suite_summary.json`, `suite_summary.md`, and per-run results

## Data-incremental extension check

Use this to verify whether continuing an already known class preserves earlier knowledge:

1. Train the class normally and keep the `final/` checkpoint.
2. Extend from that checkpoint with `--extend --load_dir ...`.
3. Compare `pre_extend_metrics`, `mixed_extension_delta`, and
   `target_extension_delta` in the second run's `results.json`.

Example:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m det_lora.train \
  --class_name military_tank \
  --epochs 10

PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m det_lora.train \
  --class_name military_tank \
  --extend \
  --load_dir experiments/train_military_tank_YYYYMMDD_HHMMSS/final \
  --epochs 5 \
  --stability_loss_weight 1e-5
```

The second run stores:

- `pre_extend_metrics`
- `pre_extend_target_metrics`
- `test_metrics`
- `test_target_metrics`
- `mixed_extension_delta`
- `target_extension_delta`

The main benchmark and the extension pool stay separate, so the extension
images are never mixed back into the core test set.

For the thesis suite, the staged extension uses the image budgets configured in
`configs/iterations/iteration1_base.json` under `extension.stages`. The stages are
cumulative, so later stages see more images than earlier stages.
Place the extra extension images in `data/extension/raw/Images/` and the
corresponding annotations in `data/extension/raw/Labels/CSV Format/train_labels.csv`.
