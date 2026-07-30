# Suite configurations

Each design science iteration of the thesis maps to exactly one configuration file. Canonical entry point:

```bash
uv run python scripts/run_iteration.py --list   # show the mapping
uv run python scripts/run_iteration.py 1        # run iteration 1
```

## Iterations

| Iteration | Config | Suite name | Delta to previous iteration |
|---|---|---|---|
| 1: base architecture | `iterations/iteration1_base.json` | `thesis_l40_main` | baseline: frozen RF-DETR, per-class adapters (`default` footprint), two-stage calibration, forward-only hard negatives. Also runs the generic baselines (`methods: finetuning, ewc, replay`) and Track B |
| 2: symmetric hard negatives | `iterations/iteration2_symmetric_hard_negatives.json` | `thesis_l40_symhn` | `+ symmetric_hard_negatives: true` (Track A and Track B) |
| 3: conflict gate (post hoc) | `iterations/iteration3_conflict_gate.json` | none (evaluation only) | no training: re-evaluates the final checkpoints of iterations 1 and 2 with and without the pairwise Mahalanobis gate (`scripts/ablations/gate_posthoc_sweep.py`). Requires completed iterations 1 and 2 |
| 4: extended adapter footprint | `iterations/iteration4_extended_footprint.json` | `thesis_l40_iter4` | `+ lora_target_preset: localization_box_ffn` (decoder FFN `linear1/linear2`, `bbox_embed`, `sampling_offsets`, `attention_weights`); symmetric hard negatives stay active |
| 5: shared adapter (CL-LoRA) | `iterations/iteration5_shared_adapter.json` | `thesis_l40_cllora` | `+ use_shared_adapter: true, shared_drift_weight: 5.0`; otherwise identical to iteration 4. Documented negative result |

## Reference methods

| Method | Config | Suite name | Role |
|---|---|---|---|
| CL-DETR | `baselines/cldetr.json` | `thesis_l40_cldetr` | detector-specific, replay-based baseline (`cl_detr_top_k: 10`, `cl_detr_iou_lambda: 0.7`) |
| joint fine-tuning | `baselines/joint.json` | `thesis_l40_joint_baseline` | upper reference (joint training on all classes) |

Naive fine-tuning, EWC, and experience replay run as `methods` inside the iteration 1 suite and are not retrained for the later iterations.

## Shared protocol

All suites share the five RF-DETR variants (nano to large), seeds 42/43/44, the `l40_final` preset, the official dataset split (`data/raw`), and Track B via `extension` (exception: joint). Results are written to `experiments/suites/<suite_name>/`; the fully resolved configuration manifest is serialized per run.
