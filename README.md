# Det-LoRA

Parameter-efficient class-incremental object detection with RF-DETR and per-class LoRA adapters.

This repository contains the reference implementation and the full experiment suite for the master thesis *Det-LoRA: Parameter-Efficient Class-Incremental Object Detection with RF-DETR* (Richard Zimmermann, FOM Hochschule, 2026).

## Idea

Det-LoRA learns object classes one after another without forgetting the classes it already knows. The base detector, RF-DETR (DINOv2 backbone plus transformer decoder), stays completely frozen. Each new class is trained into its own LoRA adapter on the decoder attention layers. Adapters are activated and deactivated at inference time and never merged into the base model, so a finished class can no longer be changed by later training. Forgetting is ruled out by construction rather than mitigated.

Two further pieces make this work in practice:

- an expandable classification head with gradient masking, so training class *n* cannot shift the output neurons of classes *1..n-1*
- per-adapter logistic score calibration, with an optional shared quality calibrator on top, so scores from independently trained adapters become comparable during joint inference

Joint inference over all classes shares one encoder pass and only swaps the decoder adapter per class, which keeps the per-class cost at the decoder share instead of a full forward pass.

The benchmark dataset is the Mendeley *Military and Civilian Vehicles* set with six classes: `military_tank`, `military_truck`, `military_aircraft`, `military_helicopter`, `civilian_car`, `civilian_aircraft`.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/). On Apple Silicon, set `PYTORCH_ENABLE_MPS_FALLBACK=1` for every training command.

```bash
uv sync                 # install dependencies
uv run pytest -q        # test suite, runs without GPU and without data

# pipeline smoke test with synthetic data, no real data needed
uv run python -m det_lora.train --class_name test --epochs 2 --synthetic
```

The RF-DETR weights (`rf-detr-*.pth`) are downloaded automatically by the `rfdetr` library on first use. The raw dataset is expected under `data/raw/` and is not part of this repository.

## Running experiments

```bash
# train a single class
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m det_lora.train \
  --class_name military_tank --epochs 10

# full class-incremental experiment (sequential, resumable)
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m det_lora.run_experiment \
  --classes military_tank military_truck --epochs 10

# thesis iterations (canonical entry point, see configs/README.md)
uv run python scripts/run_iteration.py --list   # iteration -> config mapping
uv run python scripts/run_iteration.py 1        # run iteration 1

# baseline comparison (fine-tuning, EWC, replay vs. Det-LoRA)
uv run python -m det_lora.baselines.compare \
  --classes military_tank military_truck --epochs 10
```

Results are written to `experiments/` with per-run logs, checkpoints, metrics, and aggregated comparison tables. `docs/EXPERIMENT_PROTOCOL.md` documents the benchmark protocol (classes, seeds, metrics, splits); `docs/CLUSTER_RUNS.md` describes how the production runs were executed on an NVIDIA L40 and how the evaluation artifacts are generated.

The aggregated outcomes of the six production suites reported in the thesis are checked in under `results/`: per suite the `suite_summary.json`, `suite_results.json`, and `resolved_config.json`, plus the merged CSV tables under `results/tables/`. The full run trees with checkpoints and logs are too large for git and stay local.

## Repository layout

| Path | Content |
|---|---|
| `src/det_lora/` | core library |
| `src/det_lora/model/` | `detector.py` (RF-DETR wrapper, head expansion), `det_lora.py` (adapter lifecycle, calibration) |
| `src/det_lora/train.py` | single-class training loop, including the `--extend` mode |
| `src/det_lora/run_experiment.py` | sequential class-incremental experiment with resume |
| `src/det_lora/final_runner.py` | config-driven suite runner (variants x seeds x methods) |
| `src/det_lora/evaluation/` | continual evaluation (forgetting, mAP), COCO metrics, calibration |
| `src/det_lora/baselines/` | fine-tuning, EWC, replay, CL-DETR (CL-DETR runs via `configs/baselines/cldetr.json`) |
| `src/det_lora/sdk.py` | adapter SDK (CLI and Python API) for checkpoint management |
| `scripts/` | suite and cluster scripts, evaluation, figure generation |
| `scripts/ablations/` | ablation and probe scripts, plus stored raw logs under `data/` |
| `configs/` | suite configurations; `configs/README.md` maps each design iteration to its config |
| `results/` | published suite summaries, resolved configs, and aggregated tables of the production runs |
| `tests/` | pytest suite, runs without GPU |
| `data/`, `experiments/`, `figures/`, `models/` | runtime artifacts, not versioned |

## Design notes

Class IDs: RF-DETR keeps its 90 COCO outputs, new classes get IDs from 91 upward (tank = 91, truck = 92, and so on).

LoRA targets: `cross_attn.value_proj`, `cross_attn.output_proj`, and `self_attn.out_proj` in the decoder; the backbone is never touched. Iteration 4 widens the footprint to the decoder FFN and the localization layers (`bbox_embed`, `sampling_offsets`, `attention_weights`).

Gradient masking: while a new class trains, gradients of all previously trained head neurons are zeroed. The head parameter group runs with `weight_decay = 0`, otherwise AdamW would still drift the masked neurons.

Checkpoint layout: a finalized checkpoint directory holds `det_lora_registry.json` (class to adapter mapping), `adapter_calibration.json` (calibrators), `head_state.pt`, and one PEFT weight directory per adapter.

A note on the figure scripts: `scripts/make_thesis_figures.py`, `scripts/make_symhn_figure.py`, and `scripts/make_gate_figure.py` generate pgfplots graphics with German axis labels, since they feed directly into the German thesis manuscript.

## License

MIT, see `LICENSE`.
