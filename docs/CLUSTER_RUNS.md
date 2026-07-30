# Production runs on the L40 cluster

The thesis results were produced on a rented NVIDIA L40 pod. This document describes how those runs were started and how the evaluation artifacts are generated. Everything is resumable: if a pod restarts, rerun the same command and the suite continues from its last unfinished run.

## Setup and health checks

```bash
./scripts/bootstrap_cluster.sh      # install uv, sync deps, swap opencv for the headless build
./scripts/healthcheck_cluster.sh    # GPU, CUDA, data layout, test suite
```

The container has no libGL, so `bootstrap_cluster.sh` replaces `opencv-python` with `opencv-python-headless` at the pinned version.

Data is expected under `data/raw/` (`Images/` and `Labels/CSV Format/`) and, for the extension track, under `data/extension/raw/`. The RF-DETR weights (`rf-detr-*.pth`) must be present or downloadable.

## Running the suites

Each suite is one config file (see `configs/README.md`). The generic entry point is:

```bash
uv run python scripts/run_final_suite.py --config configs/iterations/iteration1_base.json
```

For long runs, start through `nohup` so the suite survives a dropped SSH session:

```bash
nohup bash scripts/run_all_cluster.sh > run_all.log 2>&1 &   # iterations 4 + 5, then gate sweep
nohup bash scripts/run_cldetr_cluster.sh > run_cldetr.log 2>&1 &   # CL-DETR baseline suite
```

`run_all_cluster.sh` chains install, environment fixes, GPU and data checks, the test suite, both training suites, and finally the post-hoc gate sweep over all produced checkpoints. `run_l40_thesis_experiments.sh` is the equivalent wrapper for the joint baseline plus the iteration 1 suite.

### Resume behavior

- finished runs are skipped
- interrupted Det-LoRA runs resume from their `progress.json`
- interrupted baseline runs resume from their run directory
- suite status is tracked in `suite_state.json`

## Output layout

```text
experiments/suites/<suite_name>/
  model_<variant>/seed_<n>/...        # Track A runs
  extend/                             # Track B (staged class extension), own summary tree
  suite_manifest.json                 # requested configuration
  resolved_config.json                # effective configuration after overrides
  hardware.json                       # machine, git, and GPU context
  suite_state.json                    # per-run status, used for resume
  suite_events.jsonl                  # append-only event log
  suite_results.json                  # raw per-run results
  suite_summary.json / .md            # aggregated results
```

Per-run directories additionally hold `run.log`, per-class PR curves at IoU 0.5 and 0.95 inside the evaluation payloads, and mAP@0.95 next to the other scalar metrics.

The aggregated outcome of each production suite (`suite_summary.json`, `suite_results.json`, `resolved_config.json`) is preserved in the repository under `results/suites/`, together with the merged CSV tables under `results/tables/`. Only the full run trees with checkpoints and logs stay off git.

## Evaluation artifacts

After the training suites finished:

```bash
./scripts/healthcheck_evaluation.sh
./scripts/run_l40_thesis_evaluation.sh
```

This writes tables (`tables/*.csv`), plots (`plots/*.png`, `plots/*.pdf`), qualitative inference renders (`inference/*/report.json` plus annotated images), `evaluation_report.md`, and a `manifest.json` to `experiments/analysis/thesis_l40_evaluation/`. The qualitative renders use the best completed main run and the best grow-freeze stage 2 run per extension class.

To move results off the pod, zip the suite directories and the logs:

```bash
zip -r results.zip experiments/suites/<suite_name> experiments/logs
```
