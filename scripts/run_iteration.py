#!/usr/bin/env python3
"""Canonical entry point for the five thesis design-science iterations.

Each iteration of the thesis maps to exactly one config file:

    iteration 1  configs/iterations/iteration1_base.json                     suite thesis_l40_main
    iteration 2  configs/iterations/iteration2_symmetric_hard_negatives.json suite thesis_l40_symhn
    iteration 3  configs/iterations/iteration3_conflict_gate.json            eval-only gate sweep
    iteration 4  configs/iterations/iteration4_extended_footprint.json       suite thesis_l40_iter4
    iteration 5  configs/iterations/iteration5_shared_adapter.json           suite thesis_l40_cllora
    cldetr       configs/baselines/cldetr.json                               suite thesis_l40_cldetr
    joint        configs/baselines/joint.json                                suite thesis_l40_joint_baseline

Iterations 1, 2, 4, 5 and the baselines launch the full training suite via
scripts/run_final_suite.py. Iteration 3 is an evaluation-only step: it
re-evaluates the final checkpoints of iterations 1 and 2 with the pairwise
Mahalanobis conflict gate (scripts/ablations/gate_posthoc_sweep.py) and
therefore requires those suites to have finished first.

Usage:
    uv run python scripts/run_iteration.py --list
    uv run python scripts/run_iteration.py 1
    uv run python scripts/run_iteration.py 3
    uv run python scripts/run_iteration.py cldetr
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_BY_TARGET = {
    "1": "configs/iterations/iteration1_base.json",
    "2": "configs/iterations/iteration2_symmetric_hard_negatives.json",
    "3": "configs/iterations/iteration3_conflict_gate.json",
    "4": "configs/iterations/iteration4_extended_footprint.json",
    "5": "configs/iterations/iteration5_shared_adapter.json",
    "cldetr": "configs/baselines/cldetr.json",
    "joint": "configs/baselines/joint.json",
}


def run_command(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_suite(config_path: Path) -> None:
    run_command([sys.executable, "scripts/run_final_suite.py", "--config", str(config_path)])


def run_gate_sweep(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    missing_suites = []
    for suite in config["source_suites"]:
        suite_dir = REPO_ROOT / "experiments" / "suites" / suite
        if not suite_dir.exists():
            missing_suites.append(suite)
    if missing_suites:
        raise SystemExit(
            "Iteration 3 re-evaluates existing checkpoints, but these suites "
            f"have not been run yet: {', '.join(missing_suites)}. "
            "Run iterations 1 and 2 first."
        )
    for suite in config["source_suites"]:
        for model in config["models"]:
            for seed in config["seeds"]:
                checkpoint = (
                    REPO_ROOT
                    / "experiments"
                    / "suites"
                    / suite
                    / f"model_{model}"
                    / f"seed_{seed}"
                    / "det_lora"
                    / "final"
                )
                if not checkpoint.is_dir():
                    print(f"SKIP (no checkpoint): {checkpoint}", flush=True)
                    continue
                run_command(
                    [
                        sys.executable,
                        config["runner"],
                        "--suite",
                        suite,
                        "--variant",
                        model,
                        "--seed",
                        str(seed),
                        "--data_dir",
                        config["data_dir"],
                        "--fit_max_samples",
                        str(config["fit_max_samples"]),
                        "--cache_dir",
                        config["cache_dir"],
                        "--output_dir",
                        config["output_dir"],
                    ]
                )


def print_mapping() -> None:
    for target, relative_path in CONFIG_BY_TARGET.items():
        config = json.loads((REPO_ROOT / relative_path).read_text())
        suite_name = config.get("suite_name", "(evaluation-only, no suite)")
        print(f"{target:>6}  {relative_path:<58} {suite_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "target",
        nargs="?",
        choices=sorted(CONFIG_BY_TARGET),
        help="Iteration number (1-5) or baseline name (cldetr, joint)",
    )
    parser.add_argument("--list", action="store_true", help="Print the iteration-to-config mapping")
    args = parser.parse_args()

    if args.list or args.target is None:
        print_mapping()
        if args.target is None and not args.list:
            parser.error("missing target (or use --list)")
        return

    config_path = REPO_ROOT / CONFIG_BY_TARGET[args.target]
    if args.target == "3":
        run_gate_sweep(config_path)
    else:
        run_suite(config_path)


if __name__ == "__main__":
    main()
