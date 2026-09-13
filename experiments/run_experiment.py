from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


EXPERIMENTS = [
    "E1_supervised",
    "E2_semigan",
    "E3_semigan_vat",
    "E4_semigan_vat_rotation",
    "E5_full",
]


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments" / "experiment_configs.yaml"


def load_matrix():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_experiment_config(experiment_name):
    matrix = load_matrix()
    experiments = matrix["experiments"]

    if experiment_name not in experiments:
        raise ValueError(
            f"Unknown experiment: {experiment_name}. "
            f"Expected one of: {EXPERIMENTS}"
        )

    return matrix, experiments[experiment_name]


def get_run_name(experiment_name, label_pct, seed):
    return f"{experiment_name}_lbl{label_pct}_s{seed}"


def get_run_dir(experiment_name, label_pct, seed):
    return ROOT / "runs" / get_run_name(
        experiment_name,
        label_pct,
        seed,
    )


def build_run_config(experiment_name, label_pct, seed):
    matrix, experiment = load_experiment_config(experiment_name)

    return {
        "experiment": experiment_name,
        "description": experiment["description"],
        "label_pct_requested": label_pct,
        "seed": seed,
        "dataset": matrix["dataset"],
        "features": {
            "use_gan": experiment["use_gan"],
            "use_vat": experiment["use_vat"],
            "use_rotation_ssl": experiment["use_rotation_ssl"],
            "use_feature_matching": experiment["use_feature_matching"],
            "use_noise_injection": experiment["use_noise_injection"],
        },
    }


def save_run_config(experiment_name, label_pct, seed):
    run_dir = get_run_dir(
        experiment_name,
        label_pct,
        seed,
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = run_dir / "config.json"

    config = build_run_config(
        experiment_name,
        label_pct,
        seed,
    )

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            config,
            handle,
            indent=2,
        )

    return config_path


def validate_matrix():
    matrix = load_matrix()

    experiments = list(matrix["experiments"].keys())
    label_budgets = matrix["label_budgets"]
    seeds = matrix["seeds"]

    if experiments != EXPERIMENTS:
        raise ValueError(
            f"Experiment matrix mismatch.\n"
            f"Expected: {EXPERIMENTS}\n"
            f"Found: {experiments}"
        )

    if label_budgets != [1, 5, 10, 20, 100]:
        raise ValueError(
            f"Unexpected label budgets: {label_budgets}"
        )

    if seeds != [42]:
        raise ValueError(
            f"Unexpected seeds: {seeds}"
        )

    total_runs = (
        len(experiments)
        * len(label_budgets)
        * len(seeds)
    )

    print("=" * 70)
    print("EXPERIMENT MATRIX")
    print("=" * 70)
    print(f"Experiments: {len(experiments)}")
    print(f"Label budgets: {label_budgets}")
    print(f"Seeds: {seeds}")
    print(f"Total planned runs: {total_runs}")
    print("")

    for name in experiments:
        print(
            f"{name}: "
            f"{matrix['experiments'][name]['description']}"
        )

    print("")
    print("MATRIX VALIDATION PASSED")


def dry_run(experiment_name, label_pct, seed):
    config_path = save_run_config(
        experiment_name,
        label_pct,
        seed,
    )

    run_dir = get_run_dir(
        experiment_name,
        label_pct,
        seed,
    )

    print("")
    print("=" * 70)
    print("DRY RUN PASSED")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Label budget: {label_pct}%")
    print(f"Seed: {seed}")
    print(f"Run directory: {run_dir}")
    print(f"Configuration: {config_path}")


def run_training(experiment_name, label_pct, seed):
    run_dir = get_run_dir(
        experiment_name,
        label_pct,
        seed,
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = save_run_config(
        experiment_name,
        label_pct,
        seed,
    )

    log_path = run_dir / "training.log"

    command = [
        sys.executable,
        "train.py",
        "--config",
        "configs/default.yaml",
        "--exp",
        experiment_name,
        "--label_pct",
        str(label_pct),
        "--seed",
        str(seed),
        "--experiment",
        experiment_name,
    ]

    print("")
    print("=" * 70)
    print("STARTING EXPERIMENT")
    print("=" * 70)
    print(f"Experiment: {experiment_name}")
    print(f"Label budget: {label_pct}%")
    print(f"Seed: {seed}")
    print(f"Run directory: {run_dir}")
    print("")
    print("Command:")
    print(" ".join(command))
    print("")

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_handle:

        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
            log_handle.flush()

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Training failed with exit code "
            f"{return_code}. "
            f"See {log_path}"
        )

    expected_results = run_dir / "results.json"

    # train.py uses the same canonical directory naming.
    # Verify that the expected research artifact exists.
    if not expected_results.exists():
        raise RuntimeError(
            "Training finished but results.json was not found at "
            f"{expected_results}"
        )

    print("")
    print("=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)
    print(f"Results directory: {run_dir}")
    print(f"Training log: {log_path}")
    print(f"Config: {config_path}")
    print(f"Results: {expected_results}")


def main():
    parser = argparse.ArgumentParser(
        description="SemiGAN-MelanoPath experiment runner"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--run",
        action="store_true",
    )

    parser.add_argument(
        "--experiment",
        default="E5_full",
        choices=EXPERIMENTS,
    )

    parser.add_argument(
        "--label-pct",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.validate:
        validate_matrix()
        return

    if args.dry_run:
        dry_run(
            args.experiment,
            args.label_pct,
            args.seed,
        )
        return

    if args.run:
        run_training(
            args.experiment,
            args.label_pct,
            args.seed,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
