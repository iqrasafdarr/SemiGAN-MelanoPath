from pathlib import Path
import argparse
import copy
import json
import subprocess
import sys

import yaml


CONFIG_PATH = Path("experiments/experiment_configs.yaml")


EXPERIMENTS = [
    "E1_supervised",
    "E2_semigan",
    "E3_semigan_vat",
    "E4_semigan_vat_rotation",
    "E5_full",
]


def load_matrix():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_experiment_config(experiment_name):
    config = load_matrix()
    experiments = config["experiments"]

    if experiment_name not in experiments:
        available = ", ".join(experiments.keys())
        raise ValueError(
            f"Unknown experiment '{experiment_name}'. "
            f"Available experiments: {available}"
        )

    experiment = copy.deepcopy(
        experiments[experiment_name]
    )

    return {
        "name": experiment_name,
        "description": experiment["description"],
        "features": experiment,
        "label_budgets": config["label_budgets"],
        "seeds": config["seeds"],
        "dataset": config["dataset"],
    }


def build_run_config(experiment_name, label_pct, seed):
    config = load_experiment_config(experiment_name)

    if label_pct not in config["label_budgets"]:
        raise ValueError(
            f"Label budget {label_pct}% is not configured. "
            f"Available: {config['label_budgets']}"
        )

    return {
        "experiment": config["name"],
        "description": config["description"],
        "label_pct_requested": label_pct,
        "seed": seed,
        "dataset": config["dataset"],
        "features": {
            "use_gan": config["features"]["use_gan"],
            "use_vat": config["features"]["use_vat"],
            "use_rotation_ssl": config["features"]["use_rotation_ssl"],
            "use_feature_matching": config["features"]["use_feature_matching"],
            "use_noise_injection": config["features"]["use_noise_injection"],
        },
    }


def get_run_dir(experiment_name, label_pct, seed):
    return (
        Path("runs")
        / experiment_name
        / f"label_{label_pct}pct"
        / f"seed_{seed}"
    )


def save_run_config(
    experiment_name,
    label_pct,
    seed,
):
    run_config = build_run_config(
        experiment_name,
        label_pct,
        seed,
    )

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

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            run_config,
            f,
            indent=2,
        )

    return run_config, config_path


def validate_matrix():
    config = load_matrix()

    experiments = config["experiments"]
    label_budgets = config["label_budgets"]
    seeds = config["seeds"]

    assert len(experiments) == 5
    assert set(experiments.keys()) == set(
        EXPERIMENTS
    )

    assert label_budgets == [
        1,
        5,
        10,
        20,
        100,
    ]

    assert seeds == [42]

    for experiment_name in EXPERIMENTS:
        experiment = experiments[experiment_name]

        required_keys = [
            "description",
            "use_gan",
            "use_vat",
            "use_rotation_ssl",
            "use_feature_matching",
            "use_noise_injection",
        ]

        for key in required_keys:
            assert key in experiment

    total_runs = (
        len(experiments)
        * len(label_budgets)
        * len(seeds)
    )

    print("")
    print("=" * 70)
    print("EXPERIMENT MATRIX")
    print("=" * 70)
    print(f"Experiments: {len(experiments)}")
    print(f"Label budgets: {label_budgets}")
    print(f"Seeds: {seeds}")
    print(f"Total planned runs: {total_runs}")
    print("")

    for name in EXPERIMENTS:
        print(
            f"{name}: "
            f"{experiments[name]['description']}"
        )

    print("")
    print("MATRIX VALIDATION PASSED")


def dry_run(
    experiment_name,
    label_pct,
    seed,
):
    _, config_path = save_run_config(
        experiment_name,
        label_pct,
        seed,
    )

    print("")
    print("DRY RUN PASSED")
    print(f"Configuration: {config_path}")


def run_training(
    experiment_name,
    label_pct,
    seed,
):
    run_config, config_path = save_run_config(
        experiment_name,
        label_pct,
        seed,
    )

    run_dir = get_run_dir(
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
        run_dir.as_posix(),
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
    ) as log_file:

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Training failed with exit code "
            f"{return_code}. "
            f"See {log_path}"
        )

    print("")
    print("=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)
    print(f"Results directory: {run_dir}")
    print(f"Training log: {log_path}")
    print(f"Config: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "SemiGAN-MelanoPath experiment runner"
        )
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
