import argparse
import copy
import json
from pathlib import Path

import yaml


CONFIG_PATH = Path("experiments/experiment_configs.yaml")


def load_experiment_config(experiment_name):
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    experiments = config["experiments"]

    if experiment_name not in experiments:
        available = ", ".join(experiments.keys())
        raise ValueError(
            f"Unknown experiment '{experiment_name}'. "
            f"Available experiments: {available}"
        )

    experiment = copy.deepcopy(experiments[experiment_name])

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


def main():
    parser = argparse.ArgumentParser(
        description="Validate SemiGAN-MelanoPath experiment configuration."
    )

    parser.add_argument(
        "--experiment",
        default="E5_full",
        choices=[
            "E1_supervised",
            "E2_semigan",
            "E3_semigan_vat",
            "E4_semigan_vat_rotation",
            "E5_full",
        ],
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

    run_config = build_run_config(
        args.experiment,
        args.label_pct,
        args.seed,
    )

    print("\n===== EXPERIMENT CONFIGURATION =====")
    print(json.dumps(run_config, indent=2))

    run_dir = (
        Path("runs")
        / args.experiment
        / f"label_{args.label_pct}pct"
        / f"seed_{args.seed}"
    )

    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nRun directory: {run_dir}")
    print(f"Configuration saved: {config_path}")
    print("\nEXPERIMENT CONFIG TEST PASSED")


if __name__ == "__main__":
    main()
