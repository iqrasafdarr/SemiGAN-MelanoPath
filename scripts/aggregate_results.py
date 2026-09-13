import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


EXPERIMENTS = [
    "E1_supervised",
    "E2_semigan",
    "E3_semigan_vat",
    "E4_semigan_vat_rotation",
    "E5_full",
]

RUN_PATTERN = re.compile(
    r"^(E[1-5]_[a-z_]+)_lbl(\d+)_s(\d+)$"
)


def load_results(results_dir):
    """Load canonical results.json files from all experiment run directories."""
    results_dir = Path(results_dir)
    records = []

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        match = RUN_PATTERN.match(run_dir.name)
        if not match:
            continue

        experiment, label_pct, seed = match.groups()
        results_file = run_dir / "results.json"

        if not results_file.exists():
            continue

        try:
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"WARNING: Could not read {results_file}: {exc}")
            continue

        metrics = data.get("test_metrics", {})

        record = {
            "run": run_dir.name,
            "experiment": experiment,
            "label_pct_requested": int(label_pct),
            "seed": int(seed),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
            "actual_patient_pct": data.get("actual_patient_pct"),
            "actual_image_pct": data.get("actual_image_pct"),
            "n_labeled_patients": data.get("n_labeled_patients"),
            "n_labeled_images": data.get("n_labeled_images"),
            "best_epoch": data.get("best_epoch"),
        }

        records.append(record)

    return records


def print_runs(records):
    print("\n" + "=" * 100)
    print("CANONICAL EXPERIMENT RESULTS")
    print("=" * 100)

    if not records:
        print("No canonical results.json files found.")
        return

    for r in records:
        print(
            f"{r['run']:<42} "
            f"Acc={format_metric(r['accuracy'])} "
            f"F1={format_metric(r['f1'])} "
            f"AUC={format_metric(r['roc_auc'])}"
        )


def format_metric(value):
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def aggregate_by_experiment(records):
    """Aggregate metrics by experiment."""
    print("\n" + "=" * 100)
    print("AGGREGATION BY EXPERIMENT")
    print("=" * 100)

    for experiment in EXPERIMENTS:
        rows = [r for r in records if r["experiment"] == experiment]

        if not rows:
            continue

        print(f"\n{experiment} ({len(rows)} runs)")

        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            values = [
                float(r[metric])
                for r in rows
                if r[metric] is not None
            ]

            if values:
                mean = np.mean(values)
                std = np.std(values)

                print(
                    f"  {metric:<10}: "
                    f"{mean:.4f} +/- {std:.4f}"
                )


def aggregate_by_label_budget(records):
    """Aggregate metrics by requested label budget."""
    print("\n" + "=" * 100)
    print("AGGREGATION BY LABEL BUDGET")
    print("=" * 100)

    budgets = sorted(
        set(r["label_pct_requested"] for r in records)
    )

    for budget in budgets:
        rows = [
            r for r in records
            if r["label_pct_requested"] == budget
        ]

        print(f"\nRequested labels: {budget}%")

        for experiment in EXPERIMENTS:
            exp_rows = [
                r for r in rows
                if r["experiment"] == experiment
            ]

            if not exp_rows:
                continue

            f1_values = [
                float(r["f1"])
                for r in exp_rows
                if r["f1"] is not None
            ]

            auc_values = [
                float(r["roc_auc"])
                for r in exp_rows
                if r["roc_auc"] is not None
            ]

            f1 = (
                f"{np.mean(f1_values):.4f}"
                if f1_values else "N/A"
            )
            auc = (
                f"{np.mean(auc_values):.4f}"
                if auc_values else "N/A"
            )

            print(
                f"  {experiment:<30} "
                f"F1={f1}  ROC-AUC={auc}"
            )


def save_csv(records, output_path):
    """Save canonical results as CSV."""
    if not records:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "run",
        "experiment",
        "label_pct_requested",
        "seed",
        "actual_patient_pct",
        "actual_image_pct",
        "n_labeled_patients",
        "n_labeled_images",
        "best_epoch",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved CSV: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate SemiGAN-MelanoPath canonical experiment results"
    )

    parser.add_argument(
        "--results-dir",
        default="runs",
        help="Directory containing canonical experiment runs",
    )

    parser.add_argument(
        "--output",
        default="results/aggregated_results.csv",
        help="Output CSV path",
    )

    parser.add_argument(
        "--exp",
        choices=["all", "e1", "e2", "e3", "e4", "e5"],
        default="all",
        help="Experiment filter",
    )

    args = parser.parse_args()

    records = load_results(args.results_dir)

    mapping = {
        "e1": "E1_supervised",
        "e2": "E2_semigan",
        "e3": "E3_semigan_vat",
        "e4": "E4_semigan_vat_rotation",
        "e5": "E5_full",
    }

    if args.exp != "all":
        records = [
            r for r in records
            if r["experiment"] == mapping[args.exp]
        ]

    print("\n" + "=" * 100)
    print("SemiGAN-MelanoPath v2: Results Aggregation")
    print("=" * 100)

    print(f"\nCanonical runs discovered: {len(records)}")

    print_runs(records)
    aggregate_by_experiment(records)
    aggregate_by_label_budget(records)
    save_csv(records, args.output)

    print("\n" + "=" * 100)
    print("Aggregation complete")
    print("=" * 100)


if __name__ == "__main__":
    main()
