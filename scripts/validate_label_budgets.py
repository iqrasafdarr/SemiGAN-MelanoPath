import sys
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "train.py"
CONFIG = ROOT / "configs" / "default.yaml"

sys.path.insert(0, str(ROOT))

# Load root train.py explicitly
spec = importlib.util.spec_from_file_location(
    "semigan_root_train",
    TRAIN_FILE
)
train_module = importlib.util.module_from_spec(spec)
sys.modules["semigan_root_train"] = train_module
spec.loader.exec_module(train_module)

SemiGANTrainer = train_module.SemiGANTrainer

from data.loaders import (
    BreakHisDataset,
    get_patient_train_val_test_split,
)

budgets = [1, 5, 10, 20, 100]
seed = 42

print("=" * 80)
print("SEMI GAN-MELANOPATH V2 - LABEL BUDGET VALIDATION")
print("=" * 80)

trainer = SemiGANTrainer(
    config_path=str(CONFIG),
    exp_name="budget_validation",
    label_pct=1,
    seed=seed,
    experiment="E5_full",
)

print("\nLoading full BreakHis dataset...")

full_dataset = BreakHisDataset(
    root_dir=trainer.config["data"]["breakhis_root"]
)

print(f"Full dataset images: {len(full_dataset)}")

# IMPORTANT:
# First create the patient-level train/val/test split.
split = get_patient_train_val_test_split(
    full_dataset,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=seed,
)

train_indices = split["train_indices"]
val_indices = split["val_indices"]
test_indices = split["test_indices"]

print(f"Train images: {len(train_indices)}")
print(f"Validation images: {len(val_indices)}")
print(f"Test images: {len(test_indices)}")

# IMPORTANT:
# _get_train_label_split() must receive the TRAIN dataset,
# not the complete 7,909-image dataset.
train_dataset = BreakHisDataset(
    root_dir=trainer.config["data"]["breakhis_root"],
    indices=train_indices,
)

print(f"Training dataset object: {len(train_dataset)} images")

all_passed = True

for budget in budgets:

    print(f"\n--- Requested budget: {budget}% ---")

    trainer.label_pct = budget
    trainer.seed = seed

    budget_split = trainer._get_train_label_split(train_dataset)

    labeled = set(budget_split["labeled_indices"])
    unlabeled = set(budget_split["unlabeled_indices"])

    train_set = set(range(len(train_dataset)))

    checks = {
        "non_empty_labeled": len(labeled) > 0,

        "labeled_indices_valid": (
            labeled.issubset(train_set)
        ),

        "unlabeled_indices_valid": (
            unlabeled.issubset(train_set)
        ),

        "labeled_unlabeled_disjoint": (
            labeled.isdisjoint(unlabeled)
        ),

        "all_train_images_accounted": (
            labeled | unlabeled
        ) == train_set,

        "patient_count_valid": (
            0 < budget_split["n_labeled_patients"]
            <= budget_split["n_train_patients"]
        ),
    }

    if budget == 100:
        checks["100pct_all_images_labeled"] = (
            budget_split["n_labeled"]
            == budget_split["n_train_images"]
        )

        checks["100pct_no_unlabeled_images"] = (
            budget_split["n_unlabeled"] == 0
        )

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(
            f"  {name:<42} {status}"
        )

        if not passed:
            all_passed = False

    print(
        f"  Labeled patients : "
        f"{budget_split['n_labeled_patients']} / "
        f"{budget_split['n_train_patients']}"
    )

    print(
        f"  Labeled images   : "
        f"{budget_split['n_labeled']} / "
        f"{budget_split['n_train_images']}"
    )

    print(
        f"  Actual patient % : "
        f"{budget_split['actual_patient_pct']:.2f}%"
    )

    print(
        f"  Actual image %   : "
        f"{budget_split['actual_image_pct']:.2f}%"
    )

print("\n" + "=" * 80)

if all_passed:
    print("ALL LABEL-BUDGET CHECKS PASSED")
else:
    print("LABEL-BUDGET VALIDATION FAILED")

print("=" * 80)

raise SystemExit(0 if all_passed else 1)
