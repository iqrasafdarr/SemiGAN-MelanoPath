import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from data.loaders import (
    BreakHisDataset,
    get_patient_train_val_test_split,
)


ROOT = r"D:\archive (1)\BreaKHis_v1\histology_slides\breast"

dataset = BreakHisDataset(ROOT)

split = get_patient_train_val_test_split(
    dataset,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
)

train_patients = set(split["train_patients"])
val_patients = set(split["val_patients"])
test_patients = set(split["test_patients"])

print("=" * 60)
print("PATIENT-LEVEL SPLIT TEST")
print("=" * 60)

print(f"Total images:    {len(dataset)}")
print(f"Total patients:  {len(set(s['patient_id'] for s in dataset.samples))}")

print(f"\nTrain patients:  {len(train_patients)}")
print(f"Train images:    {len(split['train_indices'])}")

print(f"\nValidation patients: {len(val_patients)}")
print(f"Validation images:   {len(split['val_indices'])}")

print(f"\nTest patients:   {len(test_patients)}")
print(f"Test images:     {len(split['test_indices'])}")

print("\nPatient overlap:")
print("Train ∩ Val: ", len(train_patients & val_patients))
print("Train ∩ Test:", len(train_patients & test_patients))
print("Val ∩ Test:  ", len(val_patients & test_patients))

assert len(train_patients & val_patients) == 0
assert len(train_patients & test_patients) == 0
assert len(val_patients & test_patients) == 0

all_indices = set(
    split["train_indices"]
) | set(
    split["val_indices"]
) | set(
    split["test_indices"]
)

assert len(all_indices) == len(dataset)

print("\n" + "=" * 60)
print("PATIENT SPLIT TEST PASSED")
print("=" * 60)
