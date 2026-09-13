import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class BreakHisDataset(Dataset):
    """BreakHis histopathology dataset with patient-level grouping."""
    
    def __init__(self, root_dir, split='train', transform=None, indices=None):
        self.root_dir = root_dir
        self.transform = transform or self._default_transform()
        self.split = split
        self.samples = []

        if not os.path.exists(root_dir):
            logger.warning(f"BreakHis not found at {root_dir}. Using dummy dataset.")
            self.samples = self._create_dummy_samples(500)
            return

        # Actual BreakHis structure:
        #
        # breast/
        #   benign/
        #       SOB/
        #           adenosis/
        #               SOB_B_.../
        #                   40X/*.png
        #                   100X/*.png
        #                   200X/*.png
        #                   400X/*.png
        #
        #   malignant/
        #       SOB/
        #           ductal_carcinoma/
        #               SOB_M_.../
        #                   40X/*.png
        #
        # We recursively search the class directories so the
        # intermediate histological subtype directory is handled.

        class_roots = [
            ("benign", 0),
            ("malignant", 1),
        ]

        for class_name, label in class_roots:
            class_root = os.path.join(root_dir, class_name, "SOB")

            if not os.path.isdir(class_root):
                logger.warning(
                    f"BreakHis class directory not found: {class_root}"
                )
                continue

            for current_root, _, files in os.walk(class_root):
                for img_file in files:
                    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue

                    img_path = os.path.join(current_root, img_file)

                    # Patient directory is two levels below the subtype:
                    # SOB/<subtype>/<patient>/<magnification>/<image>
                    relative_path = os.path.relpath(img_path, class_root)
                    parts = relative_path.split(os.sep)

                    if len(parts) >= 4:
                        patient_id = parts[1]
                    else:
                        # Fallback: use the nearest parent directory.
                        patient_id = os.path.basename(
                            os.path.dirname(current_root)
                        )

                    self.samples.append({
                        'path': img_path,
                        'label': label,
                        'patient_id': patient_id
                    })

        logger.info(
            f"Loaded BreakHis: {len(self.samples)} images, "
            f"{len(set(s['patient_id'] for s in self.samples))} patients"
        )

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def _create_dummy_samples(self, n_samples):
        """Create dummy samples for testing when real data unavailable."""
        samples = []
        for i in range(n_samples):
            patient_id = f"PATIENT_{i // 5}"  # 5 images per patient
            label = i % 2
            samples.append({
                'path': None,
                'label': label,
                'patient_id': patient_id
            })
        return samples
    
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if sample['path'] is None:
            # Dummy data
            img = torch.randn(3, 128, 128)
        else:
            img = Image.open(sample['path']).convert('RGB')
            img = self.transform(img)
        
        return {
            'image': img,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'patient_id': sample['patient_id']
        }


class MHISTDataset(Dataset):
    """MHIST external validation dataset with patient-level grouping."""
    
    def __init__(self, root_dir, transform=None, indices=None):
        self.root_dir = root_dir
        self.transform = transform or self._default_transform()
        self.samples = []
        
        if not os.path.exists(root_dir):
            logger.warning(f"MHIST not found at {root_dir}. Using dummy dataset.")
            self.samples = self._create_dummy_samples(200)
            return
        
        # MHIST structure: images_dir/image.png + labels.csv
        labels_file = os.path.join(root_dir, 'labels.csv')
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        img_name, label = parts[0], int(parts[1])
                        img_path = os.path.join(root_dir, img_name)
                        if os.path.exists(img_path):
                            patient_id = img_name.split('_')[0]
                            self.samples.append({
                                'path': img_path,
                                'label': label,
                                'patient_id': patient_id
                            })
        
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
    
    def _create_dummy_samples(self, n_samples):
        samples = []
        for i in range(n_samples):
            patient_id = f"MHIST_{i // 3}"
            label = i % 2
            samples.append({
                'path': None,
                'label': label,
                'patient_id': patient_id
            })
        return samples
    
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if sample['path'] is None:
            img = torch.randn(3, 128, 128)
        else:
            img = Image.open(sample['path']).convert('RGB')
            img = self.transform(img)
        
        return {
            'image': img,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'patient_id': sample['patient_id']
        }


def get_train_val_split(dataset, label_pct=100, seed=42):
    """
    Create a patient-level labeled/unlabeled split.

    The split is performed entirely at the patient level to prevent
    image-level leakage between labeled and unlabeled subsets.

    For extreme label scarcity, the percentage is interpreted as a
    target patient-level budget. Because BreakHis contains a finite
    number of patients, at least one patient from each class is retained
    when a non-zero label percentage is requested.
    """
    if not 0 < label_pct <= 100:
        raise ValueError("label_pct must be in the range (0, 100].")

    samples = dataset.samples

    if not samples:
        raise ValueError("Dataset contains no samples.")

    patient_ids = np.array([s["patient_id"] for s in samples])
    labels = np.array([s["label"] for s in samples])

    unique_patients = np.unique(patient_ids)
    patient_labels = np.array([
        labels[patient_ids == patient][0]
        for patient in unique_patients
    ])

    # 100% means every patient is labeled.
    if label_pct == 100:
        labeled_patients = unique_patients
        unlabeled_patients = np.array([], dtype=unique_patients.dtype)

    else:
        rng = np.random.RandomState(seed)

        classes = np.unique(patient_labels)

        # Target number of labeled patients.
        target_n = max(
            1,
            int(round(len(unique_patients) * label_pct / 100.0))
        )

        # At extreme scarcity, preserve representation from every class.
        min_required = len(classes)

        if target_n < min_required:
            target_n = min_required

        selected = []

        # Select one patient from each class first.
        for cls in classes:
            candidates = np.where(patient_labels == cls)[0]
            selected.append(rng.choice(candidates))

        selected = list(dict.fromkeys(selected))

        # Fill remaining patient budget using a shuffled pool.
        remaining = [
            idx for idx in range(len(unique_patients))
            if idx not in selected
        ]
        rng.shuffle(remaining)

        if len(selected) < target_n:
            selected.extend(
                remaining[:target_n - len(selected)]
            )

        labeled_patient_indices = np.array(
            selected,
            dtype=int
        )

        labeled_patients = unique_patients[labeled_patient_indices]
        unlabeled_patients = np.array([
            patient
            for patient in unique_patients
            if patient not in set(labeled_patients)
        ])

    labeled_mask = np.isin(patient_ids, labeled_patients)
    unlabeled_mask = np.isin(patient_ids, unlabeled_patients)

    labeled_indices = np.where(labeled_mask)[0]
    unlabeled_indices = np.where(unlabeled_mask)[0]

    return {
        "labeled_indices": labeled_indices,
        "unlabeled_indices": unlabeled_indices,
        "n_labeled": len(labeled_indices),
        "n_unlabeled": len(unlabeled_indices),
    }

def create_loaders(dataset, labeled_indices, unlabeled_indices, batch_size=32):
    """Create DataLoaders for labeled and unlabeled subsets."""
    labeled_subset = Subset(dataset, labeled_indices)
    unlabeled_subset = Subset(dataset, unlabeled_indices)
    
    labeled_loader = DataLoader(
        labeled_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    unlabeled_loader = DataLoader(
        unlabeled_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return labeled_loader, unlabeled_loader
def get_patient_train_val_test_split(
    dataset,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    """
    Create deterministic patient-level train/validation/test splits.

    No patient can occur in more than one split.

    The split is performed at patient level first, then image indices
    are assigned to the corresponding patient split.
    """
    if not np.isclose(
        train_ratio + val_ratio + test_ratio,
        1.0
    ):
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must equal 1."
        )

    if min(
        train_ratio,
        val_ratio,
        test_ratio
    ) <= 0:
        raise ValueError(
            "All split ratios must be greater than zero."
        )

    samples = dataset.samples

    if not samples:
        raise ValueError("Dataset contains no samples.")

    patient_to_indices = {}

    for idx, sample in enumerate(samples):
        patient_id = sample["patient_id"]
        patient_to_indices.setdefault(
            patient_id,
            []
        ).append(idx)

    patients = np.array(
        list(patient_to_indices.keys())
    )

    # Determine one class label per patient.
    patient_labels = np.array([
        samples[patient_to_indices[p][0]]["label"]
        for p in patients
    ])

    rng = np.random.RandomState(seed)

    train_patients = []
    val_patients = []
    test_patients = []

    # Stratify by patient class so both classes are represented
    # whenever the number of patients allows it.
    for cls in np.unique(patient_labels):
        cls_patients = patients[
            patient_labels == cls
        ].copy()

        rng.shuffle(cls_patients)

        n = len(cls_patients)

        n_test = max(
            1,
            int(round(n * test_ratio))
        )

        n_val = max(
            1,
            int(round(n * val_ratio))
        )

        # Ensure at least one patient remains for training.
        if n_test + n_val >= n:
            n_test = 1
            n_val = 1

        test_cls = cls_patients[:n_test]
        val_cls = cls_patients[
            n_test:n_test + n_val
        ]
        train_cls = cls_patients[
            n_test + n_val:
        ]

        train_patients.extend(train_cls.tolist())
        val_patients.extend(val_cls.tolist())
        test_patients.extend(test_cls.tolist())

    # Shuffle patient ordering inside each split.
    rng.shuffle(train_patients)
    rng.shuffle(val_patients)
    rng.shuffle(test_patients)

    train_patients = np.array(train_patients)
    val_patients = np.array(val_patients)
    test_patients = np.array(test_patients)

    train_indices = np.array([
        idx
        for patient in train_patients
        for idx in patient_to_indices[patient]
    ], dtype=int)

    val_indices = np.array([
        idx
        for patient in val_patients
        for idx in patient_to_indices[patient]
    ], dtype=int)

    test_indices = np.array([
        idx
        for patient in test_patients
        for idx in patient_to_indices[patient]
    ], dtype=int)

    # Hard safety check: no patient leakage.
    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)

    if train_set & val_set:
        raise RuntimeError(
            "Patient leakage detected: train/validation overlap."
        )

    if train_set & test_set:
        raise RuntimeError(
            "Patient leakage detected: train/test overlap."
        )

    if val_set & test_set:
        raise RuntimeError(
            "Patient leakage detected: validation/test overlap."
        )

    return {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "train_patients": train_patients,
        "val_patients": val_patients,
        "test_patients": test_patients,
    }
