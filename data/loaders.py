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
        
        # Parse BreakHis structure: SOB_B_A-14-13411B/40X/SOB_B_A-14-13411B-20150629132155.png
        for patient_dir in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            
            # Extract patient ID (e.g., "SOB_B_A-14-13411B" from folder name)
            patient_id = patient_dir.split('/')[0]
            label = 0 if 'B' in patient_id else 1  # B=benign, M=malignant
            
            for mag in ['40X', '100X', '200X', '400X']:
                mag_dir = os.path.join(patient_path, mag)
                if not os.path.isdir(mag_dir):
                    continue
                
                for img_file in os.listdir(mag_dir):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(mag_dir, img_file)
                        self.samples.append({
                            'path': img_path,
                            'label': label,
                            'patient_id': patient_id
                        })
        
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
    Split dataset into labeled/unlabeled using GroupShuffleSplit.
    Stratifies by class, groups by patient ID.
    """
    samples = dataset.samples
    patient_ids = np.array([s['patient_id'] for s in samples])
    labels = np.array([s['label'] for s in samples])
    
    # Group by patient
    unique_patients = np.unique(patient_ids)
    patient_labels = np.array([labels[patient_ids == p][0] for p in unique_patients])
    
    # Stratified split by class at patient level
    n_labeled = max(1, int(len(unique_patients) * label_pct / 100))
    
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=1-label_pct/100, 
                                      random_state=seed)
    labeled_patients_idx, unlabeled_patients_idx = next(
        splitter.split(unique_patients, patient_labels)
    )
    
    labeled_patients = unique_patients[labeled_patients_idx]
    unlabeled_patients = unique_patients[unlabeled_patients_idx]
    
    # Map back to image indices
    labeled_mask = np.isin(patient_ids, labeled_patients)
    unlabeled_mask = np.isin(patient_ids, unlabeled_patients)
    
    labeled_indices = np.where(labeled_mask)[0]
    unlabeled_indices = np.where(unlabeled_mask)[0]
    
    return {
        'labeled_indices': labeled_indices,
        'unlabeled_indices': unlabeled_indices,
        'n_labeled': len(labeled_indices),
        'n_unlabeled': len(unlabeled_indices)
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
