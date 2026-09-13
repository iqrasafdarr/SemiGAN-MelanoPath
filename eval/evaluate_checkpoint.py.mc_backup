#!/usr/bin/env python3
"""
Evaluate a trained SemiGAN-MelanoPath checkpoint.
Includes MC-Dropout uncertainty quantification.
"""

import os
import sys
import torch
import argparse
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loaders import BreakHisDataset, MHISTDataset, get_train_val_split
from models.architectures import Generator, Discriminator
from eval.evaluator import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CheckpointEvaluator:
    """Evaluate trained discriminator on test set with MC-Dropout."""
    
    def __init__(self, ckpt_path, device='cuda'):
        self.ckpt_path = ckpt_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.evaluator = Evaluator(device=self.device)
        self.checkpoint = None
        self.discriminator = None
    
    def load_checkpoint(self, config):
        """Load checkpoint and instantiate model."""
        self.checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        
        # Instantiate discriminator
        self.discriminator = Discriminator(
            base_channels=config['discriminator']['base_channels'],
            num_classes=config['data']['num_classes'],
            num_rotations=config['discriminator']['num_rotations'],
            classifier_dropout=config['discriminator']['classifier_dropout']
        ).to(self.device)
        
        # Load weights
        self.discriminator.load_state_dict(self.checkpoint['D'])
        self.discriminator.eval()
        
        logger.info(f"Loaded checkpoint from epoch {self.checkpoint.get('epoch', '?')}")
    
    def evaluate_test_set(self, test_loader, mc_passes=20):
        """
        Evaluate on test set with MC-Dropout uncertainty.
        
        Args:
            test_loader: DataLoader with test samples
            mc_passes: number of stochastic forward passes
        
        Returns:
            dict with metrics, uncertainties, etc.
        """
        logger.info(f"Evaluating on test set ({len(test_loader.dataset)} samples)")
        logger.info(f"MC-Dropout passes: {mc_passes}")
        
        all_labels = []
        all_predictions = []
        all_logits = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader)):
                x = batch['image'].to(self.device)
                y = batch['label'].to(self.device)
                
                all_labels.append(y.cpu().numpy())
                
                # MC-Dropout passes
                mc_logits = []
                
                self.discriminator.train()  # Keep dropout enabled
                for mc_pass in range(mc_passes):
                    class_logits, _ = self.discriminator(x)
                    mc_logits.append(class_logits.cpu().numpy())
                
                self.discriminator.eval()
                
                # Stack: (mc_passes, batch_size, num_classes)
                mc_logits = np.stack(mc_logits, axis=0)
                
                # Mean and variance
                mean_logits = mc_logits.mean(axis=0)
                uncertainty = mc_logits.var(axis=0)
                
                all_logits.append(mean_logits)
                all_uncertainties.append(uncertainty)
                all_predictions.append(mean_logits.argmax(axis=1))
        
        # Concatenate
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_predictions)
        y_logits = np.concatenate(all_logits)
        y_uncertainties = np.concatenate(all_uncertainties)
        
        # Softmax probabilities
        y_probs = self._softmax(y_logits)
        
        # Classification metrics
        metrics = self.evaluator.evaluate_classification(y_true, y_pred, y_probs)
        
        # ECE (Expected Calibration Error)
        y_logits_torch = torch.from_numpy(y_logits).float()
        y_true_torch = torch.from_numpy(y_true).long()
        y_probs_torch = torch.from_numpy(y_probs).float()
        
        ece = self.evaluator.compute_ece(y_true_torch, y_probs_torch, n_bins=10)
        
        # Abstention curve
        coverage, accuracy = self.evaluator.abstention_curve(
            y_true_torch,
            y_probs_torch,
            torch.from_numpy(y_uncertainties.max(axis=1))
        )
        
        metrics['ece'] = ece
        metrics['mean_uncertainty'] = y_uncertainties.mean()
        metrics['max_uncertainty'] = y_uncertainties.max()
        metrics['coverage_at_95_acc'] = self._find_coverage_at_accuracy(
            coverage, accuracy, target_acc=0.95
        )
        
        return {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_probs': y_probs,
            'uncertainties': y_uncertainties,
            'coverage': coverage,
            'accuracy_retained': accuracy
        }
    
    def _softmax(self, logits):
        """Compute softmax from logits."""
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)
    
    def _find_coverage_at_accuracy(self, coverage, accuracy, target_acc=0.95):
        """Find coverage required to achieve target accuracy."""
        for cov, acc in zip(coverage, accuracy):
            if acc >= target_acc:
                return cov
        return coverage[-1]
    
    def print_results(self, results):
        """Print evaluation results."""
        metrics = results['metrics']
        
        logger.info("\n" + "="*60)
        logger.info("EVALUATION RESULTS")
        logger.info("="*60)
        
        logger.info("\nClassification Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        logger.info(f"  F1 (weighted): {metrics['f1']:.4f}")
        
        if 'auc' in metrics:
            logger.info(f"  AUC-ROC: {metrics['auc']:.4f}")
        
        logger.info("\nCalibration (MC-Dropout, 20 passes):")
        logger.info(f"  ECE: {metrics['ece']:.4f}")
        logger.info(f"  Mean Uncertainty: {metrics['mean_uncertainty']:.4f}")
        logger.info(f"  Max Uncertainty: {metrics['max_uncertainty']:.4f}")
        logger.info(f"  Coverage @ 95% Accuracy: {metrics['coverage_at_95_acc']:.4f}")
        
        logger.info("\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        logger.info(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        logger.info(f"  FN={cm[1,0]}, TP={cm[1,1]}")
        
        logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SemiGAN-MelanoPath checkpoint'
    )
    parser.add_argument('--ckpt', required=True,
                       help='Path to checkpoint .pt file')
    parser.add_argument('--test_data', default='./data/BreakHis',
                       help='Path to test dataset')
    parser.add_argument('--dataset', choices=['breakhis', 'mhist'],
                       default='breakhis', help='Dataset type')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mc_passes', type=int, default=20,
                       help='Number of MC-Dropout passes')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--config', default='configs/default.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    if args.dataset == 'breakhis':
        dataset = BreakHisDataset(root_dir=args.test_data)
    else:
        dataset = MHISTDataset(root_dir=args.test_data)
    
    # Create dataloader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate
    evaluator = CheckpointEvaluator(args.ckpt, device=args.device)
    evaluator.load_checkpoint(config)
    
    results = evaluator.evaluate_test_set(test_loader, mc_passes=args.mc_passes)
    evaluator.print_results(results)
    
    # Save results
    import json
    results_file = Path(args.ckpt).parent.parent / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'metrics': results['metrics'],
            'coverage_at_95_acc': results['metrics']['coverage_at_95_acc']
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
