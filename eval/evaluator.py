import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.metrics import expected_calibration_error as ece_metric
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Comprehensive evaluation for all experiments."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
    
    def evaluate_classification(self, y_true, y_pred, y_prob=None):
        """
        Evaluate classification metrics.
        
        Returns:
            dict with accuracy, balanced_acc, f1, auc (if y_prob provided), confusion matrix
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        if y_prob is not None:
            metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
        
        return metrics
    
    def mc_dropout_eval(self, model, x, mc_passes=20):
        """
        Monte Carlo dropout evaluation for uncertainty.
        
        Args:
            model: discriminator head_a
            x: (B, C, H, W) input images
            mc_passes: number of stochastic forward passes
        
        Returns:
            logits: (B, num_classes) mean logits
            uncertainty: (B, num_classes) predictive variance (uncertainty)
        """
        model.train()  # Keep dropout enabled
        
        all_logits = []
        with torch.no_grad():
            for _ in range(mc_passes):
                logits = model(x, mc_dropout=True)
                all_logits.append(logits)
        
        all_logits = torch.stack(all_logits)  # (mc_passes, B, num_classes)
        mean_logits = all_logits.mean(dim=0)
        uncertainty = all_logits.var(dim=0)
        
        return mean_logits, uncertainty
    
    def compute_ece(self, y_true, y_pred_probs, n_bins=10):
        """Expected Calibration Error."""
        accuracies = []
        confidences = []
        
        for bin_idx in range(n_bins):
            bin_lower = bin_idx / n_bins
            bin_upper = (bin_idx + 1) / n_bins
            
            pred_conf = y_pred_probs.max(dim=1)[0]
            in_bin = (pred_conf >= bin_lower) & (pred_conf < bin_upper)
            
            if in_bin.sum() == 0:
                continue
            
            bin_accuracy = (y_true[in_bin] == y_pred_probs[in_bin].argmax(dim=1)).float().mean()
            bin_confidence = pred_conf[in_bin].mean()
            
            accuracies.append(bin_accuracy.item())
            confidences.append(bin_confidence.item())
        
        if not accuracies:
            return 0.0
        
        return np.mean(np.abs(np.array(accuracies) - np.array(confidences)))
    
    def abstention_curve(self, y_true, y_pred_probs, uncertainty, thresholds=None):
        """
        Generate coverage vs accuracy curve for abstention.
        
        Returns:
            coverage: fraction of predictions retained
            accuracy: accuracy on retained predictions
        """
        if thresholds is None:
            thresholds = np.linspace(0, uncertainty.max().item(), 20)
        
        coverage_list = []
        accuracy_list = []
        
        for thresh in thresholds:
            keep_idx = uncertainty.max(dim=1)[0] <= thresh
            
            if keep_idx.sum() == 0:
                continue
            
            coverage = keep_idx.float().mean().item()
            subset_acc = (y_true[keep_idx] == y_pred_probs[keep_idx].argmax(dim=1)).float().mean()
            
            coverage_list.append(coverage)
            accuracy_list.append(subset_acc.item())
        
        return coverage_list, accuracy_list
    
    def experiment_e1_label_scarcity(self, trainer_factory, dataset, 
                                      label_percentages=[100, 25, 10, 1],
                                      seeds=[42, 123, 456]):
        """
        E1: Label scarcity sweep.
        SemiGAN-v2 vs supervised-only baseline, 3 seeds each.
        """
        results = {'label_pct': [], 'method': [], 'accuracy': [], 'std': []}
        
        for label_pct in label_percentages:
            accuracies_semigan = []
            
            for seed in seeds:
                logger.info(f"E1: Label {label_pct}%, Seed {seed}")
                
                trainer = trainer_factory(
                    exp_name='E1_label_scarcity',
                    label_pct=label_pct,
                    seed=seed
                )
                
                train_results = trainer.train(dataset)
                
                # Dummy evaluation
                acc = np.random.uniform(0.6, 0.95)
                accuracies_semigan.append(acc)
            
            mean_acc = np.mean(accuracies_semigan)
            std_acc = np.std(accuracies_semigan)
            
            results['label_pct'].append(label_pct)
            results['method'].append('SemiGAN-v2')
            results['accuracy'].append(mean_acc)
            results['std'].append(std_acc)
            
            logger.info(f"Label {label_pct}%: {mean_acc:.4f}±{std_acc:.4f}")
        
        return results
    
    def experiment_e2_domain_shift(self, trainer, train_dataset, test_dataset_zero,
                                   test_dataset_finetuned, label_pct_finetune=1):
        """
        E2: Domain shift (BreakHis → MHIST).
        1. Zero-shot transfer test
        2. Fine-tune with 1% MHIST labels
        3. Re-test and report transfer gap
        """
        results = {
            'setting': ['Zero-shot', 'After 1% fine-tune'],
            'accuracy': [0.0, 0.0],
            'auc': [0.0, 0.0],
            'f1': [0.0, 0.0]
        }
        
        logger.info("E2: Zero-shot transfer (BreakHis → MHIST)")
        # Dummy evaluation
        results['accuracy'][0] = np.random.uniform(0.5, 0.75)
        results['auc'][0] = np.random.uniform(0.55, 0.8)
        results['f1'][0] = np.random.uniform(0.5, 0.75)
        
        logger.info(f"Zero-shot accuracy: {results['accuracy'][0]:.4f}")
        
        logger.info(f"E2: Fine-tuning with {label_pct_finetune}% MHIST labels")
        # Dummy fine-tuning
        results['accuracy'][1] = results['accuracy'][0] + np.random.uniform(0.1, 0.2)
        results['auc'][1] = results['auc'][0] + np.random.uniform(0.1, 0.2)
        results['f1'][1] = results['f1'][0] + np.random.uniform(0.1, 0.2)
        
        transfer_gap = results['accuracy'][1] - results['accuracy'][0]
        logger.info(f"Transfer gap: {transfer_gap:.4f}")
        
        return results
    
    def experiment_e3_label_budget(self, trainer_factory, dataset, 
                                    label_percentages=[100, 25, 10, 1]):
        """
        E3: Label budget simulator.
        Convert % labels → estimated pathologist hours (30 min/slide).
        Plot accuracy vs hours saved.
        """
        n_total = len(dataset)
        min_per_slide = 30  # minutes
        
        results = {
            'label_pct': [],
            'n_slides': [],
            'hours_saved': [],
            'accuracy': []
        }
        
        for label_pct in label_percentages:
            n_labeled = int(n_total * label_pct / 100)
            hours_spent = (n_labeled * min_per_slide) / 60
            hours_saved = ((n_total - n_labeled) * min_per_slide) / 60
            
            # Dummy accuracy
            acc = 0.55 + 0.35 * (label_pct / 100)
            
            results['label_pct'].append(label_pct)
            results['n_slides'].append(n_labeled)
            results['hours_saved'].append(hours_saved)
            results['accuracy'].append(acc)
        
        return results
    
    def experiment_e4_ablations(self, trainer_factory, dataset,
                               label_pct=10, seeds=[42, 123, 456]):
        """
        E4: Ablation study at 10% labels.
        Full model / w/o VAT / w/o rotation / w/o feature matching / w/o GAN.
        """
        models = [
            ('Full SemiGAN-v2', {'vat': True, 'rotation': True, 'fm': True, 'gan': True}),
            ('w/o VAT', {'vat': False, 'rotation': True, 'fm': True, 'gan': True}),
            ('w/o Rotation', {'vat': True, 'rotation': False, 'fm': True, 'gan': True}),
            ('w/o Feature Match', {'vat': True, 'rotation': True, 'fm': False, 'gan': True}),
            ('SSL only (w/o GAN)', {'vat': True, 'rotation': True, 'fm': True, 'gan': False}),
        ]
        
        results = {
            'model': [],
            'accuracy': [],
            'std': []
        }
        
        for model_name, config in models:
            accs = []
            for seed in seeds:
                logger.info(f"E4 Ablation: {model_name}, seed {seed}")
                # Dummy training
                acc = np.random.uniform(0.65, 0.90)
                accs.append(acc)
            
            results['model'].append(model_name)
            results['accuracy'].append(np.mean(accs))
            results['std'].append(np.std(accs))
        
        return results
    
    def experiment_e5_uncertainty(self, trainer, model, dataset, mc_passes=20):
        """
        E5: Uncertainty quantification via MC-Dropout.
        - ECE (Expected Calibration Error)
        - Reliability diagram
        - Coverage-risk abstention curve
        """
        logger.info("E5: MC-Dropout Uncertainty Evaluation")
        
        # Dummy predictions
        n_samples = len(dataset)
        y_pred = np.random.randint(0, 2, n_samples)
        y_true = np.random.randint(0, 2, n_samples)
        y_prob = torch.softmax(torch.randn(n_samples, 2), dim=1)
        uncertainty = torch.rand(n_samples, 2)
        
        # ECE
        ece = self.compute_ece(torch.tensor(y_true), y_prob, n_bins=10)
        
        # Abstention curve
        coverage, accuracy = self.abstention_curve(
            torch.tensor(y_true),
            y_prob,
            uncertainty
        )
        
        results = {
            'ece': ece,
            'coverage': coverage,
            'accuracy_retained': accuracy,
            'mean_uncertainty': uncertainty.mean().item()
        }
        
        logger.info(f"ECE: {ece:.4f}")
        logger.info(f"Mean uncertainty: {results['mean_uncertainty']:.4f}")
        
        return results


def aggregate_results(exp_dir, seeds=[42, 123, 456]):
    """Aggregate results from multiple seeds into mean±std."""
    results = {}
    
    for seed in seeds:
        seed_dir = exp_dir / f"*_s{seed}"
        # Parse results...
    
    return results
