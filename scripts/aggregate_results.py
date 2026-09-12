#!/usr/bin/env python3
"""
Aggregate results from multiple experiments and seeds.
Computes mean±std for E1-E5 across 3 seeds.
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys


def aggregate_e1_label_scarcity(results_dir='runs'):
    """Aggregate E1 label scarcity results across seeds."""
    print("\n" + "="*70)
    print("E1: Label Scarcity Sweep (100% → 1%)")
    print("="*70)
    
    # Find all E1 result files
    e1_files = glob.glob(os.path.join(results_dir, 'E1_label_scarcity_*.csv'))
    
    if not e1_files:
        print("No E1 results found.")
        return None
    
    # Combine
    dfs = [pd.read_csv(f) for f in e1_files]
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Aggregate by label percentage
    summary = df_combined.groupby('label_pct').agg({
        'accuracy': ['mean', 'std', 'count']
    }).round(4)
    
    print(summary)
    return summary


def aggregate_e2_domain_shift(results_dir='runs'):
    """Aggregate E2 domain shift results."""
    print("\n" + "="*70)
    print("E2: Domain Shift (BreakHis → MHIST)")
    print("="*70)
    
    e2_files = glob.glob(os.path.join(results_dir, 'E2_domain_shift_*.json'))
    
    if not e2_files:
        print("No E2 results found.")
        return None
    
    results = defaultdict(list)
    
    for f in e2_files:
        with open(f) as fp:
            data = json.load(fp)
            for key in data.keys():
                if key in ['setting', 'accuracy', 'auc', 'f1']:
                    continue
                if isinstance(data[key], (list, dict)):
                    for i, val in enumerate(data[key]):
                        if isinstance(val, (int, float)):
                            results[f"{key}_{i}"].append(val)
    
    print("Zero-shot accuracy:", np.mean(results.get('accuracy_0', [0])))
    print("Fine-tuned accuracy (+1%):", np.mean(results.get('accuracy_1', [0])))
    print("Transfer gap:", 
          np.mean(results.get('accuracy_1', [0])) - np.mean(results.get('accuracy_0', [0])))
    
    return results


def aggregate_e3_label_budget(results_dir='runs'):
    """Aggregate E3 label budget results."""
    print("\n" + "="*70)
    print("E3: Label Budget Calculator")
    print("="*70)
    
    e3_files = glob.glob(os.path.join(results_dir, 'E3_label_budget_*.csv'))
    
    if not e3_files:
        print("No E3 results found.")
        return None
    
    dfs = [pd.read_csv(f) for f in e3_files]
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Show unique label percentages and hours saved
    summary = df_combined[['label_pct', 'hours_saved', 'accuracy']].drop_duplicates()
    summary = summary.sort_values('label_pct')
    
    print(summary.to_string(index=False))
    return summary


def aggregate_e4_ablations(results_dir='runs'):
    """Aggregate E4 ablation results."""
    print("\n" + "="*70)
    print("E4: Ablation Study (10% labels)")
    print("="*70)
    
    e4_files = glob.glob(os.path.join(results_dir, 'E4_ablations_*.csv'))
    
    if not e4_files:
        print("No E4 results found.")
        return None
    
    dfs = [pd.read_csv(f) for f in e4_files]
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Aggregate by model variant
    summary = df_combined.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'std': 'mean'
    }).round(4)
    
    print(summary)
    return summary


def aggregate_e5_uncertainty(results_dir='runs'):
    """Aggregate E5 uncertainty results."""
    print("\n" + "="*70)
    print("E5: Uncertainty & Calibration (MC-Dropout)")
    print("="*70)
    
    e5_files = glob.glob(os.path.join(results_dir, 'E5_uncertainty_*.json'))
    
    if not e5_files:
        print("No E5 results found.")
        return None
    
    ece_list = []
    uncertainty_list = []
    coverage_95_list = []
    
    for f in e5_files:
        with open(f) as fp:
            data = json.load(fp)
            if 'ece' in data:
                ece_list.append(float(data['ece']))
            if 'mean_uncertainty' in data:
                uncertainty_list.append(float(data['mean_uncertainty']))
    
    if ece_list:
        print(f"ECE (mean±std): {np.mean(ece_list):.4f}±{np.std(ece_list):.4f}")
    if uncertainty_list:
        print(f"Mean Uncertainty: {np.mean(uncertainty_list):.4f}±{np.std(uncertainty_list):.4f}")
    
    return {
        'ece': ece_list,
        'uncertainty': uncertainty_list
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Aggregate SemiGAN-MelanoPath results'
    )
    parser.add_argument('--results_dir', default='runs',
                       help='Directory containing results')
    parser.add_argument('--exp', choices=['all', 'e1', 'e2', 'e3', 'e4', 'e5'],
                       default='all', help='Which experiment to aggregate')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SemiGAN-MelanoPath v2: Results Aggregation")
    print("="*70)
    
    if args.exp in ['all', 'e1']:
        aggregate_e1_label_scarcity(args.results_dir)
    
    if args.exp in ['all', 'e2']:
        aggregate_e2_domain_shift(args.results_dir)
    
    if args.exp in ['all', 'e3']:
        aggregate_e3_label_budget(args.results_dir)
    
    if args.exp in ['all', 'e4']:
        aggregate_e4_ablations(args.results_dir)
    
    if args.exp in ['all', 'e5']:
        aggregate_e5_uncertainty(args.results_dir)
    
    print("\n" + "="*70)
    print("Aggregation complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
