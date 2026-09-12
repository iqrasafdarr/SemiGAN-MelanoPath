#!/usr/bin/env python3
"""Inspect saved checkpoints and training logs."""

import torch
import sys
from pathlib import Path
import argparse


def inspect_checkpoint(ckpt_path):
    """Load and inspect checkpoint contents."""
    print(f"\n{'='*70}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*70}\n")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    print(f"Epoch: {ckpt.get('epoch', 'Unknown')}")
    print(f"\nCheckpoint keys: {list(ckpt.keys())}\n")
    
    if 'G' in ckpt:
        print("Generator State Dict:")
        print(f"  Num parameters: {sum(p.numel() for p in ckpt['G'].values() if isinstance(p, torch.Tensor))}")
        print(f"  Keys: {list(ckpt['G'].keys())[:5]}... (showing first 5)\n")
    
    if 'D' in ckpt:
        print("Discriminator State Dict:")
        print(f"  Num parameters: {sum(p.numel() for p in ckpt['D'].values() if isinstance(p, torch.Tensor))}")
        print(f"  Keys: {list(ckpt['D'].keys())[:5]}... (showing first 5)\n")
    
    if 'opt_G' in ckpt:
        print("Generator Optimizer State:")
        print(f"  Keys: {list(ckpt['opt_G'].keys())}\n")
    
    if 'opt_D' in ckpt:
        print("Discriminator Optimizer State:")
        print(f"  Keys: {list(ckpt['opt_D'].keys())}\n")
    
    print(f"{'='*70}\n")


def inspect_training_log(log_path):
    """Parse and display training log."""
    print(f"\n{'='*70}")
    print(f"Training Log: {log_path}")
    print(f"{'='*70}\n")
    
    try:
        with open(log_path) as f:
            lines = f.readlines()
        
        if not lines:
            print("Log is empty.")
            return
        
        print(f"Total epochs: {len(lines)}")
        print(f"\nFirst 5 epochs:")
        print("Epoch\tLoss_D\t\tLoss_G")
        print("-" * 50)
        
        for i, line in enumerate(lines[:5]):
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                print(f"{parts[0]}\t{parts[1]}\t\t{parts[2]}")
        
        if len(lines) > 5:
            print("\n...\n")
            print(f"Last 5 epochs:")
            print("Epoch\tLoss_D\t\tLoss_G")
            print("-" * 50)
            for line in lines[-5:]:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    print(f"{parts[0]}\t{parts[1]}\t\t{parts[2]}")
        
        # Compute statistics
        losses_d = []
        losses_g = []
        
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                try:
                    losses_d.append(float(parts[1]))
                    losses_g.append(float(parts[2]))
                except:
                    pass
        
        if losses_d:
            print(f"\nD Loss Statistics:")
            print(f"  Min: {min(losses_d):.6f} (epoch {losses_d.index(min(losses_d))+1})")
            print(f"  Max: {max(losses_d):.6f}")
            print(f"  Mean: {sum(losses_d)/len(losses_d):.6f}")
        
        if losses_g:
            print(f"\nG Loss Statistics:")
            print(f"  Min: {min(losses_g):.6f} (epoch {losses_g.index(min(losses_g))+1})")
            print(f"  Max: {max(losses_g):.6f}")
            print(f"  Mean: {sum(losses_g)/len(losses_g):.6f}")
        
        print(f"\n{'='*70}\n")
    
    except Exception as e:
        print(f"Error parsing log: {e}\n")


def list_experiment_checkpoints(exp_dir):
    """List all checkpoints in an experiment directory."""
    exp_path = Path(exp_dir)
    
    if not exp_path.exists():
        print(f"Experiment directory not found: {exp_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"Experiment: {exp_dir}")
    print(f"{'='*70}\n")
    
    # Checkpoints
    ckpt_dir = exp_path / 'checkpoints'
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob('*.pt'))
        print(f"Checkpoints ({len(ckpts)}):")
        for ckpt in ckpts[:10]:  # Show first 10
            size_mb = ckpt.stat().st_size / 1e6
            print(f"  {ckpt.name} ({size_mb:.1f}MB)")
        if len(ckpts) > 10:
            print(f"  ... and {len(ckpts)-10} more")
    
    # Samples
    sample_dir = exp_path / 'samples'
    if sample_dir.exists():
        samples = sorted(sample_dir.glob('*.pt'))
        print(f"\nGenerated Samples ({len(samples)}):")
        for sample in samples[:5]:  # Show first 5
            size_mb = sample.stat().st_size / 1e6
            print(f"  {sample.name} ({size_mb:.1f}MB)")
        if len(samples) > 5:
            print(f"  ... and {len(samples)-5} more")
    
    # Results
    result_file = exp_path / 'result.txt'
    if result_file.exists():
        print(f"\nTraining Log: {result_file.name}")
        inspect_training_log(result_file)
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Inspect SemiGAN checkpoints')
    parser.add_argument('--ckpt', help='Path to checkpoint .pt file')
    parser.add_argument('--log', help='Path to training log .txt file')
    parser.add_argument('--exp', help='Path to experiment directory')
    args = parser.parse_args()
    
    if not any([args.ckpt, args.log, args.exp]):
        parser.print_help()
        return
    
    if args.ckpt:
        inspect_checkpoint(args.ckpt)
    
    if args.log:
        inspect_training_log(args.log)
    
    if args.exp:
        list_experiment_checkpoints(args.exp)


if __name__ == '__main__':
    main()
