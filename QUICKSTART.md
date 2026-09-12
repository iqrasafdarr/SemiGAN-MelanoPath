# SemiGAN-MelanoPath v2: Quick Start Guide

**Semi-supervised GAN for Label-Scarce Histopathology** | MITACS 54710

---

## 1. Installation (< 5 minutes)

```bash
# Clone and enter directory
cd semigan-melanopathy

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torchvision; print('✓ torchvision ok')"
```

---

## 2. Prepare Data

### Option A: Use Real Datasets
```bash
# BreakHis (training)
mkdir -p data/BreakHis
# Download from: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database/
# Extract to data/BreakHis/

# MHIST (external validation)
mkdir -p data/MHIST
# Download from: https://github.com/adamri/mhist
# Extract to data/MHIST/
```

### Option B: Use Dummy Data (for testing)
No need to download; code auto-generates synthetic data for testing:
```bash
python train.py --label_pct 100 --seed 42
```

---

## 3. Train a Model (< 1 hour on GPU)

### Single Training Run (100% labels)
```bash
python train.py \
    --config configs/default.yaml \
    --exp my_experiment \
    --label_pct 100 \
    --seed 42
```

Output saved to: `runs/my_experiment_lbl100_s42/`

### Train with Different Label Scarcity
```bash
# 10% labels
python train.py --exp my_exp --label_pct 10 --seed 42

# 1% labels (extreme scarcity)
python train.py --exp my_exp --label_pct 1 --seed 42
```

---

## 4. Run Full Experiment Suite (E1-E5)

Orchestrates all 5 experiments with 3 seeds each (~4-6 hours on GPU):

```bash
# Run all experiments
python run_experiments.py --exp all --results_dir ./runs

# Or run individual experiments
python run_experiments.py --exp e1  # Label scarcity (E1)
python run_experiments.py --exp e2  # Domain shift (E2)
python run_experiments.py --exp e3  # Label budget (E3)
python run_experiments.py --exp e4  # Ablations (E4)
python run_experiments.py --exp e5  # Uncertainty (E5)
```

Results auto-saved to: `runs/E*_*.csv` and `runs/SUMMARY_*.txt`

---

## 5. Evaluate Checkpoint

After training, evaluate a checkpoint on test data with MC-Dropout uncertainty:

```bash
python eval/evaluate_checkpoint.py \
    --ckpt runs/my_experiment_lbl100_s42/checkpoints/ckpt_epoch_50.pt \
    --test_data ./data/BreakHis \
    --dataset breakhis \
    --mc_passes 20 \
    --batch_size 32
```

Output: classification metrics + ECE + calibration curve

---

## 6. Aggregate & Visualize Results

### Aggregate Results from Multiple Seeds
```bash
python scripts/aggregate_results.py --results_dir ./runs --exp all
```

Outputs mean±std for all experiments.

### Visualize in Jupyter
```bash
jupyter notebook notebooks/analysis.ipynb
```

Generates plots for:
- E1: Label scarcity curve
- E2: Domain shift bars
- E3: Label budget ROI
- E4: Ablation rankings
- E5: Reliability diagram + abstention curve

---

## 7. Typical Workflow

### Reproducing Paper Results (4-6 hours)
```bash
# Run full suite
python run_experiments.py --exp all

# Wait for completion...

# Aggregate
python scripts/aggregate_results.py --results_dir ./runs

# Visualize
jupyter notebook notebooks/analysis.ipynb
```

Results summary: `runs/SUMMARY_<timestamp>.txt`

### Quick Test (5 minutes)
```bash
# Test on dummy data
python train.py --label_pct 100 --seed 42

# Should complete in ~1-2 minutes
# Check: runs/semigan_v2_lbl100_s42/result.txt
```

### Custom Experiment
```bash
# Train with 25% labels, 3 seeds
for seed in 42 123 456; do
    python train.py --exp custom_exp --label_pct 25 --seed $seed &
done
wait

# Evaluate
python scripts/aggregate_results.py --results_dir ./runs
```

---

## 8. Key Files & Structure

```
semigan-melanopathy/
├── train.py               ← Main training script
├── run_experiments.py     ← E1-E5 orchestrator
├── README.md              ← Full documentation
├── QUICKSTART.md          ← This file
├── configs/
│   └── default.yaml       ← Hyperparameters (tweak here!)
├── models/
│   ├── architectures.py   ← Generator + Discriminator (dual heads)
│   └── losses.py          ← VAT + SSL + GAN losses
├── data/
│   └── loaders.py         ← BreakHis/MHIST dataset loaders
├── eval/
│   ├── evaluator.py       ← E1-E5 experiments
│   └── evaluate_checkpoint.py  ← Checkpoint eval + MC-Dropout
├── notebooks/
│   └── analysis.ipynb     ← Visualization & results parsing
├── scripts/
│   └── aggregate_results.py    ← Result aggregation
└── runs/                  ← All results saved here
    ├── E1_label_scarcity_*.csv
    ├── E2_domain_shift_*.json
    ├── E4_ablations_*.csv
    └── SUMMARY_*.txt
```

---

## 9. Hyperparameter Tuning

Edit `configs/default.yaml`:

```yaml
losses:
  vat_weight: 3.0        # VAT strength (↑ = more regularization)
  rotation_weight: 0.5   # SSL rotation (↑ = stronger SSL)
  feature_match_weight: 10.0
  
optimizer:
  lr: 2.0e-4            # Learning rate
  d_weight_decay: 1.0e-4 # L2 on discriminator only
```

Then retrain:
```bash
python train.py --config configs/default.yaml --label_pct 10 --seed 42
```

---

## 10. Common Issues & Fixes

### "CUDA out of memory"
```bash
# Reduce batch size in configs/default.yaml
batch_size: 16  # Default: 32
```

### "No data found (using dummy data)"
→ Either download real datasets OR just run test with dummy data (it's fine!)

### "Results not aggregating"
```bash
# Check file paths
ls runs/E1_label_scarcity_*.csv
python scripts/aggregate_results.py --results_dir ./runs
```

### Checkpoint not loading
```bash
# Verify file exists
ls -lh runs/my_exp_lbl100_s42/checkpoints/

# Re-evaluate with explicit path
python eval/evaluate_checkpoint.py --ckpt ./runs/my_exp_lbl100_s42/checkpoints/ckpt_epoch_50.pt
```

---

## 11. Expected Outputs

After running `python train.py --label_pct 100 --seed 42`:

```
runs/semigan_v2_lbl100_s42/
├── checkpoints/
│   ├── ckpt_epoch_10.pt
│   ├── ckpt_epoch_20.pt
│   └── ... (every 10 epochs)
├── samples/
│   ├── samples_epoch_5.pt
│   └── ... (every 5 epochs)
└── result.txt
   (epoch, loss_D, loss_G per line)
```

After running `python run_experiments.py --exp all`:

```
runs/
├── E1_label_scarcity_20240115_120345.csv
├── E2_domain_shift_20240115_121234.json
├── E3_label_budget_20240115_122145.csv
├── E4_ablations_20240115_130456.csv
├── E5_uncertainty_20240115_133012.json
└── SUMMARY_20240115_134512.txt
```

---

## 12. Support & Debugging

### Enable Detailed Logging
```bash
export PYTHONUNBUFFERED=1
python train.py --label_pct 100 --seed 42 2>&1 | tee training.log
```

### Profile GPU Memory
```bash
python -c "import torch; print(torch.cuda.memory_allocated() / 1e9, 'GB')"
```

### Check PyTorch/CUDA
```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name())"
```

---

## 13. Next Steps

1. ✅ Install dependencies
2. ✅ Quick test on dummy data (5 min)
3. ✅ Download BreakHis/MHIST (if available)
4. ✅ Run single experiment (E1 label scarcity)
5. ✅ Run full suite (4-6 hours)
6. ✅ Visualize results (analysis.ipynb)
7. ✅ Customize hyperparameters
8. ✅ Deploy checkpoint on new cohorts

---

**Questions?** → Check `README.md` for detailed methodology & references.

**Paper?** → All results in `runs/SUMMARY_*.txt`

**Reproduce?** → Follow exact seeds (42, 123, 456) and configs/default.yaml
