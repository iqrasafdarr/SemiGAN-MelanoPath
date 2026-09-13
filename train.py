import os
import sys
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import itertools

from data.loaders import BreakHisDataset, get_train_val_split, create_loaders
from models.architectures import Generator, Discriminator
from models.losses import (
    VAT, SupervisedCE, RotationPredictionLoss, AdversarialLoss,
    FeatureMatchingLoss, NoiseInjection, create_rotations
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemiGANTrainer:
    """Main trainer for SemiGAN-MelanoPath v2."""
    
    def __init__(self, config_path, exp_name, label_pct, seed):
        self.seed = seed
        self.label_pct = label_pct
        self.exp_name = exp_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self._set_seed()
        self._setup_paths()
        self._init_models()
        self._init_optimizers()
        self._init_losses()
    
    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
    
    def _setup_paths(self):
        self.exp_dir = Path(f"runs/{self.exp_name}_lbl{self.label_pct}_s{self.seed}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.sample_dir = self.exp_dir / "samples"
        self.ckpt_dir.mkdir(exist_ok=True)
        self.sample_dir.mkdir(exist_ok=True)
    
    def _init_models(self):
        # Generator
        self.G = Generator(
            z_dim=self.config['generator']['z_dim'],
            base_channels=self.config['generator']['base_channels'],
            use_spectral_norm=self.config['generator']['spectral_norm']
        ).to(self.device)
        
        # Discriminator
        self.D = Discriminator(
            base_channels=self.config['discriminator']['base_channels'],
            num_classes=self.config['data']['num_classes'],
            num_rotations=self.config['discriminator']['num_rotations'],
            classifier_dropout=self.config['discriminator']['classifier_dropout'],
            use_spectral_norm=self.config['discriminator']['spectral_norm']
        ).to(self.device)
        
        # Input noise
        self.noise_layer = NoiseInjection(self.config['losses']['noise_sigma'])
    
    def _init_optimizers(self):
        g_lr = self.config['optimizer']['lr']
        d_lr = self.config['optimizer']['lr']
        betas = tuple(self.config['optimizer']['betas'])
        
        self.opt_G = optim.Adam(self.G.parameters(), lr=g_lr, betas=betas,
                               weight_decay=self.config['optimizer']['g_weight_decay'])
        
        # L2 weight decay ONLY on D
        self.opt_D = optim.Adam(self.D.parameters(), lr=d_lr, betas=betas,
                               weight_decay=self.config['optimizer']['d_weight_decay'])
    
    def _init_losses(self):
        self.supervised_ce = SupervisedCE()
        self.rotation_ce = RotationPredictionLoss()
        self.adversarial_loss = AdversarialLoss()
        self.feature_match = FeatureMatchingLoss()
        self.vat = VAT(
            eps=self.config['losses']['vat_eps'],
            beta=self.config['losses']['vat_beta']
        )
    
    def train(self, train_dataset, val_dataset=None):
        """Train SemiGAN-MelanoPath v2."""
        # Split into labeled/unlabeled
        split_info = get_train_val_split(
            train_dataset, 
            label_pct=self.label_pct,
            seed=self.seed
        )
        
        logger.info(f"Label %: {self.label_pct}% | Labeled: {split_info['n_labeled']} | "
                   f"Unlabeled: {split_info['n_unlabeled']}")
        
        labeled_loader, unlabeled_loader = create_loaders(
            train_dataset,
            split_info['labeled_indices'],
            split_info['unlabeled_indices'],
            batch_size=self.config['training']['batch_size']
        )
        
        num_epochs = self.config['training']['num_epochs']
        n_critic = self.config['training']['n_critic']
        
        # Training loop
        results = {
            'train_loss_D': [],
            'train_loss_G': [],
            'epoch': []
        }
        
        for epoch in range(num_epochs):
            epoch_loss_D = 0.0
            epoch_loss_G = 0.0
            n_batches = 0
            
            # Zip labeled and unlabeled loaders (cycle if different lengths)
            labeled_iter = iter(labeled_loader)
            unlabeled_iter = iter(unlabeled_loader)
            
            max_batches = max(len(labeled_loader), len(unlabeled_loader))
            
            pbar = tqdm(range(max_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for _ in pbar:
                # Get batch
                try:
                    labeled_batch = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    labeled_batch = next(labeled_iter)
                
                try:
                    unlabeled_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    unlabeled_batch = next(unlabeled_iter)
                
                # Train Discriminator
                for _ in range(n_critic):
                    loss_D = self._train_discriminator(labeled_batch, unlabeled_batch)
                    epoch_loss_D += loss_D.item()
                
                # Train Generator
                loss_G = self._train_generator()
                epoch_loss_G += loss_G.item()
                
                n_batches += 1
                pbar.update(1)
            
            epoch_loss_D /= (n_batches * n_critic)
            epoch_loss_G /= n_batches
            
            results['epoch'].append(epoch + 1)
            results['train_loss_D'].append(epoch_loss_D)
            results['train_loss_G'].append(epoch_loss_G)
            
            logger.info(f"Epoch {epoch+1}: D_loss={epoch_loss_D:.4f}, G_loss={epoch_loss_G:.4f}")
            
            # Sample generation
            if (epoch + 1) % self.config['logging']['sample_interval'] == 0:
                self._save_samples(epoch)
            
            # Checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)
        
        return results
    
    def _train_discriminator(self, labeled_batch, unlabeled_batch):
        """Single discriminator step."""
        self.D.train()
        self.opt_D.zero_grad()
        
        # Labeled data: supervised classification
        x_labeled = labeled_batch['image'].to(self.device)
        y_labeled = labeled_batch['label'].to(self.device)
        
        x_labeled = self.noise_layer(x_labeled)

        # Keep the real image batch for generator feature matching.
        self._real_batch_for_g = x_labeled.detach()
        class_logits, fake_logits = self.D(x_labeled)
        
        loss_supervised = self.supervised_ce(class_logits, y_labeled)
        loss_D = self.config['losses']['supervised_weight'] * loss_supervised
        
        # Unlabeled data: rotation prediction (self-supervised)
        x_unlabeled = unlabeled_batch['image'].to(self.device)
        x_unlabeled = self.noise_layer(x_unlabeled)
        
        # Create rotations
        x_rot, y_rot = create_rotations(x_unlabeled)
        x_rot = x_rot.to(self.device)
        y_rot = y_rot.to(self.device)
        
        class_logits_rot, rotation_logits, _, _ = self.D(
            x_rot, return_rotation=True
        )
        loss_rotation = self.rotation_ce(rotation_logits, y_rot)
        loss_D += self.config['losses']['rotation_weight'] * loss_rotation
        
        # Adversarial loss (real vs fake)
        z = torch.randn(x_labeled.size(0), self.config['generator']['z_dim']).to(self.device)
        fake_images = self.G(z).detach()
        
        _, fake_logits_fake = self.D(fake_images)
        loss_fake = self.adversarial_loss(fake_logits_fake, torch.zeros(x_labeled.size(0)))
        
        _, fake_logits_real = self.D(x_labeled)
        loss_real = self.adversarial_loss(fake_logits_real, torch.ones(x_labeled.size(0)))
        
        loss_adversarial = (loss_fake + loss_real) / 2
        loss_D += self.config['losses']['adversarial_weight'] * loss_adversarial
        
        # VAT on unlabeled data
        def logit_fn(x):
            class_logits, _ = self.D(x)
            return class_logits
        
        loss_vat = self.vat(self.D, x_unlabeled, logit_fn)
        loss_D += self.config['losses']['vat_weight'] * loss_vat
        
        loss_D.backward()
        self.opt_D.step()
        
        return loss_D
    
    def _train_generator(self):
        """Single generator step."""
        self.G.train()
        self.opt_G.zero_grad()
        
        z = torch.randn(self.config['training']['batch_size'], 
                       self.config['generator']['z_dim']).to(self.device)
        fake_images = self.G(z)
        
        # Feature matching: compare generated features with real image features.
        with torch.no_grad():
            real_feat, _, _ = self.D(
                self._real_batch_for_g,
                return_features=True
            )

        fake_feat, _, fake_logits_fake = self.D(
            fake_images,
            return_features=True
        )
        loss_feature_match = self.feature_match(fake_feat, real_feat)
        
        # Adversarial loss
        loss_adversarial = self.adversarial_loss(fake_logits_fake, torch.ones(z.size(0)))
        
        loss_G = (
            self.config['losses']['feature_match_weight'] * loss_feature_match +
            self.config['losses']['adversarial_weight'] * loss_adversarial
        )
        
        loss_G.backward()
        self.opt_G.step()
        
        return loss_G
    
    def _save_samples(self, epoch):
        """Save generated sample grid."""
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(16, self.config['generator']['z_dim']).to(self.device)
            fake_images = self.G(z)
        
        # Save as images (dummy implementation)
        torch.save(fake_images, self.sample_dir / f"samples_epoch_{epoch+1}.pt")
    
    def _save_checkpoint(self, epoch):
        """Save model checkpoint."""
        ckpt = {
            'epoch': epoch,
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D': self.opt_D.state_dict()
        }
        torch.save(ckpt, self.ckpt_dir / f"ckpt_epoch_{epoch+1}.pt")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--exp', default='semigan_v2')
    parser.add_argument('--label_pct', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    trainer = SemiGANTrainer(args.config, args.exp, args.label_pct, args.seed)
    
    dataset = BreakHisDataset(
        root_dir=trainer.config['data']['breakhis_root'],
        transform=None
    )
    
    results = trainer.train(dataset)
    
    # Save results
    with open(trainer.exp_dir / 'result.txt', 'w') as f:
        for i, (epoch, loss_d, loss_g) in enumerate(
            zip(results['epoch'], results['train_loss_D'], results['train_loss_G'])
        ):
            f.write(f"{epoch}\t{loss_d:.6f}\t{loss_g:.6f}\n")


if __name__ == '__main__':
    main()



