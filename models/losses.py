import torch
import torch.nn as nn
import torch.nn.functional as F


class VAT(nn.Module):
    """
    Virtual Adversarial Training (VAT) for semi-supervised learning.
    Finds worst-case perturbation that maximizes KL divergence, then
    adds consistency loss to minimize it.
    
    Key reference: Miyato et al., "Virtual Adversarial Training for SSL"
    """
    
    def __init__(self, eps=1.0, beta=0.1, num_power_iter=1):
        super().__init__()
        self.eps = eps  # perturbation magnitude
        self.beta = beta  # power iteration step size
        self.num_power_iter = num_power_iter
    
    def forward(self, discriminator, x, logit_model_fn):
        """
        Args:
            discriminator: the model to regularize
            x: input image tensor (B, C, H, W)
            logit_model_fn: function that returns logits given image
        
        Returns:
            vat_loss: scalar consistency loss
        """
        # Get original logits (detached)
        with torch.no_grad():
            logit_orig = logit_model_fn(x).detach()
            prob_orig = F.softmax(logit_orig, dim=1).detach()
        
        # Initialize random perturbation
        d = torch.randn_like(x)
        d = d / (torch.norm(d, p=2, dim=[1, 2, 3], keepdim=True) + 1e-12)
        
        # Power iteration to find worst-case perturbation
        for _ in range(self.num_power_iter):
            d.requires_grad_(True)
            logit_perturbed = logit_model_fn(x + self.beta * d)
            prob_perturbed = F.softmax(logit_perturbed, dim=1)
            
            # KL divergence: KL(p_orig || p_perturbed)
            kl = F.kl_div(
                F.log_softmax(logit_perturbed, dim=1),
                prob_orig,
                reduction='batchmean'
            )
            
            grad = torch.autograd.grad(kl, d, create_graph=True)[0]
            d = grad / (torch.norm(grad, p=2, dim=[1, 2, 3], keepdim=True) + 1e-12)
            d = d.detach()
        
        # Final adversarial perturbation
        r_vat = self.eps * d
        
        # Consistency loss: minimize KL(p(x) || p(x + r_vat))
        with torch.no_grad():
            logit_perturbed = logit_model_fn(x + r_vat)
            prob_perturbed = F.softmax(logit_perturbed, dim=1)
        
        logit_orig_2 = logit_model_fn(x)
        vat_loss = F.kl_div(
            F.log_softmax(logit_orig_2, dim=1),
            prob_perturbed,
            reduction='batchmean'
        )
        
        return vat_loss


class SupervisedCE(nn.Module):
    """Supervised cross-entropy loss on labeled data."""
    
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels):
        """
        Args:
            logits: (B, num_classes) class logits
            labels: (B,) ground truth labels
        """
        return self.ce(logits, labels)


class RotationPredictionLoss(nn.Module):
    """Self-supervised rotation prediction loss."""
    
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, rotation_logits, rotation_labels):
        """
        Args:
            rotation_logits: (B, num_rotations) rotation class logits
            rotation_labels: (B,) rotation class labels
        """
        return self.ce(rotation_logits, rotation_labels)


class AdversarialLoss(nn.Module):
    """GAN adversarial loss (BCE for real/fake discrimination)."""
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, fake_logits, is_real):
        """
        Args:
            fake_logits: (B, 1) output from real/fake head
            is_real: (B,) binary labels (1 for real, 0 for fake)
        """
        return self.bce(fake_logits, is_real.float().unsqueeze(1))


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for generator (Salimans et al.)."""
    
    def __init__(self):
        super().__init__()
        self.l2 = nn.MSELoss()
    
    def forward(self, fake_features, real_features):
        """
        Args:
            fake_features: (B, C) features from discriminator on fake images
            real_features: (B, C) features from discriminator on real images
        
        Returns:
            feature_match_loss: scalar loss
        """
        # Match mean features
        real_mean = real_features.mean(dim=0)
        fake_mean = fake_features.mean(dim=0)
        return self.l2(fake_mean, real_mean)


class NoiseInjection(nn.Module):
    """Input noise injection for regularization."""
    
    def __init__(self, sigma=0.05):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, x):
        """Add Gaussian noise to input."""
        if self.training:
            return x + torch.randn_like(x) * self.sigma
        return x


def create_rotations(x):
    """
    Create 4 rotations of input image: 0, 90, 180, 270 degrees.
    
    Args:
        x: (B, C, H, W) image tensor
    
    Returns:
        rotations: (4*B, C, H, W) concatenated rotations
        rotation_labels: (4*B,) rotation class labels
    """
    rot_0 = x
    rot_90 = torch.rot90(x, 1, [2, 3])
    rot_180 = torch.rot90(x, 2, [2, 3])
    rot_270 = torch.rot90(x, 3, [2, 3])
    
    rotations = torch.cat([rot_0, rot_90, rot_180, rot_270], dim=0)
    labels = torch.cat([
        torch.zeros(x.size(0), dtype=torch.long),
        torch.ones(x.size(0), dtype=torch.long),
        torch.full((x.size(0),), 2, dtype=torch.long),
        torch.full((x.size(0),), 3, dtype=torch.long)
    ], dim=0)
    
    return rotations, labels


def kl_divergence(p, q):
    """Compute KL(p || q) where p and q are probability distributions."""
    return (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(dim=1).mean()
