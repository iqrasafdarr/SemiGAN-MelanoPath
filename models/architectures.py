import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


class SpectralNorm(object):
    """Spectral normalization wrapper."""
    def __init__(self, module, use_spectral_norm=True):
        if use_spectral_norm:
            self.module = spectral_norm(module)
        else:
            self.module = module
    
    def __call__(self, x):
        return self.module(x)


class Generator(nn.Module):
    """
    DCGAN-style generator with spectral normalization.
    Maps z (100-dim) to 128x128 images.
    """
    
    def __init__(self, z_dim=100, base_channels=32, num_channels=3, 
                 use_spectral_norm=True):
        super().__init__()
        self.z_dim = z_dim
        self.base_channels = base_channels
        
        self.fc = nn.Linear(z_dim, base_channels * 8 * 4 * 4)
        
        self.conv_layers = nn.Sequential(
            # 4x4 -> 8x8
            self._conv_block(base_channels * 8, base_channels * 4, use_spectral_norm),
            # 8x8 -> 16x16
            self._conv_block(base_channels * 4, base_channels * 2, use_spectral_norm),
            # 16x16 -> 32x32
            self._conv_block(base_channels * 2, base_channels, use_spectral_norm),
            # 32x32 -> 64x64
            self._conv_block(base_channels, base_channels // 2, use_spectral_norm),
            # 64x64 -> 128x128
            self._conv_block(base_channels // 2, base_channels // 4, use_spectral_norm),
        )
        
        # Final layer
        conv_final = nn.Conv2d(base_channels // 4, num_channels, 3, padding=1)
        if use_spectral_norm:
            conv_final = spectral_norm(conv_final)
        
        self.conv_layers.add_module('final_conv', conv_final)
        self.conv_layers.add_module('tanh', nn.Tanh())
    
    def _conv_block(self, in_channels, out_channels, use_spectral_norm):
        layers = []
        conv = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        if use_spectral_norm:
            conv = spectral_norm(conv)
        layers.append(conv)
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.base_channels * 8, 4, 4)
        x = self.conv_layers(x)
        return x


class DiscriminatorHeadA(nn.Module):
    """
    Discriminator Head A: K-class classifier with dropout for MC evaluation.
    Predicts class labels (benign/malignant) and optionally rotation angle.
    """
    
    def __init__(self, in_channels, num_classes=2, num_rotations=4, 
                 dropout_rate=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.num_rotations = num_rotations
        self.dropout_rate = dropout_rate
        
        # Classification head
        self.class_fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.Dropout(dropout_rate),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(dropout_rate),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
        # Rotation prediction head (self-supervised)
        self.rotation_fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.Dropout(dropout_rate),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(dropout_rate),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_rotations)
        )
    
    def forward(self, features, return_rotation=False, mc_dropout=False):
        """
        Args:
            features: (B, C) feature vector from shared discriminator
            return_rotation: if True, also return rotation logits
            mc_dropout: if True, keep dropout enabled for MC estimation
        """
        if not mc_dropout:
            self.eval()
        
        class_logits = self.class_fc(features)
        
        if return_rotation:
            rotation_logits = self.rotation_fc(features)
            return class_logits, rotation_logits
        
        return class_logits


class DiscriminatorHeadB(nn.Module):
    """Discriminator Head B: Real/Fake binary classification."""
    
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
    
    def forward(self, features):
        return self.fc(features)


class Discriminator(nn.Module):
    """
    Discriminator with dual heads:
    - Head A: K-class classifier + rotation prediction (self-supervised)
    - Head B: Real/fake binary head
    Shared feature extractor with spectral normalization.
    """
    
    def __init__(self, base_channels=32, num_classes=2, num_rotations=4,
                 classifier_dropout=0.5, use_spectral_norm=True):
        super().__init__()
        self.base_channels = base_channels
        self.use_spectral_norm = use_spectral_norm
        
        # Shared feature extractor
        self.features = self._build_feature_extractor(base_channels, use_spectral_norm)
        
        # Global average pooling then flatten
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        feature_dim = base_channels * 16
        
        # Dual heads
        self.head_a = DiscriminatorHeadA(
            feature_dim, 
            num_classes=num_classes,
            num_rotations=num_rotations,
            dropout_rate=classifier_dropout
        )
        self.head_b = DiscriminatorHeadB(feature_dim)
    
    def _build_feature_extractor(self, base_channels, use_spectral_norm):
        layers = []
        
        # 128x128 -> 64x64
        conv = nn.Conv2d(3, base_channels, 4, stride=2, padding=1)
        if use_spectral_norm:
            conv = spectral_norm(conv)
        layers.append(conv)
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # 64x64 -> 32x32
        conv = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        if use_spectral_norm:
            conv = spectral_norm(conv)
        layers.append(conv)
        layers.append(nn.BatchNorm2d(base_channels * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # 32x32 -> 16x16
        conv = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)
        if use_spectral_norm:
            conv = spectral_norm(conv)
        layers.append(conv)
        layers.append(nn.BatchNorm2d(base_channels * 4))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # 16x16 -> 8x8
        conv = nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=2, padding=1)
        if use_spectral_norm:
            conv = spectral_norm(conv)
        layers.append(conv)
        layers.append(nn.BatchNorm2d(base_channels * 8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # 8x8 -> 4x4
        conv = nn.Conv2d(base_channels * 8, base_channels * 16, 4, stride=2, padding=1)
        if use_spectral_norm:
            conv = spectral_norm(conv)
        layers.append(conv)
        layers.append(nn.BatchNorm2d(base_channels * 16))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, return_features=False, return_rotation=False, 
                mc_dropout=False):
        """
        Args:
            x: input image (B, 3, H, W)
            return_features: if True, return features before heads
            return_rotation: if True, also return rotation logits
            mc_dropout: if True, keep dropout enabled for MC estimation
        
        Returns:
            If return_features=True: (features, class_logits, fake_logits)
            If return_rotation=True: (class_logits, rotation_logits, fake_logits)
            Otherwise: (class_logits, fake_logits)
        """
        feat = self.features(x)
        feat = self.gap(feat)
        feat = feat.view(feat.size(0), -1)
        
        if return_rotation:
            class_logits, rotation_logits = self.head_a(
                feat, return_rotation=True, mc_dropout=mc_dropout
            )
            fake_logits = self.head_b(feat)
            return class_logits, rotation_logits, fake_logits, feat
        
        class_logits = self.head_a(feat, mc_dropout=mc_dropout)
        fake_logits = self.head_b(feat)
        
        if return_features:
            return feat, class_logits, fake_logits
        
        return class_logits, fake_logits
